"""Kafka-to-Kafka PyFlink streaming feature job.

The Flink image intentionally does not bundle a Kafka connector JAR. Install a
compatible connector through scripts/install_flink_kafka_connector.ps1 and
submit this job through scripts/submit_realtime_flink_job.ps1. The local Kafka
feature worker is only a development fallback when this PyFlink job is not
submitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

from pyflink.common import Duration, WatermarkStrategy
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.common.serialization import ByteArraySchema, SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.base import DeliveryGuarantee
from pyflink.datastream.connectors.kafka import KafkaOffsetsInitializer, KafkaRecordSerializationSchema, KafkaSink, KafkaSource
from pyflink.datastream.functions import KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.typeinfo import Types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_metadata import MARKET_USDM, canonical_event_metadata, normalize_message_metadata, normalize_symbol


RAW_TOPICS = ("market.trade.v1", "market.book-ticker.v1", "market.depth.delta.v1")
FEATURE_TOPIC = "market.features.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-only one-minute features with PyFlink Kafka source/sink.")
    parser.add_argument("--bootstrap-servers", default="kafka:29092")
    parser.add_argument("--consumer-group", default="flink-realtime-features-v1")
    parser.add_argument("--checkpoint-ms", type=int, default=60_000)
    parser.add_argument("--max-out-of-orderness-seconds", type=int, default=5)
    parser.add_argument("--allowed-lateness-seconds", type=int, default=10)
    parser.add_argument("--idle-timeout-seconds", type=int, default=30)
    return parser.parse_args()


def minute_start(timestamp_ms: int) -> int:
    return timestamp_ms - (timestamp_ms % 60_000)


def fresh_state(minute: int) -> dict[str, Any]:
    return {
        "minute": minute,
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": 0.0,
        "trade_count": 0,
        "taker_buy_volume": 0.0,
        "taker_sell_volume": 0.0,
        "bid_price": None,
        "ask_price": None,
        "book_imbalance_top5": None,
        "first_trade_time_ms": None,
        "last_trade_time_ms": None,
        "book_ticker_time_ms": None,
        "depth_time_ms": None,
        "closes": [],
    }


def window_deadline(minute: int, allowed_lateness_ms: int) -> int:
    return minute + 60_000 + allowed_lateness_ms


def is_too_late(minute: int, current_watermark: int, allowed_lateness_ms: int) -> bool:
    return current_watermark >= window_deadline(minute, allowed_lateness_ms)


def apply_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    payload = event.get("payload", {})
    event_type = event.get("event_type")
    event_time_ms = int(event["event_time_ms"])
    if event_type in {"agg_trade", "trade"}:
        price = float(payload["price"])
        quantity = float(payload["quantity"])
        if state["first_trade_time_ms"] is None or event_time_ms < int(state["first_trade_time_ms"]):
            state["first_trade_time_ms"] = event_time_ms
            state["open"] = price
        if state["last_trade_time_ms"] is None or event_time_ms >= int(state["last_trade_time_ms"]):
            state["last_trade_time_ms"] = event_time_ms
            state["close"] = price
        state["high"] = price if state["high"] is None else max(float(state["high"]), price)
        state["low"] = price if state["low"] is None else min(float(state["low"]), price)
        state["volume"] += quantity
        state["trade_count"] += 1
        if bool(payload.get("buyer_is_maker")):
            state["taker_sell_volume"] += quantity
        else:
            state["taker_buy_volume"] += quantity
    elif event_type == "book_ticker":
        if state["book_ticker_time_ms"] is None or event_time_ms >= int(state["book_ticker_time_ms"]):
            state["book_ticker_time_ms"] = event_time_ms
            state["bid_price"] = float(payload["bid_price"])
            state["ask_price"] = float(payload["ask_price"])
    elif event_type == "depth_delta" and payload.get("sequence_status") == "synchronized":
        if state["depth_time_ms"] is not None and event_time_ms < int(state["depth_time_ms"]):
            return
        bids = payload.get("top_bids", [])[:5]
        asks = payload.get("top_asks", [])[:5]
        if bids and asks:
            bid_volume = sum(float(level[1]) for level in bids)
            ask_volume = sum(float(level[1]) for level in asks)
            total = bid_volume + ask_volume
            state["depth_time_ms"] = event_time_ms
            state["book_imbalance_top5"] = (bid_volume - ask_volume) / total if total else 0.0


class MarketEventTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, value: str, record_timestamp: int) -> int:
        return int(json.loads(value)["event_time_ms"])


class OneMinuteFeatureFunction(KeyedProcessFunction):
    """Buffer event-time minutes and close them after watermark plus allowed lateness."""

    def __init__(self, allowed_lateness_ms: int):
        self.allowed_lateness_ms = allowed_lateness_ms

    def open(self, runtime_context):
        self.state = runtime_context.get_state(ValueStateDescriptor("minute-feature-state", Types.STRING()))
        metrics = runtime_context.get_metrics_group()
        self.accepted_events = metrics.counter("accepted_events")
        self.dropped_late_events = metrics.counter("dropped_late_events")

    def process_element(self, value: str, ctx: KeyedProcessFunction.Context):
        event = normalize_message_metadata(json.loads(value))
        event_time_ms = int(event["event_time_ms"])
        minute = minute_start(event_time_ms)
        timer_service = ctx.timer_service()
        if is_too_late(minute, timer_service.current_watermark(), self.allowed_lateness_ms):
            self.dropped_late_events.inc()
            return
        root_text = self.state.value()
        root = json.loads(root_text) if root_text else {"minutes": {}, "closes": []}
        minute_key = str(minute)
        minute_state = root["minutes"].get(minute_key, fresh_state(minute))
        apply_event(minute_state, event)
        root["minutes"][minute_key] = minute_state
        self.state.update(json.dumps(root, separators=(",", ":")))
        timer_service.register_event_time_timer(window_deadline(minute, self.allowed_lateness_ms))
        self.accepted_events.inc()

    def on_timer(self, timestamp: int, ctx: KeyedProcessFunction.OnTimerContext):
        root_text = self.state.value()
        if not root_text:
            return
        root = json.loads(root_text)
        ready = sorted(
            int(minute_key)
            for minute_key in root["minutes"]
            if window_deadline(int(minute_key), self.allowed_lateness_ms) <= timestamp
        )
        closes = list(root.get("closes", []))
        for minute in ready:
            minute_state = root["minutes"].pop(str(minute))
            minute_state["closes"] = closes
            feature = self.finish_feature(ctx.get_current_key(), minute_state)
            if minute_state.get("close") is not None:
                closes = (closes + [float(minute_state["close"])])[-5:]
            if feature is not None:
                yield json.dumps(feature, separators=(",", ":"))
        root["closes"] = closes
        if root["minutes"] or closes:
            self.state.update(json.dumps(root, separators=(",", ":")))
        else:
            self.state.clear()

    @staticmethod
    def finish_feature(symbol: str, state: dict[str, Any]) -> dict[str, Any] | None:
        close = state.get("close")
        if close is None or state.get("open") is None:
            return None
        closes = deque(state.get("closes", []), maxlen=5)
        previous_close = closes[-1] if closes else None
        closes.append(float(close))
        bid_price = state.get("bid_price")
        ask_price = state.get("ask_price")
        total_taker = float(state["taker_buy_volume"]) + float(state["taker_sell_volume"])
        metadata = canonical_event_metadata(
            symbol=symbol,
            market=MARKET_USDM,
            timeframe="1m",
            event_time_ms=int(state["minute"]),
        )
        return {
            "schema_version": "realtime_feature_v1",
            "feature_id": f"{normalize_symbol(symbol)}:1m:{state['minute']}",
            "open": state["open"],
            "high": state["high"],
            "low": state["low"],
            "close": close,
            "volume": state["volume"],
            "trade_count": state["trade_count"],
            "taker_volume_imbalance": (
                (float(state["taker_buy_volume"]) - float(state["taker_sell_volume"])) / total_taker
                if total_taker
                else 0.0
            ),
            "book_mid": (float(bid_price) + float(ask_price)) / 2 if bid_price is not None and ask_price is not None else None,
            "book_spread": float(ask_price) - float(bid_price) if bid_price is not None and ask_price is not None else None,
            "book_imbalance_top5": state.get("book_imbalance_top5"),
            "ma_5": sum(closes) / len(closes) if len(closes) == 5 else None,
            "return_1m": float(close) / previous_close - 1.0 if previous_close else None,
            "feature_ready": len(closes) == 5 and previous_close is not None,
            "source": "pyflink_kafka_streaming",
            **metadata,
        }


def main() -> None:
    args = parse_args()
    if args.checkpoint_ms <= 0:
        raise ValueError("--checkpoint-ms must be positive.")
    if args.max_out_of_orderness_seconds < 0 or args.allowed_lateness_seconds < 0:
        raise ValueError("Watermark delay and allowed lateness must not be negative.")
    if args.idle_timeout_seconds <= 0:
        raise ValueError("--idle-timeout-seconds must be positive.")
    env = StreamExecutionEnvironment.get_execution_environment()
    env.enable_checkpointing(args.checkpoint_ms)
    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(args.bootstrap_servers)
        .set_topics(*RAW_TOPICS)
        .set_group_id(args.consumer_group)
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )
    watermark_strategy = (
        WatermarkStrategy.for_bounded_out_of_orderness(
            Duration.of_seconds(args.max_out_of_orderness_seconds)
        )
        .with_timestamp_assigner(MarketEventTimestampAssigner())
        .with_idleness(Duration.of_seconds(args.idle_timeout_seconds))
    )
    stream = env.from_source(source, watermark_strategy, "Kafka raw market source")
    serializer = (
        KafkaRecordSerializationSchema.builder()
        .set_topic(FEATURE_TOPIC)
        .set_value_serialization_schema(ByteArraySchema())
        .build()
    )
    sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(args.bootstrap_servers)
        .set_delivery_guarantee(DeliveryGuarantee.EXACTLY_ONCE)
        .set_transactional_id_prefix("realtime-features-v1")
        .set_property("transaction.timeout.ms", "300000")
        .set_record_serializer(serializer)
        .build()
    )
    features = stream.key_by(lambda value: normalize_symbol(json.loads(value)["symbol"])).process(
        OneMinuteFeatureFunction(args.allowed_lateness_seconds * 1_000)
    )
    features.map(
        lambda feature_json: feature_json.encode("utf-8"),
        output_type=Types.PRIMITIVE_ARRAY(Types.BYTE()),
    ).sink_to(sink)
    env.execute("realtime-kafka-feature-job-paper-only")


if __name__ == "__main__":
    main()
