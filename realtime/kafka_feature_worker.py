"""Development streaming feature worker for the realtime Kafka contract.

This worker is intentionally paper-only. It is a local development bridge while
the Flink 2.2 Kafka connector dependency is provisioned for the cluster.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaTimeoutError

from realtime.contracts import FEATURE_SCHEMA_VERSION, RAW_TOPICS, TOPIC_FEATURES, decode_json, json_bytes, safe_symbol, utc_now_iso
from market_metadata import MARKET_USDM, canonical_event_metadata


@dataclass
class MinuteState:
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume: float = 0.0
    trade_count: int = 0
    taker_buy_volume: float = 0.0
    taker_sell_volume: float = 0.0
    bid_price: float | None = None
    ask_price: float | None = None
    book_imbalance_top5: float | None = None
    event_ids: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-minute paper-only market features from Kafka.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--consumer-group", default="realtime-feature-worker-v1")
    parser.add_argument("--grace-seconds", type=int, default=5)
    parser.add_argument("--from-beginning", action="store_true")
    parser.add_argument("--metrics-path", default="realtime_runtime/feature_worker_metrics.json")
    return parser.parse_args()


def create_consumer(args: argparse.Namespace) -> KafkaConsumer:
    while True:
        try:
            return KafkaConsumer(
                *RAW_TOPICS,
                bootstrap_servers=args.bootstrap_servers,
                group_id=args.consumer_group,
                auto_offset_reset="earliest" if args.from_beginning else "latest",
                enable_auto_commit=False,
                value_deserializer=decode_json,
            )
        except KafkaTimeoutError:
            print("Kafka not ready for feature worker; retrying in 5 seconds.")
            time.sleep(5)


def minute_start(timestamp_ms: int) -> int:
    return timestamp_ms - (timestamp_ms % 60_000)


def update_trade(state: MinuteState, payload: dict[str, Any]) -> None:
    price = float(payload["price"])
    quantity = float(payload["quantity"])
    state.open_price = price if state.open_price is None else state.open_price
    state.high_price = price if state.high_price is None else max(state.high_price, price)
    state.low_price = price if state.low_price is None else min(state.low_price, price)
    state.close_price = price
    state.volume += quantity
    state.trade_count += 1
    if bool(payload.get("buyer_is_maker")):
        state.taker_sell_volume += quantity
    else:
        state.taker_buy_volume += quantity


def update_book_ticker(state: MinuteState, payload: dict[str, Any]) -> None:
    state.bid_price = float(payload["bid_price"])
    state.ask_price = float(payload["ask_price"])


def update_depth(state: MinuteState, payload: dict[str, Any]) -> None:
    if payload.get("sequence_status") != "synchronized":
        return
    bids = payload.get("top_bids", [])[:5]
    asks = payload.get("top_asks", [])[:5]
    if not bids or not asks:
        return
    bid_volume = sum(float(level[1]) for level in bids)
    ask_volume = sum(float(level[1]) for level in asks)
    total = bid_volume + ask_volume
    state.book_imbalance_top5 = (bid_volume - ask_volume) / total if total else 0.0


def build_feature(
    *,
    symbol: str,
    minute: int,
    state: MinuteState,
    closes: deque[float],
) -> dict[str, Any] | None:
    if state.close_price is None or state.open_price is None or state.high_price is None or state.low_price is None:
        return None
    previous_close = closes[-1] if closes else None
    closes.append(state.close_price)
    ma_5 = sum(closes) / len(closes) if len(closes) == 5 else None
    total_taker = state.taker_buy_volume + state.taker_sell_volume
    metadata = canonical_event_metadata(
        symbol=symbol,
        market=MARKET_USDM,
        timeframe="1m",
        event_time_ms=minute,
    )
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_id": f"{symbol}:1m:{minute}",
        "created_at_utc": utc_now_iso(),
        "open": state.open_price,
        "high": state.high_price,
        "low": state.low_price,
        "close": state.close_price,
        "volume": state.volume,
        "trade_count": state.trade_count,
        "taker_volume_imbalance": (
            (state.taker_buy_volume - state.taker_sell_volume) / total_taker if total_taker else 0.0
        ),
        "book_mid": (
            (state.bid_price + state.ask_price) / 2
            if state.bid_price is not None and state.ask_price is not None
            else None
        ),
        "book_spread": (
            state.ask_price - state.bid_price
            if state.bid_price is not None and state.ask_price is not None
            else None
        ),
        "book_imbalance_top5": state.book_imbalance_top5,
        "ma_5": ma_5,
        "return_1m": state.close_price / previous_close - 1.0 if previous_close else None,
        "feature_ready": ma_5 is not None and previous_close is not None,
        "source": "kafka_feature_worker_development_bridge",
        **metadata,
    }


def write_metrics(path: Path, counts: Counter[str], pending_minutes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at_utc": utc_now_iso(),
                "counts": dict(counts),
                "pending_minutes": pending_minutes,
                "mode": "paper_only_development_bridge",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.grace_seconds < 0:
        raise ValueError("--grace-seconds must not be negative.")
    consumer = create_consumer(args)
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        acks="all",
        retries=5,
        key_serializer=lambda value: str(value).encode("utf-8"),
        value_serializer=json_bytes,
    )
    states: dict[str, dict[int, MinuteState]] = defaultdict(dict)
    closes: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))
    counts: Counter[str] = Counter()
    try:
        while True:
            records = consumer.poll(timeout_ms=1_000, max_records=1_000)
            for partition_records in records.values():
                for record in partition_records:
                    event = record.value
                    event_id = str(event.get("event_id", ""))
                    symbol = safe_symbol(str(event.get("symbol", "")))
                    event_time_ms = int(event.get("event_time_ms", 0))
                    if not event_id or not symbol or event_time_ms <= 0:
                        counts["invalid_event"] += 1
                        continue
                    minute = minute_start(event_time_ms)
                    state = states[symbol].setdefault(minute, MinuteState())
                    if event_id in state.event_ids:
                        counts["duplicate_event"] += 1
                        continue
                    state.event_ids.add(event_id)
                    payload = event.get("payload", {})
                    event_type = event.get("event_type")
                    if event_type in {"agg_trade", "trade"}:
                        update_trade(state, payload)
                    elif event_type == "book_ticker":
                        update_book_ticker(state, payload)
                    elif event_type == "depth_delta":
                        update_depth(state, payload)
                    else:
                        counts["unknown_event"] += 1
                        continue
                    counts["raw_event"] += 1

            cutoff = minute_start(int(time.time() * 1000) - args.grace_seconds * 1_000)
            for symbol, symbol_states in list(states.items()):
                completed = sorted(minute for minute in symbol_states if minute < cutoff)
                for minute in completed:
                    feature = build_feature(
                        symbol=symbol,
                        minute=minute,
                        state=symbol_states.pop(minute),
                        closes=closes[symbol],
                    )
                    if feature is None:
                        counts["incomplete_minute"] += 1
                        continue
                    producer.send(TOPIC_FEATURES, key=symbol, value=feature)
                    counts["feature_event"] += 1
            producer.flush(timeout=10)
            consumer.commit()
            write_metrics(
                Path(args.metrics_path),
                counts,
                sum(len(item) for item in states.values()),
            )
    finally:
        producer.flush(timeout=20)
        producer.close(timeout=20)
        consumer.close()


if __name__ == "__main__":
    main()
