"""Send test or Binance USD-M futures OHLCV events to Kafka."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
from kafka import KafkaProducer
from kafka.serializer import Serializer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_metadata import EVENT_ID_SCHEMA_VERSION, canonical_candle_event_id


DEFAULT_TOPIC = "btc_market_events_v1"


class JsonSerializer(Serializer):
    def serialize(self, topic: str, headers: object, data: object) -> bytes:
        return json.dumps(data).encode("utf-8")


class Utf8Serializer(Serializer):
    def serialize(self, topic: str, headers: object, data: object) -> bytes:
        return str(data).encode("utf-8")


def build_test_event(
    sequence: int,
    event_time: datetime,
    run_id: str,
    timeframe: str = "1m",
) -> dict[str, object]:
    """Create a valid, reproducible OHLCV-shaped event for the assignment."""
    base_price = 65000.0 + sequence * 1.8 + math.sin(sequence / 11.0) * 45.0
    open_price = round(base_price, 2)
    close_price = round(base_price + math.sin(sequence / 5.0) * 12.0, 2)
    high_price = round(max(open_price, close_price) + 8.0 + (sequence % 5), 2)
    low_price = round(min(open_price, close_price) - 8.0 - (sequence % 3), 2)

    event_time_ms = int(event_time.timestamp() * 1000)
    return {
        "schema_version": "market_event_v1",
        "event_id_schema_version": EVENT_ID_SCHEMA_VERSION,
        "run_id": run_id,
        "event_id": canonical_candle_event_id(
            source="assignment-test",
            symbol="BTCUSDT",
            market="usdm",
            timeframe=timeframe,
            event_time_ms=event_time_ms,
        ),
        "event_sequence": sequence,
        "event_time": event_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": "BTCUSDT",
        "market": "USDT-M",
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": round(10.0 + (sequence % 31) * 0.75, 4),
        "source": "assignment4_deterministic_test",
    }


def to_utc_text(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def fetch_binance_usdm_events(count: int, timeframe: str, run_id: str) -> list[dict[str, object]]:
    """Fetch only closed BTCUSDT perpetual candles from Binance's public API."""
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    interval_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    if interval_ms <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    now_ms = exchange.milliseconds()
    latest_closed_start_ms = (now_ms // interval_ms - 1) * interval_ms
    since_ms = latest_closed_start_ms - (count - 1) * interval_ms
    rows = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe, since=since_ms, limit=count)

    closed_rows = sorted(
        {int(row[0]): row for row in rows if int(row[0]) <= latest_closed_start_ms}.values(),
        key=lambda row: int(row[0]),
    )
    expected_timestamps = list(range(since_ms, latest_closed_start_ms + interval_ms, interval_ms))
    actual_timestamps = [int(row[0]) for row in closed_rows]
    if actual_timestamps != expected_timestamps:
        missing = sorted(set(expected_timestamps) - set(actual_timestamps))
        raise RuntimeError(
            "Binance USDT-M OHLCV data is incomplete: "
            f"expected={count}, actual={len(closed_rows)}, missing={len(missing)}, "
            f"first_missing={missing[:3]}"
        )

    events: list[dict[str, object]] = []
    for sequence, row in enumerate(closed_rows):
        timestamp_ms, open_price, high_price, low_price, close_price, volume = row[:6]
        events.append(
            {
                "schema_version": "market_event_v1",
                "event_id_schema_version": EVENT_ID_SCHEMA_VERSION,
                "run_id": run_id,
                "event_id": canonical_candle_event_id(
                    source="binance",
                    symbol="BTCUSDT",
                    market="usdm",
                    timeframe=timeframe,
                    event_time_ms=int(timestamp_ms),
                ),
                "event_sequence": sequence,
                "event_time": to_utc_text(int(timestamp_ms)),
                "symbol": "BTCUSDT",
                "market": "USDT-M",
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
                "source": "binance_usdm_public_ohlcv",
            }
        )
    return events


def load_local_replay_events(
    path: Path,
    count: int,
    run_id: str,
    timeframe: str = "1m",
) -> list[dict[str, object]]:
    """Expand saved real OHLCV events without sending load to an external API."""
    source_events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not source_events:
        raise RuntimeError(f"Local replay source is empty: {path}")

    first_time = datetime.fromisoformat(
        str(source_events[0]["event_time"]).replace("Z", "+00:00")
    )
    events = []
    for sequence in range(count):
        source = source_events[sequence % len(source_events)]
        event_time = first_time + timedelta(minutes=sequence)
        event_time_ms = int(event_time.timestamp() * 1000)
        events.append(
            {
                "schema_version": "market_event_v1",
                "event_id_schema_version": EVENT_ID_SCHEMA_VERSION,
                "run_id": run_id,
                "event_id": canonical_candle_event_id(
                    source="local-replay",
                    symbol=str(source["symbol"]),
                    market=str(source["market"]),
                    timeframe=timeframe,
                    event_time_ms=event_time_ms,
                ),
                "event_sequence": sequence,
                "event_time": event_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "symbol": str(source["symbol"]),
                "market": str(source["market"]),
                "open": float(source["open"]),
                "high": float(source["high"]),
                "low": float(source["low"]),
                "close": float(source["close"]),
                "volume": float(source["volume"]),
                "source": "local_replay_from_binance_usdm_ohlcv",
                "source_event_id": str(source.get("event_id", "")),
            }
        )
    return events


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kafka market event producer")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--source",
        choices=["test", "binance-usdm", "local-replay"],
        default="test",
        help=(
            "test creates repeatable data; binance-usdm fetches public candles; "
            "local-replay expands a saved JSONL file without external load."
        ),
    )
    parser.add_argument(
        "--source-file",
        default="examples/kafka_flink/data/consumed_binance_usdm_events.jsonl",
    )
    parser.add_argument("--timeframe", default="1m", help="Binance timeframe used with --source binance-usdm.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--run-id", default="portfolio-demo-v1")
    parser.add_argument(
        "--duplicate-count",
        type=int,
        default=0,
        help="Send the first N records once before the full set to reproduce duplicate delivery.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate events, then write the report without sending to Kafka.",
    )
    parser.add_argument(
        "--report-path",
        default="examples/kafka_flink/results/producer_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_started = time.monotonic()
    if args.count < 100:
        raise ValueError("The reproducible demo requires --count to be at least 100.")
    if args.duplicate_count < 0 or args.duplicate_count > args.count:
        raise ValueError("--duplicate-count must be between 0 and --count.")

    if args.source == "binance-usdm":
        events = fetch_binance_usdm_events(args.count, args.timeframe, args.run_id)
    elif args.source == "local-replay":
        events = load_local_replay_events(
            Path(args.source_file),
            args.count,
            args.run_id,
            args.timeframe,
        )
    else:
        start_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start_time -= timedelta(minutes=args.count - 1)
        events = [
            build_test_event(
                sequence,
                start_time + timedelta(minutes=sequence),
                args.run_id,
                args.timeframe,
            )
            for sequence in range(args.count)
        ]

    if len(events) != args.count:
        raise RuntimeError(f"Expected {args.count} events, received {len(events)}")

    send_elapsed_seconds = 0.0
    sent_events = events[: args.duplicate_count] + events
    if not args.dry_run:
        send_started = time.monotonic()
        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap_servers,
            value_serializer=JsonSerializer(),
            key_serializer=Utf8Serializer(),
            acks="all",
            retries=3,
        )
        try:
            futures = [producer.send(args.topic, key=event["event_id"], value=event) for event in sent_events]
            for future in futures:
                future.get(timeout=30)
            producer.flush(timeout=30)
        finally:
            producer.close(timeout=30)
        send_elapsed_seconds = time.monotonic() - send_started

    report = {
        "status": "dry_run" if args.dry_run else "success",
        "topic": args.topic,
        "run_id": args.run_id,
        "producer_sent_count": 0 if args.dry_run else len(sent_events),
        "producer_unique_event_count": len(events),
        "intentional_duplicate_count": args.duplicate_count,
        "fetched_event_count": len(events),
        "event_schema_version": "market_event_v1",
        "event_id_schema_version": EVENT_ID_SCHEMA_VERSION,
        "source": args.source,
        "symbol": "BTCUSDT",
        "market": "USDT-M",
        "timeframe": args.timeframe,
        "event_time_start_utc": events[0]["event_time"],
        "event_time_end_utc": events[-1]["event_time"],
        "send_elapsed_seconds": round(send_elapsed_seconds, 6),
        "send_records_per_second": (
            round(len(sent_events) / send_elapsed_seconds, 3) if send_elapsed_seconds > 0 else 0.0
        ),
        "total_elapsed_seconds": round(time.monotonic() - total_started, 6),
    }
    write_report(Path(args.report_path), report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
