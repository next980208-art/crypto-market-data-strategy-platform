"""Verify committed realtime feature IDs for duplicates and one-minute gaps."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check committed realtime Kafka features after a Flink restart.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="market.features.v1")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-time-ms", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--output", default="runtime_reports/realtime_checkpoint_restart_latest.json")
    return parser.parse_args()


def analyze_features(features: list[dict[str, Any]], symbol: str, start_time_ms: int) -> dict[str, Any]:
    selected = [
        feature
        for feature in features
        if feature.get("symbol") == symbol and int(feature.get("event_time_ms", -1)) >= start_time_ms
    ]
    ids = [str(feature.get("feature_id", "")) for feature in selected]
    id_counts = Counter(ids)
    duplicate_ids = sorted(feature_id for feature_id, count in id_counts.items() if feature_id and count > 1)
    minutes = sorted({int(feature["event_time_ms"]) for feature in selected})
    gaps = [
        {"previous_event_time_ms": previous, "next_event_time_ms": current, "gap_minutes": (current - previous) // 60_000 - 1}
        for previous, current in zip(minutes, minutes[1:])
        if current - previous != 60_000
    ]
    invalid_ids = [feature_id for feature_id in ids if not feature_id]
    return {
        "symbol": symbol,
        "start_time_ms": start_time_ms,
        "records": len(selected),
        "unique_feature_ids": len(set(ids)) - (1 if invalid_ids else 0),
        "duplicate_feature_ids": duplicate_ids,
        "invalid_feature_id_count": len(invalid_ids),
        "minute_gap_count": len(gaps),
        "minute_gaps": gaps,
        "first_event_time_ms": minutes[0] if minutes else None,
        "last_event_time_ms": minutes[-1] if minutes else None,
        "healthy": bool(selected) and not duplicate_ids and not invalid_ids and not gaps,
    }


def collect_features(args: argparse.Namespace) -> list[dict[str, Any]]:
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=None,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        isolation_level="read_committed",
    )
    features: list[dict[str, Any]] = []
    deadline = time.monotonic() + args.poll_seconds
    try:
        while time.monotonic() < deadline:
            records = consumer.poll(timeout_ms=1_000, max_records=1_000)
            for partition_records in records.values():
                features.extend(json.loads(record.value.decode("utf-8")) for record in partition_records)
    finally:
        consumer.close()
    return features


def main() -> None:
    args = parse_args()
    if args.start_time_ms < 0 or args.poll_seconds <= 0:
        raise ValueError("Start time must not be negative and poll seconds must be positive.")
    report = analyze_features(collect_features(args), args.symbol, args.start_time_ms)
    report.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "topic": args.topic,
            "consumer_isolation": "read_committed",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
