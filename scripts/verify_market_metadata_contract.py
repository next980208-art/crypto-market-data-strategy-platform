"""Verify canonical metadata on recent raw and feature Kafka records."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, TopicPartition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_metadata import METADATA_SCHEMA_VERSION, TIMESTAMP_UNIT, normalize_market, normalize_symbol
from realtime.contracts import RAW_TOPICS, TOPIC_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the live Kafka market metadata contract.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--lookback-records", type=int, default=5_000)
    parser.add_argument("--output", default="runtime_reports/market_metadata_contract_latest.json")
    return parser.parse_args()


def validate_contract_record(record: dict[str, Any], expected_symbol: str) -> list[str]:
    errors: list[str] = []
    expected = normalize_symbol(expected_symbol)
    try:
        actual_symbol = normalize_symbol(str(record.get("symbol", "")))
    except ValueError:
        actual_symbol = ""
    if actual_symbol != expected or record.get("symbol") != expected:
        errors.append(f"symbol must be canonical {expected}")
    try:
        actual_market = normalize_market(str(record.get("market", "")))
    except ValueError:
        actual_market = ""
    if actual_market != "usdm" or record.get("market") != "usdm":
        errors.append("market must be canonical usdm")
    if record.get("metadata_schema_version") != METADATA_SCHEMA_VERSION:
        errors.append(f"metadata_schema_version must be {METADATA_SCHEMA_VERSION}")
    if record.get("timestamp_unit") != TIMESTAMP_UNIT:
        errors.append(f"timestamp_unit must be {TIMESTAMP_UNIT}")
    try:
        if int(record["event_time_ms"]) < 0:
            errors.append("event_time_ms must not be negative")
    except (KeyError, TypeError, ValueError):
        errors.append("event_time_ms must be an integer")
    if record.get("timeframe") != "1m":
        errors.append("timeframe must be 1m")
    return errors


def recent_consumer(bootstrap_servers: str, topics: tuple[str, ...], lookback_records: int) -> KafkaConsumer:
    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id=None,
        enable_auto_commit=False,
        isolation_level="read_committed",
    )
    partitions = [
        TopicPartition(topic, partition)
        for topic in topics
        for partition in sorted(consumer.partitions_for_topic(topic) or [])
    ]
    if not partitions:
        consumer.close()
        raise RuntimeError("No Kafka partitions were found for the metadata verification topics.")
    consumer.assign(partitions)
    end_offsets = consumer.end_offsets(partitions)
    for partition in partitions:
        consumer.seek(partition, max(0, end_offsets[partition] - lookback_records))
    return consumer


def collect_valid_samples(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    topics = (*RAW_TOPICS, TOPIC_FEATURES)
    consumer = recent_consumer(args.bootstrap_servers, topics, args.lookback_records)
    samples: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + args.poll_seconds
    try:
        while time.monotonic() < deadline and len(samples) < len(topics):
            records = consumer.poll(timeout_ms=1_000, max_records=2_000)
            for partition_records in records.values():
                for record in partition_records:
                    try:
                        payload = json.loads(record.value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if payload.get("symbol") != normalize_symbol(args.symbol):
                        continue
                    if not validate_contract_record(payload, args.symbol):
                        samples[record.topic] = payload
    finally:
        consumer.close()
    return samples


def sample_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "metadata_schema_version",
        "event_id",
        "feature_id",
        "event_type",
        "event_time_ms",
        "timestamp_unit",
        "symbol",
        "market",
        "timeframe",
        "source",
    )
    return {field: payload[field] for field in fields if field in payload}


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0 or args.lookback_records <= 0:
        raise ValueError("Poll seconds and lookback records must be positive.")
    expected_topics = (*RAW_TOPICS, TOPIC_FEATURES)
    samples = collect_valid_samples(args)
    missing_topics = [topic for topic in expected_topics if topic not in samples]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "canonical_contract": {
            "symbol": normalize_symbol(args.symbol),
            "market": "usdm",
            "timeframe": "1m",
            "timestamp_field": "event_time_ms",
            "timestamp_unit": TIMESTAMP_UNIT,
        },
        "expected_topics": list(expected_topics),
        "verified_topics": sorted(samples),
        "missing_topics": missing_topics,
        "samples": {topic: sample_summary(payload) for topic, payload in sorted(samples.items())},
        "consumer_isolation": "read_committed",
        "healthy": not missing_topics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
