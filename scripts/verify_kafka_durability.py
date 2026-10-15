"""Verify stable event IDs and Kafka restart persistence evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Kafka durability experiment artifacts")
    parser.add_argument("--first-events", required=True)
    parser.add_argument("--second-events", required=True)
    parser.add_argument("--first-producer-report", required=True)
    parser.add_argument("--first-consumer-report", required=True)
    parser.add_argument("--second-producer-report", required=True)
    parser.add_argument("--second-consumer-report", required=True)
    parser.add_argument("--recreated-events", required=True)
    parser.add_argument("--recreated-consumer-report", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def read_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    first_events = read_jsonl(args.first_events)
    second_events = read_jsonl(args.second_events)
    first_producer = read_json(args.first_producer_report)
    first_consumer = read_json(args.first_consumer_report)
    second_producer = read_json(args.second_producer_report)
    second_consumer = read_json(args.second_consumer_report)
    recreated_events = read_jsonl(args.recreated_events)
    recreated_consumer = read_json(args.recreated_consumer_report)

    first_ids = [str(event.get("event_id", "")) for event in first_events]
    second_ids = [str(event.get("event_id", "")) for event in second_events]
    recreated_ids = [str(event.get("event_id", "")) for event in recreated_events]
    first_run_ids = {str(event.get("run_id", "")) for event in first_events}
    second_run_ids = {str(event.get("run_id", "")) for event in second_events}

    checks = {
        "first_producer_sent_100": first_producer.get("producer_sent_count") == 100,
        "received_100_after_kafka_restart": first_consumer.get("consumer_received_count") == 100,
        "second_producer_sent_100": second_producer.get("producer_sent_count") == 100,
        "second_consumer_received_100": second_consumer.get("consumer_received_count") == 100,
        "received_100_after_container_recreation": (
            recreated_consumer.get("consumer_received_count") == 100
        ),
        "first_ids_unique": len(first_ids) == len(set(first_ids)) == 100,
        "second_ids_unique": len(second_ids) == len(set(second_ids)) == 100,
        "event_ids_equal_between_run_ids": first_ids == second_ids,
        "event_ids_survive_container_recreation": first_ids == recreated_ids,
        "run_ids_are_different": first_run_ids != second_run_ids,
        "event_id_schema_is_v1": all(
            event.get("event_id_schema_version") == "market_event_id_v1"
            for event in first_events + second_events
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Kafka named-volume restart and stable event identity",
        "topic": first_producer.get("topic"),
        "first_run_id": first_producer.get("run_id"),
        "second_run_id": second_producer.get("run_id"),
        "first_event_count": len(first_events),
        "second_event_count": len(second_events),
        "sample_stable_event_id": first_ids[0] if first_ids else None,
        "checks": checks,
        "errors": errors,
        "healthy": not errors,
    }
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors:
        raise RuntimeError(f"Kafka durability verification failed: {errors}")


if __name__ == "__main__":
    main()
