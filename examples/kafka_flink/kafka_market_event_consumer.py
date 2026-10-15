"""Consume one portfolio demo run from Kafka and persist the raw JSONL records."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from kafka import KafkaConsumer
from kafka.serializer import Deserializer


DEFAULT_TOPIC = "btc_market_events_v1"


class JsonDeserializer(Deserializer):
    def deserialize(self, topic: str, headers: object, data: bytes) -> dict[str, object]:
        return json.loads(data.decode("utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kafka market event consumer")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--run-id", default="portfolio-demo-v1")
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--output-path",
        default="examples/kafka_flink/data/consumed_market_events.jsonl",
    )
    parser.add_argument(
        "--report-path",
        default="examples/kafka_flink/results/consumer_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if args.expected_count < 100:
        raise ValueError("The reproducible demo requires --expected-count to be at least 100.")

    group_id = f"{args.run_id}-consumer-{uuid.uuid4().hex[:8]}"
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=JsonDeserializer(),
    )

    deadline = time.monotonic() + args.timeout_seconds
    received_by_id: dict[str, dict[str, object]] = {}
    matching_message_count = 0
    duplicate_message_count = 0
    try:
        while len(received_by_id) < args.expected_count and time.monotonic() < deadline:
            records = consumer.poll(timeout_ms=1000, max_records=500)
            for partition_records in records.values():
                for record in partition_records:
                    event = record.value
                    if event.get("run_id") != args.run_id:
                        continue
                    matching_message_count += 1
                    event_id = str(event["event_id"])
                    if event_id in received_by_id:
                        duplicate_message_count += 1
                    received_by_id[event_id] = event
    finally:
        consumer.close()

    ordered_events = sorted(received_by_id.values(), key=lambda event: int(event["event_sequence"]))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for event in ordered_events:
            output_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    elapsed_seconds = time.monotonic() - started
    report = {
        "status": "success" if len(ordered_events) == args.expected_count else "incomplete",
        "topic": args.topic,
        "run_id": args.run_id,
        "consumer_group_id": group_id,
        "consumer_received_count": len(ordered_events),
        "consumer_matching_message_count": matching_message_count,
        "duplicate_message_count": duplicate_message_count,
        "expected_count": args.expected_count,
        "raw_jsonl_path": str(output_path),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "unique_records_per_second": round(len(ordered_events) / elapsed_seconds, 3),
    }
    write_json(Path(args.report_path), report)
    print(json.dumps(report, ensure_ascii=False))

    if len(ordered_events) != args.expected_count:
        raise RuntimeError(
            f"Kafka 수신 건수가 부족합니다: expected={args.expected_count}, received={len(ordered_events)}"
        )


if __name__ == "__main__":
    main()
