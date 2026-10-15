"""Paper-only model inference service for realtime feature events."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaTimeoutError

from realtime.contracts import SIGNAL_SCHEMA_VERSION, TOPIC_FEATURES, TOPIC_SIGNALS, decode_json, json_bytes, safe_symbol, utc_now_iso
from realtime.idempotency import IdempotencyStore, offsets_after_records
from market_metadata import canonical_event_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-only trading signals from realtime features.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--consumer-group", default="realtime-inference-v1")
    parser.add_argument("--approval-manifest", help="Paper approval manifest. Missing means every signal is no_trade.")
    parser.add_argument("--metrics-path", default="realtime_runtime/inference_metrics.json")
    parser.add_argument("--idempotency-db", default="realtime_runtime/inference_idempotency.sqlite3")
    parser.add_argument("--transactional-id", default="realtime-inference-paper-v1")
    return parser.parse_args()


def load_manifest(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Approval manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("approval_status") != "paper_approved":
        raise RuntimeError("Approval manifest must set approval_status to paper_approved.")
    required = ("model_path", "feature_columns", "probability_threshold", "stop_distance_pct")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise RuntimeError(f"Approval manifest is missing: {missing}")
    return manifest


def blocked_signal(feature: dict[str, Any], reason: str) -> dict[str, Any]:
    metadata = canonical_event_metadata(
        symbol=str(feature["symbol"]),
        market=str(feature["market"]),
        timeframe=str(feature.get("timeframe", "1m")),
        event_time_ms=int(feature["event_time_ms"]),
    )
    return {
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "signal_id": f"{feature['feature_id']}:blocked",
        "created_at_utc": utc_now_iso(),
        "action": "no_trade",
        "confidence": 0.0,
        "entry_price": feature.get("close"),
        "stop_distance_pct": None,
        "reason": reason,
        "model_status": "blocked",
        "paper_only": True,
        **metadata,
    }


def write_metrics(path: Path, counters: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"updated_at_utc": utc_now_iso(), "counters": counters, "paper_only": True},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_consumer(args: argparse.Namespace) -> KafkaConsumer:
    while True:
        try:
            return KafkaConsumer(
                TOPIC_FEATURES,
                bootstrap_servers=args.bootstrap_servers,
                group_id=args.consumer_group,
                auto_offset_reset="latest",
                enable_auto_commit=False,
                isolation_level="read_committed",
            )
        except KafkaTimeoutError:
            print("Kafka not ready for inference service; retrying in 5 seconds.")
            time.sleep(5)


def create_transactional_producer(args: argparse.Namespace) -> KafkaProducer:
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        acks="all",
        retries=5,
        enable_idempotence=True,
        transactional_id=args.transactional_id,
    )
    producer.init_transactions()
    return producer


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.approval_manifest)
    if manifest is not None:
        raise RuntimeError(
            "Realtime model loading is intentionally disabled until the current model passes "
            "walk-forward and paper approval. Remove the manifest to run the safe no_trade path."
        )
    consumer = create_consumer(args)
    producer = create_transactional_producer(args)
    store = IdempotencyStore(args.idempotency_db)
    counters = {
        "features_committed": 0,
        "blocked_signals_committed": 0,
        "duplicate_features_skipped": 0,
        "invalid_features_skipped": 0,
    }
    try:
        while True:
            records = consumer.poll(timeout_ms=1_000, max_records=500)
            if not records:
                write_metrics(Path(args.metrics_path), counters)
                continue
            committed_ids: list[str] = []
            batch_ids: set[str] = set()
            batch_duplicates = 0
            batch_invalid = 0
            producer.begin_transaction()
            try:
                for partition_records in records.values():
                    for record in partition_records:
                        feature = decode_json(record.value)
                        feature_id = str(feature.get("feature_id", ""))
                        if not feature_id:
                            batch_invalid += 1
                            continue
                        if feature_id in batch_ids or store.contains("feature", feature_id):
                            batch_duplicates += 1
                            continue
                        signal = blocked_signal(feature, "no_paper_approved_model")
                        producer.send(
                            TOPIC_SIGNALS,
                            key=signal["symbol"].encode("utf-8"),
                            value=json_bytes(signal),
                        )
                        batch_ids.add(feature_id)
                        committed_ids.append(feature_id)
                producer.send_offsets_to_transaction(
                    offsets_after_records(records),
                    consumer.group_metadata(),
                )
                producer.commit_transaction()
            except Exception:
                producer.abort_transaction()
                raise
            store.mark_many("feature", committed_ids)
            counters["features_committed"] += len(committed_ids)
            counters["blocked_signals_committed"] += len(committed_ids)
            counters["duplicate_features_skipped"] += batch_duplicates
            counters["invalid_features_skipped"] += batch_invalid
            write_metrics(Path(args.metrics_path), counters)
    finally:
        producer.close(timeout=20)
        consumer.close()
        store.close()


if __name__ == "__main__":
    main()
