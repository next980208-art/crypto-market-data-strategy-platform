"""Validate synchronized depth-book events published to Kafka."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.contracts import TOPIC_DEPTH, decode_json, safe_symbol
from realtime.depth_book import DEPTH_PAYLOAD_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify synchronized local-book events in Kafka.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--minimum-events", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--collector-metrics", default="realtime_runtime/collector_metrics.json")
    parser.add_argument("--report-path", default="runtime_reports/depth_kafka_validation_latest.json")
    return parser.parse_args()


def validate_event(event: dict[str, Any], symbol: str, previous_u: int | None) -> tuple[int, list[str]]:
    errors: list[str] = []
    payload = event.get("payload", {})
    if event.get("event_type") != "depth_delta":
        errors.append("Unexpected event_type.")
    if safe_symbol(event.get("symbol", "")) != symbol:
        errors.append("Unexpected symbol.")
    if payload.get("depth_schema_version") != DEPTH_PAYLOAD_SCHEMA_VERSION:
        errors.append("Unexpected depth payload schema.")
    if payload.get("sequence_status") != "synchronized":
        errors.append("Depth payload is not synchronized.")
    try:
        final_update_id = int(payload["final_update_id"])
        previous_final_update_id = int(payload["previous_final_update_id"])
    except (KeyError, TypeError, ValueError):
        return 0, errors + ["Invalid update IDs."]
    if previous_u is not None and previous_final_update_id != previous_u:
        errors.append(f"Sequence gap: pu={previous_final_update_id}, previous_u={previous_u}")
    bids = payload.get("top_bids", [])
    asks = payload.get("top_asks", [])
    if not bids or not asks:
        errors.append("Missing top bids or asks.")
    else:
        try:
            bid_prices = [float(level[0]) for level in bids]
            ask_prices = [float(level[0]) for level in asks]
            if bid_prices != sorted(bid_prices, reverse=True):
                errors.append("Bids are not sorted descending.")
            if ask_prices != sorted(ask_prices):
                errors.append("Asks are not sorted ascending.")
            if bid_prices[0] >= ask_prices[0]:
                errors.append("Best bid must be below best ask.")
        except (TypeError, ValueError, IndexError):
            errors.append("Invalid top-level price data.")
    return final_update_id, errors


def main() -> None:
    args = parse_args()
    if args.minimum_events <= 0 or args.timeout_seconds <= 0:
        raise ValueError("minimum-events and timeout-seconds must be positive.")
    metrics_path = Path(args.collector_metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    session_id = str(metrics.get("session_id", ""))
    if not session_id:
        raise RuntimeError(f"Collector session_id is missing from {metrics_path}")

    symbol = safe_symbol(args.symbol)
    consumer = KafkaConsumer(
        TOPIC_DEPTH,
        bootstrap_servers=args.bootstrap_servers,
        group_id=f"depth-validation-{uuid.uuid4().hex}",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=args.timeout_seconds * 1000,
    )
    checked = 0
    previous_u: int | None = None
    errors: list[str] = []
    first_update_id: int | None = None
    last_update_id: int | None = None
    try:
        for record in consumer:
            event = decode_json(record.value)
            if event.get("session_id") != session_id:
                continue
            final_update_id, event_errors = validate_event(event, symbol, previous_u)
            if first_update_id is None:
                first_update_id = final_update_id
            last_update_id = final_update_id
            previous_u = final_update_id
            checked += 1
            errors.extend(f"event {checked}: {error}" for error in event_errors)
            if checked >= args.minimum_events:
                break
    finally:
        consumer.close()

    if checked < args.minimum_events:
        errors.append(f"Expected at least {args.minimum_events} events, found {checked}.")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "symbol": symbol,
        "topic": TOPIC_DEPTH,
        "checked_events": checked,
        "first_update_id": first_update_id,
        "last_update_id": last_update_id,
        "errors": errors,
        "healthy": not errors,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), **report}, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError("Kafka depth event validation failed.")


if __name__ == "__main__":
    main()
