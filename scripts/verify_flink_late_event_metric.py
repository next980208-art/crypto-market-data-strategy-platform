"""Inject one late Kafka event and verify Flink drops it without feature output."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from kafka import KafkaConsumer, KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_metadata import canonical_event_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the PyFlink dropped_late_events metric.")
    parser.add_argument("--flink-rest", default="http://localhost:8081")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--raw-topic", default="market.trade.v1")
    parser.add_argument("--feature-topic", default="market.features.v1")
    parser.add_argument("--late-by-seconds", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--output", default="runtime_reports/flink_late_event_metric_latest.json")
    return parser.parse_args()


def find_running_job_and_vertex(flink_rest: str) -> tuple[str, str]:
    overview = requests.get(f"{flink_rest}/jobs/overview", timeout=10)
    overview.raise_for_status()
    candidates = [
        job
        for job in overview.json().get("jobs", [])
        if job.get("state") == "RUNNING" and job.get("name") == "realtime-kafka-feature-job-paper-only"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one running realtime feature job, found={len(candidates)}")
    job_id = str(candidates[0]["jid"])
    detail = requests.get(f"{flink_rest}/jobs/{job_id}", timeout=10)
    detail.raise_for_status()
    vertices = [
        vertex
        for vertex in detail.json().get("vertices", [])
        if "KEYED PROCESS" in str(vertex.get("name", ""))
    ]
    if len(vertices) != 1:
        raise RuntimeError(f"Expected exactly one keyed-process vertex, found={len(vertices)}")
    return job_id, str(vertices[0]["id"])


def discover_metric_id(flink_rest: str, job_id: str, vertex_id: str, suffix: str) -> str | None:
    response = requests.get(f"{flink_rest}/jobs/{job_id}/vertices/{vertex_id}/metrics", timeout=10)
    response.raise_for_status()
    matches = [str(metric["id"]) for metric in response.json() if str(metric.get("id", "")).endswith(suffix)]
    if len(matches) > 1:
        raise RuntimeError(f"Metric suffix is ambiguous: suffix={suffix}, matches={matches}")
    return matches[0] if matches else None


def metric_value(flink_rest: str, job_id: str, vertex_id: str, metric_id: str | None) -> int:
    if metric_id is None:
        return 0
    response = requests.get(
        f"{flink_rest}/jobs/{job_id}/vertices/{vertex_id}/metrics",
        params={"get": metric_id},
        timeout=10,
    )
    response.raise_for_status()
    values = response.json()
    return int(float(values[0]["value"])) if values else 0


def send_probe(bootstrap_servers: str, topic: str, symbol: str, event_time_ms: int) -> dict[str, Any]:
    event = {
        "schema_version": "market_raw_v1",
        "event_id": f"late-probe-{uuid.uuid4().hex}",
        "event_type": "agg_trade",
        "received_at_ms": int(time.time() * 1000),
        "payload": {
            "price": "100.0",
            "quantity": "1.0",
            "buyer_is_maker": False,
        },
        "test_probe": True,
        **canonical_event_metadata(
            symbol=symbol,
            market="usdm",
            timeframe="1m",
            event_time_ms=event_time_ms,
        ),
    }
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        acks="all",
        retries=5,
    )
    try:
        producer.send(
            topic,
            key=symbol.encode("utf-8"),
            value=json.dumps(event, separators=(",", ":")).encode("utf-8"),
        ).get(timeout=20)
        producer.flush(timeout=20)
    finally:
        producer.close(timeout=20)
    return event


def count_probe_features(bootstrap_servers: str, topic: str, symbol: str, poll_seconds: int = 5) -> int:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=None,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        isolation_level="read_committed",
    )
    count = 0
    deadline = time.monotonic() + poll_seconds
    try:
        while time.monotonic() < deadline:
            records = consumer.poll(timeout_ms=1_000, max_records=1_000)
            for partition_records in records.values():
                for record in partition_records:
                    feature = json.loads(record.value.decode("utf-8"))
                    if feature.get("symbol") == symbol:
                        count += 1
    finally:
        consumer.close()
    return count


def evaluate_result(
    *,
    baseline_dropped: int,
    final_dropped: int,
    baseline_accepted: int,
    final_accepted: int,
    output_features: int,
) -> dict[str, Any]:
    dropped_delta = final_dropped - baseline_dropped
    accepted_delta = final_accepted - baseline_accepted
    return {
        "baseline_dropped_late_events": baseline_dropped,
        "final_dropped_late_events": final_dropped,
        "dropped_late_events_delta": dropped_delta,
        "baseline_accepted_events": baseline_accepted,
        "final_accepted_events": final_accepted,
        "accepted_events_delta": accepted_delta,
        "probe_feature_outputs": output_features,
        "healthy": dropped_delta == 1 and accepted_delta >= 0 and output_features == 0,
    }


def main() -> None:
    args = parse_args()
    if args.late_by_seconds <= 70 or args.timeout_seconds <= 0:
        raise ValueError("Probe must be over 70 seconds late and timeout must be positive.")
    job_id, vertex_id = find_running_job_and_vertex(args.flink_rest)
    accepted_id = discover_metric_id(args.flink_rest, job_id, vertex_id, "accepted_events")
    watermark_id = discover_metric_id(
        args.flink_rest, job_id, vertex_id, "KEYED_PROCESS__Map.currentInputWatermark"
    )
    if accepted_id is None or watermark_id is None:
        raise RuntimeError("Required accepted-events or input-watermark metric is missing.")
    dropped_id = discover_metric_id(args.flink_rest, job_id, vertex_id, "dropped_late_events")
    baseline_accepted = metric_value(args.flink_rest, job_id, vertex_id, accepted_id)
    baseline_dropped = metric_value(args.flink_rest, job_id, vertex_id, dropped_id)
    watermark = metric_value(args.flink_rest, job_id, vertex_id, watermark_id)
    if watermark <= 0:
        raise RuntimeError(f"Input watermark has not started: {watermark}")

    probe_symbol = f"LATEPROBE{int(time.time())}"
    event_time_ms = watermark - args.late_by_seconds * 1_000
    event = send_probe(args.bootstrap_servers, args.raw_topic, probe_symbol, event_time_ms)

    deadline = time.monotonic() + args.timeout_seconds
    final_dropped = baseline_dropped
    final_accepted = baseline_accepted
    while time.monotonic() < deadline:
        time.sleep(1)
        dropped_id = dropped_id or discover_metric_id(
            args.flink_rest, job_id, vertex_id, "dropped_late_events"
        )
        final_dropped = metric_value(args.flink_rest, job_id, vertex_id, dropped_id)
        final_accepted = metric_value(args.flink_rest, job_id, vertex_id, accepted_id)
        if final_dropped > baseline_dropped:
            break

    output_features = count_probe_features(args.bootstrap_servers, args.feature_topic, probe_symbol)
    report = evaluate_result(
        baseline_dropped=baseline_dropped,
        final_dropped=final_dropped,
        baseline_accepted=baseline_accepted,
        final_accepted=final_accepted,
        output_features=output_features,
    )
    report.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "vertex_id": vertex_id,
            "watermark_before_probe": watermark,
            "probe_symbol": probe_symbol,
            "probe_event_id": event["event_id"],
            "probe_event_time_ms": event_time_ms,
            "late_by_seconds": args.late_by_seconds,
            "raw_topic": args.raw_topic,
            "feature_topic": args.feature_topic,
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
