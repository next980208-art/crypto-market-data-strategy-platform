"""Inject duplicate feature/signal IDs and count committed downstream outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_metadata import canonical_event_metadata
from realtime.contracts import (
    FEATURE_SCHEMA_VERSION,
    SIGNAL_SCHEMA_VERSION,
    TOPIC_FEATURES,
    TOPIC_PAPER_EXECUTIONS,
    TOPIC_RISK_AUDIT,
    TOPIC_SIGNALS,
    json_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify inference and risk business-ID idempotency.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--send-count", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--settle-seconds", type=int, default=5)
    parser.add_argument("--minimum-inference-skips", type=int, default=1)
    parser.add_argument("--minimum-risk-skips", type=int, default=1)
    parser.add_argument("--inference-metrics", default="realtime_runtime/inference_metrics.json")
    parser.add_argument("--risk-metrics", default="realtime_runtime/risk_metrics.json")
    parser.add_argument("--output", default="runtime_reports/realtime_idempotency_latest.json")
    return parser.parse_args()


def probe_feature(probe_id: str, event_time_ms: int) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_id": f"{probe_id}:feature",
        "close": 100.0,
        "feature_ready": True,
        "source": "idempotency_verification_probe",
        "test_probe": True,
        **canonical_event_metadata(
            symbol="BTCUSDT",
            market="usdm",
            timeframe="1m",
            event_time_ms=event_time_ms,
        ),
    }


def direct_risk_signal(probe_id: str, event_time_ms: int) -> dict[str, Any]:
    return {
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "signal_id": f"{probe_id}:direct-risk-signal",
        "action": "no_trade",
        "confidence": 0.0,
        "entry_price": 100.0,
        "stop_distance_pct": None,
        "reason": "idempotency_verification_probe",
        "model_status": "blocked",
        "paper_only": True,
        "test_probe": True,
        **canonical_event_metadata(
            symbol="BTCUSDT",
            market="usdm",
            timeframe="1m",
            event_time_ms=event_time_ms,
        ),
    }


def inject_duplicates(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    event_time_ms = int(time.time() * 1000)
    feature = probe_feature(args.probe_id, event_time_ms)
    signal = direct_risk_signal(args.probe_id, event_time_ms)
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        acks="all",
        retries=5,
        enable_idempotence=True,
    )
    try:
        for _ in range(args.send_count):
            producer.send(
                TOPIC_FEATURES,
                key=b"BTCUSDT",
                value=json_bytes(feature),
            )
            producer.send(
                TOPIC_SIGNALS,
                key=b"BTCUSDT",
                value=json_bytes(signal),
            )
        producer.flush(timeout=20)
    finally:
        producer.close(timeout=20)
    return feature, signal


def collect_probe_outputs(
    bootstrap_servers: str,
    feature_signal_id: str,
    direct_signal_id: str,
    poll_seconds: int,
) -> dict[str, int]:
    consumer = KafkaConsumer(
        TOPIC_SIGNALS,
        TOPIC_RISK_AUDIT,
        TOPIC_PAPER_EXECUTIONS,
        bootstrap_servers=bootstrap_servers,
        group_id=None,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        isolation_level="read_committed",
    )
    counts = {
        "inference_signals": 0,
        "inference_risk_audits": 0,
        "direct_risk_audits": 0,
        "probe_paper_orders": 0,
    }
    deadline = time.monotonic() + poll_seconds
    try:
        while time.monotonic() < deadline:
            records = consumer.poll(timeout_ms=1_000, max_records=2_000)
            for partition_records in records.values():
                for record in partition_records:
                    try:
                        payload = json.loads(record.value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if record.topic == TOPIC_SIGNALS and payload.get("signal_id") == feature_signal_id:
                        counts["inference_signals"] += 1
                    elif record.topic == TOPIC_RISK_AUDIT:
                        if payload.get("signal_id") == feature_signal_id:
                            counts["inference_risk_audits"] += 1
                        elif payload.get("signal_id") == direct_signal_id:
                            counts["direct_risk_audits"] += 1
                    elif record.topic == TOPIC_PAPER_EXECUTIONS and payload.get("signal_id") in {
                        feature_signal_id,
                        direct_signal_id,
                    }:
                        counts["probe_paper_orders"] += 1
            if (
                counts["inference_signals"] >= 1
                and counts["inference_risk_audits"] >= 1
                and counts["direct_risk_audits"] >= 1
            ):
                break
    finally:
        consumer.close()
    return counts


def evaluate_counts(
    counts: dict[str, int],
    inference_metrics: dict[str, Any] | None = None,
    risk_metrics: dict[str, Any] | None = None,
    *,
    minimum_inference_skips: int = 0,
    minimum_risk_skips: int = 0,
) -> bool:
    outputs_healthy = counts == {
        "inference_signals": 1,
        "inference_risk_audits": 1,
        "direct_risk_audits": 1,
        "probe_paper_orders": 0,
    }
    inference_counters = (inference_metrics or {}).get("counters", {})
    risk_counters = (risk_metrics or {}).get("counters", {})
    return (
        outputs_healthy
        and int(inference_counters.get("duplicate_features_skipped", 0)) >= minimum_inference_skips
        and int(risk_counters.get("duplicate_signals_skipped", 0)) >= minimum_risk_skips
    )


def load_metrics(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def wait_for_metrics(
    inference_path: str,
    risk_path: str,
    *,
    injected_at: datetime,
    minimum_inference_skips: int,
    minimum_risk_skips: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    latest_inference: dict[str, Any] = {}
    latest_risk: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest_inference = load_metrics(inference_path)
        latest_risk = load_metrics(risk_path)
        try:
            inference_updated = datetime.fromisoformat(str(latest_inference["updated_at_utc"]))
            risk_updated = datetime.fromisoformat(str(latest_risk["updated_at_utc"]))
        except (KeyError, TypeError, ValueError):
            time.sleep(1)
            continue
        inference_skips = int(
            latest_inference.get("counters", {}).get("duplicate_features_skipped", 0)
        )
        risk_skips = int(latest_risk.get("counters", {}).get("duplicate_signals_skipped", 0))
        if (
            inference_updated >= injected_at
            and risk_updated >= injected_at
            and inference_skips >= minimum_inference_skips
            and risk_skips >= minimum_risk_skips
        ):
            return latest_inference, latest_risk
        time.sleep(1)
    return latest_inference, latest_risk


def main() -> None:
    args = parse_args()
    if args.send_count <= 0 or args.poll_seconds <= 0 or args.settle_seconds < 0:
        raise ValueError("Send count and poll seconds must be positive; settle seconds must not be negative.")
    injected_at = datetime.now(timezone.utc)
    feature, signal = inject_duplicates(args)
    time.sleep(args.settle_seconds)
    feature_signal_id = f"{feature['feature_id']}:blocked"
    direct_signal_id = str(signal["signal_id"])
    counts = collect_probe_outputs(
        args.bootstrap_servers,
        feature_signal_id,
        direct_signal_id,
        args.poll_seconds,
    )
    inference_metrics, risk_metrics = wait_for_metrics(
        args.inference_metrics,
        args.risk_metrics,
        injected_at=injected_at,
        minimum_inference_skips=args.minimum_inference_skips,
        minimum_risk_skips=args.minimum_risk_skips,
        timeout_seconds=args.poll_seconds,
    )
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_id": args.probe_id,
        "injected_at_utc": injected_at.isoformat(),
        "duplicates_sent_per_input": args.send_count,
        "feature_id": feature["feature_id"],
        "inference_signal_id": feature_signal_id,
        "direct_risk_signal_id": direct_signal_id,
        "committed_output_counts": counts,
        "inference_counters": inference_metrics.get("counters", {}),
        "risk_counters": risk_metrics.get("counters", {}),
        "required_duplicate_skips": {
            "inference": args.minimum_inference_skips,
            "risk": args.minimum_risk_skips,
        },
        "consumer_isolation": "read_committed",
        "healthy": evaluate_counts(
            counts,
            inference_metrics,
            risk_metrics,
            minimum_inference_skips=args.minimum_inference_skips,
            minimum_risk_skips=args.minimum_risk_skips,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
