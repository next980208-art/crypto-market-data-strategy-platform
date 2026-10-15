"""Run saved input through Kafka, PyFlink, Parquet, and the read-only serving layer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
SOURCE_FILE = DEMO_ROOT / "input" / "binance_usdm_btcusdt_1000.jsonl"
OUTPUT_FILE = DEMO_ROOT / "output" / "latest_features.parquet"
RESULT_FILE = DEMO_ROOT / "results" / "latest_end_to_end_run.json"
SERVING_RESULT_FILE = DEMO_ROOT / "results" / "latest_serving_response.json"
CAPACITY_RUNNER = PROJECT_ROOT / "scripts" / "run_pipeline_capacity_test.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.serving.feature_store_api import build_dataset_view, serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end local portfolio demonstration")
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--skip-docker-start", action="store_true")
    parser.add_argument(
        "--verify-snapshot-only",
        action="store_true",
        help="Verify the committed input, Parquet, and result files without Docker.",
    )
    parser.add_argument("--serve", action="store_true", help="Start the dashboard after the run.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    return parser.parse_args()


def run(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr
        + f"\n\nRETURN_CODE\n{completed.returncode}"
        + f"\n\nELAPSED_SECONDS\n{time.monotonic() - started:.6f}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return completed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def collect_feature_files(stage: dict[str, Any]) -> list[Path]:
    files = []
    for path_text in stage["flink"]["feature_files"]:
        path = Path(path_text)
        files.append(path if path.is_absolute() else PROJECT_ROOT / path)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Flink output files are missing: {missing}")
    return files


def verify_committed_snapshot() -> dict[str, Any]:
    """Validate the small public evidence bundle without external services."""
    input_count = sum(1 for line in SOURCE_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
    run_result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    serving_view = build_dataset_view(OUTPUT_FILE)
    counts = run_result.get("counts", {})
    expected_count = int(run_result.get("requested_count", 0))
    pipeline_counts = [
        int(counts.get(name, -1))
        for name in (
            "input",
            "producer_sent",
            "consumer_received",
            "pyflink_input",
            "parquet_saved",
            "serving_read",
        )
    ]
    checks = {
        "successful_recorded_run": run_result.get("status") == "success",
        "input_matches_expected": input_count == expected_count,
        "pipeline_counts_match": bool(expected_count)
        and all(count == expected_count for count in pipeline_counts),
        "parquet_matches_expected": int(serving_view["row_count"]) == expected_count,
        "no_duplicate_timestamps": int(serving_view["duplicate_timestamps"]) == 0,
        "no_missing_required_values": int(serving_view["missing_required_values"]) == 0,
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "committed_snapshot_verification",
        "external_services_required": False,
        "expected_count": expected_count,
        "checks": checks,
    }
    if summary["status"] != "passed":
        raise RuntimeError(f"Committed demo snapshot verification failed: {summary}")
    return summary


def main() -> None:
    args = parse_args()
    if args.count < 100 or args.count > 1_000:
        raise ValueError("The short presentation demo supports 100 to 1,000 events.")
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Submission input sample is missing: {SOURCE_FILE}")
    if args.verify_snapshot_only:
        print(json.dumps(verify_committed_snapshot(), ensure_ascii=False, indent=2))
        return

    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime_root = PROJECT_ROOT / "runtime_reports" / "portfolio_demo" / execution_id
    log_root = runtime_root / "orchestrator_logs"
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    report: dict[str, Any] = {
        "demo": "저장된 공개 입력부터 Feature Store 조회까지 이어진 실행",
        "execution_id": execution_id,
        "started_at_utc": started_at.isoformat(),
        "status": "running",
        "external_api_called": False,
        "input_source": project_relative(SOURCE_FILE),
        "input_description": "Previously saved public Binance USDT-M BTCUSDT 1-minute candles",
        "requested_count": args.count,
        "pipeline": ["saved_input", "kafka", "consumer", "pyflink", "parquet", "serving_read"],
    }
    try:
        if not args.skip_docker_start:
            run(
                [
                    "docker",
                    "compose",
                    "--profile",
                    "streaming",
                    "up",
                    "-d",
                    "zookeeper",
                    "kafka",
                    "jobmanager",
                    "taskmanager",
                    "flink-client",
                ],
                log_root / "docker_start.log",
            )

        run(
            [
                sys.executable,
                str(CAPACITY_RUNNER),
                "--counts",
                str(args.count),
                "--bootstrap-servers",
                args.bootstrap_servers,
                "--topic-prefix",
                "portfolio.market.events.v1",
                "--source-file",
                str(SOURCE_FILE),
                "--flink-client-service",
                "flink-client",
                "--output-root",
                str(runtime_root / "pipeline"),
            ],
            log_root / "pipeline.log",
        )

        capacity_report_path = runtime_root / "pipeline" / "capacity_test_report.json"
        capacity_report = json.loads(capacity_report_path.read_text(encoding="utf-8"))
        stage = capacity_report["stages"][0]
        feature_files = collect_feature_files(stage)
        frame = pd.concat([pd.read_parquet(path) for path in feature_files], ignore_index=True)
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if len(frame) != args.count:
            raise RuntimeError(f"Final Parquet row mismatch: expected={args.count}, actual={len(frame)}")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(OUTPUT_FILE, index=False)

        serving_view = build_dataset_view(OUTPUT_FILE)
        write_json(SERVING_RESULT_FILE, serving_view)
        counts = {
            "input": args.count,
            "producer_sent": int(stage["kafka"]["producer"]["producer_sent_count"]),
            "consumer_received": int(stage["kafka"]["consumer"]["consumer_received_count"]),
            "pyflink_input": int(stage["flink_input"]["valid_flink_input_count"]),
            "parquet_saved": int(stage["flink"]["flink_output_processed_count"]),
            "serving_read": int(serving_view["row_count"]),
            "unprocessed": args.count - int(serving_view["row_count"]),
        }
        report.update(
            {
                "status": "success",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_elapsed_seconds": round(time.monotonic() - started, 6),
                "counts": counts,
                "all_counts_match": len(set(counts.values())) == 2
                and counts["unprocessed"] == 0,
                "topic": stage["kafka"]["topic"],
                "run_id": stage["kafka"]["run_id"],
                "flink_job_id": stage["flink"]["flink_job_id"],
                "storage_format": "parquet",
                "parquet_output": project_relative(OUTPUT_FILE),
                "serving_result": project_relative(SERVING_RESULT_FILE),
                "serving_status": serving_view["status"],
                "duplicate_timestamps": serving_view["duplicate_timestamps"],
                "missing_required_values": serving_view["missing_required_values"],
                "runtime_report": project_relative(capacity_report_path),
            }
        )
        if not all(
            counts[name] == args.count
            for name in ("input", "producer_sent", "consumer_received", "pyflink_input", "parquet_saved", "serving_read")
        ):
            raise RuntimeError(f"End-to-end counts do not match: {counts}")
        report["all_counts_match"] = True
        write_json(RESULT_FILE, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_elapsed_seconds": round(time.monotonic() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(runtime_root / "demo_run.json", report)
        raise

    if args.serve:
        serve(OUTPUT_FILE, RESULT_FILE, args.host, args.port)


if __name__ == "__main__":
    main()
