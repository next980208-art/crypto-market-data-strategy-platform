"""Measure local Kafka-to-PyFlink pipeline capacity in increasing stages."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runtime_reports" / "pipeline_capacity"
EXAMPLE_ROOT = PROJECT_ROOT / "examples" / "kafka_flink"
SOURCE_FILE = EXAMPLE_ROOT / "data" / "consumed_binance_usdm_events.jsonl"
PRODUCER = EXAMPLE_ROOT / "kafka_market_event_producer.py"
CONSUMER = EXAMPLE_ROOT / "kafka_market_event_consumer.py"
PREPARE = EXAMPLE_ROOT / "prepare_flink_input.py"
FLINK_REPORT = EXAMPLE_ROOT / "write_flink_report.py"
RESOURCE_SERVICES = ("kafka", "jobmanager", "taskmanager")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved market events through Kafka and PyFlink at increasing volumes."
    )
    parser.add_argument("--counts", nargs="+", type=int, default=[10_000, 50_000, 100_000])
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic-prefix", default="capacity.market.events.v1")
    parser.add_argument("--consumer-timeout", type=int, default=600)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--flink-client-service",
        default="airflow",
        help="Running Docker Compose service used only to submit the finite PyFlink job.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=SOURCE_FILE,
        help="Saved Binance-derived JSONL used for local replay.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def container_path(path: Path) -> str:
    return "/opt/airflow/project/" + project_relative(path)


def run_command(
    command: list[str],
    log_path: Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
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
        + f"\n\nELAPSED_SECONDS\n{time.monotonic() - started:.6f}\n",
        encoding="utf-8",
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return completed


def wait_for_kafka(bootstrap_servers: str, timeout_seconds: int = 90) -> None:
    host, port_text = bootstrap_servers.rsplit(":", 1)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, int(port_text)), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Kafka did not become reachable: {bootstrap_servers}")


def parse_percent(value: str) -> float:
    return float(value.rstrip("%") or 0)


class DockerResourceSampler:
    """Collect non-invasive Docker resource snapshots while one stage runs."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._container_ids: list[str] = []

    def start(self) -> None:
        for service in RESOURCE_SERVICES:
            completed = subprocess.run(
                ["docker", "compose", "ps", "-q", service],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )
            container_id = completed.stdout.strip()
            if completed.returncode == 0 and container_id:
                self._container_ids.append(container_id)
            else:
                self.errors.append(f"Could not resolve container for service={service}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._container_ids:
                completed = subprocess.run(
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{json .}}",
                        *self._container_ids,
                    ],
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if completed.returncode == 0:
                    captured_at = datetime.now(timezone.utc).isoformat()
                    for line in completed.stdout.splitlines():
                        if line.strip():
                            try:
                                sample = json.loads(line)
                                sample["captured_at_utc"] = captured_at
                                self.samples.append(sample)
                            except json.JSONDecodeError as exc:
                                self.errors.append(f"Invalid docker stats JSON: {exc}")
                elif completed.stderr.strip():
                    self.errors.append(completed.stderr.strip())
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        by_container: dict[str, dict[str, Any]] = {}
        for sample in self.samples:
            name = str(sample.get("Name") or sample.get("Container") or "unknown")
            summary = by_container.setdefault(
                name,
                {
                    "sample_count": 0,
                    "max_cpu_percent": 0.0,
                    "max_memory_percent": 0.0,
                    "last_memory_usage": "",
                },
            )
            summary["sample_count"] += 1
            summary["max_cpu_percent"] = max(
                float(summary["max_cpu_percent"]), parse_percent(str(sample.get("CPUPerc", "0%")))
            )
            summary["max_memory_percent"] = max(
                float(summary["max_memory_percent"]), parse_percent(str(sample.get("MemPerc", "0%")))
            )
            summary["last_memory_usage"] = str(sample.get("MemUsage", ""))
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "containers": by_container,
            "errors": sorted(set(self.errors)),
            "raw_samples": self.samples,
        }


def run_kafka_stage(
    *,
    count: int,
    topic: str,
    run_id: str,
    output_root: Path,
    bootstrap_servers: str,
    timeout_seconds: int,
    source_file: Path,
) -> dict[str, Any]:
    stage_name = f"events_{count:06d}"
    data_dir = output_root / "data"
    result_dir = output_root / "results"
    log_dir = output_root / "logs"
    consumed_path = data_dir / f"{stage_name}_consumed.jsonl"
    producer_report_path = result_dir / f"{stage_name}_producer.json"
    consumer_report_path = result_dir / f"{stage_name}_consumer.json"

    consumer_command = [
        sys.executable,
        str(CONSUMER),
        "--bootstrap-servers",
        bootstrap_servers,
        "--topic",
        topic,
        "--run-id",
        run_id,
        "--expected-count",
        str(count),
        "--timeout-seconds",
        str(timeout_seconds),
        "--output-path",
        str(consumed_path),
        "--report-path",
        str(consumer_report_path),
    ]
    started = time.monotonic()
    consumer_process = subprocess.Popen(
        consumer_command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    time.sleep(2)
    producer_command = [
        sys.executable,
        str(PRODUCER),
        "--bootstrap-servers",
        bootstrap_servers,
        "--topic",
        topic,
        "--source",
        "local-replay",
        "--source-file",
        str(source_file),
        "--count",
        str(count),
        "--duplicate-count",
        "0",
        "--run-id",
        run_id,
        "--report-path",
        str(producer_report_path),
    ]
    try:
        run_command(producer_command, log_dir / f"{stage_name}_producer.log")
        stdout, stderr = consumer_process.communicate(timeout=timeout_seconds + 30)
    except BaseException:
        consumer_process.kill()
        consumer_process.communicate()
        raise

    (log_dir / f"{stage_name}_consumer.log").write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(consumer_command)
        + "\n\nSTDOUT\n"
        + stdout
        + "\n\nSTDERR\n"
        + stderr,
        encoding="utf-8",
    )
    if consumer_process.returncode != 0:
        raise RuntimeError(f"Consumer failed for {stage_name}")

    producer_report = json.loads(producer_report_path.read_text(encoding="utf-8"))
    consumer_report = json.loads(consumer_report_path.read_text(encoding="utf-8"))
    return {
        "topic": topic,
        "run_id": run_id,
        "expected_count": count,
        "kafka_end_to_end_seconds": round(time.monotonic() - started, 6),
        "producer": producer_report,
        "consumer": consumer_report,
        "consumed_jsonl": project_relative(consumed_path),
    }


def prepare_flink_input(count: int, output_root: Path) -> tuple[Path, dict[str, Any], float]:
    stage_name = f"events_{count:06d}"
    input_path = output_root / "data" / f"{stage_name}_consumed.jsonl"
    csv_path = output_root / "data" / f"{stage_name}_flink_input.csv"
    report_path = output_root / "results" / f"{stage_name}_flink_input.json"
    started = time.monotonic()
    run_command(
        [
            sys.executable,
            str(PREPARE),
            "--input-path",
            str(input_path),
            "--output-path",
            str(csv_path),
            "--report-path",
            str(report_path),
        ],
        output_root / "logs" / f"{stage_name}_prepare.log",
    )
    return csv_path, json.loads(report_path.read_text(encoding="utf-8")), time.monotonic() - started


def run_flink_stage(
    count: int,
    csv_path: Path,
    output_root: Path,
    flink_client_service: str,
) -> dict[str, Any]:
    stage_name = f"events_{count:06d}"
    feature_folder = output_root / "output" / stage_name
    input_report_path = output_root / "results" / f"{stage_name}_flink_input.json"
    input_report = json.loads(input_report_path.read_text(encoding="utf-8"))
    run_id = str(input_report["run_id"])
    started = time.monotonic()
    run_command(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "-w",
            "/opt/airflow/project",
            flink_client_service,
            "python",
            "flink_batch_submitter.py",
            "--raw-file",
            container_path(csv_path),
            "--feature-folder",
            container_path(feature_folder),
            "--keep-raw",
            "--reset-stale-staging",
        ],
        output_root / "logs" / f"{stage_name}_flink.log",
    )
    flink_elapsed = time.monotonic() - started

    marker_path = feature_folder / "_markers" / f"_SUCCESS_{run_id}.json"
    report_path = output_root / "results" / f"{stage_name}_flink.json"
    run_command(
        [
            sys.executable,
            str(FLINK_REPORT),
            "--input-report",
            str(input_report_path),
            "--marker-path",
            str(marker_path),
            "--project-root",
            str(PROJECT_ROOT),
            "--report-path",
            str(report_path),
        ],
        output_root / "logs" / f"{stage_name}_flink_report.log",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["flink_elapsed_seconds"] = round(flink_elapsed, 6)
    write_json(report_path, report)
    return report


def validate_stage(stage: dict[str, Any]) -> list[str]:
    expected = int(stage["requested_count"])
    errors = []
    producer_count = int(stage["kafka"]["producer"]["producer_sent_count"])
    consumer_count = int(stage["kafka"]["consumer"]["consumer_received_count"])
    flink_input_count = int(stage["flink_input"]["valid_flink_input_count"])
    flink_output_count = int(stage["flink"]["flink_output_processed_count"])
    for label, actual in (
        ("producer", producer_count),
        ("consumer", consumer_count),
        ("flink_input", flink_input_count),
        ("flink_output", flink_output_count),
    ):
        if actual != expected:
            errors.append(f"{label} count mismatch: expected={expected}, actual={actual}")
    if int(stage["kafka"]["consumer"]["duplicate_message_count"]) != 0:
        errors.append("Unexpected duplicate event IDs were observed")
    return errors


def main() -> None:
    args = parse_args()
    counts = sorted(set(args.counts))
    if not counts or any(count < 100 for count in counts):
        raise ValueError("Every capacity stage must contain at least 100 events.")
    output_root = args.output_root.resolve()
    output_root.relative_to(PROJECT_ROOT)
    source_file = args.source_file.resolve()
    source_file.relative_to(PROJECT_ROOT)
    if not source_file.exists():
        raise FileNotFoundError(f"Saved local replay source is missing: {source_file}")
    for name in ("data", "results", "logs", "output"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    wait_for_kafka(args.bootstrap_servers)
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_report_path = output_root / "capacity_test_report.json"
    report: dict[str, Any] = {
        "experiment": "local Kafka and PyFlink staged capacity test",
        "execution_id": execution_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_service_load_sent": False,
        "data_source": project_relative(source_file),
        "requested_stages": counts,
        "pipeline": "saved Binance-derived JSONL -> Kafka -> consumer JSONL -> PyFlink -> Parquet",
        "stages": [],
        "status": "running",
    }
    write_json(final_report_path, report)

    for count in counts:
        stage_started = time.monotonic()
        stage_name = f"events_{count:06d}"
        topic = f"{args.topic_prefix}.{count}.{execution_id.lower()}"
        run_id = f"capacity-{count}-{execution_id}"
        sampler = DockerResourceSampler()
        sampler.start()
        stage: dict[str, Any] = {
            "name": stage_name,
            "requested_count": count,
            "status": "running",
        }
        print(f"[{stage_name}] Kafka replay started", flush=True)
        try:
            stage["kafka"] = run_kafka_stage(
                count=count,
                topic=topic,
                run_id=run_id,
                output_root=output_root,
                bootstrap_servers=args.bootstrap_servers,
                timeout_seconds=args.consumer_timeout,
                source_file=source_file,
            )
            csv_path, input_report, prepare_elapsed = prepare_flink_input(count, output_root)
            stage["flink_input"] = input_report
            stage["prepare_elapsed_seconds"] = round(prepare_elapsed, 6)
            print(f"[{stage_name}] PyFlink processing started", flush=True)
            stage["flink"] = run_flink_stage(
                count,
                csv_path,
                output_root,
                args.flink_client_service,
            )
            stage["resource_usage"] = sampler.stop()
            stage["total_pipeline_seconds"] = round(time.monotonic() - stage_started, 6)
            errors = validate_stage(stage)
            stage["validation"] = {"healthy": not errors, "errors": errors}
            stage["status"] = "success" if not errors else "failed"
        except BaseException as exc:
            stage["resource_usage"] = sampler.stop()
            stage["total_pipeline_seconds"] = round(time.monotonic() - stage_started, 6)
            stage["status"] = "failed"
            stage["validation"] = {"healthy": False, "errors": [f"{type(exc).__name__}: {exc}"]}
            report["stages"].append(stage)
            report["status"] = "failed"
            report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(final_report_path, report)
            raise

        report["stages"].append(stage)
        write_json(final_report_path, report)
        print(
            f"[{stage_name}] completed: healthy={stage['validation']['healthy']}, "
            f"seconds={stage['total_pipeline_seconds']}",
            flush=True,
        )
        if not stage["validation"]["healthy"]:
            report["status"] = "failed"
            break

    healthy_stages = [
        int(stage["requested_count"])
        for stage in report["stages"]
        if stage.get("validation", {}).get("healthy")
    ]
    report["largest_verified_event_count"] = max(healthy_stages, default=0)
    report["capacity_limit_found"] = report["largest_verified_event_count"] < max(counts)
    report["status"] = "success" if len(healthy_stages) == len(counts) else "failed"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(final_report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "largest_verified_event_count": report["largest_verified_event_count"],
                "capacity_limit_found": report["capacity_limit_found"],
                "report": project_relative(final_report_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if report["status"] != "success":
        raise RuntimeError("One or more capacity stages failed; inspect the incremental report.")


if __name__ == "__main__":
    main()
