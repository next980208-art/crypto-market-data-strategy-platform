"""Create a compact execution report from a PyFlink success marker and Parquet output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a report for the completed PyFlink batch")
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--marker-path", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--report-path",
        default="examples/kafka_flink/results/flink_report.json",
    )
    return parser.parse_args()


def resolve_feature_path(path_text: str, project_root: Path) -> Path:
    """Translate a shared Airflow container path to the local project path when needed."""
    normalized = path_text.replace("\\", "/")
    container_prefix = "/opt/airflow/project/"
    if normalized.startswith(container_prefix):
        return project_root.joinpath(*normalized[len(container_prefix) :].split("/"))
    return Path(path_text)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_report = json.loads(Path(args.input_report).read_text(encoding="utf-8"))
    marker = json.loads(Path(args.marker_path).read_text(encoding="utf-8"))
    feature_files = [resolve_feature_path(str(path), project_root) for path in marker["feature_files"]]
    frame = pd.concat([pd.read_parquet(path) for path in feature_files], ignore_index=True)

    expected_count = int(input_report["valid_flink_input_count"])
    if len(frame) != expected_count:
        raise RuntimeError(f"Flink output count mismatch: expected={expected_count}, actual={len(frame)}")
    required_features = {"ma_5", "return_1m", "feature_schema_version"}
    if not required_features.issubset(frame.columns):
        raise RuntimeError(f"Flink output is missing features: {sorted(required_features - set(frame.columns))}")

    def relative_to_project(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)

    report = {
        "status": "success",
        "processor": marker["processor"],
        "flink_job_id": marker["job_id"],
        "run_id": marker["run_id"],
        "flink_input_raw_count": input_report["raw_jsonl_count"],
        "flink_input_valid_count": expected_count,
        "flink_output_processed_count": len(frame),
        "dropped_count_before_flink": input_report["raw_jsonl_count"] - expected_count,
        "storage_format": "parquet",
        "feature_schema_version": marker["feature_schema_version"],
        "feature_files": [relative_to_project(path) for path in feature_files],
        "final_columns": frame.columns.tolist(),
        "flink_features": ["ma_5", "return_1m"],
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
