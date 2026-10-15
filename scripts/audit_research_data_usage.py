"""Inventory recorded data exposure; absence of a record never certifies a holdout."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts import automatic_research as runner
from scripts import search_trade_patterns as prior

ROOT = runner.search.ROOT
MINUTE = 60000


def ranges(values):
    values = np.unique(np.asarray(values, dtype=np.int64))
    if not len(values):
        return []
    cuts = np.flatnonzero(np.diff(values) != MINUTE) + 1
    return [{"start_ms": int(p[0]), "end_exclusive_ms": int(p[-1]) + MINUTE, "rows": len(p)}
            for p in np.split(values, cuts)]


def declarations(value, pointer="$"):
    found = []
    if isinstance(value, dict):
        for role, left, right in (("training_declared", "train_start", "validation_start"),
                                  ("validation_declared", "validation_start", "validation_end"),
                                  ("test_declared", "test_start", "test_end")):
            a, b = value.get(left), value.get(right)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and 946684800000 <= a < b <= 4102444800000:
                found.append({"role": role, "start_ms": int(a), "end_exclusive_ms": int(b), "json_pointer": pointer})
        for key, child in value.items():
            found.extend(declarations(child, pointer + "." + key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(declarations(child, pointer + f"[{i}]"))
    return found


def build():
    metadata_path = ROOT / "models_5y_20260828/direction_model_BTCUSDT_USDM_1m_metadata.json"
    metadata = prior.read(metadata_path)
    dataset = ROOT / metadata["dataset_file"]
    timestamps = pd.read_parquet(dataset, columns=["timestamp"]).timestamp.to_numpy()
    if not np.all(np.diff(timestamps) == MINUTE):
        raise ValueError("Unexpected dataset time grid")
    start, end = int(timestamps[0]), int(timestamps[-1]) + MINUTE
    split = metadata["test_start_timestamp"]
    if metadata["train_rows"] + metadata["test_rows"] + metadata["purged_train_rows"] != len(timestamps):
        raise ValueError("Baseline metadata row reconciliation failed")
    prediction_files = []
    for directory in sorted(ROOT.iterdir()):
        if directory.is_dir() and directory.name.startswith("predictions"):
            for path in sorted(directory.rglob("*.parquet")):
                if "timestamp" not in pq.ParquetFile(path).schema.names:
                    continue
                values = pd.read_parquet(path, columns=["timestamp"]).timestamp.to_numpy()
                prediction_files.append({"file": str(path.relative_to(ROOT)), "sha256": runner.checksum(path),
                                         "role": "predictions_materialized_not_proof_of_human_inspection",
                                         "rows": len(values), "ranges": ranges(values)})
    baseline = next(p for p in prediction_files if p["file"].startswith("predictions_5y_20260828"))
    if baseline["rows"] != metadata["test_rows"] or baseline["ranges"] != [{"start_ms": split, "end_exclusive_ms": end, "rows": metadata["test_rows"]}]:
        raise ValueError("Baseline test prediction range mismatch")
    protocols = []
    for path in sorted((ROOT / "runtime_reports").glob("*/protocol.json")):
        document = prior.read(path)
        protocols.append({"file": str(path.relative_to(ROOT)), "sha256": runner.checksum(path),
                          "declared_ranges": declarations(document),
                          "note": "Declarations alone do not prove execution; referenced/inherited ranges may be absent here."})
    now = datetime.now(timezone.utc)
    prospective = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return {"status": "completed", "created_utc": now.isoformat(),
            "dataset": {"file": str(dataset.relative_to(ROOT)), "sha256": runner.checksum(dataset),
                        "rows": len(timestamps), "start_ms": start, "end_exclusive_ms": end},
            "baseline_evidence": {"metadata": str(metadata_path.relative_to(ROOT)), "metadata_sha256": runner.checksum(metadata_path),
                "recorded_train_rows": metadata["train_rows"], "purged_rows": metadata["purged_train_rows"],
                "recorded_test_rows": metadata["test_rows"], "test_start_ms": split, "test_end_exclusive_ms": end,
                "measured_test_accuracy": metadata["accuracy"],
                "conclusion": "The final year already has saved predictions and measured test performance. Recent-run exclusion does not reset project-level exposure."},
            "prediction_files": prediction_files, "protocols": protocols,
            "clean_historical_holdout_certified": False,
            "limitations": "This is evidence inventory, not a complete history of every manual inspection. Training exposure also disqualifies the same rows as untouched future evaluation; purge gaps are not a meaningful new holdout.",
            "longer_training_feasibility": {"first_recent_fold_history_days": 120,
                "requested_windows_days": [90, 180, 365],
                "same_first_fold_180_365_possible": False,
                "reason": "The first recent validation starts 120 days after the five-year dataset begins. A 180/365-day comparison needs earlier data or a different predeclared development schedule."},
            "prospective_reservation": {"start_utc": prospective.isoformat(), "minimum_observation_end_utc": (prospective + timedelta(days=60)).isoformat(),
                "status": "planned_only_not_collected", "note": "A scheduling reservation, not sufficient evidence or a guarantee; candidate and decision rules must be frozen before evaluation."},
            "orders_sent": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError("Use a project subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    result = build()
    result["audit_code_sha256"] = runner.checksum(Path(__file__))
    runner.search.write_json(output / "audit.json", result)
    print({"status": result["status"], "prediction_files": len(result["prediction_files"]),
           "protocols": len(result["protocols"]), "clean_historical_holdout_certified": False,
           "prospective_reservation": result["prospective_reservation"]})


if __name__ == "__main__":
    main()
