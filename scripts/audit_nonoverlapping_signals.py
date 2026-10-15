"""Count fixed-horizon nonoverlapping signal samples; no training, collection or orders."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_cost_aware_gate as gate
from scripts.verify_cost_aware_gate import verify

MINUTE = gate.search.MINUTE
HORIZON_MINUTES = 240


def validate_signals(frame):
    for column in ("timestamp", "entry_timestamp", "max_holding_minutes"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or not values.eq(np.floor(values)).all():
            raise ValueError(f"Invalid integer column: {column}")
    if not frame.timestamp.is_unique or not frame.entry_timestamp.is_unique:
        raise ValueError("Duplicate signal or entry timestamp")
    if not (frame.entry_timestamp-frame.timestamp).eq(MINUTE).all():
        raise ValueError("Expected next-minute entry")
    if not frame.max_holding_minutes.eq(HORIZON_MINUTES).all():
        raise ValueError("Expected fixed 240-minute horizon")


def select_nonoverlapping(frame):
    validate_signals(frame)
    ordered = frame.sort_values("entry_timestamp").reset_index(drop=True)
    keep, previous_end = [], None
    # Treat the horizon as closed: even its final bar cannot be reused by the next sample.
    for index, entry in enumerate(ordered.entry_timestamp):
        if previous_end is None or entry > previous_end:
            keep.append(index)
            previous_end = int(entry) + HORIZON_MINUTES*MINUTE
    return ordered.iloc[keep].reset_index(drop=True).copy()


def describe(frame):
    net = pd.to_numeric(frame.actual_net_label_r, errors="raise")
    if not np.isfinite(net).all():
        raise ValueError("Invalid outcome label")
    positive = int(net.gt(0).sum())
    nonpositive = len(net)-positive
    return {"signals": len(frame), "positive": positive, "nonpositive": nonpositive,
            "positive_rate": float(net.gt(0).mean()) if len(net) else None,
            "mean_net_label_r": float(net.mean()) if len(net) else None,
            "distinct_entry_days_utc": int(pd.to_datetime(frame.entry_timestamp, unit="ms", utc=True).dt.normalize().nunique()),
            "meets_existing_meta_training_minimum": len(net) >= gate.MIN_SIGNALS and min(positive, nonpositive) >= gate.MIN_CLASS}


def count_union_time_windows(frames):
    if not frames:
        return {"summed_rows": 0, "unique_entry_times": 0, "repeated_entry_times": 0, "nonoverlapping_time_windows": 0}
    # Only combine timing metadata. Different folds' models and targets are not pooled for training.
    times = pd.concat([f[["timestamp", "entry_timestamp", "max_holding_minutes"]] for f in frames], ignore_index=True)
    unique = times.drop_duplicates("entry_timestamp")
    return {"summed_rows": len(times), "unique_entry_times": len(unique),
            "repeated_entry_times": len(times)-len(unique),
            "nonoverlapping_time_windows": len(select_nonoverlapping(unique))}


def run(source, output):
    search, diagnosis = gate.search, gate.diagnosis
    inputs = [source / name for name in ("protocol.json", "analysis.json", "execution.json")]
    inputs += [source / f"{prefix}_fold_{fold}.parquet" for fold in (1, 2, 3)
               for prefix in ("gate_training_signals", "validation_signals", "gated_trades")]
    plan = {"created_utc": search.utc_now(), "source": str(source.resolve()),
            "method": "earliest eligible entry, closed fixed 240-minute horizon, next entry strictly after prior horizon",
            "selection_uses": ["timestamp", "entry_timestamp", "max_holding_minutes"],
            "selection_ignores": ["actual_net_label_r", "actual_holding_minutes", "outcome", "gate_probability"],
            "validation_gate_accepted_scope": "subset decided by the previously frozen gate, then fixed-horizon sampling",
            "minimum_meta_signals": gate.MIN_SIGNALS, "minimum_each_class": gate.MIN_CLASS,
            "statistical_independence_proven": False, "model_training_requested": False,
            "independent_test_evaluated": False, "orders_sent": 0, "collection_started": False,
            "input_sha256": {str(p.resolve()): diagnosis.digest(p) for p in inputs},
            "code_sha256": {name: diagnosis.digest(search.ROOT / name) for name in
                            ("scripts/audit_nonoverlapping_signals.py", "scripts/verify_cost_aware_gate.py", "scripts/run_cost_aware_gate.py")}}
    search.write_json(output / "protocol.json", plan)
    for name in plan["code_sha256"]:
        target = output / "code_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((search.ROOT / name).read_bytes())
    verification = verify(source)
    search.write_json(output / "source_verification.json", verification)
    old = json.loads((source / "analysis.json").read_text(encoding="utf-8"))
    rows, meta_frames, validation_frames, accepted_frames = [], [], [], []
    for fold in old["folds"]:
        number = fold["fold"]
        meta = pd.read_parquet(source / f"gate_training_signals_fold_{number}.parquet")
        validation = pd.read_parquet(source / f"validation_signals_fold_{number}.parquet")
        accepted = validation[validation.gate_accepted].copy()
        meta_frames.append(meta)
        validation_frames.append(validation)
        accepted_frames.append(accepted)
        groups = {}
        for kind, original in (("meta_training", meta), ("validation_all", validation), ("validation_gate_accepted", accepted)):
            selected = select_nonoverlapping(original)
            audit = selected[["timestamp", "entry_timestamp", "max_holding_minutes", "pred_action", "actual_net_label_r"]].copy()
            audit["selection_horizon_end_ms"] = audit.entry_timestamp + HORIZON_MINUTES*MINUTE
            path = output / f"{kind}_fold_{number}.parquet"
            audit.to_parquet(path, index=False)
            saved = pd.read_parquet(path)
            pd.testing.assert_frame_equal(audit, saved)
            if len(saved) > 1:
                if not (saved.entry_timestamp.iloc[1:].to_numpy() > saved.selection_horizon_end_ms.iloc[:-1].to_numpy()).all():
                    raise AssertionError("Selected horizons overlap")
            groups[kind] = {"before": describe(original), "after": describe(selected),
                            "excluded_overlapping": len(original)-len(selected),
                            "retained_fraction": len(selected)/len(original) if len(original) else None}
        row = {"fold": number, "groups": groups, "previous_executed_trades": fold["gated"]["trades"]}
        rows.append(row)
        m = groups["meta_training"]
        print(f"Fold {number}: meta {m['before']['signals']} -> {m['after']['signals']}; positives={m['after']['positive']}; training_minimum={m['after']['meets_existing_meta_training_minimum']}", flush=True)
    for path, checksum in plan["input_sha256"].items():
        if diagnosis.digest(Path(path)) != checksum:
            raise ValueError("Source artifact changed during audit")
    for name, checksum in plan["code_sha256"].items():
        if diagnosis.digest(search.ROOT / name) != checksum:
            raise ValueError("Audit code changed during execution")
    result = {"status": "completed", "finished_utc": search.utc_now(), "folds": rows,
              "training_folds_meeting_minimum": sum(r["groups"]["meta_training"]["after"]["meets_existing_meta_training_minimum"] for r in rows),
              "union_timing_only": {"meta_training": count_union_time_windows(meta_frames),
                                    "validation_all": count_union_time_windows(validation_frames),
                                    "validation_gate_accepted": count_union_time_windows(accepted_frames)},
              "statistical_independence_proven": False, "models_trained": 0, "orders_sent": 0,
              "independent_test_evaluated": False, "model_approved": False, "live_policy": "unchanged_no_trade"}
    search.write_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runtime_reports/cost_aware_gate_20260914"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = "failed"
    try:
        run(args.source, args.output)
        status = "completed"
    finally:
        execution = {"status": status, "elapsed_seconds": round(time.monotonic()-started, 3),
                     "output_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())}
        gate.search.write_json(args.output / "execution.json", execution)
        print(json.dumps(execution))


if __name__ == "__main__":
    main()
