"""Research-informed fixed sampling experiment; no collection, orders or automatic deployment."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import traceback

import joblib
import numpy as np
import pandas as pd

from scripts import search_training_variants as windows
from scripts import search_trade_patterns as prior
from scripts import automatic_research as runner
from scripts import automatic_research_worker as worker
from scripts import run_bounded_profitability_search as search
from scripts import expected_return_research as returns
from scripts import audit_nonoverlapping_signals as overlap

SOURCES = [
    {"title": "Bagging in Financial Machine Learning: Sequential Bootstrapping", "url": "https://hudsonthames.org/bagging-in-financial-machine-learning-sequential-bootstrapping-python/",
     "use": "Motivation: overlapping outcome windows repeat information. This experiment uses deterministic spacing, NOT sequential bootstrapping."},
    {"title": "TimeSeriesSplit", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
     "use": "Chronological evaluation and gap principle. Reuse project's timestamp/outcome purging, not sklearn TimeSeriesSplit."},
    {"title": "The Probability of Backtest Overfitting", "url": "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf",
     "use": "Record all tested configurations and distinguish reused validation from independent evidence. PBO/CSCV not implemented."}]


def configurations():
    return [{"id": f"{family}_{sampling}_entry{threshold}", "variant": f"{family}_{sampling}",
             "family": family, "sampling": sampling, "threshold_r": threshold, "target_r": 1.5}
            for family in ("shallow90", "ridge90") for sampling in ("gap5", "closed240") for threshold in (.05, .15, .3)]


def sample(frame, mode):
    overlap.validate_signals(frame)
    ordered = frame.sort_values("entry_timestamp").reset_index(drop=True)
    if mode == "closed240":
        return overlap.select_nonoverlapping(ordered)
    if mode != "gap5":
        raise ValueError("Unknown sampling mode")
    keep, previous = [], None
    for index, entry in enumerate(ordered.entry_timestamp):
        if previous is None or entry-previous >= 5*search.MINUTE:
            keep.append(index)
            previous = int(entry)
    return ordered.iloc[keep].reset_index(drop=True).copy()


def verify_inputs(root):
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for file, digest in plan["hashes"].items():
        if runner.checksum(Path(file)) != digest:
            raise ValueError("Frozen input changed: "+file)
    reference = Path(plan["reference"])
    _, source, old = windows.check_inputs(reference)
    previous = prior.read(reference/"results.json")
    if previous["status"] != "completed":
        raise ValueError("Reference research incomplete")
    for file, digest in previous["artifact_sha256"].items():
        if runner.checksum(reference/file) != digest:
            raise ValueError("Reference artifact changed: "+file)
    return plan, reference, source, old


def fit_models(root, source, old):
    training = search.load("13_train_selective_direction_model.py", "sample_features")
    extended = search.load("15_two_stage_walk_forward_model.py", "sample_extended")
    parent = Path(prior.read(source/"protocol.json")["source_run"])
    audit = worker.get_result(parent, prior.read(parent/"state.json"), "audit")
    metadata = []
    for fold in audit["folds"]:
        n = fold["fold"]
        raw = pd.read_parquet(old["dataset"], filters=[("timestamp", ">=", max(audit["start_ms"], fold["start"]-240*search.MINUTE)),
                                                     ("timestamp", "<", fold["validation_end"])])
        frame = worker.rebuild_labels(raw.sort_values("timestamp").reset_index(drop=True))
        del raw
        frame, features = extended.add_two_stage_features(frame, training)
        if features != old["features"]:
            raise ValueError("Feature order changed")
        train, future = search.purged_parts(frame, {"train_start": fold["validation_start"]-90*search.DAY,
                                                   "validation_start": fold["validation_start"], "validation_end": fold["validation_end"]})
        validation = pd.read_parquet(source/f"validation_{n}.parquet")
        np.testing.assert_array_equal(validation.timestamp, future.timestamp)
        np.testing.assert_allclose(validation[features], future[features], rtol=0, atol=0)
        del frame, future
        gc.collect()
        for mode in ("gap5", "closed240"):
            selected = sample(train, mode)
            if len(selected) < 200:
                raise ValueError("Fewer than 200 training rows after sampling")
            columns = list(dict.fromkeys(["timestamp", "entry_timestamp", "max_holding_minutes", "stop_distance_pct"]+features+returns.label_columns(1.5)))
            selected[columns].to_parquet(root/"samples"/f"{mode}_{n}.parquet", index=False)
            # Retain the original candidate times to replay sampling without loading labels or raw data.
            train[["timestamp", "entry_timestamp", "max_holding_minutes"]].to_parquet(root/"samples"/f"source_times_{n}.parquet", index=False)
            for family in ("shallow90", "ridge90"):
                variant = f"{family}_{mode}"
                values = {"timestamp": validation.timestamp.to_numpy()}
                for offset, side in enumerate(("long", "short")):
                    key = f"{variant}_{n}_{side}"
                    model = windows.make_model(family, old, 242+(n*10+3)*10+offset)
                    model.fit(selected[features], returns.target_returns(selected, side, 1.5, 5))
                    values[side] = model.predict(validation[features])
                    file = root/"models"/(key+".joblib")
                    joblib.dump(model, file)
                    metadata.append({"id": key, "variant": variant, "family": family, "sampling": mode, "fold": n, "side": side,
                                     "source_rows": len(train), "training_rows": len(selected), "sample_path": f"samples/{mode}_{n}.parquet",
                                     "minimum_entry_gap_minutes": float(selected.entry_timestamp.diff().dropna().min()/search.MINUTE),
                                     "last_train_outcome_ms": int((selected.entry_timestamp+240*search.MINUTE).max()),
                                     "validation_start_ms": fold["validation_start"], "model_sha256": runner.checksum(file)})
                    search.write_json(root/"training.json", metadata)
                    del model
                pd.DataFrame(values).to_parquet(root/f"predictions_{variant}_{n}.parquet", index=False)
                print(f"Fitted {variant}, fold {n}, samples={len(selected)}/{len(train)}", flush=True)
            del selected
        del train, validation
        gc.collect()


def run_worker(root, verify):
    plan, reference, source, old = verify_inputs(root)
    if verify:
        saved = prior.read(root/"results.json")
        for file, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/file) != digest:
                raise ValueError("Artifact changed: "+file)
        for row in prior.read(root/"training.json"):
            n = row["fold"]
            selected = pd.read_parquet(root/row["sample_path"])
            times = pd.read_parquet(root/"samples"/f"source_times_{n}.parquet")
            expected = sample(times, row["sampling"])
            np.testing.assert_array_equal(expected.timestamp, selected.timestamp)
            if len(selected) != row["training_rows"] or not (selected.entry_timestamp+240*search.MINUTE).lt(row["validation_start_ms"]).all():
                raise ValueError("Training sample count or purge mismatch")
            file = root/"models"/(row["id"]+".joblib")
            if runner.checksum(file) != row["model_sha256"]:
                raise ValueError("Model changed")
            model = joblib.load(file)
            validation = pd.read_parquet(source/f"validation_{n}.parquet")
            values = pd.read_parquet(root/f"predictions_{row['variant']}_{n}.parquet")
            np.testing.assert_allclose(model.predict(validation[old["features"]]), values[row["side"]], atol=1e-12, rtol=0)
    else:
        fit_models(root, source, old)
    action = search.load("16_action_profitability_walk_forward_model.py", "sample_action")
    backtest = search.load("8_model_signal_backtest.py", "sample_backtest")
    results = []
    for config in plan["configurations"]:
        file = root/"trials"/(config["id"]+".json")
        if not verify:
            search.write_json(file, {"status": "running", "config": config})
        try:
            result = windows.evaluate(root, source, config, action, backtest, verify)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if prior.read(file) != result:
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(file, result)
        results.append(result)
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if saved["trials"] != results:
            raise ValueError("Results mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "models_reloaded": 24, "sampling_replayed": True,
                          "policies_replayed": 12, "metric_backtests": 72, "trade_files": 36, "models_fitted": 0,
                          "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
    else:
        files = [p for p in root.rglob("*") if p.is_file() and (p.suffix in (".parquet", ".joblib") or p.parent.name == "trials")]
        files.append(root/"training.json")
        search.write_json(root/"results.json", {"status": "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors",
                          "trials": results, "candidate_ids": [r["config"]["id"] for r in results if r.get("candidate")],
                          "orders_sent": 0, "models_fitted": 24, "model_approved": False, "independent_test_evaluated": False,
                          "artifact_sha256": {str(p.relative_to(root)): runner.checksum(p) for p in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runtime_reports/training_variant_search_20260917"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT) or not 60 <= args.max_seconds <= 3600:
        raise ValueError("Use project subdirectory and 60..3600 seconds")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 64, "minimum_free_gib": 5}
    if not args.verify:
        reference = args.reference.resolve()
        windows.check_inputs(reference)
        if shutil.disk_usage(reference).free < 5*1024**3:
            raise ValueError("Insufficient disk space")
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("samples", "models", "trades", "trials"):
            (root/folder).mkdir()
        files = [reference/name for name in ("protocol.json", "state.json", "results.json")]+[Path(__file__).resolve(), Path(windows.__file__).resolve()]
        plan = {"created_utc": search.utc_now(), "reference": str(reference), "hashes": {str(p): runner.checksum(p) for p in files},
                "configurations": configurations(), "literature": SOURCES, "sources_accessed": "2026-09-17",
                "sampling": "Earliest entry first, >=5 minute gap or closed 240 minute outcome nonoverlap; no outcome-based selection",
                "training_days": 90, "minimum_training_rows": 200, "target_r": 1.5, "features": "existing 45 causal features",
                "model_config": "reuse frozen shallow90 and ridge90 from reference; same seeds/params",
                "limits": {**limits, "max_seconds": args.max_seconds},
                "admission": ">=20 trades, mean R/return positive, MDD >=-20% in all 3 folds under 15/20bps total costs",
                "data_reuse": "Repeated validation, exploratory; independence not proven; PBO/CSCV not computed",
                "orders_allowed": False, "collection_allowed": False, "deployment_allowed": False}
        search.write_json(root/"protocol.json", plan)
        (root/"code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json")})
        for config in configurations():
            search.write_json(root/"trials"/(config["id"]+".json"), {"status": "pending", "config": config})
    with runner.exclusive_lock(root):
        state = prior.read(root/"state.json")
        log = root/("verify_log" if args.verify else "run_log")
        log.mkdir(exist_ok=True)
        reason = runner.limit_reason(0, args.max_seconds, 0, runner.output_size(root), shutil.disk_usage(root).free, limits)
        if reason:
            raise ValueError(reason)
        command = [sys.executable, "-u", "-m", "scripts.search_sampled_training", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root/"state.json", state)
        try:
            measured = runner.monitor(command, root, log, args.max_seconds, limits)
        except BaseException as exc:
            if not args.verify:
                state.update(status="interrupted", reason=type(exc).__name__)
                search.write_json(root/"state.json", state)
            raise
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root/"results.json")["status"] if success else "stopped", measurement=measured,
                         finished_utc=search.utc_now())
            search.write_json(root/"state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
