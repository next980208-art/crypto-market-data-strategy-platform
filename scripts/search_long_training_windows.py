"""Predeclared 90/180/365-day training comparison on reused development data."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts import collect_long_window_marks as collection
from scripts import validate_funding_event_execution as replay
from scripts import search_training_variants as windows

search, prior, runner = replay.search, replay.prior, replay.runner
consensus, sizing = replay.consensus, replay.sizing
sampled = replay.research.sampled


def configurations():
    return [{"id": f"{family}_{days}d_entry{threshold}", "family": family, "days": days,
             "threshold_r": threshold, "target_r": 1.5, "risk_per_trade_pct": .005}
            for family in ("shallow", "ridge") for days in (90, 180, 365) for threshold in (.05, .15, .3)]


def numeric_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def select_window(frame, schedule, days):
    spec = {**schedule, "train_start": schedule["validation_start"]-days*search.DAY}
    train, future = search.purged_parts(frame, spec)
    if train.timestamp.min() > spec["train_start"]+search.DAY:
        raise ValueError("Insufficient history for declared training window")
    selected = sampled.sample(train, "gap5")
    if (selected.entry_timestamp+selected.max_holding_minutes*search.MINUTE).max() >= schedule["validation_start"]:
        raise ValueError("Training outcome overlaps evaluation")
    return selected, future, spec


def bounded_read(path, start, end):
    pieces = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16384):
        stamps = batch.column(batch.schema.get_field_index("timestamp"))
        mask = pc.and_(pc.greater_equal(stamps, start), pc.less(stamps, end))
        part = batch.filter(mask)
        if part.num_rows:
            pieces.append(part.to_pandas())
    if not pieces:
        raise ValueError("Empty source interval")
    frame = pd.concat(pieces, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    np.testing.assert_array_equal(frame.timestamp, np.arange(start, end, search.MINUTE))
    return frame


def prepared_frame(plan, schedule, columns, extended, training_code):
    start, end = schedule["validation_start"]-365*search.DAY, schedule["validation_end"]
    parts = []
    # Compute labels before thinning. Warmup and future label bars never become training inputs.
    for left in range(start, end, 30*search.DAY):
        right = min(left+30*search.DAY, end)
        raw = bounded_read(plan["dataset"], left-2*search.DAY, min(right+241*search.MINUTE, end))
        frame = windows.worker.rebuild_labels(raw)
        del raw
        frame, features = extended.add_two_stage_features(frame, training_code)
        if features != plan["features"] or not np.isfinite(frame[features].to_numpy()).all():
            raise ValueError("Feature contract mismatch")
        keep = frame.timestamp.ge(left) & frame.timestamp.lt(right)
        keep &= frame.timestamp.ge(schedule["validation_start"]) | frame.timestamp.mod(5*search.MINUTE).eq(0)
        parts.append(frame.loc[keep, columns].copy())
        del frame
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def save(root, rows, status):
    consensus.save_progress(root, rows, status)
    report = prior.read(root / "results.json")
    paths = list((root / "models").glob("*.joblib"))+list((root / "samples").glob("*.parquet"))+list(root.glob("*.parquet"))
    if (root / "training.json").exists():
        paths.append(root / "training.json")
        report["models_fitted"] = len(prior.read(root / "training.json"))
    report["artifact_sha256"].update({str(p.relative_to(root)): runner.checksum(p) for p in paths})
    search.write_json(root / "results.json", report)


def training(root, plan, verify):
    training_code = search.load("13_train_selective_direction_model.py", "longwindow_features")
    extended = search.load("15_two_stage_walk_forward_model.py", "longwindow_extended")
    columns = list(dict.fromkeys(["timestamp", "entry_timestamp", "entry_price", "entry_rule", "label_horizon_complete",
                                "max_holding_minutes", "stop_distance_pct", "round_trip_cost_bps"]+plan["features"]+sizing.returns.label_columns(1.5)))
    metadata = []
    for schedule in plan["schedules"]:
        n = schedule["fold"]
        print(f"Preparing fold{n} in 30-day chunks", flush=True)
        frame = prepared_frame(plan, schedule, columns, extended, training_code)
        features = plan["features"]
        if not frame.max_holding_minutes.eq(240).all() or not frame.round_trip_cost_bps.eq(10).all():
            raise ValueError("Unexpected labels")
        future_reference = None
        for days in (90, 180, 365):
            train, future, spec = select_window(frame, schedule, days)
            future = future.reset_index(drop=True)
            if future_reference is None:
                future_reference = future.copy()
                path = root / f"validation_{n}.parquet"
                if verify:
                    pd.testing.assert_frame_equal(future, pd.read_parquet(path), check_exact=True)
                else:
                    future.to_parquet(path, index=False)
            else:
                pd.testing.assert_frame_equal(future, future_reference, check_exact=True)
            path = root / "samples" / f"train_{days}d_{n}.parquet"
            sample_times = train[["timestamp", "entry_timestamp", "max_holding_minutes"]].copy()
            feature_digest = numeric_digest(train[features])
            if verify:
                pd.testing.assert_frame_equal(sample_times, pd.read_parquet(path), check_exact=True)
            else:
                sample_times.to_parquet(path, index=False)
            for family in ("shallow", "ridge"):
                values = {"timestamp": future.timestamp.to_numpy()}
                for offset, side in enumerate(("long", "short")):
                    key = f"{family}_{days}d_{n}_{side}"
                    path = root / "models" / (key+".joblib")
                    seed = 902+n*10+offset
                    target = sizing.returns.target_returns(train, side, 1.5, 5)
                    if verify:
                        model = joblib.load(path)
                    else:
                        model = windows.make_model(family+"90", plan, seed)
                        model.fit(train[features], target)
                        joblib.dump(model, path)
                    values[side] = model.predict(future[features])
                    metadata.append({"id": key, "fold": n, "days": days, "family": family, "side": side, "seed": seed,
                                     "train_start_ms": spec["train_start"], "training_rows": len(train), "validation_rows": len(future),
                                     "last_train_outcome_ms": int((train.entry_timestamp+240*search.MINUTE).max()),
                                     "validation_start_ms": schedule["validation_start"], "model_sha256": runner.checksum(path),
                                     "training_features_sha256": feature_digest, "target_sha256": numeric_digest(target),
                                     "target_mean_r": float(target.mean()),
                                     "quality": windows.quality(values[side], sizing.returns.target_returns(future, side, 1.5, 5))})
                    if not verify:
                        search.write_json(root / "training.json", metadata)
                    del model
                path = root / f"predictions_{family}_{days}d_{n}.parquet"
                values = pd.DataFrame(values)
                if verify:
                    pd.testing.assert_frame_equal(values, pd.read_parquet(path), check_exact=True)
                else:
                    values.to_parquet(path, index=False)
                print(f"fold{n} {family} {days}d: train={len(train)} validation={len(future)}", flush=True)
            del train, future
            gc.collect()
        del frame, future_reference
        gc.collect()
    if verify and metadata != prior.read(root / "training.json"):
        raise ValueError("Training metadata replay mismatch")


def worker(root, verify):
    plan, state = prior.read(root / "protocol.json"), prior.read(root / "state.json")
    replay.check_hashes(plan, state, root)
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Package changed: "+name)
    if verify:
        saved = prior.read(root / "results.json")
        if saved["status"] != "completed":
            raise ValueError("Incomplete experiment")
        for name, digest in saved["artifact_sha256"].items():
            if runner.checksum(root / name) != digest:
                raise ValueError("Artifact changed: "+name)
    rows = [{"status": "pending", "config": c} for c in plan["configurations"]]
    try:
        training(root, plan, verify)
    except Exception as exc:
        if verify:
            raise
        error = {"error": str(exc), "traceback": traceback.format_exc(), "stage": "training"}
        search.write_json(root / "training_error.json", error)
        rows = [{"status": "error", "config": c, **error} for c in plan["configurations"]]
        for row in rows:
            search.write_json(root / "trials" / (row["config"]["id"]+".json"), row)
        save(root, rows, "completed_with_errors")
        return
    action = search.load("16_action_profitability_walk_forward_model.py", "longwindow_action")
    backtest = search.load("8_model_signal_backtest.py", "longwindow_backtest")
    markets, frames = {}, {}
    valuation = Path(plan["valuation"])
    for schedule in plan["schedules"]:
        n = schedule["fold"]
        markets[n] = replay.engine.FundingMarket(prior.read(valuation / f"funding_{n}.json"), pd.read_parquet(valuation / f"mark1m_{n}.parquet"))
        frames[n] = pd.read_parquet(root / f"validation_{n}.parquet")
    for i, config in enumerate(plan["configurations"]):
        path = root / "trials" / (config["id"]+".json")
        if not verify:
            rows[i] = {"status": "running", "config": config}
            search.write_json(path, rows[i])
            save(root, rows, "running")
        try:
            folds, diagnostics = [], []
            for schedule in plan["schedules"]:
                n = schedule["fold"]
                values = pd.read_parquet(root / f"predictions_{config['family']}_{config['days']}d_{n}.parquet")
                np.testing.assert_array_equal(values.timestamp, frames[n].timestamp)
                prediction = sizing.returns.prediction_frame(frames[n], values.long, values.short, config["threshold_r"], 0)
                fold = {"fold": n}
                for cost, extra in consensus.COSTS:
                    metrics, summary, tables = replay.evaluate(prediction, action, backtest, extra, markets[n])
                    weekly = replay.cash_weekly(tables["events"], schedule["validation_start"], schedule["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold[cost] = metrics
                    diagnostics.append({"fold": n, "cost_case": cost, "summary": summary, "weekly": weekly})
                    for name, table in tables.items():
                        dest = root / "trades" / f"{config['id']}_{n}_{cost}_{name}.parquet"
                        if verify:
                            pd.testing.assert_frame_equal(table, pd.read_parquet(dest), check_exact=True)
                        else:
                            table.to_parquet(dest, index=False)
                folds.append(fold)
            result = consensus.summarize(config, folds, diagnostics)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(path):
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(path, result)
        rows[i] = result
        if not verify:
            save(root, rows, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if saved["trials"] != rows:
            raise ValueError("Results mismatch")
        search.write_json(root / "verification.json", {"status": "passed", "models_reloaded": 36,
                          "training_samples_rebuilt": 9, "evaluation_frames_rebuilt": 3, "policies_replayed": 18,
                          "backtests_replayed": 162, "trade_ledger_files": 486, "weekly_reconciliations": 162,
                          "results_sha256": runner.checksum(root / "results.json"), "orders_sent": 0})
    else:
        save(root, rows, "completed" if all(r["status"] == "completed" for r in rows) else "completed_with_errors")


def initialize(root, valuation, limits):
    report = prior.read(valuation / "collection.json")
    if report["status"] != "completed" or report["protocol_sha256"] != runner.checksum(valuation / "protocol.json"):
        raise ValueError("Valuation collection incomplete or changed")
    for task in report["tasks"]:
        for name, digest in task["hashes"].items():
            if runner.checksum(valuation / name) != digest:
                raise ValueError("Valuation file changed")
    _, _, _, old, _ = replay.context_run.inputs(search.ROOT / "runtime_reports/market_context_validation_20260918")
    if shutil.disk_usage(search.ROOT).free < 5*1024**3:
        raise ValueError("Insufficient disk space")
    files = [Path(old["dataset"]), valuation / "collection.json", valuation / "protocol.json",
             search.ROOT / "runtime_reports/data_usage_audit_20260920/audit.json"]
    files += list(valuation.glob("*.parquet"))+list(valuation.glob("funding_*.json"))
    files += [Path(m.__file__).resolve() for m in (replay, replay.engine, collection, windows, windows.worker, sampled,
                                                search, prior, runner, consensus, sizing, sizing.returns)]
    files += [Path(__file__).resolve()]+[search.ROOT / name for name in ("4_triple_barrier_labeler.py", "13_train_selective_direction_model.py",
                                 "15_two_stage_walk_forward_model.py", "16_action_profitability_walk_forward_model.py", "8_model_signal_backtest.py")]
    plan = {"created_utc": search.utc_now(), "dataset": old["dataset"], "features": old["features"], "base_config": old["base_config"],
            "schedules": collection.schedules(), "configurations": configurations(), "valuation": str(valuation),
            "hashes": {str(p): runner.checksum(p) for p in files}, "limits": {**limits, "seconds": 600},
            "packages": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "scikit-learn", "xgboost", "pyarrow", "joblib")},
            "sampling": "Clock-aligned 5-minute training rows after causal features/labels; purged boundaries; 30-day chunks with 2-day warmup and 241-minute label tail; identical evaluation rows across all windows",
            "sample_storage": "Timestamp lists plus numeric feature/target SHA256; reconstruct source rows during verification instead of copying every feature",
            "admission": "All 3 folds x 3 costs: >=20 trades, positive mean R/return, cash MDD>=-20%, >=3/4 positive full weeks",
            "limitations": "Reused development data, not independent holdout. Funding mark proxies, cash-only risk guards, no margin/liquidation/fill model.",
            "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False}
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("models", "samples", "trades", "trials"):
        (root / directory).mkdir()
    search.write_json(root / "protocol.json", plan)
    search.write_json(root / "state.json", {"status": "ready", "protocol_sha256": runner.checksum(root / "protocol.json")})
    shutil.copyfile(__file__, root / "code_snapshot.py")
    rows = [{"status": "pending", "config": c} for c in configurations()]
    for row in rows:
        search.write_json(root / "trials" / (row["config"]["id"]+".json"), row)
    save(root, rows, "ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valuation", type=Path, default=Path("runtime_reports/long_window_marks_20260920_v2"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use project subdirectory")
    if args.worker:
        worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 256, "minimum_free_gib": 5}
    if not args.verify:
        initialize(root, args.valuation.resolve(), limits)
    with runner.exclusive_lock(root):
        state = prior.read(root / "state.json")
        folder = root / ("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.search_long_training_windows", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root / "state.json", state)
        try:
            measured = runner.monitor(command, root, folder, 600, limits)
        except BaseException as exc:
            if not args.verify:
                state.update(status="interrupted", reason=type(exc).__name__)
                search.write_json(root / "state.json", state)
            raise
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root / ("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root / "results.json")["status"] if success else "stopped", measurement=measured,
                         finished_utc=search.utc_now())
            search.write_json(root / "state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
