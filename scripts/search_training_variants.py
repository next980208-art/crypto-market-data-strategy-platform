"""Bounded training-window/model comparison and prediction diagnostics. Offline only."""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts import automatic_research as runner
from scripts import automatic_research_worker as worker
from scripts import search_base_targets as base
from scripts import search_trade_patterns as prior
from scripts import expected_return_research as returns
from scripts import audit_nonoverlapping_signals as overlap
from scripts import run_bounded_profitability_search as search

VARIANTS = {"xgb90": {"days": 90, "family": "xgb"},
            "shallow90": {"days": 90, "family": "shallow"},
            "ridge30": {"days": 30, "family": "ridge"},
            "ridge90": {"days": 90, "family": "ridge"}}


def configurations():
    return [{"id": f"{name}_entry{t}", "variant": name, "threshold_r": t, "target_r": 1.5}
            for name in VARIANTS for t in (.05, .15, .3)]


def quality(predicted, actual):
    p, a = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if p.ndim != 1 or p.shape != a.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("Invalid prediction/outcome vectors")
    if not len(p):
        return {"rows": 0, "predicted_mean_r": None, "actual_mean_r": None, "bias_r": None, "mae_r": None, "rank_correlation": None}
    corr = pd.Series(p).rank().corr(pd.Series(a).rank()) if np.std(p) > 0 and np.std(a) > 0 else None
    return {"rows": len(p), "predicted_mean_r": float(p.mean()), "actual_mean_r": float(a.mean()),
            "bias_r": float((p-a).mean()), "mae_r": float(np.abs(p-a).mean()),
            "rank_correlation": float(corr) if corr is not None and np.isfinite(corr) else None}


def diagnostic(frame, predictions):
    np.testing.assert_array_equal(frame.timestamp.to_numpy(), predictions.timestamp.to_numpy())
    long, short = predictions.long.to_numpy(), predictions.short.to_numpy()
    long_wins = long > short
    selected = np.maximum(long, short)
    actual = np.where(long_wins, returns.target_returns(frame, "long", 1.5, 5), returns.target_returns(frame, "short", 1.5, 5))
    # Ties are not actionable, and label outcomes never affect sampling or bin assignment.
    eligible = long != short
    compact = frame.loc[eligible, ["timestamp", "entry_timestamp", "max_holding_minutes"]].copy()
    compact["predicted"] = selected[eligible]
    compact["actual"] = actual[eligible]
    boundaries = [-float("inf"), 0, .05, .15, .3, float("inf")]
    names = ["negative", "0_to_0.05", "0.05_to_0.15", "0.15_to_0.3", "0.3_plus"]
    result = {}
    for scope, rows in (("all_overlapping_signals", compact), ("closed_240m_sample", overlap.select_nonoverlapping(compact))):
        bins = []
        for i, name in enumerate(names):
            group = rows[(rows.predicted >= boundaries[i]) & (rows.predicted < boundaries[i+1])]
            bins.append({"bin": name, **quality(group.predicted, group.actual)})
        result[scope] = {"overall": quality(rows.predicted, rows.actual), "bins": bins}
    return result


def make_model(name, old, seed):
    spec = VARIANTS[name]
    if spec["family"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100., solver="lsqr"))
    params = old["base_config"]["params"].copy()
    if spec["family"] == "shallow":
        params.update(max_depth=2, min_child_weight=100, reg_lambda=20., reg_alpha=1.)
    return XGBRegressor(objective="reg:squarederror", eval_metric="mae", tree_method="hist", n_jobs=2,
                        random_state=seed, **params)


def check_inputs(root):
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for file, digest in plan["hashes"].items():
        if runner.checksum(Path(file)) != digest:
            raise ValueError("Frozen input changed: "+file)
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Package changed: "+name)
    source = Path(plan["source"])
    _, old = base.verify_inputs(source)
    report = prior.read(source/"results.json")
    if report["status"] != "completed":
        raise ValueError("Reference incomplete")
    for file, digest in report["artifact_sha256"].items():
        if runner.checksum(source/file) != digest:
            raise ValueError("Reference artifact changed: "+file)
    return plan, source, old


def train(root, source, old):
    training = search.load("13_train_selective_direction_model.py", "window_features")
    extended = search.load("15_two_stage_walk_forward_model.py", "window_extended")
    old_training = prior.read(source/"training.json")
    audit_source = Path(prior.read(source/"protocol.json")["source_run"])
    audit = worker.get_result(audit_source, prior.read(audit_source/"state.json"), "audit")
    diagnostics, metadata = [], []
    for row in old_training:
        number, fold = row["fold"], row["schedule"]
        validation = pd.read_parquet(source/f"validation_{number}.parquet")
        reference = pd.read_parquet(source/f"predictions_{number}_1.5.parquet")
        diag = {"fold": number, "reference_validation": diagnostic(validation, reference)}
        raw = pd.read_parquet(old["dataset"], filters=[("timestamp", ">=", max(audit["start_ms"], fold["start"]-240*search.MINUTE)),
                                                     ("timestamp", "<", fold["validation_end"])])
        frame = worker.rebuild_labels(raw.sort_values("timestamp").reset_index(drop=True))
        del raw
        frame, features = extended.add_two_stage_features(frame, training)
        if features != old["features"]:
            raise ValueError("Feature order changed")
        # Assert identical evaluation rows/features despite the longer training selection.
        _, same_validation = search.purged_parts(frame, base.training_spec(fold))
        np.testing.assert_array_equal(same_validation.timestamp, validation.timestamp)
        np.testing.assert_allclose(same_validation[features], validation[features], rtol=0, atol=0)
        del same_validation
        reference_train, _ = search.purged_parts(frame, base.training_spec(fold))
        diag["reference_sides"] = []
        for side in ("long", "short"):
            model = XGBRegressor(n_jobs=2)
            model.load_model(source/"models"/f"fold{number}_target1.5_{side}.json")
            actual = returns.target_returns(reference_train, side, 1.5, 5)
            future_actual = returns.target_returns(validation, side, 1.5, 5)
            diag["reference_sides"].append({"side": side, "in_sample": quality(model.predict(reference_train[features]), actual),
                                            "future": quality(reference[side], future_actual),
                                            "constant_train_mean_future_mae": float(np.abs(future_actual-actual.mean()).mean())})
        del reference_train
        diagnostics.append(diag)
        search.write_json(root/"diagnostics.json", diagnostics)
        for name, variant in VARIANTS.items():
            spec = {**base.training_spec(fold), "train_start": fold["validation_start"]-variant["days"]*search.DAY}
            train_frame, unused = search.purged_parts(frame, spec)
            del unused
            x = train_frame[features]
            if not np.isfinite(x.to_numpy()).all():
                raise ValueError("Nonfinite training features")
            values = {"timestamp": validation.timestamp.to_numpy()}
            for offset, side in enumerate(("long", "short")):
                key = f"{name}_{number}_{side}"
                model = make_model(name, old, 242+(number*10+3)*10+offset)
                target = returns.target_returns(train_frame, side, 1.5, 5)
                model.fit(x, target)
                values[side] = model.predict(validation[features])
                path = root/"models"/(key+".joblib")
                joblib.dump(model, path)
                metadata.append({"id": key, "variant": name, "fold": number, "side": side,
                                 "training_rows": len(x), "train_start_ms": spec["train_start"],
                                 "last_train_outcome_ms": int((train_frame.entry_timestamp+240*search.MINUTE).max()),
                                 "validation_start_ms": fold["validation_start"], "model_sha256": runner.checksum(path),
                                 "training_fit": quality(model.predict(x), target),
                                 "future": quality(values[side], returns.target_returns(validation, side, 1.5, 5))})
                search.write_json(root/"training.json", metadata)
                del model
            pd.DataFrame(values).to_parquet(root/f"predictions_{name}_{number}.parquet", index=False)
            print(f"Fitted {name}, fold {number}", flush=True)
            del x, train_frame
            gc.collect()
        del frame, validation
        gc.collect()


def evaluate(root, source, config, action, backtest, verify):
    folds = []
    for number in (1, 2, 3):
        frame = pd.read_parquet(source/f"validation_{number}.parquet")
        values = pd.read_parquet(root/f"predictions_{config['variant']}_{number}.parquet")
        np.testing.assert_array_equal(frame.timestamp, values.timestamp)
        prediction = returns.prediction_frame(frame, values.long, values.short, config["threshold_r"], 0)
        folds.append({"fold": number, "normal": action.evaluate_candidate(backtest, prediction, 5, 1.5),
                      "stress": action.evaluate_candidate(backtest, prediction, 10, 1.5)})
        args = action.backtest_args(5)
        args.take_profit_r = 1.5
        trades, _, _ = backtest.simulate(prediction[prediction.pred_action.ne("no_trade")], args)
        trades = pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"])
        path = root/"trades"/f"{config['id']}_{number}.parquet"
        if verify:
            pd.testing.assert_frame_equal(trades, pd.read_parquet(path))
        else:
            trades.to_parquet(path, index=False)
    result = prior.summarize(config, folds)
    result.update(status="completed", independent_test_evaluated=False, model_approved=False, orders_sent=0)
    return result


def run_worker(root, verify):
    plan, source, old = check_inputs(root)
    action = search.load("16_action_profitability_walk_forward_model.py", "window_action")
    backtest = search.load("8_model_signal_backtest.py", "window_backtest")
    if verify:
        saved = prior.read(root/"results.json")
        for file, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/file) != digest:
                raise ValueError("Artifact changed: "+file)
        for row in prior.read(root/"training.json"):
            if row["last_train_outcome_ms"] >= row["validation_start_ms"]:
                raise ValueError("Training outcomes cross evaluation boundary")
            file = root/"models"/(row["id"]+".joblib")
            if runner.checksum(file) != row["model_sha256"]:
                raise ValueError("Model changed")
            # Load only this run's trusted, hash-verified local model files.
            model = joblib.load(file)
            validation = pd.read_parquet(source/f"validation_{row['fold']}.parquet")
            values = pd.read_parquet(root/f"predictions_{row['variant']}_{row['fold']}.parquet")
            np.testing.assert_allclose(model.predict(validation[old["features"]]), values[row["side"]], rtol=0, atol=1e-12)
        diagnostics = prior.read(root/"diagnostics.json")
        for row in diagnostics:
            n = row["fold"]
            recomputed = diagnostic(pd.read_parquet(source/f"validation_{n}.parquet"), pd.read_parquet(source/f"predictions_{n}_1.5.parquet"))
            if row["reference_validation"] != recomputed:
                raise ValueError("Diagnostic replay mismatch")
    else:
        train(root, source, old)
    trials = []
    for config in plan["configurations"]:
        path = root/"trials"/(config["id"]+".json")
        if not verify:
            search.write_json(path, {"status": "running", "config": config})
        try:
            result = evaluate(root, source, config, action, backtest, verify)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(path):
                raise ValueError("Trial replay differs")
        else:
            search.write_json(path, result)
        trials.append(result)
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if trials != saved["trials"]:
            raise ValueError("Summary differs")
        search.write_json(root/"verification.json", {"status": "passed", "models_reloaded": 24, "models_fitted": 0,
                          "policies_replayed": 12, "metric_backtests": 72, "trade_files": 36,
                          "diagnostic_folds_replayed": 3, "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
    else:
        files = [p for p in root.rglob("*") if p.is_file() and (p.suffix in (".joblib", ".parquet") or p.parent.name == "trials")]
        files += [root/"diagnostics.json", root/"training.json"]
        search.write_json(root/"results.json", {"status": "completed" if all(r["status"] == "completed" for r in trials) else "completed_with_errors",
                          "trials": trials, "candidate_ids": [r["config"]["id"] for r in trials if r.get("candidate")],
                          "models_fitted": 24, "orders_sent": 0, "model_approved": False, "independent_test_evaluated": False,
                          "artifact_sha256": {str(p.relative_to(root)): runner.checksum(p) for p in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runtime_reports/base_target_search_20260916"))
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
        source = args.source.resolve()
        base.verify_inputs(source)
        if shutil.disk_usage(source).free < 5*1024**3:
            raise ValueError("Insufficient disk space")
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("models", "trials", "trades"):
            (root/folder).mkdir()
        files = [source/name for name in ("protocol.json", "state.json", "results.json", "training.json")]
        files += [Path(__file__).resolve(), search.ROOT/"scripts/research_ledger.py"]
        plan = {"created_utc": search.utc_now(), "source": str(source), "hashes": {str(p): runner.checksum(p) for p in files},
                "configurations": configurations(), "variants": VARIANTS, "limits": {**limits, "max_seconds": args.max_seconds},
                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "pyarrow")},
                "ridge": {"alpha": 100., "solver": "lsqr", "scale_fit": "training_only"},
                "shallow_changes": {"max_depth": 2, "min_child_weight": 100, "reg_lambda": 20., "reg_alpha": 1.},
                "data_reuse": "Repeatedly inspected validation, exploratory only. No independent final-test claims.",
                "admission": "All three folds: >=20 trades, mean R/return >0, MDD >=-20%, at both 15/20bps total costs",
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
        command = [sys.executable, "-u", "-m", "scripts.search_training_variants", "--output", str(root), "--worker"]
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
            if not success:
                for path in (root/"trials").glob("*.json"):
                    row = prior.read(path)
                    if row["status"] == "running":
                        row.update(status="interrupted", reason=measured["stop_reason"] or "worker_failed")
                        search.write_json(path, row)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
