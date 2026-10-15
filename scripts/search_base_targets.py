"""Controlled offline comparison of base return-regression targets. No orders or collection."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from scripts import automatic_research as runner
from scripts import automatic_research_worker as worker
from scripts import run_bounded_profitability_search as search
from scripts import expected_return_research as returns
from scripts import search_trade_patterns as prior


def configurations():
    return [{"id": f"target{target}_entry{threshold}", "target_r": target, "threshold_r": threshold}
            for target in (1.0, 1.5) for threshold in (.05, .15, .3)]


def training_spec(fold):
    return {"train_start": fold["validation_start"]-30*search.DAY,
            "validation_start": fold["validation_start"], "validation_end": fold["validation_end"]}


def verify_inputs(root):
    plan = prior.read(root/"protocol.json")
    state = prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for file, digest in plan["code_sha256"].items():
        if runner.checksum(search.ROOT/file) != digest:
            raise ValueError("Source changed: "+file)
    source = Path(plan["source_run"])
    old, old_state = prior.read(source/"protocol.json"), prior.read(source/"state.json")
    if runner.checksum(source/"protocol.json") != plan["source_protocol_sha256"]:
        raise ValueError("Source protocol changed")
    if runner.checksum(source/"state.json") != plan["source_state_sha256"]:
        raise ValueError("Source state changed")
    runner.verify_resume(source, old, old_state)
    return plan, old


def prepare_and_fit(root, plan, old):
    source = Path(plan["source_run"])
    state = prior.read(source/"state.json")
    audit = worker.get_result(source, state, "audit")
    training = search.load("13_train_selective_direction_model.py", "targets_features")
    extended = search.load("15_two_stage_walk_forward_model.py", "targets_extended")
    metadata = []
    for fold in audit["folds"]:
        number = fold["fold"]
        # Use the identical warm-up history as the reference, including recursive EMA inputs.
        raw = pd.read_parquet(old["dataset"], filters=[("timestamp", ">=", max(audit["start_ms"], fold["start"]-240*search.MINUTE)),
                                                     ("timestamp", "<", fold["validation_end"])])
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        frame = worker.rebuild_labels(raw)
        frame, features = extended.add_two_stage_features(frame, training)
        if features != old["features"] or not np.isfinite(frame[features].to_numpy()).all():
            raise ValueError("Feature contract mismatch")
        train, validation = search.purged_parts(frame, training_spec(fold))
        validation.to_parquet(root/f"validation_{number}.parquet", index=False)
        reference = worker.get_result(source, state, f"prepare_{number}")
        row = {"fold": number, "schedule": fold, "train_rows": len(train), "validation_rows": len(validation),
               "last_train_outcome_ms": int((train.entry_timestamp+240*search.MINUTE).max()),
               "reference_baseline": reference["baseline"], "models": []}
        for target in (1.0, 1.5):
            estimates = {}
            for offset, side in enumerate(("long", "short")):
                key = f"fold{number}_target{target}_{side}"
                seed = 242+(number*10+3)*10+offset
                model = XGBRegressor(objective="reg:squarederror", eval_metric="mae", tree_method="hist", n_jobs=2,
                                     random_state=seed, **old["base_config"]["params"])
                labels = returns.target_returns(train, side, target, 5)
                model.fit(train[features], labels)
                model.save_model(root/"models"/(key+".json"))
                estimates[side] = model.predict(validation[features])
                row["models"].append({"id": key, "side": side, "target_r": target, "seed": seed,
                                      "train_target_mean_r": float(labels.mean())})
            pd.DataFrame({"timestamp": validation.timestamp.to_numpy(), **estimates}).to_parquet(
                root/f"predictions_{number}_{target}.parquet", index=False)
        metadata.append(row)
        search.write_json(root/"training.json", metadata)
        print(f"Trained fold {number}/3: four base models", flush=True)
    return metadata


def evaluate(root, old, config, action, backtest, verify):
    folds = []
    for number in (1, 2, 3):
        validation = pd.read_parquet(root/f"validation_{number}.parquet")
        values = pd.read_parquet(root/f"predictions_{number}_{config['target_r']}.parquet")
        np.testing.assert_array_equal(validation.timestamp.to_numpy(), values.timestamp.to_numpy())
        prediction = returns.prediction_frame(validation, values.long.to_numpy(), values.short.to_numpy(), config["threshold_r"], 0)
        normal = action.evaluate_candidate(backtest, prediction, 5, config["target_r"])
        stress = action.evaluate_candidate(backtest, prediction, 10, config["target_r"])
        folds.append({"fold": number, "normal": normal, "stress": stress})
        args = action.backtest_args(5)
        args.take_profit_r = config["target_r"]
        trades, _, _ = backtest.simulate(prediction[prediction.pred_action.ne("no_trade")], args)
        trades = pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"])
        file = root/"trades"/f"{config['id']}_fold{number}.parquet"
        if verify:
            pd.testing.assert_frame_equal(pd.read_parquet(file), trades)
        else:
            trades.to_parquet(file, index=False)
    result = prior.summarize(config, folds)
    result.update(status="completed", model_approved=False, orders_sent=0, independent_test_evaluated=False)
    return result


def run_worker(root, verify):
    plan, old = verify_inputs(root)
    action = search.load("16_action_profitability_walk_forward_model.py", "targets_action")
    backtest = search.load("8_model_signal_backtest.py", "targets_backtest")
    if verify:
        saved = prior.read(root/"results.json")
        for name, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/name) != digest:
                raise ValueError("Artifact changed: "+name)
        metadata = prior.read(root/"training.json")
        for row in metadata:
            number = row["fold"]
            validation = pd.read_parquet(root/f"validation_{number}.parquet")
            if row["last_train_outcome_ms"] >= row["schedule"]["validation_start"]:
                raise ValueError("Training purge failed")
            for spec in row["models"]:
                model = XGBRegressor(n_jobs=2)
                model.load_model(root/"models"/(spec["id"]+".json"))
                values = pd.read_parquet(root/f"predictions_{number}_{spec['target_r']}.parquet")
                np.testing.assert_allclose(model.predict(validation[old["features"]]), values[spec["side"]], atol=1e-12, rtol=0)
    else:
        metadata = prepare_and_fit(root, plan, old)
    results = []
    for config in plan["configurations"]:
        path = root/"trials"/(config["id"]+".json")
        if not verify:
            search.write_json(path, {"status": "running", "config": config})
        try:
            result = evaluate(root, old, config, action, backtest, verify)
            # The reference target/threshold must exactly reproduce the earlier unfiltered baseline.
            if config["target_r"] == 1.5 and config["threshold_r"] == .05:
                if [f["normal"] for f in result["folds"]] != [m["reference_baseline"] for m in metadata]:
                    raise ValueError("Reference 1.5R baseline not reproduced")
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(path):
                raise ValueError("Replay differs")
        else:
            search.write_json(path, result)
        results.append(result)
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if results != saved["trials"]:
            raise ValueError("Summary differs")
        search.write_json(root/"verification.json", {"status": "passed", "models_reloaded": 12, "policies_replayed": 6,
                          "metric_backtests": 36, "trade_files": 18, "results_sha256": runner.checksum(root/"results.json"),
                          "models_fitted": 0, "orders_sent": 0})
    else:
        artifacts = [p for p in root.rglob("*") if p.is_file() and (p.suffix == ".parquet" or p.parent.name in ("models", "trials"))]
        artifacts.append(root/"training.json")
        search.write_json(root/"results.json", {"status": "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors",
                          "trials": results, "candidate_ids": [r["config"]["id"] for r in results if r.get("candidate")],
                          "models_fitted": 12, "orders_sent": 0, "model_approved": False, "independent_test_evaluated": False,
                          "reference_reproduced": next(r for r in results if r["config"]["id"] == "target1.5_entry0.05")["status"] == "completed",
                          "artifact_sha256": {str(p.relative_to(root)): runner.checksum(p) for p in artifacts}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("runtime_reports/automatic_research_20260915_v2"))
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT) or not 60 <= args.max_seconds <= 3600:
        raise ValueError("Use project subdirectory and 60..3600 second budget")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 128, "minimum_free_gib": 5}
    if not args.verify:
        source = args.source.resolve()
        old, old_state = prior.read(source/"protocol.json"), prior.read(source/"state.json")
        runner.verify_resume(source, old, old_state)
        if old_state["status"] != "completed":
            raise ValueError("Reference not completed")
        if shutil.disk_usage(source).free < 5*1024**3:
            raise ValueError("Insufficient disk space")
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("models", "trials", "trades"):
            (root/folder).mkdir()
        files = ["scripts/search_base_targets.py", "scripts/search_trade_patterns.py"]
        plan = {"created_utc": search.utc_now(), "source_run": str(source), "source_protocol_sha256": runner.checksum(source/"protocol.json"),
                "source_state_sha256": runner.checksum(source/"state.json"), "code_sha256": {p: runner.checksum(search.ROOT/p) for p in files},
                "configurations": configurations(), "limits": {**limits, "max_seconds": args.max_seconds},
                "training": "same 30 days, 45 features, XGBoost params/seeds as reference final chunk; targets 1R/1.5R",
                "costs_bps": [15, 20], "admission": ">=20 trades, positive mean R and return, MDD >=-20% in all 3 folds and both costs",
                "data_reuse": "Previously inspected historical validation; not independent", "orders_allowed": False}
        search.write_json(root/"protocol.json", plan)
        (root/"code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json")})
        for config in configurations():
            search.write_json(root/"trials"/(config["id"]+".json"), {"status": "pending", "config": config})
    with runner.exclusive_lock(root):
        state = prior.read(root/"state.json")
        log = root/("verify_log" if args.verify else "run_log")
        log.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.search_base_targets", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state["status"] = "running"
            search.write_json(root/"state.json", state)
        try:
            measured = runner.monitor(command, root, log, args.max_seconds, limits)
        except BaseException as exc:
            if not args.verify:
                state.update(status="interrupted", error=type(exc).__name__)
                search.write_json(root/"state.json", state)
            raise
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root/"results.json")["status"] if success else "stopped", measurement=measured)
            search.write_json(root/"state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
