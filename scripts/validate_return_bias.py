"""Chronological training/calibration comparison with frozen conditions and failure logs."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import joblib
import numpy as np
import pandas as pd

from scripts import validate_consensus as consensus

sizing, prior, runner, search = consensus.sizing, consensus.prior, consensus.runner, consensus.search
sampled = sizing.sampled


def configurations():
    return [{"id": f"{family}_{mode}_entry{threshold}", "family": family, "mode": mode,
             "threshold_r": threshold, "risk_per_trade_pct": .005, "target_r": 1.5}
            for family in ("shallow90", "ridge90") for mode in ("raw", "bias_corrected") for threshold in (.05, .15, .3)]


def split_samples(frame, validation_start):
    sampled.overlap.validate_signals(frame)
    boundary = validation_start-15*search.DAY
    outcome_end = frame.entry_timestamp+frame.max_holding_minutes*search.MINUTE
    if not (outcome_end < validation_start).all():
        raise ValueError("Calibration outcomes overlap evaluation")
    train = frame.loc[(frame.timestamp < boundary) & (outcome_end < boundary)].copy()
    calibration = sampled.overlap.select_nonoverlapping(frame.loc[frame.timestamp >= boundary])
    if len(train) < 200 or len(calibration) < 50:
        raise ValueError("Insufficient chronological train/calibration samples")
    return train, calibration


def mean_bias(predicted, actual):
    p, a = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if p.ndim != 1 or p.shape != a.shape or len(p) < 50 or not np.isfinite([p, a]).all():
        raise ValueError("Invalid calibration vectors")
    return float((p-a).mean())


def predictions(frame, values, mode, threshold):
    np.testing.assert_array_equal(frame.timestamp, values.timestamp)
    if mode not in ("raw", "bias_corrected"):
        raise ValueError("Unknown calibration mode")
    prefix = "raw" if mode == "raw" else "corrected"
    return sizing.returns.prediction_frame(frame, values[prefix+"_long"], values[prefix+"_short"], threshold, 0)


def fit_or_verify(root, reference, source, old, schedules, verify):
    metadata = []
    for n in (1, 2, 3):
        train, calibration = split_samples(pd.read_parquet(reference/f"samples/gap5_{n}.parquet"), schedules[n]["validation_start"])
        future = pd.read_parquet(source/f"validation_{n}.parquet")
        features = old["features"]
        for family in ("shallow90", "ridge90"):
            values = {"timestamp": future.timestamp.to_numpy()}
            for offset, side in enumerate(("long", "short")):
                path = root/"models"/f"{family}_{n}_{side}.joblib"
                if verify:
                    model = joblib.load(path)
                else:
                    model = sampled.windows.make_model(family, old, 7242+n*10+offset)
                    model.fit(train[features], sizing.returns.target_returns(train, side, 1.5, 5))
                    joblib.dump(model, path)
                predicted = model.predict(calibration[features])
                actual = sizing.returns.target_returns(calibration, side, 1.5, 5)
                bias = mean_bias(predicted, actual)
                raw = model.predict(future[features])
                values["raw_"+side], values["corrected_"+side] = raw, raw-bias
                metadata.append({"family": family, "fold": n, "side": side, "train_rows": len(train), "calibration_rows": len(calibration),
                                 "calibration_start_ms": schedules[n]["validation_start"]-15*search.DAY,
                                 "last_train_outcome_ms": int((train.entry_timestamp+240*search.MINUTE).max()),
                                 "last_calibration_outcome_ms": int((calibration.entry_timestamp+240*search.MINUTE).max()),
                                 "validation_start_ms": schedules[n]["validation_start"], "bias_r": bias,
                                 "calibration_predicted_mean_r": float(np.mean(predicted)), "calibration_actual_mean_r": float(np.mean(actual)),
                                 "model_sha256": runner.checksum(path)})
            table = pd.DataFrame(values)
            path = root/f"predictions_{family}_{n}.parquet"
            if verify:
                pd.testing.assert_frame_equal(table, pd.read_parquet(path), check_exact=True)
            else:
                table.to_parquet(path, index=False)
                search.write_json(root/"training.json", metadata)
            print(f"{family} fold{n}: train={len(train)}, calibration={len(calibration)}", flush=True)
    if verify and metadata != prior.read(root/"training.json"):
        raise ValueError("Calibration/model replay mismatch")


def save_progress(root, results, status):
    consensus.save_progress(root, results, status)
    report = prior.read(root/"results.json")
    files = list((root/"models").glob("*.joblib"))+list(root.glob("predictions_*.parquet"))
    if (root/"training.json").exists():
        files.append(root/"training.json")
    report["artifact_sha256"].update({str(p.relative_to(root)): runner.checksum(p) for p in files})
    report["models_fitted"] = len(list((root/"models").glob("*.joblib")))
    search.write_json(root/"results.json", report)
    (root/"LEDGER.md").write_text(consensus.ledger.render([root/"results.json"]), encoding="utf-8")


def run_worker(root, verify):
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for path, digest in plan["hashes"].items():
        if runner.checksum(Path(path)) != digest:
            raise ValueError("Frozen input changed: "+path)
    reference = Path(plan["reference"])
    source, _ = sizing.check_reference(reference)
    _, _, _, old = sampled.verify_inputs(reference)
    schedules = {r["fold"]: r["schedule"] for r in prior.read(source/"training.json")}
    if verify:
        saved = prior.read(root/"results.json")
        if saved["status"] != "completed":
            raise ValueError("Incomplete experiment")
        for path, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/path) != digest:
                raise ValueError("Artifact changed: "+path)
    fit_or_verify(root, reference, source, old, schedules, verify)
    action = search.load("16_action_profitability_walk_forward_model.py", "bias_action")
    backtest = search.load("8_model_signal_backtest.py", "bias_backtest")
    results = [{"status": "pending", "config": c} for c in plan["configurations"]]
    for i, config in enumerate(plan["configurations"]):
        path = root/"trials"/(config["id"]+".json")
        if not verify:
            results[i] = {"status": "running", "config": config}
            search.write_json(path, results[i])
            save_progress(root, results, "running")
        try:
            folds, diagnostics = [], []
            for n in (1, 2, 3):
                frame = pd.read_parquet(source/f"validation_{n}.parquet")
                values = pd.read_parquet(root/f"predictions_{config['family']}_{n}.parquet")
                prediction = predictions(frame, values, config["mode"], config["threshold_r"])
                fold = {"fold": n}
                for label, extra in consensus.COSTS:
                    metrics, summary, trades = sizing.evaluate(backtest, action, prediction, .005, extra)
                    weekly = consensus.weekly_results(trades, schedules[n]["validation_start"], schedules[n]["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold[label] = metrics
                    diagnostics.append({"fold": n, "cost_case": label, "summary": summary, "weekly": weekly})
                    trade_path = root/"trades"/f"{config['id']}_{n}_{label}.parquet"
                    if verify:
                        pd.testing.assert_frame_equal(trades, pd.read_parquet(trade_path))
                    else:
                        trades.to_parquet(trade_path, index=False)
                folds.append(fold)
            result = consensus.summarize(config, folds, diagnostics)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if prior.read(path) != result:
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(path, result)
        results[i] = result
        if not verify:
            save_progress(root, results, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if results != saved["trials"]:
            raise ValueError("Results replay mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "models_reloaded": 12, "bias_values_recomputed": 12,
                          "policies_replayed": 12, "metric_backtests": 108, "trade_files": 108, "weekly_reconciliations": 108,
                          "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
    else:
        save_progress(root, results, "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runtime_reports/literature_sampling_search_20260917"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Output must be project subdirectory")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 64, "minimum_free_gib": 5}
    if not args.verify:
        reference = args.reference.resolve()
        sizing.check_reference(reference)
        if shutil.disk_usage(reference).free < 5*1024**3:
            raise ValueError("Insufficient disk space")
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("trials", "trades", "models"):
            (root/folder).mkdir()
        files = [reference/n for n in ("protocol.json", "state.json", "results.json")]
        files += [Path(m.__file__).resolve() for m in (consensus, sizing, sampled, consensus.ledger)]+[Path(__file__).resolve()]
        plan = {"created_utc": search.utc_now(), "reference": str(reference), "configurations": configurations(),
                "hashes": {str(p): runner.checksum(p) for p in files}, "limits": {**limits, "seconds": 600},
                "split": "90-day sampled training split into first 75 days fit and last 15 days bias calibration; purge 240-minute outcomes at boundaries",
                "calibration": "Earliest-first closed-240-minute nonoverlap; minimum 50 samples; bias=mean(predicted-actual); corrected=raw-bias, no refit",
                "model_seeds": "7242+fold*10+side_offset", "minimum_train_rows": 200,
                "admission": ">=20 trades, positive mean R and return, MDD>=-20%, >=3/4 positive full weeks in each fold under total costs 15/20/30bps",
                "limitations": "Reused outer validation; not independent test; no funding/liquidation/mark-to-market; normal stops assumed",
                "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False}
        search.write_json(root/"protocol.json", plan)
        (root/"code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json")})
        rows = [{"status": "pending", "config": c} for c in configurations()]
        for row in rows:
            search.write_json(root/"trials"/(row["config"]["id"]+".json"), row)
        save_progress(root, rows, "ready")
    with runner.exclusive_lock(root):
        state = prior.read(root/"state.json")
        folder = root/("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.validate_return_bias", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root/"state.json", state)
        try:
            measured = runner.monitor(command, root, folder, 600, limits)
        except BaseException as exc:
            if not args.verify:
                state.update(status="interrupted", reason=type(exc).__name__)
                search.write_json(root/"state.json", state)
            raise
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root/"results.json")["status"] if success else "stopped", measurement=measured, finished_utc=search.utc_now())
            search.write_json(root/"state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
