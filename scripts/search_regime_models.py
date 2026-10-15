"""Finite regime/model research with a complete per-trial journal; never places orders."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from scripts import automatic_research as runner
from scripts import run_bounded_profitability_search as search
from scripts import run_cost_aware_gate as gate
from scripts import search_trade_patterns as prior

REGIMES = ("all", "low_volatility", "high_volatility", "low_volume", "high_volume", "trend_aligned", "trend_opposed")


def configurations():
    return [{"id": f"{model}_p{threshold}_{regime}", "model": model, "threshold": threshold, "regime": regime}
            for model in ("logistic", "tree") for threshold in (.45, .5, .55) for regime in REGIMES]


def cutoffs(meta):
    x = meta[gate.GATE_COLUMNS]
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Nonfinite training features")
    return {"volatility_15m": float(x.volatility_15m.median()), "volume_ratio_60": float(x.volume_ratio_60.median())}


def regime_mask(x, limits, regime):
    if regime == "all":
        return np.ones(len(x), dtype=bool)
    if regime in ("low_volatility", "high_volatility"):
        low = x.volatility_15m.to_numpy() <= limits["volatility_15m"]
        return low if regime == "low_volatility" else ~low
    if regime in ("low_volume", "high_volume"):
        low = x.volume_ratio_60.to_numpy() <= limits["volume_ratio_60"]
        return low if regime == "low_volume" else ~low
    direction = np.where(x.is_long.to_numpy() == 1, 1, -1)
    signed = x.return_60m.to_numpy()*direction
    if regime == "trend_aligned":
        return signed > 0
    if regime == "trend_opposed":
        return signed < 0
    raise ValueError("Unknown market regime")


def apply_policy(frame, probability, limits, config):
    output = prior.filter_signals(frame, probability, {"threshold": config["threshold"], "direction": "both"})
    output.loc[~regime_mask(gate.gate_features(frame), limits, config["regime"]), "pred_action"] = "no_trade"
    return output


def make_model(name):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=.1, solver="lbfgs", max_iter=1000))
    if name == "tree":
        return DecisionTreeClassifier(max_depth=2, min_samples_leaf=40, class_weight=None, random_state=915)
    raise ValueError("Unknown model family")


def disposition(row):
    if row["candidate"]:
        return "research_candidate"
    if row["normal"]["positive_folds"] > 0:
        return "partial_positive_rejected"
    return "rejected"


def verify_inputs(root):
    plan = prior.read(root/"protocol.json")
    state = prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    for name, digest in plan["input_sha256"].items():
        if runner.checksum(Path(name)) != digest:
            raise ValueError("Frozen input changed: "+name)
    if importlib.metadata.version("joblib") != plan["joblib_version"]:
        raise ValueError("Joblib version changed")
    source = Path(plan["source_patterns"])
    _, frames = prior.inputs(source)
    previous = prior.read(source/"results.json")
    for name, digest in previous["artifact_sha256"].items():
        if runner.checksum(source/name) != digest:
            raise ValueError("Previous output changed: "+name)
    return plan, frames


def model_artifacts(root, frames, verify):
    metadata_path = root/"models.json"
    metadata = prior.read(metadata_path) if verify else {}
    probabilities = {}
    for name in ("logistic", "tree"):
        for number, (meta, validation) in enumerate(frames, 1):
            key = f"{name}_{number}"
            path = root/"models"/(key+".joblib")
            probability_path = root/"models"/(key+"_probabilities.parquet")
            x, target = meta[gate.GATE_COLUMNS], meta.actual_net_label_r.gt(0).astype(int)
            limits = cutoffs(meta)
            if verify:
                if metadata[key]["cutoffs"] != limits or metadata[key]["training_rows"] != len(meta):
                    raise ValueError("Training metadata mismatch")
                if runner.checksum(path) != metadata[key]["model_sha256"]:
                    raise ValueError("Saved model hash mismatch")
                # Only load this run's locally produced, hash-verified artifacts, never uploaded pickle files.
                model = joblib.load(path)
                saved = pd.read_parquet(probability_path)
                np.testing.assert_array_equal(saved.timestamp.to_numpy(), validation.timestamp.to_numpy())
                probability = model.predict_proba(gate.gate_features(validation))[:, 1]
                np.testing.assert_allclose(saved.probability.to_numpy(), probability, atol=1e-12)
            else:
                model = make_model(name)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(x, target)
                np.testing.assert_array_equal(model.classes_, [0, 1])
                probability = model.predict_proba(gate.gate_features(validation))[:, 1]
                joblib.dump(model, path)
                pd.DataFrame({"timestamp": validation.timestamp.to_numpy(), "probability": probability}).to_parquet(probability_path, index=False)
                if name == "tree":
                    (path.with_suffix(".txt")).write_text(export_text(model, feature_names=gate.GATE_COLUMNS), encoding="utf-8")
                metadata[key] = {"family": name, "training_rows": len(meta), "positive_rows": int(target.sum()),
                                 "cutoffs": limits, "columns": gate.GATE_COLUMNS, "model_sha256": runner.checksum(path),
                                 "model_path": str(path.relative_to(root)), "probability_path": str(probability_path.relative_to(root))}
                search.write_json(metadata_path, metadata)
            probabilities[key] = probability
    return metadata, probabilities


def compare(root, frames, probabilities, config, action, backtest, save):
    folds = []
    for number, (meta, validation) in enumerate(frames, 1):
        limits = cutoffs(meta)
        filtered = apply_policy(validation, probabilities[f"{config['model']}_{number}"], limits, config)
        folds.append({"fold": number, "cutoffs": limits,
                      "normal": action.evaluate_candidate(backtest, filtered, 5, 1.5),
                      "stress": action.evaluate_candidate(backtest, filtered, 10, 1.5)})
        if save:
            args = action.backtest_args(5)
            args.take_profit_r = 1.5
            trades, _, _ = backtest.simulate(filtered[filtered.pred_action.ne("no_trade")], args)
            pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"]).to_parquet(
                root/"trades"/f"{config['id']}_fold{number}.parquet", index=False)
        else:
            args = action.backtest_args(5)
            args.take_profit_r = 1.5
            trades, _, _ = backtest.simulate(filtered[filtered.pred_action.ne("no_trade")], args)
            expected = pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"])
            pd.testing.assert_frame_equal(pd.read_parquet(root/"trades"/f"{config['id']}_fold{number}.parquet"), expected)
    result = prior.summarize(config, folds)
    result.update(status="completed", classification=disposition(result), orders_sent=0, model_approved=False,
                  independent_test_evaluated=False)
    return result


def write_report(root, results, previous):
    lines = ["# Model and Regime Research Journal", "", "All completed/rejected/partial-positive/error trials are retained.",
             "Exploratory reused validation; no independent profitability claim, orders or deployment.", "",
             "42 fixed policies; two gate families, three thresholds, seven regimes. Six gate model fits.",
             "Normal costs: 10+5bps. Stress costs: 10+10bps. Each fold must have >=20 trades, positive return/mean R and MDD >=-20%.",
             "", "## Current Batch", "", "| Trial | Classification | Returns % (fold 1/2/3) | Trades | Stress returns % | Reasons |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in results:
        if row["status"] != "completed":
            lines.append(f"| {row['config']['id']} | {row['status']} | - | - | - | {row.get('error', '')} |")
            continue
        normal = "/".join(f"{v:.3f}" for v in row["normal"]["fold_returns_pct"])
        stress = "/".join(f"{v:.3f}" for v in row["stress"]["fold_returns_pct"])
        trades = "/".join(str(f["normal"]["trades"]) for f in row["folds"])
        lines.append(f"| {row['config']['id']} | {row['classification']} | {normal} | {trades} | {stress} | {'; '.join(row['rejection_reasons'])} |")
    lines += ["", "## Previous 36 Policies (Preserved, Not Rerun)", "",
              "| Trial | Classification | Returns % | Reasons |", "| --- | --- | --- | --- |"]
    for row in previous["trials"]:
        values = "/".join(f"{v:.3f}" for v in row["normal"]["fold_returns_pct"])
        lines.append(f"| {row['config']['id']} | {disposition(row)} | {values} | {'; '.join(row['rejection_reasons'])} |")
    lines += ["", "Partial positives and zero-trade/zero-return results are not approved candidates.",
              "Models, per-fold trade files, input hashes and each per-trial JSON are retained even for rejected policies.",
              "The same validation has been searched repeatedly; selection bias/multiple testing are not corrected here."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_worker(root, verify):
    plan, frames = verify_inputs(root)
    previous = prior.read(Path(plan["source_patterns"])/"results.json")
    if verify:
        saved = prior.read(root/"results.json")
        for name, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/name) != digest:
                raise ValueError("Artifact changed: "+name)
    metadata, probabilities = model_artifacts(root, frames, verify)
    action = search.load("16_action_profitability_walk_forward_model.py", "regime_action")
    backtest = search.load("8_model_signal_backtest.py", "regime_backtest")
    results = []
    for config in plan["configurations"]:
        file = root/"trials"/(config["id"]+".json")
        if verify:
            stored = prior.read(file)
            if stored["status"] != "completed":
                raise ValueError("Cannot verify failed/incomplete trial: "+config["id"])
            result = compare(root, frames, probabilities, config, action, backtest, False)
            if result != stored:
                raise ValueError("Trial replay mismatch: "+config["id"])
        else:
            search.write_json(file, {"config": config, "status": "running", "started_utc": search.utc_now()})
            try:
                result = compare(root, frames, probabilities, config, action, backtest, True)
            except Exception as exc:
                result = {"config": config, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(), "orders_sent": 0}
            search.write_json(file, result)
        results.append(result)
        if not verify:
            write_report(root, results, previous)
        print(f"{len(results)}/42 {config['id']}: {result.get('classification', result['status'])}", flush=True)
    if verify:
        if results != saved["trials"]:
            raise ValueError("Summary and trial records differ")
        search.write_json(root/"verification.json", {"status": "passed", "verified_utc": search.utc_now(),
                          "policies_replayed": len(results), "metric_backtests_replayed": len(results)*6,
                          "trade_files_replayed": len(results)*3, "models_fitted": 0,
                          "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
        return
    candidates = [r["config"]["id"] for r in results if r.get("candidate")]
    partial = [r["config"]["id"] for r in results if r.get("classification") == "partial_positive_rejected"]
    payload = {"status": "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors",
               "finished_utc": search.utc_now(), "trials": results, "candidate_ids": candidates, "partial_positive_ids": partial,
               "previous_trial_count": len(previous["trials"]), "models_fitted": len(metadata),
               "orders_sent": 0, "collection_started": False, "model_approved": False, "independent_test_evaluated": False,
               "artifact_sha256": {str(p.relative_to(root)): runner.checksum(p) for folder in ("models", "trials", "trades")
                                   for p in (root/folder).rglob("*") if p.is_file()}}
    payload["artifact_sha256"]["models.json"] = runner.checksum(root/"models.json")
    search.write_json(root/"results.json", payload)
    search.write_json(root/"candidate_index.json", {"candidate_ids": candidates, "partial_positive_ids": partial,
                      "all_trial_ids": [r["config"]["id"] for r in results], "model_approved": False,
                      "results_sha256": runner.checksum(root/"results.json")})


def execute(args):
    root = args.output.resolve()
    if not root.is_relative_to(search.ROOT) or root == search.ROOT or not 60 <= args.max_seconds <= 3600:
        raise ValueError("Use a project subdirectory and 60..3600 seconds")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    if args.status:
        print((root/"state.json").read_text(encoding="utf-8"))
        counts = {}
        for p in (root/"trials").glob("*.json"):
            status = prior.read(p)["status"]
            counts[status] = counts.get(status, 0)+1
        print(json.dumps({"trials": counts}), flush=True)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 64, "minimum_free_gib": 5}
    if not args.verify:
        source = args.source.resolve()
        prior.inputs(source)
        previous = prior.read(source/"results.json")
        if previous["status"] != "completed":
            raise ValueError("Previous pattern search incomplete")
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("models", "trials", "trades"):
            (root/folder).mkdir()
        files = [source/"protocol.json", source/"results.json", Path(__file__).resolve(), Path(prior.__file__).resolve()]
        plan = {"created_utc": search.utc_now(), "source_patterns": str(source), "configurations": configurations(),
                "input_sha256": {str(p): runner.checksum(p) for p in files}, "joblib_version": importlib.metadata.version("joblib"),
                "model_parameters": {"logistic": {"C": .1, "solver": "lbfgs", "max_iter": 1000},
                                     "tree": {"max_depth": 2, "min_samples_leaf": 40, "random_state": 915}},
                "regime_thresholds": "Per-fold training meta sample medians only; never fit to validation",
                "selection": "All 3 folds: >=20 trades, positive mean R/return, MDD >=-20%, under 15/20bps total costs",
                "data_reuse": "Historical validation already inspected; not independent final test",
                "limits": {**limits, "max_seconds": args.max_seconds}, "orders_allowed": False, "collection_allowed": False}
        search.write_json(root/"protocol.json", plan)
        (root/"code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        state = {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json"), "orders_sent": 0}
        search.write_json(root/"state.json", state)
        for config in plan["configurations"]:
            search.write_json(root/"trials"/(config["id"]+".json"), {"config": config, "status": "pending"})
    state = prior.read(root/"state.json")
    reason = runner.limit_reason(0, args.max_seconds, 0, runner.output_size(root), shutil.disk_usage(root).free, limits)
    if reason:
        if not args.verify:
            state.update(status="stopped", reason=reason)
            search.write_json(root/"state.json", state)
        raise ValueError(reason)
    with runner.exclusive_lock(root):
        folder = root/("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.search_regime_models", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root/"state.json", state)
        try:
            measured = runner.monitor(command, root, folder, args.max_seconds, limits)
        except BaseException as exc:
            if not args.verify:
                state.update(status="interrupted", reason=type(exc).__name__)
                search.write_json(root/"state.json", state)
                for path in (root/"trials").glob("*.json"):
                    trial = prior.read(path)
                    if trial["status"] == "running":
                        trial.update(status="interrupted", reason=type(exc).__name__)
                        search.write_json(path, trial)
            raise
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        if not args.verify:
            state.update(status=prior.read(root/"results.json")["status"] if success else "stopped",
                         measurement=measured, finished_utc=search.utc_now())
            search.write_json(root/"state.json", state)
            for path in (root/"trials").glob("*.json"):
                trial = prior.read(path)
                if trial["status"] == "running":
                    trial.update(status="interrupted", reason=measured["stop_reason"] or "worker_failed")
                    search.write_json(path, trial)
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
    print(json.dumps(measured), flush=True)
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("runtime_reports/trade_pattern_search_20260915"))
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
