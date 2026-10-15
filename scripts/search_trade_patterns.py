"""Bounded exploratory gate-policy search on frozen historical predictions; no orders."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import automatic_research as runner
from scripts import automatic_research_worker as worker
from scripts import run_cost_aware_gate as gate
from scripts import run_bounded_profitability_search as search


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configurations():
    return [{"id": f"c{c}_p{p}_{side}", "C": c, "threshold": p, "direction": side}
            for c in (.01, .1, 1.0) for p in (.45, .5, .55, .6)
            for side in ("both", "long", "short")]


def filter_signals(frame, probability, config):
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(frame),) or not np.isfinite(probability).all():
        raise ValueError("Invalid probability vector")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probability outside [0,1]")
    accepted = probability >= config["threshold"]
    if config["direction"] not in ("both", "long", "short"):
        raise ValueError("Unknown direction")
    if config["direction"] != "both":
        accepted &= frame.pred_action.eq(config["direction"]).to_numpy()
    result = frame.copy()
    result.loc[~accepted, "pred_action"] = "no_trade"
    return result


def predict(parameters, frame):
    x = gate.gate_features(frame)[parameters["columns"]].to_numpy()
    z = ((x-np.array(parameters["mean"]))/np.array(parameters["scale"])) @ np.array(parameters["coef"])
    return 1/(1+np.exp(-np.clip(z+parameters["intercept"], -700, 700)))


def summarize(config, folds):
    normal = search.policy_summary([f["normal"] for f in folds])
    stress = search.policy_summary([f["stress"] for f in folds])
    reasons = []
    for label, summary in (("normal", normal), ("stress", stress)):
        if summary["folds_with_enough_trades"] != 3:
            reasons.append(label+": fewer than 20 trades in at least one fold")
        if summary["positive_folds"] != 3:
            reasons.append(label+": nonpositive mean R/return in at least one fold")
        if summary["worst_drawdown_pct"] < -20:
            reasons.append(label+": drawdown below -20%")
    return {"config": config, "folds": folds, "normal": normal, "stress": stress,
            "candidate": normal["validation_eligible"] and stress["validation_eligible"], "rejection_reasons": reasons}


def ranking(row):
    # Insufficient/zero-trade policies cannot outrank fully populated diagnostics.
    return (row["normal"]["folds_with_enough_trades"], row["stress"]["worst_fold_avg_r"],
            row["normal"]["worst_fold_avg_r"], row["normal"]["weighted_avg_r"])


def inputs(root):
    plan = read(root/"protocol.json")
    for name, checksum in plan["input_sha256"].items():
        if runner.checksum(Path(name)) != checksum:
            raise ValueError("Frozen input changed: "+name)
    source = Path(plan["source_run"])
    source_plan, state = read(source/"protocol.json"), read(source/"state.json")
    runner.verify_resume(source, source_plan, state)
    frames = []
    for number in (1, 2, 3):
        result = worker.get_result(source, state, f"prepare_{number}")
        meta = pd.read_parquet(source/result["meta_path"])
        validation = pd.read_parquet(source/result["validation_path"])
        if len(meta) < 200 or min(meta.actual_net_label_r.gt(0).sum(), meta.actual_net_label_r.le(0).sum()) < 20:
            raise ValueError("Insufficient meta samples/classes")
        if (meta.entry_timestamp+meta.max_holding_minutes*search.MINUTE).max() >= validation.timestamp.min():
            raise ValueError("Meta outcomes overlap validation")
        frames.append((meta, validation))
    return plan, frames


def run_worker(root, verify=False):
    plan, frames = inputs(root)
    action = search.load("16_action_profitability_walk_forward_model.py", "pattern_action")
    backtest = search.load("8_model_signal_backtest.py", "pattern_backtest")
    models, probabilities = {}, {}
    if verify:
        saved = read(root/"results.json")
        for name, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/name) != digest:
                raise ValueError("Saved artifact changed: "+name)
        models = read(root/"models.json")
    for c in sorted({item["C"] for item in plan["configurations"]}):
        for index, (meta, validation) in enumerate(frames, 1):
            key = f"c{c}_fold{index}"
            if not verify:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=1000))
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(meta[gate.GATE_COLUMNS], meta.actual_net_label_r.gt(0).astype(int))
                scaler, logistic = model.steps[0][1], model.steps[1][1]
                models[key] = {"columns": gate.GATE_COLUMNS, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                               "coef": logistic.coef_[0].tolist(), "intercept": float(logistic.intercept_[0])}
                probability = model.predict_proba(gate.gate_features(validation))[:, 1]
                np.testing.assert_allclose(probability, predict(models[key], validation), atol=1e-12)
                pd.DataFrame({"timestamp": validation.timestamp.to_numpy(), "probability": probability}).to_parquet(root/(key+".parquet"), index=False)
            else:
                stored = pd.read_parquet(root/(key+".parquet"))
                np.testing.assert_array_equal(stored.timestamp, validation.timestamp)
                scaler = StandardScaler().fit(meta[gate.GATE_COLUMNS])
                np.testing.assert_allclose(models[key]["mean"], scaler.mean_, atol=1e-12)
                np.testing.assert_allclose(models[key]["scale"], scaler.scale_, atol=1e-12)
                probability = stored.probability.to_numpy()
                np.testing.assert_allclose(probability, predict(models[key], validation), atol=1e-12)
            probabilities[key] = probability
    results = []
    for config in plan["configurations"]:
        folds = []
        for index, (_, validation) in enumerate(frames, 1):
            filtered = filter_signals(validation, probabilities[f"c{config['C']}_fold{index}"], config)
            folds.append({"fold": index, "normal": action.evaluate_candidate(backtest, filtered, 5, 1.5),
                          "stress": action.evaluate_candidate(backtest, filtered, 10, 1.5)})
        results.append(summarize(config, folds))
        if not verify:
            search.write_json(root/"progress.json", {"completed": len(results), "total": len(plan["configurations"]), "last": results[-1]})
        print(f"{len(results)}/{len(plan['configurations'])}: {config['id']} candidate={results[-1]['candidate']}", flush=True)
    ordered = sorted(results, key=ranking, reverse=True)
    candidate = next((r["config"]["id"] for r in ordered if r["candidate"]), None)
    if verify:
        if results != saved["trials"] or candidate != saved["candidate"]:
            raise ValueError("Replay metrics/selection mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "verified_utc": search.utc_now(),
                          "policies_replayed": len(results), "backtests_replayed": len(results)*6, "models_fitted": 0,
                          "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
        return
    search.write_json(root/"models.json", models)
    payload = {"status": "completed", "finished_utc": search.utc_now(), "trials": results,
               "candidate": candidate, "best_diagnostic": ordered[0]["config"]["id"], "models_fitted": len(models),
               "independent_test_evaluated": False, "model_approved": False, "orders_sent": 0,
               "artifact_sha256": {p.name: runner.checksum(p) for p in [root/"models.json", *root.glob("*.parquet")]}}
    search.write_json(root/"results.json", payload)
    lines = ["# Bounded Trade Pattern Search", "", "Exploratory reused validation data, not independent investment performance.",
             "No collection, orders or deployment. Returns from different folds must not be added together.", "",
             "Normal: label cost 10bps + 5bps; stress: label cost 10bps + 10bps.",
             "Candidate requires >=20 trades, positive mean R and return, MDD >=-20% in ALL 3 folds under BOTH costs.",
             "", f"Candidate: {candidate or 'none'}. Models fitted: {len(models)}. Policies: {len(results)}.", "",
             "| Config | Fold returns % | Trades per fold | Stress returns % | Candidate |",
             "| --- | --- | --- | --- | --- |"]
    for row in ordered:
        returns = lambda cost: ", ".join(f"{v:.3f}" for v in row[cost]["fold_returns_pct"])
        trades = ", ".join(str(f["normal"]["trades"]) for f in row["folds"])
        lines.append(f"| {row['config']['id']} | {returns('normal')} | {trades} | {returns('stress')} | {row['candidate']} |")
    (root/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


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
        raise ValueError("Use a project subdirectory and a 60..3600 second budget")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 64, "minimum_free_gib": 5}
    if not args.verify:
        source = args.source.resolve()
        source_plan, state = read(source/"protocol.json"), read(source/"state.json")
        if state["status"] != "completed":
            raise ValueError("Source research must be completed")
        runner.verify_resume(source, source_plan, state)
        root.mkdir(parents=True, exist_ok=False)
        files = [source/"protocol.json", source/"state.json", Path(__file__).resolve()]
        files += [source/name for task in state["tasks"].values() for name in task["artifact_sha256"]]
        plan = {"created_utc": search.utc_now(), "source_run": str(source), "configurations": configurations(),
                "input_sha256": {str(p): runner.checksum(p) for p in files}, "limits": {**limits, "max_seconds": args.max_seconds},
                "take_profit_r": 1.5, "extra_slippage_bps": [5, 10], "minimum_trades": 20,
                "data_reuse": "Previously inspected validation. Exploratory, not independent holdout.",
                "orders_allowed": False, "collection_allowed": False}
        search.write_json(root/"protocol.json", plan)
        (root/"search_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    reason = runner.limit_reason(0, args.max_seconds, 0, runner.output_size(root), shutil.disk_usage(root).free, limits)
    if reason:
        raise ValueError(reason)
    with runner.exclusive_lock(root):
        folder = root/("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.search_trade_patterns", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        measured = runner.monitor(command, root, folder, args.max_seconds, limits)
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
    print(json.dumps(measured), flush=True)
    return 0 if measured["exit_code"] == 0 and not measured["stop_reason"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
