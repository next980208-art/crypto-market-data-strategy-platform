"""Bounded offline position-sizing comparison on frozen predictions; never sends orders."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

from scripts import search_sampled_training as sampled
from scripts import search_trade_patterns as prior
from scripts import automatic_research as runner
from scripts import run_bounded_profitability_search as search
from scripts import expected_return_research as returns

RISKS = (.0025, .005, .01, .02)


def configurations():
    return [{**base, "source_id": base["id"], "id": f"{base['id']}_risk{risk*100:g}pct",
             "risk_per_trade_pct": risk} for base in sampled.configurations() for risk in RISKS]


def notional_ratio(risk, stop, cost_bps):
    if not 0 < risk <= .02 or not np.isfinite(stop) or stop <= 0 or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Invalid sizing inputs")
    return risk / (stop + cost_bps / 10000)


def check_reference(reference):
    _, _, source, _ = sampled.verify_inputs(reference)
    report = prior.read(reference / "results.json")
    if report["status"] != "completed":
        raise ValueError("Reference incomplete")
    for file, digest in report["artifact_sha256"].items():
        if runner.checksum(reference / file) != digest:
            raise ValueError("Reference artifact changed: " + file)
    return source, report


def evaluate(backtest, action, prediction, risk, extra):
    args = action.backtest_args(extra)
    args.risk_per_trade_pct = risk
    args.take_profit_r = 1.5
    backtest.validate_args(args)
    signals = prediction[prediction.pred_action.ne("no_trade")].copy()
    trades, curve, skipped = backtest.simulate(signals, args)
    summary = backtest.summarize(trades, curve, skipped, signals, args)
    long_count = int(signals.pred_action.eq("long").sum())
    short_count = int(signals.pred_action.eq("short").sum())
    metrics = {"signals": len(signals), "long_signals": long_count, "short_signals": short_count,
               "maximum_direction_share": max(long_count, short_count)/len(signals) if len(signals) else 1.0}
    metrics.update({key: summary[key] for key in ("trades", "win_rate_pct", "avg_return_r", "total_return_pct", "max_drawdown_pct")})
    table = pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"])
    return metrics, summary, table


def run_worker(root, verify):
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for file, digest in plan["hashes"].items():
        if runner.checksum(Path(file)) != digest:
            raise ValueError("Frozen input changed: " + file)
    reference = Path(plan["reference"])
    source, previous = check_reference(reference)
    baselines = {r["config"]["id"]: r for r in previous["trials"]}
    if verify:
        saved = prior.read(root/"results.json")
        if saved["status"] != "completed":
            raise ValueError("Cannot verify incomplete experiment")
        for file, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/file) != digest:
                raise ValueError("Artifact changed: " + file)
    action = search.load("16_action_profitability_walk_forward_model.py", "risk_action")
    backtest = search.load("8_model_signal_backtest.py", "risk_backtest")
    frames = {n: pd.read_parquet(source/f"validation_{n}.parquet") for n in (1, 2, 3)}
    predictions, current_source = {}, None
    results = []
    for config in plan["configurations"]:
        file = root/"trials"/(config["id"]+".json")
        if not verify:
            search.write_json(file, {"status": "running", "config": config})
        try:
            if current_source != config["source_id"]:
                predictions.clear()
                for n, frame in frames.items():
                    values = pd.read_parquet(reference/f"predictions_{config['variant']}_{n}.parquet")
                    np.testing.assert_array_equal(frame.timestamp, values.timestamp)
                    predictions[n] = returns.prediction_frame(frame, values.long, values.short, config["threshold_r"], 0)
                current_source = config["source_id"]
            folds, diagnostics = [], []
            for n in (1, 2, 3):
                fold = {"fold": n}
                prediction = predictions[n]
                for label, extra in (("normal", 5), ("stress", 10)):
                    metrics, summary, trades = evaluate(backtest, action, prediction, config["risk_per_trade_pct"], extra)
                    fold[label] = metrics
                    if config["risk_per_trade_pct"] == .02 and metrics != baselines[config["source_id"]]["folds"][n-1][label]:
                        raise ValueError("2% baseline no longer reproduces reference")
                    path = root/"trades"/f"{config['id']}_{n}_{label}.parquet"
                    if verify:
                        pd.testing.assert_frame_equal(trades, pd.read_parquet(path))
                    else:
                        trades.to_parquet(path, index=False)
                    ratios = [notional_ratio(config["risk_per_trade_pct"], s, c+extra)
                              for s, c in prediction[["stop_distance_pct", "round_trip_cost_bps"]].drop_duplicates().itertuples(index=False, name=None)]
                    diagnostics.append({"fold": n, "cost_case": label, "summary": summary,
                                        "theoretical_notional_equity_min": min(ratios), "theoretical_notional_equity_max": max(ratios)})
                folds.append(fold)
            result = prior.summarize(config, folds)
            result.update(status="completed", diagnostics=diagnostics, independent_test_evaluated=False,
                          model_approved=False, orders_sent=0)
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
            raise ValueError("Results replay mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "policies_replayed": len(results),
                          "metric_backtests": len(results)*6, "trade_files": len(results)*6,
                          "baseline_policies_matched": 12, "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
    else:
        files = list((root/"trials").glob("*.json"))+list((root/"trades").glob("*.parquet"))
        search.write_json(root/"results.json", {"status": "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors",
                          "trials": results, "candidate_ids": [r["config"]["id"] for r in results if r.get("candidate")],
                          "models_fitted": 0, "orders_sent": 0, "model_approved": False, "independent_test_evaluated": False,
                          "artifact_sha256": {str(p.relative_to(root)): runner.checksum(p) for p in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runtime_reports/literature_sampling_search_20260917"))
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
        check_reference(reference)
        if shutil.disk_usage(reference).free < 5*1024**3:
            raise ValueError("Insufficient disk space")
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("trials", "trades"):
            (root/folder).mkdir()
        files = [reference/n for n in ("protocol.json", "state.json", "results.json")]+[Path(__file__).resolve(), Path(sampled.__file__).resolve()]
        plan = {"created_utc": search.utc_now(), "reference": str(reference), "hashes": {str(p): runner.checksum(p) for p in files},
                "configurations": configurations(), "limits": {**limits, "max_seconds": args.max_seconds},
                "admission": ">=20 trades, mean R/return positive, MDD >=-20% in all 3 folds under 15/20bps total costs",
                "sizing": "risk / (stop_distance + total_cost_bps/10000); not exchange leverage or margin simulation",
                "daily_stop_pct": .04, "hard_daily_stop_pct": .06, "max_entries_per_day": 5,
                "data_reuse": "Exploratory reused validation; no new models and no independent test",
                "limitations": ["No funding, liquidation, margin or intratrade mark-to-market", "Normal stop fill assumed; gaps excluded",
                                "Fixed daily guards may change trade selection when risk changes", "No guaranteed stable profitability"],
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
        command = [sys.executable, "-u", "-m", "scripts.search_risk_sizing", "--output", str(root), "--worker"]
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
