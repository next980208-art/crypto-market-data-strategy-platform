"""Bounded ensemble validation with persistent failures; offline research only."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

from scripts import search_risk_sizing as sizing
from scripts import research_ledger as ledger

prior, runner, search = sizing.prior, sizing.runner, sizing.search
COSTS = (("normal", 5), ("stress", 10), ("severe", 20))


def configurations():
    return [{"id": f"gap5_{mode}_entry{threshold}", "mode": mode, "threshold_r": threshold,
             "risk_per_trade_pct": .005, "target_r": 1.5}
            for mode in ("mean", "agreement") for threshold in (.05, .15, .3)]


def combine(frame, first, second, config):
    for values in (first, second):
        np.testing.assert_array_equal(frame.timestamp, values.timestamp)
        if not np.isfinite(values[["long", "short"]].to_numpy()).all():
            raise ValueError("Nonfinite predictions")
    long_values = (first.long.to_numpy()+second.long.to_numpy())/2
    short_values = (first.short.to_numpy()+second.short.to_numpy())/2
    result = sizing.returns.prediction_frame(frame, long_values, short_values, config["threshold_r"], 0)
    if config["mode"] == "agreement":
        one = sizing.returns.prediction_frame(frame, first.long, first.short, config["threshold_r"], 0)
        two = sizing.returns.prediction_frame(frame, second.long, second.short, config["threshold_r"], 0)
        accepted = one.pred_action.eq(two.pred_action) & one.pred_action.ne("no_trade")
        result.loc[~accepted, "pred_action"] = "no_trade"
    elif config["mode"] != "mean":
        raise ValueError("Unknown ensemble mode")
    return result


def weekly_results(trades, start_ms, end_ms):
    start, end = pd.to_datetime(start_ms, unit="ms", utc=True), pd.to_datetime(end_ms, unit="ms", utc=True)
    if start != start.normalize() or end != end.normalize() or end <= start:
        raise ValueError("Expected positive interval on UTC day boundaries")
    days = pd.date_range(start, end, freq="D", inclusive="left")
    daily = pd.Series(0., index=days)
    if not trades.empty:
        exits = pd.to_datetime(trades.exit_time_utc, utc=True)
        if not ((exits >= start) & (exits < end)).all():
            raise ValueError("Trade exits outside evaluation period")
        values = trades.return_pct.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= -1).any():
            raise ValueError("Invalid account returns")
        daily.update(pd.Series(values, index=exits.dt.floor("D")).groupby(level=0).apply(lambda x: (1+x).prod()-1))
    blocks = []
    for offset in range(0, len(days), 7):
        block = daily.iloc[offset:offset+7]
        blocks.append({"start_utc": block.index[0].isoformat(), "days": len(block),
                       "complete_week": len(block) == 7, "return_pct": float(((1+block).prod()-1)*100)})
    complete = [b for b in blocks if b["complete_week"]]
    return {"blocks": blocks, "complete_weeks": len(complete),
            "positive_complete_weeks": sum(b["return_pct"] > 0 for b in complete),
            "total_return_pct": float(((1+daily).prod()-1)*100),
            "criterion_passed": len(complete) >= 4 and sum(b["return_pct"] > 0 for b in complete) >= 3}


def summarize(config, folds, diagnostics):
    result = prior.summarize(config, folds)
    severe = search.policy_summary([f["severe"] for f in folds])
    result["severe"] = severe
    if not severe["validation_eligible"]:
        result["rejection_reasons"].append("severe: minimum trades/positive return/mean R/drawdown criteria failed")
    for item in diagnostics:
        if not item["weekly"]["criterion_passed"]:
            result["rejection_reasons"].append(f"fold{item['fold']} {item['cost_case']}: fewer than 3 positive full weeks out of 4")
    result.update(candidate=result["candidate"] and severe["validation_eligible"] and all(d["weekly"]["criterion_passed"] for d in diagnostics),
                  diagnostics=diagnostics, status="completed", independent_test_evaluated=False, model_approved=False, orders_sent=0)
    return result


def save_progress(root, results, status):
    artifacts = list((root/"trials").glob("*.json"))+list((root/"trades").glob("*.parquet"))
    search.write_json(root/"results.json", {"status": status, "trials": results,
                      "candidate_ids": [r["config"]["id"] for r in results if r.get("candidate")],
                      "orders_sent": 0, "models_fitted": 0, "model_approved": False, "independent_test_evaluated": False,
                      "artifact_sha256": {str(p.relative_to(root)): runner.checksum(p) for p in artifacts}})
    (root/"LEDGER.md").write_text(ledger.render([root/"results.json"]), encoding="utf-8")


def run_worker(root, verify):
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for file, digest in plan["hashes"].items():
        if runner.checksum(Path(file)) != digest:
            raise ValueError("Frozen input changed: "+file)
    reference = Path(plan["reference"])
    source, _ = sizing.check_reference(reference)
    schedules = {r["fold"]: r["schedule"] for r in prior.read(source/"training.json")}
    if verify:
        saved = prior.read(root/"results.json")
        if saved["status"] != "completed":
            raise ValueError("Incomplete run")
        for file, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/file) != digest:
                raise ValueError("Artifact changed: "+file)
    action = search.load("16_action_profitability_walk_forward_model.py", "consensus_action")
    backtest = search.load("8_model_signal_backtest.py", "consensus_backtest")
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
                first = pd.read_parquet(reference/f"predictions_shallow90_gap5_{n}.parquet")
                second = pd.read_parquet(reference/f"predictions_ridge90_gap5_{n}.parquet")
                prediction = combine(frame, first, second, config)
                fold = {"fold": n}
                for label, extra in COSTS:
                    metrics, summary, trades = sizing.evaluate(backtest, action, prediction, .005, extra)
                    weekly = weekly_results(trades, schedules[n]["validation_start"], schedules[n]["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold[label] = metrics
                    diagnostics.append({"fold": n, "cost_case": label, "weekly": weekly, "summary": summary})
                    trade_path = root/"trades"/f"{config['id']}_{n}_{label}.parquet"
                    if verify:
                        pd.testing.assert_frame_equal(trades, pd.read_parquet(trade_path))
                    else:
                        trades.to_parquet(trade_path, index=False)
                folds.append(fold)
            result = summarize(config, folds, diagnostics)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(path):
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(path, result)
        results[i] = result
        if not verify:
            save_progress(root, results, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if saved["trials"] != results:
            raise ValueError("Results replay mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "policies_replayed": 6,
                          "metric_backtests": 54, "trade_files": 54, "weekly_reconciliations": 54,
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
        for folder in ("trials", "trades"):
            (root/folder).mkdir()
        files = [reference/n for n in ("protocol.json", "state.json", "results.json")]
        files += [Path(__file__).resolve(), Path(sizing.__file__).resolve(), Path(ledger.__file__).resolve(), Path(sizing.sampled.__file__).resolve()]
        plan = {"created_utc": search.utc_now(), "reference": str(reference), "configurations": configurations(),
                "hashes": {str(p): runner.checksum(p) for p in files}, "limits": {**limits, "seconds": 600},
                "admission": "All 3 folds, total costs 15/20/30bps: >=20 trades, positive mean R and return, MDD>=-20%; >=3 positive 7-day blocks of 4 per fold/cost",
                "weekly": "Realized exit UTC day, zero-return days included, 7-day blocks anchored to fold start; trailing 2 days reported but not full-week criterion",
                "risk_per_trade_pct": .005, "models": "frozen shallow90_gap5 and ridge90_gap5, no refitting",
                "limitations": "Reused validation, no independent test, no funding/liquidation/intratrade drawdown, normal stop fills assumed; no stable profit guarantee",
                "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False}
        search.write_json(root/"protocol.json", plan)
        (root/"code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json")})
        pending = [{"status": "pending", "config": c} for c in configurations()]
        for row in pending:
            search.write_json(root/"trials"/(row["config"]["id"]+".json"), row)
        save_progress(root, pending, "ready")
    with runner.exclusive_lock(root):
        state = prior.read(root/"state.json")
        log = root/("verify_log" if args.verify else "run_log")
        log.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.validate_consensus", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root/"state.json", state)
        try:
            measured = runner.monitor(command, root, log, 600, limits)
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
