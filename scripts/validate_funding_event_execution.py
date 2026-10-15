"""Bounded chronological funding replay of 30 frozen policies; no orders or fitting."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import traceback

import numpy as np
import pandas as pd

from scripts import funding_event_backtest as engine
from scripts import search_context_ablation as research

context_run = research.context_run
search, prior, runner = research.search, research.prior, research.runner
consensus, sizing = research.consensus, research.sizing


def evaluate(prediction, action, backtest, extra, market):
    args = action.backtest_args(extra)
    args.risk_per_trade_pct, args.take_profit_r = .005, 1.5
    signals = prediction[prediction.pred_action.ne("no_trade")].copy()
    trades, curve, skipped, events, daily = engine.simulate(signals, args, backtest, market)
    summary = backtest.summarize(trades, curve, skipped, signals, args)
    counts = [int(signals.pred_action.eq(side).sum()) for side in ("long", "short")]
    metrics = {"signals": len(signals), "long_signals": counts[0], "short_signals": counts[1],
               "maximum_direction_share": max(counts)/len(signals) if len(signals) else 1.}
    metrics.update({k: summary[k] for k in ("trades", "win_rate_pct", "avg_return_r", "total_return_pct", "max_drawdown_pct")})
    tables = {"trades": pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"]),
              "events": pd.DataFrame(events, columns=list(dict.fromkeys(k for e in events for k in e)) if events else ["exit_time_utc", "return_pct"]),
              "daily": pd.DataFrame(daily, columns=list(daily[0]) if daily else ["day_start_ms", "realized_pnl"])}
    return metrics, summary, tables


def references(plan):
    return {name: Path(path) for name, path in plan["references"].items()}


def cash_weekly(events, start, end):
    table = events.copy()
    if not table.empty:
        table["exit_time_utc"] = pd.to_datetime(table.timestamp, unit="ms", utc=True)
    return consensus.weekly_results(table, start, end)


def prepare(plan):
    roots = references(plan)
    controls = {name: research.verified_reference(root) for name, root in roots.items()}
    _, _, source, _, context = context_run.inputs(roots["context"])
    marks_root = Path(plan["marks"])
    collection = prior.read(marks_root / "collection.json")
    if collection["status"] != "completed":
        raise ValueError("Incomplete marks collection")
    frames, markets = {}, {}
    with closing(sqlite3.connect((context / "context.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        for n in (1, 2, 3):
            sources = {family: pd.read_parquet(context / f"{family}_{n}.parquet") for family in context_run.collection.SPECS}
            frame, _ = context_run.enrich(pd.read_parquet(source / f"validation_{n}.parquet"), sources)
            expected = pd.read_parquet(roots["context"] / f"context_features_{n}.parquet")
            pd.testing.assert_frame_equal(frame[expected.columns], expected, check_exact=True)
            frames[n] = frame
            events = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM observations WHERE family=? ORDER BY event_ms", (f"funding_{n}",))]
            marks = pd.read_parquet(marks_root / f"mark1m_{n}.parquet")
            if len(marks) != 43200 or not np.diff(marks.event_ms.to_numpy()).tolist() == [engine.MINUTE]*(len(marks)-1):
                raise ValueError("Incomplete one-minute marks")
            markets[n] = engine.FundingMarket(events, marks)
    schedules = {r["fold"]: r["schedule"] for r in prior.read(source / "training.json")}
    return roots, controls, frames, markets, schedules


def check_hashes(plan, state, root):
    if runner.checksum(root / "protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for name, digest in plan["hashes"].items():
        if runner.checksum(Path(name)) != digest:
            raise ValueError("Frozen input changed: " + name)


def run_worker(root, verify):
    plan, state = prior.read(root / "protocol.json"), prior.read(root / "state.json")
    check_hashes(plan, state, root)
    if verify:
        saved = prior.read(root / "results.json")
        if saved["status"] != "completed":
            raise ValueError("Incomplete replay")
        for name, digest in saved["artifact_sha256"].items():
            if runner.checksum(root / name) != digest:
                raise ValueError("Replay artifact changed: " + name)
    roots, controls, frames, markets, schedules = prepare(plan)
    action = search.load("16_action_profitability_walk_forward_model.py", "funded_action")
    backtest = search.load("8_model_signal_backtest.py", "funded_backtest")
    results = [{"status": "pending", "config": c} for c in plan["configurations"]]
    for i, config in enumerate(plan["configurations"]):
        trial_path = root / "trials" / (config["id"] + ".json")
        if not verify:
            results[i] = {"status": "running", "config": config}
            search.write_json(trial_path, results[i])
            consensus.save_progress(root, results, "running")
        try:
            original = next(r for r in controls[config["reference"]]["trials"] if r["config"]["id"] == config["source_id"])
            old_folds = {r["fold"]: r for r in original["folds"]}
            folds, diagnostics = [], []
            for n, frame in frames.items():
                path = roots[config["reference"]] / f"predictions_{config['family']}_{config['group']}_{n}.parquet"
                values = pd.read_parquet(path)
                np.testing.assert_array_equal(frame.timestamp, values.timestamp)
                if not np.isfinite(values[["long", "short"]].to_numpy()).all():
                    raise ValueError("Nonfinite saved prediction")
                prediction = sizing.returns.prediction_frame(frame, values.long, values.short, config["threshold_r"], 0)
                fold = {"fold": n}
                for cost, extra in consensus.COSTS:
                    metrics, summary, tables = evaluate(prediction, action, backtest, extra, markets[n])
                    control, control_summary, control_tables = evaluate(prediction, action, backtest, extra, None)
                    weekly = cash_weekly(tables["events"], schedules[n]["validation_start"], schedules[n]["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold[cost] = metrics
                    trade_rows = tables["trades"].to_dict("records")
                    diagnostics.append({"fold": n, "cost_case": cost, "summary": summary, "weekly": weekly,
                                        "chronological_no_funding": control, "control_summary": control_summary,
                                        "funding_return_delta_pp": metrics["total_return_pct"]-control["total_return_pct"],
                                        "chronology_return_delta_pp": control["total_return_pct"]-old_folds[n][cost]["total_return_pct"],
                                        "funding_events": sum(t["funding_events"] for t in trade_rows),
                                        "proxy_events": sum(t["funding_proxy_events"] for t in trade_rows),
                                        "boundary_uncertain_events": sum(t["boundary_uncertain_events"] for t in trade_rows)})
                    for name, table in {**tables, "control_trades": control_tables["trades"]}.items():
                        path = root / "trades" / f"{config['id']}_{n}_{cost}_{name}.parquet"
                        if verify:
                            pd.testing.assert_frame_equal(table, pd.read_parquet(path), check_exact=True)
                        else:
                            table.to_parquet(path, index=False)
                folds.append(fold)
            result = consensus.summarize(config, folds, diagnostics)
            result["funding_cashflows_modeled"] = True
            result["independent_holdout_certified"] = False
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(trial_path):
                raise ValueError("Trial replay mismatch: " + config["id"])
        else:
            search.write_json(trial_path, result)
        results[i] = result
        if not verify:
            consensus.save_progress(root, results, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if saved["trials"] != results:
            raise ValueError("Results mismatch")
        search.write_json(root / "verification.json", {"status": "passed", "policies_replayed": len(results),
                          "funded_backtests": len(results)*9, "control_backtests": len(results)*9,
                          "parquet_files_replayed": len(results)*36, "weekly_reconciliations": len(results)*9,
                          "results_sha256": runner.checksum(root / "results.json"), "orders_sent": 0})
    else:
        consensus.save_progress(root, results, "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors")


def initialize(root, roots, marks, usage, limits):
    reports = {name: research.verified_reference(path) for name, path in roots.items()}
    _, _, source, _, context = context_run.inputs(roots["context"])
    if shutil.disk_usage(search.ROOT).free < limits["minimum_free_gib"] * 1024**3:
        raise ValueError("Insufficient disk space")
    configs = [{**r["config"], "source_id": r["config"]["id"], "id": r["config"]["id"] + "_funded",
                "reference": name} for name, report in reports.items() for r in report["trials"]]
    if len(configs) != 30 or len({c["id"] for c in configs}) != 30:
        raise ValueError("Expected exactly 30 unique frozen policies")
    files = [Path(__file__).resolve(), Path(engine.__file__).resolve(), usage, context / "context.sqlite3",
             marks / "collection.json", source / "training.json",
             search.ROOT / "8_model_signal_backtest.py", search.ROOT / "16_action_profitability_walk_forward_model.py"]
    files += [Path(m.__file__).resolve() for m in (research, context_run, search, prior, runner, consensus, sizing, sizing.returns)]
    for path in roots.values():
        files += [path / name for name in ("protocol.json", "state.json", "results.json", "verification.json")]
        files += list(path.glob("predictions_*.parquet"))
    for n in (1, 2, 3):
        files += [source / f"validation_{n}.parquet", roots["context"] / f"context_features_{n}.parquet", marks / f"mark1m_{n}.parquet"]
        files += [context / f"{family}_{n}.parquet" for family in context_run.collection.SPECS]
    root.mkdir(parents=True, exist_ok=False)
    for name in ("trials", "trades"):
        (root / name).mkdir()
    plan = {"created_utc": search.utc_now(), "references": {k: str(v) for k, v in roots.items()},
            "marks": str(marks), "data_usage_audit": str(usage), "configurations": configs,
            "hashes": {str(p): runner.checksum(p) for p in files}, "limits": {**limits, "seconds": 600},
            "admission": "All 3 folds x costs15/20/30bps: >=20 trades, mean R and return>0, cash MDD>=-20%, >=3/4 positive weeks",
            "funding": "Actual rate; historical settlement mark where present, otherwise previous completed 1m mark close; fixed entry quantity",
            "event_order": "Exit first, funding second, entry third on ties; strictly internal settlements only",
            "control": "Same chronological engine without funding; compare separately to original exit-booking engine",
            "limitations": "Reused development folds; no independent holdout; proxy marks and boundary timing uncertain; costs at exit; realized-cash guards not MTM; no margin/liquidation/fill simulation",
            "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False, "models_fitted": 0}
    search.write_json(root / "protocol.json", plan)
    search.write_json(root / "state.json", {"status": "ready", "protocol_sha256": runner.checksum(root / "protocol.json")})
    for name, source_file in (("code_snapshot.py", Path(__file__)), ("engine_snapshot.py", Path(engine.__file__))):
        shutil.copyfile(source_file, root / name)
    pending = [{"status": "pending", "config": c} for c in configs]
    for row in pending:
        search.write_json(root / "trials" / (row["config"]["id"] + ".json"), row)
    consensus.save_progress(root, pending, "ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use a project subdirectory")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 96, "minimum_free_gib": 5}
    if not args.verify:
        base = search.ROOT / "runtime_reports"
        initialize(root, {"context": base / "market_context_validation_20260918", "ablation": base / "context_ablation_20260919"},
                   base / "valuation_marks_20260919", base / "data_usage_audit_20260920/audit.json", limits)
    with runner.exclusive_lock(root):
        state = prior.read(root / "state.json")
        folder = root / ("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.validate_funding_event_execution", "--output", str(root), "--worker"]
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
            state.update(status=prior.read(root / "results.json")["status"] if success else "stopped",
                         measurement=measured, finished_utc=search.utc_now())
            search.write_json(root / "state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
