"""Bounded low-frequency rule baselines with funding and persistent ledgers."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

from scripts import search_low_frequency_targets as lowfreq

consensus = lowfreq.consensus
prior, replay, runner, search = lowfreq.prior, lowfreq.replay, lowfreq.runner, lowfreq.search

RULES = ("trend", "breakout", "momentum", "rsi_reversal")


def configurations(target_specs):
    return [{"id": f"{spec['id']}_{rule}", "target_id": spec["id"], "rule": rule,
             "target_r": 1.5, "risk_per_trade_pct": .005}
            for spec in target_specs for rule in RULES]


def apply_rule(frame, rule):
    result = frame.copy()
    if rule == "trend":
        long_mask = (result.trend_ma_12_48 > 0) & (result.return_12bar > 0)
        short_mask = (result.trend_ma_12_48 < 0) & (result.return_12bar < 0)
    elif rule == "breakout":
        long_mask = result.range_position_48bar >= .9
        short_mask = result.range_position_48bar <= .1
    elif rule == "momentum":
        long_mask = (result.return_12bar >= .005) & (result.return_3bar > 0)
        short_mask = (result.return_12bar <= -.005) & (result.return_3bar < 0)
    elif rule == "rsi_reversal":
        long_mask = result.rsi_14bar <= .25
        short_mask = result.rsi_14bar >= .75
    else:
        raise ValueError("Unknown rule: "+rule)
    if (long_mask & short_mask).any():
        raise ValueError("Rule emitted both directions")
    result["pred_action"] = np.select([long_mask, short_mask], ["long", "short"], default="no_trade")
    result["pred_confidence"] = 1.
    return result


def save(root, rows, status):
    consensus.save_progress(root, rows, status)


def run_worker(root, verify):
    plan = prior.read(root/"protocol.json")
    state = prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for file, digest in plan["hashes"].items():
        if runner.checksum(Path(file)) != digest:
            raise ValueError("Frozen input changed: "+file)
    source = Path(plan["source"])
    source_plan = prior.read(source/"protocol.json")
    if prior.read(source/"verification.json")["status"] != "passed":
        raise ValueError("Source verification is not passed")
    if verify:
        saved = prior.read(root/"results.json")
        if saved["status"] != "completed":
            raise ValueError("Incomplete run")
        for file, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/file) != digest:
                raise ValueError("Artifact changed: "+file)
    action = search.load("16_action_profitability_walk_forward_model.py", "lowfreq_rule_action")
    backtest = search.load("8_model_signal_backtest.py", "lowfreq_rule_backtest")
    valuation = Path(source_plan["valuation"])
    markets = {n: replay.engine.FundingMarket(prior.read(valuation/f"funding_{n}.json"),
               pd.read_parquet(valuation/f"mark1m_{n}.parquet")) for n in (1, 2, 3)}
    schedules = {row["fold"]: row for row in source_plan["schedules"]}
    rows = [{"status": "pending", "config": config} for config in plan["configurations"]]
    for index, config in enumerate(plan["configurations"]):
        trial_path = root/"trials"/(config["id"]+".json")
        if not verify:
            rows[index] = {"status": "running", "config": config}
            search.write_json(trial_path, rows[index])
            save(root, rows, "running")
        try:
            folds, diagnostics = [], []
            for fold in (1, 2, 3):
                frame = pd.read_parquet(source/f"validation_{config['target_id']}_{fold}.parquet")
                prediction = apply_rule(frame, config["rule"])
                fold_result = {"fold": fold}
                for cost_case, extra_bps in consensus.COSTS:
                    metrics, summary, tables = lowfreq.evaluate(prediction, action, backtest, extra_bps, markets[fold])
                    schedule = schedules[fold]
                    weekly = replay.cash_weekly(tables["events"], schedule["validation_start"], schedule["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold_result[cost_case] = metrics
                    diagnostics.append({"fold": fold, "cost_case": cost_case, "summary": summary, "weekly": weekly})
                    for name, table in tables.items():
                        destination = root/"trades"/f"{config['id']}_{fold}_{cost_case}_{name}.parquet"
                        if verify:
                            pd.testing.assert_frame_equal(table, pd.read_parquet(destination), check_exact=True)
                        else:
                            table.to_parquet(destination, index=False)
                folds.append(fold_result)
            result = consensus.summarize(config, folds, diagnostics)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc),
                      "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(trial_path):
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(trial_path, result)
        rows[index] = result
        if not verify:
            save(root, rows, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if saved["trials"] != rows:
            raise ValueError("Results replay mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "policies_replayed": len(rows),
            "backtests_replayed": len(rows)*9, "trade_ledger_files": len(rows)*27,
            "weekly_reconciliations": len(rows)*9, "results_sha256": runner.checksum(root/"results.json"),
            "orders_sent": 0})
    else:
        save(root, rows, "completed" if all(row["status"] == "completed" for row in rows) else "completed_with_errors")


def initialize(root, source, limits):
    if shutil.disk_usage(search.ROOT).free < limits["minimum_free_gib"]*1024**3:
        raise ValueError("Insufficient disk space")
    source_plan = prior.read(source/"protocol.json")
    source_verification = prior.read(source/"verification.json")
    if source_verification["status"] != "passed":
        raise ValueError("Source verification is not passed")
    specs = source_plan["target_specs"]
    validation_files = [source/f"validation_{spec['id']}_{fold}.parquet" for spec in specs for fold in (1, 2, 3)]
    files = [Path(__file__).resolve(), Path(lowfreq.__file__).resolve(), Path(lowfreq.engine.__file__).resolve(),
             source/"protocol.json", source/"results.json", source/"verification.json"]+validation_files
    valuation = Path(source_plan["valuation"])
    files += [valuation/f"funding_{fold}.json" for fold in (1, 2, 3)]
    files += [valuation/f"mark1m_{fold}.parquet" for fold in (1, 2, 3)]
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("trials", "trades"):
        (root/folder).mkdir()
    configs = configurations(specs)
    plan = {"created_utc": search.utc_now(), "source": str(source), "target_specs": specs,
            "configurations": configs, "rules": {
                "trend": "trend_ma_12_48 and return_12bar have the same nonzero direction",
                "breakout": "range_position_48bar >=0.9 long or <=0.1 short",
                "momentum": "abs(return_12bar)>=0.5% with return_3bar confirmation",
                "rsi_reversal": "RSI14 <=0.25 long or >=0.75 short"},
            "hashes": {str(file): runner.checksum(file) for file in files},
            "limits": {**limits, "seconds": 600},
            "costs": "10bps in source labels plus 5/10/20bps and chronological funding cashflows",
            "admission": "All 3 folds and all costs: >=20 trades, positive mean R/return, cash MDD>=-20%, >=3/4 positive full weeks",
            "limitations": "Reused development data; fixed rule baselines are not an independent test; source OHLC and funding limitations remain",
            "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False}
    search.write_json(root/"protocol.json", plan)
    search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json")})
    shutil.copyfile(__file__, root/"code_snapshot.py")
    pending = [{"status": "pending", "config": config} for config in configs]
    for row in pending:
        search.write_json(root/"trials"/(row["config"]["id"]+".json"), row)
    save(root, pending, "ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runtime_reports/low_frequency_targets_20260922"))
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
    limits = {"memory_mib": 2048, "output_mib": 256, "minimum_free_gib": 5}
    if not args.verify:
        initialize(root, args.source.resolve(), limits)
    with runner.exclusive_lock(root):
        state = prior.read(root/"state.json")
        log = root/("verify_log" if args.verify else "run_log")
        log.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.search_low_frequency_rules", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root/"state.json", state)
        measured = runner.monitor(command, root, log, 600, limits)
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root/"results.json")["status"] if success else "stopped",
                         measurement=measured, finished_utc=search.utc_now())
            search.write_json(root/"state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
