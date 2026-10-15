"""Bounded asymmetric-payoff and long-hold research with actual funding; no orders."""

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from scripts import search_low_frequency_targets as lowfreq

consensus = lowfreq.consensus
engine = lowfreq.engine
prior, replay, runner, search = lowfreq.prior, lowfreq.replay, lowfreq.runner, lowfreq.search

HOLDS = (3*1440, 7*1440, 14*1440)
STOPS = (.01, .02)
TARGETS = (2., 3., 5., 8.)
RULES = ("trend", "breakout", "pullback", "reversal")
RISK_PER_TRADE = .005
OUTPUT_LIMIT = 256*1024**2


def configurations():
    rows = []
    for hold in HOLDS:
        for stop in STOPS:
            for target in TARGETS:
                for rule in RULES:
                    rows.append({
                        "id": f"hold{hold}_stop{stop*100:g}pct_tp{target:g}_{rule}",
                        "max_holding_minutes": hold,
                        "stop_distance_pct": stop,
                        "target_r": target,
                        "rule": rule,
                        "risk_per_trade_pct": RISK_PER_TRADE,
                    })
    return rows


def apply_rule(frame, rule):
    result = frame.copy()
    if rule == "trend":
        long_mask = (result.trend_ma_12_48 > 0) & (result.return_12bar > 0)
        short_mask = (result.trend_ma_12_48 < 0) & (result.return_12bar < 0)
    elif rule == "breakout":
        long_mask = result.range_position_48bar >= .9
        short_mask = result.range_position_48bar <= .1
    elif rule == "pullback":
        long_mask = (result.trend_ma_12_48 > 0) & (result.range_position_48bar <= .25) & (result.return_3bar > 0)
        short_mask = (result.trend_ma_12_48 < 0) & (result.range_position_48bar >= .75) & (result.return_3bar < 0)
    elif rule == "reversal":
        long_mask = (result.range_position_48bar <= .1) & (result.rsi_14bar <= .25) & (result.return_1bar > 0)
        short_mask = (result.range_position_48bar >= .9) & (result.rsi_14bar >= .75) & (result.return_1bar < 0)
    else:
        raise ValueError("Unknown rule: "+rule)
    if (long_mask & short_mask).any():
        raise ValueError("Rule emitted both directions")
    result["pred_action"] = np.select([long_mask, short_mask], ["long", "short"], default="no_trade")
    result["pred_confidence"] = 1.
    return result


def build_frames(plan):
    labeler = search.load("4_triple_barrier_labeler.py", "asymmetric_long_hold_labels")
    frames = {}
    warmup = 2*search.DAY
    for schedule in plan["schedules"]:
        fold, start, end = schedule["fold"], schedule["validation_start"], schedule["validation_end"]
        raw = pd.read_parquet(plan["dataset"], columns=["timestamp", "open", "high", "low", "close", "volume"],
                              filters=[("timestamp", ">=", start-warmup), ("timestamp", "<", end)])
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        np.testing.assert_array_equal(raw.timestamp, np.arange(start-warmup, end, search.MINUTE))
        bars = lowfreq.aggregate(raw, 15)
        features, _ = lowfreq.add_features(bars)
        for hold in HOLDS:
            for stop in STOPS:
                labels = labeler.add_labels(bars, list(TARGETS), stop, hold, 10, 1024, 10)
                columns = [c for c in labels if c not in ("datetime_utc", "symbol", "market", "timeframe", "close")]
                frame = features.merge(labels[columns], on="timestamp", how="inner", validate="one_to_one")
                frame = frame[frame.timestamp >= start].reset_index(drop=True)
                if frame.empty or not frame.label_horizon_complete.eq(True).all():
                    raise ValueError(f"No complete labels for fold{fold}, hold={hold}, stop={stop}")
                frames[fold, hold, stop] = frame
        print(f"fold{fold}: bars={len(bars)} prepared={len(HOLDS)*len(STOPS)}", flush=True)
    return frames


def markets(plan):
    root = Path(plan["valuation"])
    collection = prior.read(root/"collection.json")
    if collection["status"] != "completed":
        raise ValueError("Valuation collection is incomplete")
    return {
        fold: engine.FundingMarket(prior.read(root/f"funding_{fold}.json"), pd.read_parquet(root/f"mark1m_{fold}.parquet"))
        for fold in (1, 2, 3)
    }


def evaluate(prediction, target_r, extra_bps, market):
    action = search.load("16_action_profitability_walk_forward_model.py", "asymmetric_long_hold_action")
    backtest = search.load("8_model_signal_backtest.py", "asymmetric_long_hold_backtest")
    args = action.backtest_args(extra_bps)
    args.risk_per_trade_pct = RISK_PER_TRADE
    args.take_profit_r = target_r
    args.timeframe = "15m"
    signals = prediction[prediction.pred_action.ne("no_trade")].copy()
    trades, curve, skipped, events, daily = engine.simulate(signals, args, backtest, market)
    summary = backtest.summarize(trades, curve, skipped, signals, args)
    counts = [int(signals.pred_action.eq(side).sum()) for side in ("long", "short")]
    metrics = {
        "signals": len(signals),
        "long_signals": counts[0],
        "short_signals": counts[1],
        "maximum_direction_share": max(counts)/len(signals) if len(signals) else 1.,
        "trades": int(summary["trades"]),
        "win_rate_pct": float(summary["win_rate_pct"]),
        "avg_return_r": float(summary["avg_return_r"]),
        "total_return_pct": float(summary["total_return_pct"]),
        "max_drawdown_pct": float(summary["max_drawdown_pct"]),
        "max_consecutive_losses": int(summary["max_consecutive_losses"]),
        "funding_cash_delta_pct_of_initial": float(sum(row["funding_pnl"] for row in trades)*100),
        "funding_events": int(sum(row["funding_events"] for row in trades)),
        "average_holding_minutes": float(np.mean([row["holding_minutes"] for row in trades])) if trades else 0.,
    }
    columns = {
        "trades": list(trades[0]) if trades else ["signal_timestamp", "equity"],
        "events": list(dict.fromkeys(k for row in events for k in row)) if events else ["timestamp", "exit_time_utc", "return_pct"],
    }
    tables = {"trades": pd.DataFrame(trades, columns=columns["trades"]),
              "events": pd.DataFrame(events, columns=columns["events"])}
    return metrics, tables


def summarize(config, folds):
    costs = {}
    for name, _ in consensus.COSTS:
        values = [row[name] for row in folds]
        costs[name] = {
            "fold_returns_pct": [row["total_return_pct"] for row in values],
            "fold_trades": [row["trades"] for row in values],
            "fold_win_rates_pct": [row["win_rate_pct"] for row in values],
            "average_return_pct": float(np.mean([row["total_return_pct"] for row in values])),
            "worst_return_pct": float(min(row["total_return_pct"] for row in values)),
            "positive_folds": int(sum(row["total_return_pct"] > 0 for row in values)),
            "worst_avg_return_r": float(min(row["avg_return_r"] for row in values)),
            "worst_drawdown_pct": float(min(row["max_drawdown_pct"] for row in values)),
            "total_trades": int(sum(row["trades"] for row in values)),
            "funding_cash_delta_pct_of_initial": float(sum(row["funding_cash_delta_pct_of_initial"] for row in values)),
        }
    reasons = []
    if costs["normal"]["positive_folds"] != 3 or costs["normal"]["worst_avg_return_r"] <= 0:
        reasons.append("normal costs were not positive in all three folds")
    if costs["stress"]["positive_folds"] != 3 or costs["stress"]["worst_avg_return_r"] <= 0:
        reasons.append("stress costs were not positive in all three folds")
    if costs["severe"]["positive_folds"] < 2 or costs["severe"]["worst_drawdown_pct"] < -20:
        reasons.append("severe costs failed the two-of-three positive or drawdown gate")
    if min(costs["normal"]["fold_trades"]) < 3 or costs["normal"]["total_trades"] < 15:
        reasons.append("too few trades for even the exploratory gate")
    survivor = not reasons
    return {
        "status": "completed",
        "config": config,
        "folds": folds,
        "costs": costs,
        "development_survivor": survivor,
        "rejection_reasons": reasons,
        "independent_test_evaluated": False,
        "model_approved": False,
        "orders_sent": 0,
    }


def directory_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def initialize(root, source, valuation):
    if shutil.disk_usage(search.ROOT).free < 5*1024**3:
        raise ValueError("Insufficient free space")
    source_plan = prior.read(source/"protocol.json")
    if prior.read(source/"verification.json")["status"] != "passed":
        raise ValueError("Source verification is not passed")
    if prior.read(valuation/"collection.json")["status"] != "completed":
        raise ValueError("Funding valuation is not complete")
    root.mkdir(parents=True, exist_ok=False)
    (root/"trials").mkdir()
    (root/"ledgers").mkdir()
    inputs = [Path(__file__).resolve(), Path(source_plan["dataset"]), source/"protocol.json", source/"verification.json",
              valuation/"collection.json"]
    inputs += [valuation/f"funding_{fold}.json" for fold in (1, 2, 3)]
    inputs += [valuation/f"mark1m_{fold}.parquet" for fold in (1, 2, 3)]
    plan = {
        "created_utc": search.utc_now(),
        "dataset": source_plan["dataset"],
        "source": str(source),
        "valuation": str(valuation),
        "schedules": source_plan["schedules"],
        "configurations": configurations(),
        "rules": {
            "trend": "12/48-bar trend and 12-bar return agree",
            "breakout": "48-bar range position >=0.9 long or <=0.1 short",
            "pullback": "trend-aligned lower/upper quartile with 3-bar rebound confirmation",
            "reversal": "range extreme plus RSI extreme and one-bar reversal confirmation",
        },
        "holds_minutes": list(HOLDS),
        "stops_pct": list(STOPS),
        "targets_r": list(TARGETS),
        "costs": {name: bps for name, bps in consensus.COSTS},
        "label_round_trip_cost_bps": 10,
        "risk_per_trade_pct": RISK_PER_TRADE,
        "admission": "Development survivor only: all normal/stress folds positive with positive mean R; >=2 severe folds positive; MDD >=-20%; >=3 trades/fold and >=15 total",
        "limitations": "Reused 30-day development folds; long holds reduce independent trades; no trailing stop or fill/latency evidence; any survivor requires frozen future validation",
        "research_basis": [
            "https://fairmodel.econ.yale.edu/ec439/mosk.pdf",
            "https://www.research.ed.ac.uk/en/publications/anatomy-of-cryptocurrency-perpetual-futures-returns/",
        ],
        "limits": {"output_bytes": OUTPUT_LIMIT, "minimum_free_gib": 5},
        "hashes": {str(path): runner.checksum(path) for path in inputs},
        "orders_allowed": False,
        "deployment_allowed": False,
        "collection_allowed": False,
    }
    search.write_json(root/"protocol.json", plan)
    search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json"),
                                           "orders_sent": 0})


def run(root, verify):
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for path, digest in plan["hashes"].items():
        if runner.checksum(Path(path)) != digest:
            raise ValueError("Frozen input changed: "+path)
    saved = prior.read(root/"results.json") if verify else None
    if verify:
        for name, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/name) != digest:
                raise ValueError("Artifact changed: "+name)
    prepared, valuation = build_frames(plan), markets(plan)
    schedules = {row["fold"]: row for row in plan["schedules"]}
    results = []
    for index, config in enumerate(plan["configurations"], 1):
        try:
            folds = []
            for fold in (1, 2, 3):
                frame = prepared[fold, config["max_holding_minutes"], config["stop_distance_pct"]]
                prediction = apply_rule(frame, config["rule"])
                fold_result = {"fold": fold}
                for cost, extra in consensus.COSTS:
                    metrics, tables = evaluate(prediction, config["target_r"], extra, valuation[fold])
                    weekly = replay.cash_weekly(tables["events"], schedules[fold]["validation_start"], schedules[fold]["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    metrics["weekly"] = weekly
                    fold_result[cost] = metrics
                    for name, table in tables.items():
                        path = root/"ledgers"/f"{config['id']}_{fold}_{cost}_{name}.parquet"
                        if verify:
                            pd.testing.assert_frame_equal(table, pd.read_parquet(path), check_exact=True)
                        else:
                            table.to_parquet(path, index=False)
                folds.append(fold_result)
            result = summarize(config, folds)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc),
                      "traceback": traceback.format_exc(), "model_approved": False, "orders_sent": 0}
        trial_path = root/"trials"/(config["id"]+".json")
        if verify:
            if result != prior.read(trial_path):
                raise ValueError("Trial replay mismatch: "+config["id"])
        else:
            search.write_json(trial_path, result)
            if directory_bytes(root) > plan["limits"]["output_bytes"]:
                raise ValueError("Output size limit exceeded")
        results.append(result)
        print(f"{index}/{len(plan['configurations'])} {config['id']}: survivor={result.get('development_survivor')}", flush=True)
    completed = [row for row in results if row["status"] == "completed"]
    ranked = sorted(completed, key=lambda row: (row["costs"]["normal"]["average_return_pct"],
                                                 row["costs"]["stress"]["worst_return_pct"]), reverse=True)
    single = max(completed, key=lambda row: max(row["costs"]["normal"]["fold_returns_pct"]))
    report = {
        "status": "completed" if len(completed) == len(results) else "completed_with_errors",
        "policy_count": len(results),
        "backtest_count": len(results)*len(plan["schedules"])*len(consensus.COSTS),
        "development_survivor_ids": [row["config"]["id"] for row in completed if row["development_survivor"]],
        "highest_average_normal": ranked[0] if ranked else None,
        "highest_single_fold": {"id": single["config"]["id"],
                                "return_pct": max(single["costs"]["normal"]["fold_returns_pct"])} if completed else None,
        "trials": results,
        "independent_test_evaluated": False,
        "model_approved": False,
        "orders_sent": 0,
    }
    if verify:
        if saved["trials"] != results or saved["development_survivor_ids"] != report["development_survivor_ids"]:
            raise ValueError("Results replay mismatch")
        search.write_json(root/"verification.json", {
            "status": "passed", "policies_replayed": len(results), "backtests_replayed": report["backtest_count"],
            "ledger_files_replayed": len(list((root/"ledgers").glob("*.parquet"))),
            "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0,
        })
    else:
        artifacts = list((root/"trials").glob("*.json"))+list((root/"ledgers").glob("*.parquet"))
        report["artifact_sha256"] = {str(path.relative_to(root)): runner.checksum(path) for path in artifacts}
        search.write_json(root/"results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runtime_reports/low_frequency_targets_20260922"))
    parser.add_argument("--valuation", type=Path, default=Path("runtime_reports/long_window_marks_20260920_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Output must be a project subdirectory")
    started = time.monotonic()
    if not args.verify:
        initialize(root, args.source.resolve(), args.valuation.resolve())
        state = prior.read(root/"state.json")
        state.update(status="running", started_utc=search.utc_now())
        search.write_json(root/"state.json", state)
    report = run(root, args.verify)
    if not args.verify:
        state = prior.read(root/"state.json")
        state.update(status=report["status"], finished_utc=search.utc_now(), elapsed_seconds=time.monotonic()-started,
                     output_bytes=directory_bytes(root), orders_sent=0)
        search.write_json(root/"state.json", state)
    print(json.dumps({"status": report["status"], "policies": report["policy_count"],
                      "survivors": report["development_survivor_ids"], "orders_sent": 0}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
