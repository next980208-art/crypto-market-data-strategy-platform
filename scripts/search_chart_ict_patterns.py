"""Causal chart-pattern and ICT/SMC proxy comparison with funding; no orders."""

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from scripts import search_asymmetric_long_holds as asymmetric
from scripts import search_low_frequency_targets as lowfreq

consensus = lowfreq.consensus
prior, replay, runner, search = lowfreq.prior, lowfreq.replay, lowfreq.runner, lowfreq.search

PATTERNS = (
    "ma_trend", "donchian_breakout", "support_rejection", "engulfing", "hammer",
    "repeated_level", "triangle_breakout", "inside_breakout",
    "ict_liquidity_sweep", "ict_fvg_displacement", "ict_order_block_proxy",
    "ict_mss_after_sweep", "ict_premium_discount", "ict_session_sweep",
)
EXITS = (
    {"id": "standard", "max_holding_minutes": 1440, "stop_distance_pct": .01, "target_r": 1.5},
    {"id": "runner", "max_holding_minutes": 4320, "stop_distance_pct": .02, "target_r": 5.},
)
OUTPUT_LIMIT = 192*1024**2


def configurations():
    return [{**exit_spec, "id": f"{exit_spec['id']}_{pattern}", "pattern": pattern,
             "risk_per_trade_pct": asymmetric.RISK_PER_TRADE}
            for exit_spec in EXITS for pattern in PATTERNS]


def pattern_masks(frame):
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    open_, high, low, close = (pd.to_numeric(frame[name], errors="raise") for name in ("open", "high", "low", "close"))
    bullish, bearish = close > open_, close < open_
    body = (close-open_).abs()
    atr = frame.atr_14bar_pct*close
    prior_high20 = high.shift(1).rolling(20, min_periods=20).max()
    prior_low20 = low.shift(1).rolling(20, min_periods=20).min()
    prior_high48 = high.shift(1).rolling(48, min_periods=48).max()
    prior_low48 = low.shift(1).rolling(48, min_periods=48).min()
    prior_high12 = high.shift(1).rolling(12, min_periods=12).max()
    prior_low12 = low.shift(1).rolling(12, min_periods=12).min()
    old_low = low.shift(4).rolling(44, min_periods=44).min()
    old_high = high.shift(4).rolling(44, min_periods=44).max()
    recent_high = high.shift(1).rolling(4, min_periods=4).max()
    recent_low = low.shift(1).rolling(4, min_periods=4).min()
    range12 = prior_high12-prior_low12
    range48 = prior_high48-prior_low48
    prior_body_mean = body.shift(1).rolling(20, min_periods=20).mean()

    sweep_long = (low < prior_low20) & (close > prior_low20) & bullish & (frame.close_location >= .6)
    sweep_short = (high > prior_high20) & (close < prior_high20) & bearish & (frame.close_location <= .4)
    sweep_long_recent = sweep_long.shift(1).rolling(4, min_periods=1).max().fillna(0).astype(bool)
    sweep_short_recent = sweep_short.shift(1).rolling(4, min_periods=1).max().fillna(0).astype(bool)
    fvg_long = (low > high.shift(2)) & bullish & (body > 1.2*prior_body_mean)
    fvg_short = (high < low.shift(2)) & bearish & (body > 1.2*prior_body_mean)
    utc = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    session = utc.dt.hour.isin([7, 8, 9, 13, 14, 15])

    masks = {
        "ma_trend": ((frame.trend_ma_12_48 > 0) & (frame.return_12bar > 0),
                     (frame.trend_ma_12_48 < 0) & (frame.return_12bar < 0)),
        "donchian_breakout": (close > prior_high20, close < prior_low20),
        "support_rejection": (sweep_long, sweep_short),
        "engulfing": (bullish & bearish.shift(1, fill_value=False) & (open_ <= close.shift(1)) &
                      (close >= open_.shift(1)) & (frame.range_position_48bar <= .35),
                      bearish & bullish.shift(1, fill_value=False) & (open_ >= close.shift(1)) &
                      (close <= open_.shift(1)) & (frame.range_position_48bar >= .65)),
        "hammer": (bullish & (frame.lower_wick_pct >= 2*body/open_) &
                   (frame.upper_wick_pct <= body/open_) & (frame.range_position_48bar <= .25),
                   bearish & (frame.upper_wick_pct >= 2*body/open_) &
                   (frame.lower_wick_pct <= body/open_) & (frame.range_position_48bar >= .75)),
        "repeated_level": ((low-old_low).abs() <= .25*atr) & (recent_high >= old_low+atr) & bullish,
        "triangle_breakout": ((range12 <= .55*range48) & (close > prior_high12),
                              (range12 <= .55*range48) & (close < prior_low12)),
        "inside_breakout": (((high.shift(1) < high.shift(2)) & (low.shift(1) > low.shift(2)) &
                             (close > high.shift(1))),
                            ((high.shift(1) < high.shift(2)) & (low.shift(1) > low.shift(2)) &
                             (close < low.shift(1)))),
        "ict_liquidity_sweep": (sweep_long & (frame.volume_ratio_12bar > 0),
                                sweep_short & (frame.volume_ratio_12bar > 0)),
        "ict_fvg_displacement": (fvg_long, fvg_short),
        "ict_order_block_proxy": (bearish.shift(1, fill_value=False) & bullish &
                                  (close > high.shift(1)) & (body > 1.5*prior_body_mean),
                                  bullish.shift(1, fill_value=False) & bearish &
                                  (close < low.shift(1)) & (body > 1.5*prior_body_mean)),
        "ict_mss_after_sweep": (sweep_long_recent & (close > prior_high12) & fvg_long,
                                sweep_short_recent & (close < prior_low12) & fvg_short),
        "ict_premium_discount": ((frame.trend_ma_12_48 > 0) & (frame.range_position_48bar <= .25) & bullish,
                                 (frame.trend_ma_12_48 < 0) & (frame.range_position_48bar >= .75) & bearish),
        "ict_session_sweep": (sweep_long & session, sweep_short & session),
    }
    # repeated_level is built symmetrically after its readable long expression above.
    masks["repeated_level"] = (masks["repeated_level"],
                               ((high-old_high).abs() <= .25*atr) & (recent_low <= old_high-atr) & bearish)
    return {name: (long_mask.fillna(False), short_mask.fillna(False)) for name, (long_mask, short_mask) in masks.items()}


def apply_pattern(frame, pattern):
    result = frame.copy().sort_values("timestamp").reset_index(drop=True)
    long_mask, short_mask = pattern_masks(result)[pattern]
    if (long_mask & short_mask).any():
        raise ValueError("Pattern emitted both directions: "+pattern)
    result["pred_action"] = np.select([long_mask, short_mask], ["long", "short"], default="no_trade")
    result["pred_confidence"] = 1.
    return result


def build_runner_frames(plan):
    labeler = search.load("4_triple_barrier_labeler.py", "chart_ict_runner_labels")
    frames = {}
    for schedule in plan["schedules"]:
        fold, start, end = schedule["fold"], schedule["validation_start"], schedule["validation_end"]
        warmup = 2*search.DAY
        raw = pd.read_parquet(plan["dataset"], columns=["timestamp", "open", "high", "low", "close", "volume"],
                              filters=[("timestamp", ">=", start-warmup), ("timestamp", "<", end)])
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        np.testing.assert_array_equal(raw.timestamp, np.arange(start-warmup, end, search.MINUTE))
        bars = lowfreq.aggregate(raw, 15)
        features, _ = lowfreq.add_features(bars)
        labels = labeler.add_labels(bars, [5.], .02, 4320, 10, 1024, 10)
        columns = [c for c in labels if c not in ("datetime_utc", "symbol", "market", "timeframe", "close")]
        frame = features.merge(labels[columns], on="timestamp", how="inner", validate="one_to_one")
        frames[fold] = frame[frame.timestamp >= start].reset_index(drop=True)
    return frames


def build_frames(plan):
    source = Path(plan["source"])
    frames = {"standard": {fold: pd.read_parquet(source/f"validation_15m_hold1440_stop1pct_{fold}.parquet")
                           for fold in (1, 2, 3)},
              "runner": build_runner_frames(plan)}
    for exit_id, values in frames.items():
        for fold, frame in values.items():
            if frame.empty or not frame.label_horizon_complete.eq(True).all():
                raise ValueError(f"Invalid {exit_id} fold{fold}")
    return frames


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
    inputs = [Path(__file__).resolve(), Path(asymmetric.__file__).resolve(), Path(source_plan["dataset"]),
              source/"protocol.json", source/"verification.json", valuation/"collection.json"]
    inputs += [source/f"validation_15m_hold1440_stop1pct_{fold}.parquet" for fold in (1, 2, 3)]
    inputs += [valuation/f"funding_{fold}.json" for fold in (1, 2, 3)]
    inputs += [valuation/f"mark1m_{fold}.parquet" for fold in (1, 2, 3)]
    plan = {
        "created_utc": search.utc_now(), "dataset": source_plan["dataset"], "source": str(source),
        "valuation": str(valuation), "schedules": source_plan["schedules"],
        "patterns": list(PATTERNS), "exits": list(EXITS), "configurations": configurations(),
        "pattern_contract": "Every signal uses completed current/past 15m bars and enters at the next bar open",
        "ict_notice": "ICT/SMC names are deterministic OHLCV proxies, not a claim of canonical or institutional order-flow reconstruction",
        "costs": {name: bps for name, bps in consensus.COSTS}, "label_round_trip_cost_bps": 10,
        "admission": "All normal/stress folds positive with positive mean R; >=2 severe folds positive; MDD >=-20%; >=3 trades/fold and >=15 total",
        "research_sources": [
            "https://papers.ssrn.com/sol3/Delivery.cfm/nber_w7613.pdf?abstractid=228099",
            "https://www.sciencedirect.com/science/article/pii/S1544612319308852",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC3967202/",
            "https://www.youtube.com/watch?v=Bkt8B3kLATQ",
            "https://www.youtube.com/watch?v=BxR3tVRS_FI",
        ],
        "limitations": "Reused three 30-day development folds; proxy definitions overlap; multiple-comparison bias; OHLCV cannot reconstruct order book or institutional intent",
        "limits": {"output_bytes": OUTPUT_LIMIT, "minimum_free_gib": 5},
        "hashes": {str(path): runner.checksum(path) for path in inputs},
        "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False,
    }
    search.write_json(root/"protocol.json", plan)
    search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json"),
                                          "orders_sent": 0})


def valuation_markets(plan):
    root = Path(plan["valuation"])
    return {fold: lowfreq.engine.FundingMarket(prior.read(root/f"funding_{fold}.json"),
                                               pd.read_parquet(root/f"mark1m_{fold}.parquet"))
            for fold in (1, 2, 3)}


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
    frames, markets = build_frames(plan), valuation_markets(plan)
    schedules = {row["fold"]: row for row in plan["schedules"]}
    results = []
    for index, config in enumerate(plan["configurations"], 1):
        try:
            folds = []
            for fold in (1, 2, 3):
                prediction = apply_pattern(frames[config["id"].split("_", 1)[0]][fold], config["pattern"])
                fold_result = {"fold": fold}
                for cost, extra in consensus.COSTS:
                    metrics, tables = asymmetric.evaluate(prediction, config["target_r"], extra, markets[fold])
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
            result = asymmetric.summarize(config, folds)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc),
                      "traceback": traceback.format_exc(), "model_approved": False, "orders_sent": 0}
        trial = root/"trials"/(config["id"]+".json")
        if verify:
            if result != prior.read(trial):
                raise ValueError("Trial replay mismatch: "+config["id"])
        else:
            search.write_json(trial, result)
            if directory_bytes(root) > plan["limits"]["output_bytes"]:
                raise ValueError("Output size limit exceeded")
        results.append(result)
        print(f"{index}/{len(plan['configurations'])} {config['id']}: survivor={result.get('development_survivor')}", flush=True)
    completed = [row for row in results if row["status"] == "completed"]
    ranked = sorted(completed, key=lambda row: (row["costs"]["stress"]["worst_return_pct"],
                                                 row["costs"]["normal"]["average_return_pct"]), reverse=True)
    highest = max(completed, key=lambda row: max(row["costs"]["normal"]["fold_returns_pct"]))
    report = {
        "status": "completed" if len(completed) == len(results) else "completed_with_errors",
        "policy_count": len(results), "backtest_count": len(results)*9,
        "development_survivor_ids": [row["config"]["id"] for row in completed if row["development_survivor"]],
        "best_worst_stress": ranked[0] if ranked else None,
        "highest_single_fold": {"id": highest["config"]["id"],
                                "return_pct": max(highest["costs"]["normal"]["fold_returns_pct"])} if completed else None,
        "trials": results, "independent_test_evaluated": False, "model_approved": False, "orders_sent": 0,
    }
    if verify:
        if saved["trials"] != results or saved["development_survivor_ids"] != report["development_survivor_ids"]:
            raise ValueError("Results replay mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "policies_replayed": len(results),
            "backtests_replayed": report["backtest_count"], "ledger_files_replayed": len(list((root/"ledgers").glob("*.parquet"))),
            "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
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
