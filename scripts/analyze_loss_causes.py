"""Reproduce one frozen validation policy and diagnose costs and direction; no orders."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from scripts import expected_return_research as returns
from scripts import run_bounded_profitability_search as search

REFERENCE = search.ROOT / "runtime_reports/expected_return_search_20260910"
TRIAL = "base_return_tp_1_5"
THRESHOLD, MARGIN = .05, 0.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flip_predictions(frame):
    result = frame.copy()
    if not result.pred_action.isin(["long", "short", "no_trade"]).all():
        raise ValueError("Unexpected action")
    result["pred_action"] = result.pred_action.map({"long": "short", "short": "long", "no_trade": "no_trade"})
    return result


def decompose(trades, predictions, args, backtest):
    source = predictions.set_index("timestamp", verify_integrity=True)
    previous_equity = 1.0
    rows = []
    for trade in trades:
        row = source.loc[trade["signal_timestamp"]]
        side = trade["action"]
        fee_r = float(row.round_trip_cost_bps) / 10000 / float(row.stop_distance_pct)
        slip_r = args.extra_slippage_bps / 10000 / float(row.stop_distance_pct)
        gross_r = trade["label_return_r_net"] + fee_r
        # Freeze the original position's capital exposure, including its original compounding.
        exposure = previous_equity * args.risk_per_trade_pct / (1 + fee_r + slip_r)
        gross_pnl, fee_pnl, slip_pnl = exposure * gross_r, exposure * fee_r, exposure * slip_r
        net_pnl = previous_equity * trade["return_pct"]
        if not np.isclose(gross_pnl - fee_pnl - slip_pnl, net_pnl, atol=1e-12):
            raise AssertionError("Cost decomposition mismatch")
        opposite = "short" if side == "long" else "long"
        opposite_label = float(row[backtest.return_column(opposite, args.take_profit_r)])
        rows.append({**trade, "equity_before": previous_equity, "gross_label_r": gross_r,
                     "net_label_r_after_slippage": trade["label_return_r_net"] - slip_r,
                     "fee_label_r": fee_r, "slippage_label_r": slip_r,
                     "gross_pnl_initial_capital": gross_pnl, "fee_pnl_initial_capital": fee_pnl,
                     "slippage_pnl_initial_capital": slip_pnl, "net_pnl_initial_capital": net_pnl,
                     "original_signal_expected_r": float(max(row.predicted_long_net_label_r, row.predicted_short_net_label_r)),
                     "volatility_group": str(row.volatility_group), "score_group": str(row.score_group),
                     "opposite_same_entry_net_label_r": opposite_label - slip_r,
                     "opposite_same_entry_holding_minutes": int(row[backtest.holding_column(opposite, args.take_profit_r)]),
                     "opposite_same_entry_outcome": str(row[backtest.outcome_column(opposite, args.take_profit_r)])})
        previous_equity = trade["equity"]
    return pd.DataFrame(rows)


def group_statistics(frame):
    result = []
    if frame.empty:
        return result
    for field in ("action", "date_utc", "outcome", "volatility_group", "score_group"):
        for name, group in frame.groupby(field, observed=True):
            win = group[group.return_r > 0].return_r
            loss = group[group.return_r < 0].return_r
            result.append({"field": field, "value": str(name), "trades": len(group),
                           "small_sample_under_20": len(group) < 20,
                           "win_rate_pct": float((group.return_r > 0).mean()*100),
                           "avg_r": float(group.return_r.mean()),
                           "mean_gross_label_r": float(group.gross_label_r.mean()),
                           "mean_net_label_r": float(group.net_label_r_after_slippage.mean()),
                           "mean_predicted_label_r": float(group.original_signal_expected_r.mean()),
                           "mean_winning_r": float(win.mean()) if len(win) else None,
                           "mean_losing_r": float(loss.mean()) if len(loss) else None,
                           "mean_holding_minutes": float(group.holding_minutes.mean())})
    return result


def verify_reference(protocol):
    if digest(Path(protocol["dataset"])) != protocol["dataset_sha256"]:
        raise ValueError("Reference dataset changed")
    for package, version in protocol["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Package version changed: {package}")
    for name, checksum in protocol["code_sha256"].items():
        snapshot = REFERENCE / "code_snapshot" / name
        if digest(snapshot) != checksum:
            raise ValueError(f"Corrupt source snapshot: {name}")
        current = search.ROOT / name
        if name != "scripts/run_bounded_profitability_search.py":
            if digest(current) != checksum:
                raise ValueError(f"Research code changed: {name}")
        else:
            # The runner gained file-write retries after the original run. Used analysis helpers must match.
            trees = [ast.parse(p.read_text(encoding="utf-8")) for p in (snapshot, current)]
            for function in ("purged_parts", "load", "policy_summary"):
                nodes = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == function) for t in trees]
                if ast.dump(nodes[0]) != ast.dump(nodes[1]):
                    raise ValueError(f"Reference helper changed: {function}")


def run_worker(output):
    protocol = json.loads((REFERENCE / "protocol.json").read_text(encoding="utf-8"))
    original = json.loads((REFERENCE / f"{TRIAL}.json").read_text(encoding="utf-8"))
    verify_reference(protocol)
    config = next(c for c in protocol["configurations"] if c["id"] == TRIAL)
    plan = {"created_utc": search.utc_now(), "reference": str(REFERENCE), "trial": TRIAL,
            "threshold": THRESHOLD, "margin": MARGIN, "config": config, "folds": protocol["folds"],
            "dataset_sha256": protocol["dataset_sha256"], "packages": protocol["packages"],
            "reference_protocol_sha256": digest(REFERENCE / "protocol.json"),
            "reference_trial_sha256": digest(REFERENCE / f"{TRIAL}.json"),
            "analysis_code_sha256": digest(Path(__file__)),
            "reason_for_policy": "Previously discussed approximately -20 percent base 1.5R diagnostic policy; no new policy search",
            "cost_method": "additive PnL on original executed trades and original capital exposure",
            "matched_reverse_method": "opposite labels at original executed entries; overlaps possible; NOT a portfolio return",
            "reverse_method": "flip all actionable original predictions, preserve no_trade, resimulate position and daily limits",
            "volatility_groups": "training-only volatility_15m quantiles 1/3 and 2/3 per fold",
            "score_groups": "fixed predicted net-label-R bins <0.10, 0.10..0.20, >=0.20",
            "independent_test_evaluated": False, "orders_sent": 0, "registry_modified": False}
    search.write_json(output / "protocol.json", plan)
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    action = search.load("16_action_profitability_walk_forward_model.py", "loss_action")
    walk = search.load("14_walk_forward_direction_model.py", "loss_walk")
    training = search.load("13_train_selective_direction_model.py", "loss_training")
    extended = search.load("15_two_stage_walk_forward_model.py", "loss_extended")
    backtest = search.load("8_model_signal_backtest.py", "loss_backtest")
    dataset = Path(protocol["dataset"])
    trade_columns = action.verified_trade_columns(protocol["manifest"], dataset)
    result = {"status": "running", "folds": [], "independent_test_evaluated": False,
              "model_approved": False, "orders_sent": 0, "live_policy": "unchanged_no_trade"}
    frames = []
    if len(protocol["folds"]) != len(original["folds"]):
        raise ValueError("Reference fold count changed")
    for fold, old in zip(protocol["folds"], original["folds"]):
        if fold["fold"] != old["fold"] or fold["validation_end"] > protocol["holdout_start_ms"]:
            raise ValueError("Invalid fold alignment or reserved period overlap")
        raw = walk.load_fold_source(dataset, {**fold, "test_end": fold["validation_end"]},
                                    protocol["start_ms"], trade_columns + returns.label_columns(config["take_profit_r"]))
        frame, features = extended.add_two_stage_features(raw, training)
        if config["include_trade"]:
            features += trade_columns
        if set(features) & set(returns.label_columns(config["take_profit_r"])) or not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError("Invalid features or label leakage")
        if not frame.label_horizon_complete.eq(True).all() or not frame.max_holding_minutes.eq(240).all():
            raise ValueError("Incomplete labels or unexpected holding horizon")
        train, validation = search.purged_parts(frame, fold)
        if (len(train), len(validation), features) != (old["train_rows"], old["validation_rows"], old["feature_columns"]):
            raise ValueError("Reference rows/features changed")
        if not frame.round_trip_cost_bps.eq(protocol["round_trip_cost_bps"]).all():
            raise ValueError("Unexpected label costs")
        predicted = {}
        for offset, side in enumerate(("long", "short")):
            model = XGBRegressor(objective="reg:squarederror", eval_metric="mae", tree_method="hist",
                                 n_jobs=protocol["n_jobs"], random_state=protocol["seed"]+fold["fold"]*10+offset,
                                 **config["params"])
            model.fit(train[features], returns.target_returns(train, side, config["take_profit_r"], protocol["extra_slippage_bps"]))
            predicted[side] = model.predict(validation[features])
        predictions = returns.prediction_frame(validation, predicted["long"], predicted["short"], THRESHOLD, MARGIN)
        cuts = train.volatility_15m.quantile([1/3, 2/3]).to_numpy()
        predictions["volatility_group"] = np.where(predictions.volatility_15m < cuts[0], "low", np.where(predictions.volatility_15m < cuts[1], "middle", "high"))
        scores = np.maximum(predicted["long"], predicted["short"])
        predictions["score_group"] = np.where(scores < .1, "under_0.10", np.where(scores < .2, "0.10_to_0.20", "at_least_0.20"))
        args = action.backtest_args(protocol["extra_slippage_bps"])
        args.take_profit_r = config["take_profit_r"]
        signals = predictions[predictions.pred_action.ne("no_trade")]
        trades, equity, skipped = backtest.simulate(signals, args)
        metrics = backtest.summarize(trades, equity, skipped, signals, args)
        previous = next(p["base"] for p in old["policies"] if p["threshold"] == THRESHOLD and p["margin"] == MARGIN)
        for key in ("trades", "win_rate_pct", "avg_return_r", "total_return_pct", "max_drawdown_pct"):
            if not np.isclose(metrics[key], previous[key], atol=1e-10, rtol=1e-10):
                raise ValueError(f"Reference metrics mismatch: fold {fold['fold']}, {key}")
        decomposed = decompose(trades, predictions, args, backtest)
        if decomposed.empty:
            raise ValueError("Reference policy has no executed trades")
        decomposed["fold"] = fold["fold"]
        decomposed.to_parquet(output / f"original_trades_fold_{fold['fold']}.parquet", index=False)
        frames.append(decomposed)
        costs = {k: float(decomposed[k+"_pnl_initial_capital"].sum()*100) for k in ("gross", "fee", "slippage", "net")}
        if not np.isclose(costs["net"], metrics["total_return_pct"], atol=1e-10):
            raise AssertionError("Trade PnL does not reconcile with portfolio")
        reversed_signals = flip_predictions(signals)
        reversed_trades, reversed_equity, reversed_skipped = backtest.simulate(reversed_signals, args)
        reversed_metrics = backtest.summarize(reversed_trades, reversed_equity, reversed_skipped, reversed_signals, args)
        pd.DataFrame(reversed_trades).to_parquet(output / f"reversed_trades_fold_{fold['fold']}.parquet", index=False)
        reversed_stress = action.evaluate_candidate(backtest, reversed_signals, protocol["stress_slippage_bps"], args.take_profit_r)
        row = {"fold": fold["fold"], "reference_reproduced": True, "train_rows": len(train), "validation_rows": len(validation),
               "last_train_outcome_limit_ms": int((train.entry_timestamp+train.max_holding_minutes*search.MINUTE).max()),
               "last_validation_outcome_limit_ms": int((validation.entry_timestamp+validation.max_holding_minutes*search.MINUTE).max()),
               "volatility_cuts_from_train": cuts.tolist(), "original": metrics, "fixed_exposure_pnl_pct_initial_capital": costs,
               "matched_entry_reverse_mean_net_label_r": float(decomposed.opposite_same_entry_net_label_r.mean()),
               "original_mean_net_label_r": float(decomposed.net_label_r_after_slippage.mean()),
               "original_mean_predicted_label_r": float(decomposed.original_signal_expected_r.mean()),
               "reversed": reversed_metrics, "reversed_stress": reversed_stress}
        result["folds"].append(row)
        search.write_json(output / "analysis.json", result)
        print(f"Fold {fold['fold']}: reproduced {len(trades)} trades; original={metrics['total_return_pct']:.3f}%, reversed={reversed_metrics['total_return_pct']:.3f}%", flush=True)
    result["groups"] = group_statistics(pd.concat(frames, ignore_index=True))
    if digest(dataset) != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed during analysis")
    result["reversed_validation_summary"] = search.policy_summary([f["reversed"] for f in result["folds"]])
    result["reversed_stress_summary"] = search.policy_summary([f["reversed_stress"] for f in result["folds"]])
    result.update(status="completed", finished_utc=search.utc_now(), total_original_trades=sum(len(f) for f in frames))
    search.write_json(output / "analysis.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker(args.output)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (args.output / "execution.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run([sys.executable, "-u", "-m", "scripts.analyze_loss_causes", "--worker", "--output", str(args.output.resolve())],
                                     cwd=search.ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            status, code = ("completed" if process.returncode == 0 else "failed"), process.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", 1
    execution = {"status": status, "exit_code": code, "elapsed_seconds": round(time.monotonic()-started, 3),
                 "timeout_seconds": 180, "output_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())}
    search.write_json(args.output / "execution.json", execution)
    print(json.dumps(execution))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
