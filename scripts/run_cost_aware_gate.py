"""One fixed chronological meta-gate experiment on the frozen validation dataset."""

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts import analyze_loss_causes as diagnosis
from scripts import expected_return_research as returns
from scripts import run_bounded_profitability_search as search

GATE_THRESHOLD = 0.5
MIN_SIGNALS, MIN_CLASS = 200, 20
CAUSAL_COLUMNS = ["volatility_15m", "volatility_240m", "volatility_ratio_15_240",
                  "return_60m", "volume_ratio_60"]
GATE_COLUMNS = ["selected_expected_r", "direction_margin_r", "is_long"] + CAUSAL_COLUMNS


def gate_features(predictions):
    if not predictions.pred_action.isin(["long", "short"]).all():
        raise ValueError("Gate features require actionable signals only")
    long = predictions.predicted_long_net_label_r
    short = predictions.predicted_short_net_label_r
    frame = predictions[CAUSAL_COLUMNS].copy()
    frame["selected_expected_r"] = np.where(predictions.pred_action.eq("long"), long, short)
    frame["direction_margin_r"] = np.abs(long - short)
    frame["is_long"] = predictions.pred_action.eq("long").astype(int)
    frame = frame[GATE_COLUMNS]
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite gate inputs")
    return frame


def selected_net_returns(predictions, target_r, slippage):
    if not predictions.pred_action.isin(["long", "short"]).all():
        raise ValueError("Targets require actionable signals only")
    return pd.Series(np.where(predictions.pred_action.eq("long"),
                              returns.target_returns(predictions, "long", target_r, slippage),
                              returns.target_returns(predictions, "short", target_r, slippage)),
                     index=predictions.index)


def inner_parts(frame, fold):
    boundary = fold["train_start"] + 30 * search.DAY
    inner = {"train_start": fold["train_start"], "validation_start": boundary,
             "validation_end": fold["validation_start"]}
    if boundary >= inner["validation_end"]:
        raise ValueError("Not enough history for chronological gate training")
    base, meta = search.purged_parts(frame, inner)
    return base, meta, inner


def fit_gate(features, target):
    if not features.index.equals(target.index) or not target.isin([0, 1]).all():
        raise ValueError("Invalid gate targets or row alignment")
    counts = target.value_counts()
    if len(target) < MIN_SIGNALS or any(counts.get(c, 0) < MIN_CLASS for c in (0, 1)):
        return None
    gate = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        gate.fit(features, target)
    return gate


def apply_gate(signals, probabilities):
    probability = np.asarray(probabilities, dtype=float)
    if probability.shape != (len(signals),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid gate probability")
    output = signals.copy()
    output["gate_probability"] = probability
    output.loc[probability < GATE_THRESHOLD, "pred_action"] = "no_trade"
    return output


def signal_diagnostics(net, probability, prior):
    target = net.gt(0).astype(int)
    accepted = probability >= GATE_THRESHOLD
    both_classes = target.nunique() == 2
    return {"signals": len(net), "positive_rate": float(target.mean()) if len(net) else None,
            "roc_auc": float(roc_auc_score(target, probability)) if both_classes else None,
            "average_precision": float(average_precision_score(target, probability)) if both_classes else None,
            "brier": float(brier_score_loss(target, probability)) if len(net) else None,
            "constant_train_prior_brier": float(brier_score_loss(target, np.full(len(net), prior))) if len(net) else None,
            "all_mean_net_label_r": float(net.mean()) if len(net) else None,
            "accepted": int(accepted.sum()),
            "accepted_positive_rate": float(target[accepted].mean()) if accepted.any() else None,
            "accepted_mean_net_label_r": float(net[accepted].mean()) if accepted.any() else None,
            "blocked_mean_net_label_r": float(net[~accepted].mean()) if (~accepted).any() else None}


def predict_base(train, future, features, protocol, config, fold):
    predictions = {}
    for offset, side in enumerate(("long", "short")):
        model = XGBRegressor(objective="reg:squarederror", eval_metric="mae", tree_method="hist",
                             n_jobs=protocol["n_jobs"], random_state=protocol["seed"]+fold["fold"]*10+offset,
                             **config["params"])
        model.fit(train[features], returns.target_returns(train, side, config["take_profit_r"], protocol["extra_slippage_bps"]))
        predictions[side] = model.predict(future[features])
    return returns.prediction_frame(future, predictions["long"], predictions["short"], diagnosis.THRESHOLD, diagnosis.MARGIN)


def save_signals(path, signals, features, net, probability=None):
    audit = signals[["timestamp", "entry_timestamp", "max_holding_minutes", "pred_action"]].copy()
    for column in features:
        audit[column] = features[column]
    audit["actual_net_label_r"] = net
    if probability is not None:
        audit["gate_probability"] = probability
        audit["gate_accepted"] = probability >= GATE_THRESHOLD
    audit.to_parquet(path, index=False)


def run_worker(output):
    protocol = json.loads((diagnosis.REFERENCE / "protocol.json").read_text(encoding="utf-8"))
    previous = json.loads((diagnosis.REFERENCE / f"{diagnosis.TRIAL}.json").read_text(encoding="utf-8"))
    diagnosis.verify_reference(protocol)
    config = next(c for c in protocol["configurations"] if c["id"] == diagnosis.TRIAL)
    plan = {"created_utc": search.utc_now(), "reference_trial": diagnosis.TRIAL,
            "reference_protocol_sha256": diagnosis.digest(diagnosis.REFERENCE / "protocol.json"),
            "reference_trial_sha256": diagnosis.digest(diagnosis.REFERENCE / f"{diagnosis.TRIAL}.json"),
            "dataset": protocol["dataset"], "dataset_sha256": protocol["dataset_sha256"], "packages": protocol["packages"],
            "folds": protocol["folds"], "holdout_start_ms": protocol["holdout_start_ms"],
            "config": config, "base_threshold": diagnosis.THRESHOLD, "base_margin": diagnosis.MARGIN,
            "hypothesis": "A fixed profitability filter trained on prior forward predictions can reject costly false-positive entries",
            "gate": {"model": "StandardScaler + LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 1000,
                     "class_weight": None, "threshold": GATE_THRESHOLD, "features": GATE_COLUMNS,
                     "minimum_signals": MIN_SIGNALS, "minimum_per_class": MIN_CLASS,
                     "inner_base_days": 30, "inner_gate_days": 15,
                     "target": "original chosen direction net label after extra slippage > 0",
                     "insufficient_training_action": "block all signals"},
            "costs_bps": {"label_round_trip": protocol["round_trip_cost_bps"], "extra": protocol["extra_slippage_bps"],
                          "stress_extra": protocol["stress_slippage_bps"]},
            "independent_test_evaluated": False, "outer_validation_previously_seen": True,
            "model_approved": False, "orders_sent": 0, "collection_started": False,
            "current_code_sha256": {name: diagnosis.digest(search.ROOT / name) for name in
                                    ["scripts/run_cost_aware_gate.py", "scripts/analyze_loss_causes.py", "scripts/run_bounded_profitability_search.py"]}}
    search.write_json(output / "protocol.json", plan)
    for name in plan["current_code_sha256"]:
        destination = output / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((search.ROOT / name).read_bytes())
    action = search.load("16_action_profitability_walk_forward_model.py", "gate_action")
    walk = search.load("14_walk_forward_direction_model.py", "gate_walk")
    training = search.load("13_train_selective_direction_model.py", "gate_training")
    extended = search.load("15_two_stage_walk_forward_model.py", "gate_extended")
    backtest = search.load("8_model_signal_backtest.py", "gate_backtest")
    dataset = Path(protocol["dataset"])
    trade_columns = action.verified_trade_columns(protocol["manifest"], dataset)
    result = {"status": "running", "folds": [], "model_approved": False, "live_policy": "unchanged_no_trade",
              "independent_test_evaluated": False, "outer_validation_previously_seen": True, "orders_sent": 0}
    if len(protocol["folds"]) != len(previous["folds"]):
        raise ValueError("Reference fold count changed")
    for fold, old in zip(protocol["folds"], previous["folds"]):
        if fold["fold"] != old["fold"] or fold["validation_end"] > protocol["holdout_start_ms"]:
            raise ValueError("Invalid fold alignment")
        raw = walk.load_fold_source(dataset, {**fold, "test_end": fold["validation_end"]}, protocol["start_ms"],
                                    trade_columns + returns.label_columns(config["take_profit_r"]))
        frame, features = extended.add_two_stage_features(raw, training)
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all() or set(features) & set(returns.label_columns(config["take_profit_r"])):
            raise ValueError("Invalid features or label leakage")
        if not frame.label_horizon_complete.eq(True).all() or not frame.max_holding_minutes.eq(240).all() or not frame.round_trip_cost_bps.eq(protocol["round_trip_cost_bps"]).all():
            raise ValueError("Invalid label horizon or costs")
        train, validation = search.purged_parts(frame, fold)
        if (len(train), len(validation), features) != (old["train_rows"], old["validation_rows"], old["feature_columns"]):
            raise ValueError("Reference rows or features changed")
        inner_train, meta, bounds = inner_parts(train, fold)
        meta_predictions = predict_base(inner_train, meta, features, protocol, config, fold)
        meta_signals = meta_predictions[meta_predictions.pred_action.ne("no_trade")].copy()
        x_meta = gate_features(meta_signals)
        net_meta = selected_net_returns(meta_signals, config["take_profit_r"], protocol["extra_slippage_bps"])
        target = net_meta.gt(0).astype(int)
        gate = fit_gate(x_meta, target)
        save_signals(output / f"gate_training_signals_fold_{fold['fold']}.parquet", meta_signals, x_meta, net_meta)
        predictions = predict_base(train, validation, features, protocol, config, fold)
        signals = predictions[predictions.pred_action.ne("no_trade")].copy()
        baseline = action.evaluate_candidate(backtest, predictions, protocol["extra_slippage_bps"], config["take_profit_r"])
        reference = next(p["base"] for p in old["policies"] if p["threshold"] == diagnosis.THRESHOLD and p["margin"] == diagnosis.MARGIN)
        for key in ("trades", "win_rate_pct", "avg_return_r", "total_return_pct", "max_drawdown_pct"):
            if not np.isclose(baseline[key], reference[key], atol=1e-10, rtol=1e-10):
                raise ValueError(f"Baseline mismatch: {key}")
        x_validation = gate_features(signals)
        probability = gate.predict_proba(x_validation)[:, 1] if gate is not None and len(signals) else np.zeros(len(signals))
        gated = apply_gate(signals, probability)
        metrics = action.evaluate_candidate(backtest, gated, protocol["extra_slippage_bps"], config["take_profit_r"])
        stress = action.evaluate_candidate(backtest, gated, protocol["stress_slippage_bps"], config["take_profit_r"])
        net_validation = selected_net_returns(signals, config["take_profit_r"], protocol["extra_slippage_bps"])
        save_signals(output / f"validation_signals_fold_{fold['fold']}.parquet", signals, x_validation, net_validation, probability)
        args = action.backtest_args(protocol["extra_slippage_bps"])
        args.take_profit_r = config["take_profit_r"]
        trades, _, _ = backtest.simulate(gated[gated.pred_action.ne("no_trade")], args)
        pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "action", "return_pct", "equity"]).to_parquet(output / f"gated_trades_fold_{fold['fold']}.parquet", index=False)
        parameters = None
        if gate is not None:
            scaler, logistic = gate.steps[0][1], gate.steps[1][1]
            parameters = {"features": GATE_COLUMNS, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                          "coefficients": logistic.coef_[0].tolist(), "intercept": float(logistic.intercept_[0]),
                          "iterations": int(logistic.n_iter_[0])}
        row = {"fold": fold["fold"], "baseline_reproduced": True, "inner_bounds": bounds,
               "inner_base_rows": len(inner_train), "meta_rows": len(meta), "meta_signals": len(target),
               "meta_positive_count": int(target.sum()), "meta_positive_rate": float(target.mean()) if len(target) else None,
               "inner_base_last_outcome_limit_ms": int((inner_train.entry_timestamp+inner_train.max_holding_minutes*search.MINUTE).max()),
               "meta_last_outcome_limit_ms": int((meta.entry_timestamp+meta.max_holding_minutes*search.MINUTE).max()),
               "validation_last_outcome_limit_ms": int((validation.entry_timestamp+validation.max_holding_minutes*search.MINUTE).max()),
               "gate_fitted": gate is not None, "gate_parameters": parameters,
               "baseline": baseline, "gated": metrics, "gated_stress": stress,
               "signal_diagnostics": signal_diagnostics(net_validation, probability, float(target.mean()) if len(target) else 0)}
        result["folds"].append(row)
        search.write_json(output / "analysis.json", result)
        print(f"Fold {fold['fold']}: baseline={baseline['total_return_pct']:.3f}%/{baseline['trades']} trades; gated={metrics['total_return_pct']:.3f}%/{metrics['trades']} trades; retained {metrics['signals']}/{len(signals)} signals", flush=True)
    result["gated_summary"] = search.policy_summary([f["gated"] for f in result["folds"]])
    result["stress_summary"] = search.policy_summary([f["gated_stress"] for f in result["folds"]])
    diagnosis.verify_reference(protocol)
    for name, checksum in plan["current_code_sha256"].items():
        if diagnosis.digest(search.ROOT / name) != checksum:
            raise ValueError("Experiment source changed during execution")
    result.update(status="completed", finished_utc=search.utc_now())
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
    start = time.monotonic()
    with (args.output / "execution.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run([sys.executable, "-u", "-m", "scripts.run_cost_aware_gate", "--worker", "--output", str(args.output.resolve())],
                                     cwd=search.ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            code = process.returncode
            status = "completed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status, code = "timeout", 1
    execution = {"status": status, "exit_code": code, "elapsed_seconds": round(time.monotonic()-start, 3),
                 "timeout_seconds": 180, "output_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())}
    search.write_json(args.output / "execution.json", execution)
    print(json.dumps(execution))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
