"""Fixed-target expected net-return regression; no collection or order submission."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from scripts import run_bounded_profitability_search as search


def configurations():
    return [{"id": f"{features}_return_tp_{str(target).replace('.', '_')}",
             "include_trade": features == "trade", "take_profit_r": target,
             "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.04,
                        "min_child_weight": 8, "gamma": 0.05, "subsample": 0.8,
                        "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5.0}}
            for features in ("base", "trade") for target in (1.0, 1.5, 3.0)]


def label_columns(take_profit_r):
    tag = str(float(take_profit_r)).replace(".", "_")
    return [f"{side}_tp_{tag}r_{field}" for side in ("long", "short")
            for field in ("return_r_net", "holding_minutes", "outcome")]


def target_returns(frame, side, take_profit_r, extra_slippage_bps):
    tag = str(float(take_profit_r)).replace(".", "_")
    stops = pd.to_numeric(frame["stop_distance_pct"], errors="raise")
    if not np.isfinite(stops).all() or (stops <= 0).any() or extra_slippage_bps < 0:
        raise ValueError("Invalid stop distance or cost")
    target = pd.to_numeric(frame[f"{side}_tp_{tag}r_return_r_net"], errors="raise")
    target = target - extra_slippage_bps / 10000 / stops
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite return target")
    return target


def prediction_frame(frame, long_returns, short_returns, threshold, margin):
    long_returns, short_returns = np.asarray(long_returns), np.asarray(short_returns)
    if not np.isfinite([threshold, margin]).all() or threshold < 0 or margin < 0:
        raise ValueError("Invalid entry policy")
    if long_returns.ndim != 1 or short_returns.ndim != 1:
        raise ValueError("Expected one-dimensional predictions")
    if len(frame) != len(long_returns) or len(frame) != len(short_returns):
        raise ValueError("Prediction length mismatch")
    if not np.isfinite(long_returns).all() or not np.isfinite(short_returns).all():
        raise ValueError("Nonfinite predictions")
    long = (long_returns >= threshold) & (long_returns > short_returns) & (long_returns - short_returns >= margin)
    short = (short_returns >= threshold) & (short_returns > long_returns) & (short_returns - long_returns >= margin)
    predictions = frame.copy()
    predictions["pred_action"] = np.where(long, "long", np.where(short, "short", "no_trade"))
    # The existing simulator requires this field, but its confidence filter is disabled.
    predictions["pred_confidence"] = 0.0
    predictions["confidence_semantics"] = "not_applicable_return_regression"
    predictions["predicted_long_net_label_r"] = long_returns
    predictions["predicted_short_net_label_r"] = short_returns
    return predictions


def run_trial(protocol_path, trial_id):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = next(row for row in protocol["configurations"] if row["id"] == trial_id)
    output = protocol_path.parent / f"{trial_id}.json"
    dataset_path = Path(protocol["dataset"])
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed after protocol was frozen")
    action = search.load("16_action_profitability_walk_forward_model.py", "return_action")
    walk = search.load("14_walk_forward_direction_model.py", "return_walk")
    training = search.load("13_train_selective_direction_model.py", "return_features")
    two_stage = search.load("15_two_stage_walk_forward_model.py", "return_extended_features")
    backtest = search.load("8_model_signal_backtest.py", "return_backtest")
    trade_columns = action.verified_trade_columns(protocol["manifest"], dataset_path)
    target_r = config["take_profit_r"]
    labels = label_columns(target_r)
    policies = [(t, m) for t in protocol["thresholds"] for m in protocol["margins"]]
    results, stress_results = {p: [] for p in policies}, {p: [] for p in policies}
    report = {"trial_id": trial_id, "status": "running", "started_utc": search.utc_now(),
              "config": config, "folds": [], "candidates": [], "test_evaluated": False,
              "real_trading_enabled": False, "registry_modified": False, "objective": "expected_net_label_return_R"}
    try:
        for fold in protocol["folds"]:
            raw = walk.load_fold_source(dataset_path, {**fold, "test_end": fold["validation_end"]},
                                        protocol["start_ms"], trade_columns + labels)
            frame, features = two_stage.add_two_stage_features(raw, training)
            if config["include_trade"]:
                features += trade_columns
            if set(features) & set(labels) or not np.isfinite(frame[features].to_numpy(dtype=float)).all():
                raise ValueError("Invalid features or label leakage")
            if not frame.label_horizon_complete.eq(True).all() or not frame.max_holding_minutes.eq(240).all():
                raise ValueError("Incomplete labels or unexpected holding horizon")
            if not frame.round_trip_cost_bps.eq(protocol["round_trip_cost_bps"]).all():
                raise ValueError("Unexpected label costs")
            train, validation = search.purged_parts(frame, fold)
            expected, diagnostics = {}, {}
            for offset, side in enumerate(("long", "short")):
                target = target_returns(train, side, target_r, protocol["extra_slippage_bps"])
                model = XGBRegressor(objective="reg:squarederror", eval_metric="mae", tree_method="hist",
                                     n_jobs=protocol["n_jobs"], random_state=protocol["seed"] + fold["fold"] * 10 + offset,
                                     **config["params"])
                model.fit(train[features], target)
                expected[side] = model.predict(validation[features])
                actual = target_returns(validation, side, target_r, protocol["extra_slippage_bps"])
                diagnostics[side] = {"train_mean_target_r": float(target.mean()),
                                     "validation_mae_r": float(np.mean(np.abs(expected[side] - actual))),
                                     "predicted_r_quantiles": [float(v) for v in np.quantile(expected[side], [0, .5, .9, 1])]}
            fold_result = {"fold": fold["fold"], "train_rows": len(train), "validation_rows": len(validation),
                           "features": len(features), "feature_columns": features, "diagnostics": diagnostics,
                           "last_train_outcome_limit_ms": int((train.entry_timestamp + train.max_holding_minutes * search.MINUTE).max()),
                           "last_validation_outcome_limit_ms": int((validation.entry_timestamp + validation.max_holding_minutes * search.MINUTE).max()),
                           "policies": []}
            for threshold, margin in policies:
                predictions = prediction_frame(validation, expected["long"], expected["short"], threshold, margin)
                metrics = action.evaluate_candidate(backtest, predictions, protocol["extra_slippage_bps"], target_r)
                stress = action.evaluate_candidate(backtest, predictions, protocol["stress_slippage_bps"], target_r)
                results[threshold, margin].append(metrics)
                stress_results[threshold, margin].append(stress)
                fold_result["policies"].append({"threshold": threshold, "margin": margin, "base": metrics, "stress": stress})
            report["folds"].append(fold_result)
            search.write_json(output, report)
            print(f"{trial_id}: fold {fold['fold']}/3 completed", flush=True)
        for pair, metrics in results.items():
            summary = search.policy_summary(metrics, protocol["minimum_validation_trades"])
            stress = search.policy_summary(stress_results[pair], protocol["minimum_validation_trades"])
            report["candidates"].append({"trial_id": trial_id, "take_profit_r": target_r,
                                         "threshold": pair[0], "margin": pair[1], **summary,
                                         "stress": stress, "stress_survived": stress["validation_eligible"]})
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_utc"] = search.utc_now()
        search.write_json(output, report)
