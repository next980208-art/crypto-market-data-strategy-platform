"""Allowlisted offline research stages. Invoked by automatic_research, never places orders."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import run_bounded_profitability_search as search
from scripts import run_cost_aware_gate as gate
from scripts import audit_nonoverlapping_signals as overlap


class Blocked(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise Blocked(reason)


def audit_dataset(plan):
    file = pq.ParquetFile(plan["dataset"])
    walk = search.load("14_walk_forward_direction_model.py", "auto_audit_walk")
    required = walk.REQUIRED_SOURCE_COLUMNS + ["symbol", "market", "timeframe", "entry_rule"]
    required += [f"{s}_tp_1_0r_{v}" for s in ("long", "short") for v in ("return_r_gross", "outcome")]
    require(set(required).issubset(file.schema.names), "missing_required_source_columns")
    columns = list(dict.fromkeys(required))
    count, first, last, previous_entry_price = 0, None, None, None
    for batch in file.iter_batches(batch_size=65536, columns=columns):
        df = batch.to_pandas()
        t = df.timestamp.to_numpy()
        require(len(t) > 0 and np.isfinite(t).all() and (t == np.floor(t)).all(), "invalid_timestamps")
        require(np.all(np.diff(t) == search.MINUTE) and (last is None or t[0]-last == search.MINUTE), "noncontinuous_minutes")
        require(df.symbol.isin(["BTCUSDT", "BTC/USDT"]).all() and df.market.eq("usdm").all() and df.timeframe.eq("1m").all(), "market_contract_mismatch")
        require(df.entry_rule.eq("next_bar_open").all() and (df.entry_timestamp-df.timestamp).eq(search.MINUTE).all(), "entry_contract_mismatch")
        require(df.stop_distance_pct.eq(.005).all() and df.round_trip_cost_bps.eq(10).all() and df.max_holding_minutes.eq(240).all(), "cost_or_horizon_mismatch")
        require(df.label_horizon_complete.eq(True).all(), "incomplete_source_labels")
        prices = df[["open", "high", "low", "close", "entry_price"]].to_numpy(dtype=float)
        require(np.isfinite(prices).all() and (prices > 0).all() and np.isfinite(df.volume).all() and df.volume.ge(0).all(), "invalid_prices_or_volume")
        require(df.high.ge(df[["open", "close", "low"]].max(axis=1)).all() and df.low.le(df[["open", "close"]].min(axis=1)).all(), "invalid_ohlc")
        require(np.allclose(df.entry_price.iloc[:-1], df.open.iloc[1:], rtol=0, atol=1e-8), "next_open_price_mismatch")
        require(previous_entry_price is None or np.isclose(previous_entry_price, df.open.iloc[0], rtol=0, atol=1e-8), "boundary_entry_price_mismatch")
        for side in ("long", "short"):
            prefix = side+"_tp_1_0r_"
            require(np.allclose(df[prefix+"return_r_gross"]-.2, df[prefix+"return_r_net"], rtol=0, atol=1e-8), "source_label_cost_mismatch")
        first = int(t[0]) if first is None else first
        last, previous_entry_price = int(t[-1]), float(df.entry_price.iloc[-1])
        count += len(df)
    require(count > 0, "empty_dataset")
    folds = make_folds(first, last+search.MINUTE)
    return {"status": "completed", "rows": count, "start_ms": first, "end_exclusive_ms": last+search.MINUTE,
            "folds": folds, "missing_target_columns": sorted(set(gate.returns.label_columns(1.5))-set(file.schema.names)),
            "label_action": "recompute 1R and 1.5R within selected windows; require stored 1R parity before training",
            "last_source_entry_price_check": "not checked beyond final file row", "source_modified": False}


def make_folds(start, end):
    folds = [{"fold": i+1, "start": start+offset*search.DAY,
              "validation_start": start+(offset+120)*search.DAY,
              "validation_end": start+(offset+150)*search.DAY}
             for i, offset in enumerate((0, 365, 730))]
    require(folds[-1]["validation_end"] <= end-365*search.DAY, "insufficient_history_for_fixed_schedule_and_excluded_final_year")
    return folds


def rebuild_labels(raw):
    labeler = search.load("4_triple_barrier_labeler.py", "auto_labels")
    labels = labeler.add_labels(raw, [1.0, 1.5], .005, 240, 10, 2048, 24)
    original = raw.set_index("timestamp", verify_integrity=True).loc[labels.timestamp].reset_index()
    for side in ("long", "short"):
        for suffix in ("return_r_net", "return_r_gross", "holding_minutes"):
            column = side+"_tp_1_0r_"+suffix
            require(np.allclose(original[column], labels[column], atol=1e-8, rtol=1e-8), "recomputed_1R_mismatch_"+column)
        column = side+"_tp_1_0r_outcome"
        require(np.array_equal(original[column].to_numpy(), labels[column].to_numpy()), "recomputed_1R_outcome_mismatch")
        for suffix in ("return_r_net", "return_r_gross", "holding_minutes", "outcome"):
            column = side+"_tp_1_5r_"+suffix
            if column in original:
                require(np.array_equal(original[column].to_numpy(), labels[column].to_numpy()) if suffix == "outcome" else
                        np.allclose(original[column], labels[column], atol=1e-8, rtol=1e-8), "recomputed_1_5R_mismatch")
            original[column] = labels[column].to_numpy()
    return original


def get_result(root, state, task):
    return json.loads((root/state["tasks"][task]["result_path"]).read_text(encoding="utf-8"))


def prepare_fold(plan, root, state, number, output):
    audit = get_result(root, state, "audit")
    fold = audit["folds"][number-1]
    raw = pd.read_parquet(plan["dataset"], filters=[("timestamp", ">=", max(audit["start_ms"], fold["start"]-240*search.MINUTE)),
                                                  ("timestamp", "<", fold["validation_end"])])
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    require(raw.timestamp.is_unique and raw.timestamp.diff().dropna().eq(search.MINUTE).all(), "window_not_continuous")
    frame = rebuild_labels(raw)
    rebuilt_rows = len(frame)
    training = search.load("13_train_selective_direction_model.py", "auto_features")
    extended = search.load("15_two_stage_walk_forward_model.py", "auto_extended")
    frame, features = extended.add_two_stage_features(frame, training)
    require(features == plan["features"] and np.isfinite(frame[features].to_numpy(dtype=float)).all(), "feature_contract_mismatch")
    config = plan["base_config"]
    protocol = {"n_jobs": 2, "seed": 242, "extra_slippage_bps": 5}
    chunks, bounds = [], []
    for index in range(4):
        spec = {"train_start": fold["start"]+index*30*search.DAY,
                "validation_start": fold["start"]+(index+1)*30*search.DAY,
                "validation_end": fold["start"]+(index+2)*30*search.DAY}
        train, future = search.purged_parts(frame, spec)
        prediction = gate.predict_base(train, future, features, protocol, config, {"fold": number*10+index})
        signals = prediction[prediction.pred_action.ne("no_trade")].copy()
        bounds.append({**spec, "train_rows": len(train), "prediction_rows": len(future), "signals": len(signals),
                       "last_train_outcome_ms": int((train.entry_timestamp+240*search.MINUTE).max()),
                       "last_prediction_outcome_ms": int((future.entry_timestamp+240*search.MINUTE).max())})
        if index < 3:
            chunks.append(signals)
        else:
            validation = signals
        print(f"prepare_{number}: forward chunk {index+1}/4, {len(signals)} signals", flush=True)
    all_meta = pd.concat(chunks, ignore_index=True)
    selected = overlap.select_nonoverlapping(all_meta)
    x_meta = gate.gate_features(selected)
    net_meta = gate.selected_net_returns(selected, 1.5, 5)
    compact_meta = selected[["timestamp", "entry_timestamp", "max_holding_minutes", "pred_action"]].copy()
    compact_meta[gate.GATE_COLUMNS] = x_meta
    compact_meta["actual_net_label_r"] = net_meta
    compact_meta.to_parquet(output/"meta.parquet", index=False)
    columns = ["timestamp", "datetime_utc", "entry_timestamp", "entry_price", "entry_rule", "label_horizon_complete",
               "max_holding_minutes", "pred_action", "pred_confidence", "direction_target", "best_action", "stop_distance_pct",
               "round_trip_cost_bps", "predicted_long_net_label_r", "predicted_short_net_label_r"]
    columns += gate.CAUSAL_COLUMNS+gate.returns.label_columns(1.5)
    validation[columns].to_parquet(output/"validation.parquet", index=False)
    action = search.load("16_action_profitability_walk_forward_model.py", "auto_action")
    backtest = search.load("8_model_signal_backtest.py", "auto_backtest")
    baseline = action.evaluate_candidate(backtest, validation, 5, 1.5)
    for name, expected in (("meta.parquet", compact_meta), ("validation.parquet", validation[columns])):
        pd.testing.assert_frame_equal(pd.read_parquet(output/name), expected.reset_index(drop=True))
    return {"status": "completed", "fold": number, "schedule": fold, "chunks": bounds, "rebuilt_rows": rebuilt_rows,
            "stored_1R_parity": True, "feature_count": len(features), "meta_signals_before": len(all_meta),
            "meta_signals_after": len(selected), "meta_positive": int(net_meta.gt(0).sum()),
            "meta_nonpositive": int(net_meta.le(0).sum()), "base_models_fitted": 8, "baseline": baseline,
            "meta_path": str((output/"meta.parquet").relative_to(root)), "validation_path": str((output/"validation.parquet").relative_to(root))}


def trial(plan, root, state, task, output):
    c = {"trial_c1": 1.0, "trial_c01": .1}[task]
    action = search.load("16_action_profitability_walk_forward_model.py", "auto_trial_action")
    backtest = search.load("8_model_signal_backtest.py", "auto_trial_backtest")
    results = []
    for number in (1, 2, 3):
        prepared = get_result(root, state, f"prepare_{number}")
        meta = pd.read_parquet(root/prepared["meta_path"])
        validation = pd.read_parquet(root/prepared["validation_path"])
        target = meta.actual_net_label_r.gt(0).astype(int)
        sufficient = len(meta) >= 200 and min(int(target.sum()), int((1-target).sum())) >= 20
        params = None
        if sufficient:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=1000))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(meta[gate.GATE_COLUMNS], target)
            x = gate.gate_features(validation)
            probability = model.predict_proba(x)[:, 1] if len(x) else np.array([])
            scaler, logistic = model.steps[0][1], model.steps[1][1]
            params = {"columns": gate.GATE_COLUMNS, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                      "coef": logistic.coef_[0].tolist(), "intercept": float(logistic.intercept_[0])}
            expected = 1/(1+np.exp(-np.clip((x.to_numpy()-scaler.mean_)/scaler.scale_ @ logistic.coef_[0]+logistic.intercept_[0], -700, 700)))
            np.testing.assert_allclose(probability, expected, atol=1e-12)
        else:
            probability = np.zeros(len(validation))
        filtered = gate.apply_gate(validation, probability)
        base = action.evaluate_candidate(backtest, filtered, 5, 1.5)
        stress = action.evaluate_candidate(backtest, filtered, 10, 1.5)
        args = action.backtest_args(5)
        args.take_profit_r = 1.5
        trades, _, skipped = backtest.simulate(filtered[filtered.pred_action.ne("no_trade")], args)
        pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"]).to_parquet(output/f"trades_{number}.parquet", index=False)
        pd.DataFrame({"timestamp": validation.timestamp.to_numpy(), "original_action": validation.pred_action.to_numpy(),
                      "probability": probability, "accepted": probability >= .5}).to_parquet(output/f"decisions_{number}.parquet", index=False)
        results.append({"fold": number, "gate_fitted": sufficient, "training_signals": len(meta), "parameters": params,
                        "baseline": prepared["baseline"], "gated": base, "stress": stress, "skipped": dict(skipped)})
        print(f"{task}: fold {number}, fitted={sufficient}, trades={base['trades']}, return={base['total_return_pct']:.3f}%", flush=True)
    summary = search.policy_summary([r["gated"] for r in results])
    stress = search.policy_summary([r["stress"] for r in results])
    return {"status": "completed", "trial": task, "C": c, "folds": results, "summary": summary, "stress_summary": stress,
            "research_candidate": all(r["gate_fitted"] for r in results) and summary["validation_eligible"] and stress["validation_eligible"],
            "model_approved": False, "orders_sent": 0, "independent_test_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=["audit", "prepare_1", "prepare_2", "prepare_3", "trial_c1", "trial_c01"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root/"protocol.json").read_text(encoding="utf-8"))
    state = json.loads((args.root/"state.json").read_text(encoding="utf-8"))
    try:
        if args.task == "audit":
            result = audit_dataset(plan)
        elif args.task.startswith("prepare_"):
            result = prepare_fold(plan, args.root, state, int(args.task[-1]), args.output)
        else:
            result = trial(plan, args.root, state, args.task, args.output)
    except Blocked as exc:
        result = {"status": "blocked", "reason": str(exc), "orders_sent": 0}
    search.write_json(args.output/"result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k in ("status", "reason", "rows", "research_candidate")}), flush=True)


if __name__ == "__main__":
    main()
