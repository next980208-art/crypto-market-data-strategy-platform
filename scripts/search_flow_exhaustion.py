"""Predeclared offline comparison of price-only and causal trade-flow features.

Existing data has prior research exposure: test results here are development
evidence, never authorization for real trading or an independent holdout claim.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts import expected_return_research as returns
from scripts import funding_event_backtest as funding
from scripts import run_bounded_profitability_search as search


DATASET = search.ROOT / "runtime_reports/trade_research_20260908/dataset/market=usdm/symbol=BTCUSDT/timeframe=1m/trade_enriched_dataset_BTCUSDT_USDM_1m.parquet"
EXPECTED_SHA = "7673bf6224889af5f67e6a590fd54641a81eb14d24110ea3c261dfa638512b71"
DAY, MINUTE = funding.DAY, funding.MINUTE
COSTS = {"normal": 5., "stress": 10., "severe": 20.}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def features(raw):
    frame = raw.copy()
    price_columns, flow_columns = [], []
    close = frame.close
    minute_return = close.pct_change(fill_method=None)
    for window in (5, 15, 60, 240):
        for name, values in {
            f"price_return_{window}": close.pct_change(window, fill_method=None),
            f"price_volatility_{window}": minute_return.rolling(window).std(),
            f"price_distance_{window}": close / close.rolling(window).mean() - 1,
            f"price_volume_ratio_{window}": frame.volume / frame.volume.rolling(window).mean(),
        }.items():
            frame[name] = values
            price_columns.append(name)
    spread = (frame.high - frame.low).replace(0, np.nan)
    for name, values in {
        "price_body": (close-frame.open)/frame.open,
        "price_upper_wick": (frame.high-frame[["close", "open"]].max(axis=1))/spread,
        "price_lower_wick": (frame[["close", "open"]].min(axis=1)-frame.low)/spread,
    }.items():
        frame[name] = values.fillna(0)
        price_columns.append(name)
    flow_columns.extend(c for c in raw.columns if c.startswith("agg_") and pd.api.types.is_numeric_dtype(raw[c]))
    imbalance = frame.agg_taker_volume_imbalance
    fast = imbalance.rolling(5).mean()
    slow = imbalance.rolling(15).mean()
    for name, values in {
        "flow_fast_change": fast-fast.shift(5),
        "flow_slow_change": slow-slow.shift(15),
        "flow_fast_minus_slow": fast-slow,
        "flow_burst_direction": frame.agg_trade_volume_ratio_15m * slow,
        "flow_price_disagreement": slow * -frame.price_return_15,
        "flow_burst_upper_wick": frame.agg_trade_volume_ratio_15m * frame.price_upper_wick,
        "flow_burst_lower_wick": frame.agg_trade_volume_ratio_15m * frame.price_lower_wick,
    }.items():
        frame[name] = values
        flow_columns.append(name)
    # Only past/current completed minutes; missing warm-up is excluded, not backfilled.
    return frame, {"price": price_columns, "flow": price_columns + flow_columns}


def period(frame, start, end):
    horizon_end = frame.entry_timestamp + frame.max_holding_minutes * MINUTE
    return frame.loc[(frame.entry_timestamp >= start) & (horizon_end < end)].copy()


def funding_market(raw):
    events = raw[["funding_rate_timestamp", "funding_rate"]].dropna().drop_duplicates()
    events = events.sort_values("funding_rate_timestamp")
    if events.funding_rate_timestamp.duplicated().any():
        raise ValueError("Conflicting funding rates")
    times = events.funding_rate_timestamp.to_numpy(dtype=np.int64)
    # This dataset uses approximately 8h settlements with millisecond jitter.
    if len(times) < 2 or (np.abs(np.diff(times)-8*60*MINUTE) > 1000).any():
        raise ValueError("Funding coverage gap or unexpected schedule")
    if times[0] > raw.entry_timestamp.min() or times[-1]+8*60*MINUTE <= raw.entry_timestamp.max():
        raise ValueError("Funding coverage does not enclose research data")
    sources = [{"fundingTime": int(row.funding_rate_timestamp), "fundingRate": float(row.funding_rate),
                "markPrice": ""} for row in events.itertuples()]
    marks = raw[["timestamp", "mark_price"]].rename(columns={"timestamp": "event_ms", "mark_price": "close"})
    return funding.FundingMarket(sources, marks), len(sources)


def funded_target(frame, side, target, market):
    values = returns.target_returns(frame, side, target, COSTS["normal"]).to_numpy().copy()
    holding = f"{side}_tp_{str(float(target)).replace('.', '_')}r_holding_minutes"
    sign = -1 if side == "long" else 1
    for index, row in enumerate(frame.to_dict("records")):
        entry = int(row["entry_timestamp"])
        exit_ms = entry + int(row[holding])*MINUTE
        charge = sum(sign*item["rate"]*item["price"] / row["entry_price"]
                     for item in market.settlements(entry, exit_ms))
        values[index] += charge / row["stop_distance_pct"]
    return values


def make_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.))
    if kind != "xgb":
        raise ValueError("Unknown model")
    return XGBRegressor(n_estimators=150, max_depth=2, learning_rate=.03,
                        min_child_weight=20, reg_lambda=10., reg_alpha=.5,
                        subsample=1., colsample_bytree=1., n_jobs=2,
                        random_state=20261001, tree_method="hist", objective="reg:squarederror")


def evaluate(predictions, target, market, backtest, action):
    metrics, detail = {}, {}
    for name, extra in COSTS.items():
        args = action.backtest_args(extra)
        args.risk_per_trade_pct = .005
        args.take_profit_r = target
        trades, curve, skipped, events, daily = funding.simulate(predictions, args, backtest, market)
        metrics[name] = backtest.summarize(trades, curve, skipped, predictions, args)
        detail[name] = {"trades": trades, "cash_curve": curve, "events": events, "daily": daily}
    return metrics, detail


def select_validation(records):
    eligible = [row for row in records if all(m["trades"] >= 10 for m in row["metrics"].values())]
    # Test results are deliberately absent from this API.
    return max(eligible, key=lambda row: (row["metrics"]["severe"]["total_return_pct"],
                                          row["metrics"]["severe"]["max_drawdown_pct"], row["id"]), default=None)


def rejection_reasons(metrics):
    reasons = []
    for name, summary in metrics.items():
        if summary["trades"] < 10:
            reasons.append(f"{name}:fewer_than_10_trades")
        if summary["total_return_pct"] <= 0:
            reasons.append(f"{name}:nonpositive_return")
        if summary["max_drawdown_pct"] < -5:
            reasons.append(f"{name}:cash_drawdown_exceeds_5pct")
    return reasons


def run(output):
    started = time.monotonic()
    if digest(DATASET) != EXPECTED_SHA:
        raise ValueError("Dataset differs from the audited source")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "created_utc": search.utc_now(), "dataset": str(DATASET.relative_to(search.ROOT)),
        "dataset_sha256": EXPECTED_SHA, "prior_research_exposure": True,
        "feature_sets": ["price", "flow"], "models": ["ridge", "xgb"],
        "targets_R": [1., 1.5, 3.], "thresholds_R": [.05, .15, .3],
        "folds": [{"train_start": f"2026-06-{day:02d}T00:00:00Z", "train_days": 45,
                   "validation_days": 10, "test_days": 10} for day in (1, 11, 21)],
        "decision_cadence_minutes": 15, "label_timeframe_minutes": 1,
        "max_holding_minutes": 240, "purge_minutes": 240,
        "risk_per_trade": .005, "stop_distance": .005,
        "round_trip_cost_bps": {k: 10+v for k, v in COSTS.items()},
        "funding": "historical rates, previous completed 1m mark close proxy, strict entry<settlement<exit",
        "selection": "per fold and feature set: highest severe return among validation policies with >=10 trades at every cost; tie cash MDD then ID",
        "diagnostic_test": "selected validation winner tested even if rejected; not deployment approval",
        "research_gate": "all 3 test folds: >=10 trades, return>0 and realized cash MDD>=-5% at all costs; each validation gate must also pass",
        "real_trading_enabled": False, "frozen_observer_modified": False,
        "maximum_runtime_seconds": 1800,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost")},
        "model_parameters": {kind: {k: str(v) for k, v in make_model(kind).get_params().items()} for kind in ("ridge", "xgb")},
        "source_hashes": {name: digest(search.ROOT / name) for name in (
            "scripts/search_flow_exhaustion.py", "scripts/funding_event_backtest.py",
            "scripts/expected_return_research.py", "8_model_signal_backtest.py",
            "16_action_profitability_walk_forward_model.py")},
    }
    search.write_json(output / "protocol.json", protocol)
    report = {"status": "running", "started_utc": search.utc_now(), "folds": [],
              "real_trading_enabled": False, "independent_holdout": False}
    try:
        raw = pd.read_parquet(DATASET)
        if not raw.timestamp.diff().dropna().eq(MINUTE).all():
            raise ValueError("Noncontiguous minute data")
        if not (raw.label_horizon_complete.eq(True).all() and raw.entry_rule.eq("next_bar_open").all()
                and raw.max_holding_minutes.eq(240).all() and raw.round_trip_cost_bps.eq(10.).all()
                and raw.stop_distance_pct.eq(.005).all()):
            raise ValueError("Unexpected labeling contract")
        market, funding_count = funding_market(raw)
        frame, columns = features(raw)
        finite = np.isfinite(frame[columns["flow"]].to_numpy(dtype=float)).all(axis=1)
        frame = frame.loc[finite & frame.entry_timestamp.mod(15*MINUTE).eq(0)].copy()
        report["data"] = {"source_rows": len(raw), "decision_rows": len(frame), "funding_events": funding_count,
                          "first_utc": str(raw.datetime_utc.min()), "last_utc": str(raw.datetime_utc.max()),
                          "feature_columns": columns, "missing_funding_rate_rows": int(raw.funding_rate.isna().sum())}
        backtest = search.load("8_model_signal_backtest.py", "flow_backtest")
        action = search.load("16_action_profitability_walk_forward_model.py", "flow_action")
        for number, definition in enumerate(protocol["folds"], 1):
            start = int(pd.Timestamp(definition["train_start"]).timestamp()*1000)
            val_start, test_start, test_end = start+45*DAY, start+55*DAY, start+65*DAY
            train, validation, test = [period(frame, a, b) for a, b in
                                       ((start, val_start), (val_start, test_start), (test_start, test_end))]
            if min(map(len, (train, validation, test))) == 0:
                raise ValueError("Empty temporal split")
            folder = output / f"fold_{number}"
            folder.mkdir()
            selected, validation_records, fitted = {}, [], {}
            for feature_set in protocol["feature_sets"]:
                for kind in protocol["models"]:
                    for target in protocol["targets_R"]:
                        if time.monotonic()-started > protocol["maximum_runtime_seconds"]:
                            raise TimeoutError("Predeclared runtime limit")
                        key = f"{feature_set}_{kind}_{target}"
                        models, predictions = {}, {}
                        for side in ("long", "short"):
                            model = make_model(kind)
                            model.fit(train[columns[feature_set]], funded_target(train, side, target, market))
                            predictions[side] = model.predict(validation[columns[feature_set]])
                            models[side] = model
                        fitted[key] = models
                        joblib.dump({"models": models, "features": columns[feature_set]}, folder / f"{key}.joblib")
                        for threshold in protocol["thresholds_R"]:
                            policy_id = f"{key}_threshold_{threshold}"
                            signals = returns.prediction_frame(validation, predictions["long"], predictions["short"], threshold, 0.)
                            metrics, _ = evaluate(signals, target, market, backtest, action)
                            record = {"id": policy_id, "key": key, "feature_set": feature_set,
                                      "target": target, "threshold": threshold, "metrics": metrics,
                                      "rejection_reasons": rejection_reasons(metrics)}
                            validation_records.append(record)
                            search.write_json(folder / "validation_trials.json", validation_records)
                        print(f"fold={number} completed={key}", flush=True)
                selected[feature_set] = select_validation([r for r in validation_records if r["feature_set"] == feature_set])
            # Persist selection before computing or inspecting any test outcome.
            search.write_json(folder / "selection_before_test.json", selected)
            fold = {"fold": number, "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
                    "test_start_utc": pd.Timestamp(test_start, unit="ms", tz="UTC").isoformat(),
                    "test_end_exclusive_utc": pd.Timestamp(test_end, unit="ms", tz="UTC").isoformat(),
                    "validation_trials": len(validation_records), "test": {}}
            for feature_set, choice in selected.items():
                if choice is None:
                    fold["test"][feature_set] = {"status": "no_validation_candidate"}
                    continue
                models = fitted[choice["key"]]
                signals = returns.prediction_frame(test, models["long"].predict(test[columns[feature_set]]),
                                                  models["short"].predict(test[columns[feature_set]]), choice["threshold"], 0.)
                signals.to_parquet(folder / f"{feature_set}_test_predictions.parquet", index=False)
                metrics, detail = evaluate(signals, choice["target"], market, backtest, action)
                search.write_json(folder / f"{feature_set}_test_ledger.json", detail)
                fold["test"][feature_set] = {"status": "diagnostic_only", "selected_policy": choice["id"],
                    "validation_rejection_reasons": choice["rejection_reasons"], "metrics": metrics,
                    "test_rejection_reasons": rejection_reasons(metrics)}
            report["folds"].append(fold)
            search.write_json(output / "result.json", report)
        report["research_gate_passed"] = {feature_set: all(
            fold["test"][feature_set].get("status") == "diagnostic_only"
            and not fold["test"][feature_set]["validation_rejection_reasons"]
            and not fold["test"][feature_set]["test_rejection_reasons"] for fold in report["folds"])
            for feature_set in protocol["feature_sets"]}
        report.update(status="completed", completed_utc=search.utc_now(), elapsed_seconds=round(time.monotonic()-started, 3))
        search.write_json(output / "result.json", report)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_seconds=round(time.monotonic()-started, 3))
        search.write_json(output / "result.json", report)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=search.ROOT / "runtime_reports/flow_exhaustion_20261001")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "research_gate_passed": result["research_gate_passed"],
                      "output": str(args.output)}, indent=2))
