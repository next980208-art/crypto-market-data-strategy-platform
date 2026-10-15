import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_SET_VERSION = "causal_stationary_v1"
ACTION_MAP = {0: "short", 1: "long"}
DEFAULT_DATASET = (
    "ml_dataset_v2_3m/market=usdm/symbol=BTCUSDT/timeframe=1m/"
    "ml_dataset_BTCUSDT_USDM_1m.parquet"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a selective BTCUSDT direction model with causal stationary features."
    )
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET)
    parser.add_argument("--model-folder", default="models_selective_v3")
    parser.add_argument("--prediction-folder", default="predictions_selective_v3")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="usdm")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--ambiguous-return-r", type=float, default=0.20)
    parser.add_argument(
        "--threshold-candidates",
        default="0.52,0.55,0.58,0.60,0.62,0.65",
    )
    parser.add_argument("--minimum-validation-trades", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def prediction_path(folder, symbol, market, timeframe):
    root = Path(folder)
    if market != "spot":
        root = root / f"market={safe_value(market)}"
    suffix = "" if market == "spot" else f"_{safe_value(market).upper()}"
    return (
        root
        / f"symbol={safe_value(symbol)}"
        / f"timeframe={safe_value(timeframe)}"
        / f"direction_predictions_{safe_value(symbol)}{suffix}_{safe_value(timeframe)}.parquet"
    )


def load_backtest_module():
    path = PROJECT_ROOT / "8_model_signal_backtest.py"
    spec = importlib.util.spec_from_file_location("model_signal_backtest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backtest module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_causal_features(frame):
    df = frame.sort_values("timestamp").copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    open_price = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    for minutes in (1, 3, 5, 15, 30, 60):
        df[f"return_{minutes}m"] = close.pct_change(minutes, fill_method=None)

    rolling_means = {}
    for minutes in (5, 15, 60):
        rolling_means[minutes] = close.rolling(minutes, min_periods=minutes).mean()
        df[f"close_to_ma_{minutes}"] = close / rolling_means[minutes] - 1.0
        df[f"volatility_{minutes}m"] = (
            df["return_1m"].rolling(minutes, min_periods=minutes).std(ddof=0)
        )

    df["trend_ma_15_60"] = rolling_means[15] / rolling_means[60] - 1.0
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["ema_12_26_spread"] = ema_12 / ema_26 - 1.0

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14_pct"] = true_range.rolling(14, min_periods=14).mean() / close
    df["candle_body_pct"] = (close - open_price) / open_price
    df["candle_range_pct"] = (high - low) / open_price
    candle_range = (high - low).replace(0.0, np.nan)
    df["close_location"] = (close - low) / candle_range
    df["upper_wick_pct"] = (high - np.maximum(open_price, close)) / open_price
    df["lower_wick_pct"] = (np.minimum(open_price, close) - low) / open_price

    log_volume = np.log1p(volume)
    df["log_volume_change_1m"] = log_volume.diff()
    for minutes in (5, 20, 60):
        rolling_volume = volume.rolling(minutes, min_periods=minutes).mean()
        df[f"volume_ratio_{minutes}"] = volume / rolling_volume - 1.0

    delta = close.diff()
    average_gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    average_loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    df["rsi_14"] = (100.0 - 100.0 / (1.0 + relative_strength)) / 100.0

    rolling_low = low.rolling(60, min_periods=60).min()
    rolling_high = high.rolling(60, min_periods=60).max()
    df["range_position_60"] = (close - rolling_low) / (rolling_high - rolling_low).replace(
        0.0, np.nan
    )

    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df["weekday_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)

    base_features = [
        "return_1m",
        "return_3m",
        "return_5m",
        "return_15m",
        "return_30m",
        "return_60m",
        "close_to_ma_5",
        "close_to_ma_15",
        "close_to_ma_60",
        "volatility_5m",
        "volatility_15m",
        "volatility_60m",
        "trend_ma_15_60",
        "ema_12_26_spread",
        "atr_14_pct",
        "candle_body_pct",
        "candle_range_pct",
        "close_location",
        "upper_wick_pct",
        "lower_wick_pct",
        "log_volume_change_1m",
        "volume_ratio_5",
        "volume_ratio_20",
        "volume_ratio_60",
        "rsi_14",
        "range_position_60",
        "minute_sin",
        "minute_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    optional_features = [
        "mark_basis_pct",
        "funding_rate",
        "funding_rate_age_minutes",
        "open_interest",
        "open_interest_value",
    ]
    features = list(base_features)
    for column in optional_features:
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any():
            df[column] = pd.to_numeric(df[column], errors="coerce")
            features.append(column)

    required_labels = [
        "long_tp_1_0r_return_r_net",
        "short_tp_1_0r_return_r_net",
        "max_holding_minutes",
        "label_horizon_complete",
    ]
    missing = [column for column in features + required_labels if column not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")

    long_return = pd.to_numeric(df["long_tp_1_0r_return_r_net"], errors="coerce")
    short_return = pd.to_numeric(df["short_tp_1_0r_return_r_net"], errors="coerce")
    df["fixed_1r_direction"] = (long_return > short_return).astype(int)
    df["fixed_1r_best_return"] = np.maximum(long_return, short_return)
    df["fixed_1r_best_action"] = np.where(df["fixed_1r_direction"].eq(1), "long", "short")

    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + required_labels).reset_index(drop=True)
    if not df["label_horizon_complete"].eq(True).all():
        raise RuntimeError("Incomplete label horizons are not allowed.")
    return df, features


def verify_causal_features(raw, full, feature_columns):
    if len(raw) < 1000:
        raise RuntimeError("At least 1,000 rows are required for causal feature verification.")
    checkpoints = sorted({min(5000, len(raw) - 1), len(raw) // 2})
    for checkpoint in checkpoints:
        prefix, prefix_features = add_causal_features(raw.iloc[: checkpoint + 1].copy())
        if prefix_features != feature_columns:
            raise RuntimeError("Feature column order changed during causal verification.")
        timestamp = int(raw.iloc[checkpoint]["timestamp"])
        expected = prefix[prefix["timestamp"].eq(timestamp)]
        actual = full[full["timestamp"].eq(timestamp)]
        if len(expected) != 1 or len(actual) != 1:
            raise RuntimeError(f"Could not match causal checkpoint timestamp={timestamp}")
        if not np.allclose(
            expected.iloc[0][feature_columns].to_numpy(dtype=float),
            actual.iloc[0][feature_columns].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=False,
        ):
            raise RuntimeError(f"Future-data leakage detected at timestamp={timestamp}")


def chronological_split(df, train_fraction, validation_fraction):
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must be below 1.")

    train_cut = int(len(df) * train_fraction)
    validation_cut = int(len(df) * (train_fraction + validation_fraction))
    validation_start = int(df.iloc[train_cut]["timestamp"])
    test_start = int(df.iloc[validation_cut]["timestamp"])
    horizon_ms = int(df["max_holding_minutes"].max()) * 60_000

    initial_train = df.iloc[:train_cut].copy()
    train = initial_train[initial_train["timestamp"] + horizon_ms < validation_start].copy()
    initial_validation = df.iloc[train_cut:validation_cut].copy()
    validation = initial_validation[
        initial_validation["timestamp"] + horizon_ms < test_start
    ].copy()
    test = df.iloc[validation_cut:].copy()
    train_validation = df.iloc[:validation_cut].copy()
    train_validation = train_validation[
        train_validation["timestamp"] + horizon_ms < test_start
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise RuntimeError("Chronological split produced an empty partition.")
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "train_validation": train_validation,
        "train_purged_rows": len(initial_train) - len(train),
        "validation_purged_rows": len(initial_validation) - len(validation),
        "final_purged_rows": validation_cut - len(train_validation),
        "validation_start_timestamp": validation_start,
        "test_start_timestamp": test_start,
        "horizon_minutes": horizon_ms // 60_000,
    }


def actionable(frame, ambiguous_return_r):
    selected = frame[frame["fixed_1r_best_return"] > ambiguous_return_r].copy()
    if selected["fixed_1r_direction"].nunique() != 2:
        raise RuntimeError("Actionable training data must contain both long and short targets.")
    return selected


def sample_weights(target):
    counts = target.value_counts().to_dict()
    total = len(target)
    return target.map(lambda value: total / (2 * counts[value])).astype(float)


def make_model(random_state):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=320,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=8,
        gamma=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=5.0,
        tree_method="hist",
        n_jobs=4,
        random_state=random_state,
    )


def train_model(frame, feature_columns, ambiguous_return_r, random_state):
    selected = actionable(frame, ambiguous_return_r)
    target = selected["fixed_1r_direction"].astype(int)
    model = make_model(random_state)
    model.fit(
        selected[feature_columns],
        target,
        sample_weight=sample_weights(target),
    )
    return model, len(selected)


def prediction_frame(model, frame, feature_columns, probability_threshold):
    probability_long = model.predict_proba(frame[feature_columns])[:, 1]
    probability_short = 1.0 - probability_long
    if probability_threshold >= 1.0:
        action = np.full(len(frame), "no_trade", dtype=object)
    else:
        action = np.where(
            probability_long >= probability_threshold,
            "long",
            np.where(probability_short >= probability_threshold, "short", "no_trade"),
        )

    predictions = frame.copy()
    predictions["original_direction_target"] = predictions["direction_target"]
    predictions["original_best_action"] = predictions["best_action"]
    predictions["direction_target"] = predictions["fixed_1r_direction"]
    predictions["best_action"] = predictions["fixed_1r_best_action"]
    predictions["pred_action"] = action
    predictions["pred_direction_target"] = np.where(
        action == "long", 1, np.where(action == "short", 0, -1)
    )
    predictions["pred_confidence"] = np.maximum(probability_long, probability_short)
    predictions["prob_long"] = probability_long
    predictions["prob_short"] = probability_short
    predictions["prob_no_trade"] = 1.0 - predictions["pred_confidence"]
    predictions["model_feature_set_version"] = FEATURE_SET_VERSION
    return predictions


def backtest_args(probability_threshold, symbol, market, timeframe):
    return SimpleNamespace(
        symbol=symbol,
        market=market,
        timeframe=timeframe,
        take_profit_r=1.0,
        risk_per_trade_pct=0.02,
        min_confidence=probability_threshold,
        min_minutes_between_entries=15,
        max_new_entries_per_day=5,
        daily_stop_pct=0.04,
        hard_daily_stop_pct=0.06,
        max_consecutive_losses=3,
        extra_slippage_bps=5.0,
    )


def select_threshold(
    model,
    validation,
    feature_columns,
    thresholds,
    minimum_trades,
    symbol,
    market,
    timeframe,
):
    backtest = load_backtest_module()
    candidates = []
    for threshold in thresholds:
        predictions = prediction_frame(model, validation, feature_columns, threshold)
        args = backtest_args(threshold, symbol, market, timeframe)
        trades, equity_curve, skipped = backtest.simulate(predictions, args)
        summary = backtest.summarize(trades, equity_curve, skipped, predictions, args)
        score = summary["total_return_pct"] - 0.5 * abs(summary["max_drawdown_pct"])
        candidates.append(
            {
                "threshold": threshold,
                "trades": summary["trades"],
                "win_rate_pct": summary["win_rate_pct"],
                "avg_return_r": summary["avg_return_r"],
                "total_return_pct": summary["total_return_pct"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "selection_score": score,
                "eligible": summary["trades"] >= minimum_trades,
            }
        )

    return choose_threshold_candidate(candidates, minimum_trades), candidates


def choose_threshold_candidate(candidates, minimum_trades):
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError(
            f"No probability threshold produced at least {minimum_trades} validation trades."
        )
    selected = max(
        eligible,
        key=lambda item: (item["selection_score"], item["avg_return_r"], item["trades"]),
    )
    if selected["avg_return_r"] <= 0 or selected["total_return_pct"] <= 0:
        return {
            "threshold": 1.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_r": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "selection_score": 0.0,
            "eligible": True,
            "deployable": False,
            "abstain_reason": "no_positive_expectancy_validation_candidate",
            "best_evaluated_candidate": selected,
        }
    selected = dict(selected)
    selected["deployable"] = True
    selected["abstain_reason"] = None
    return selected


def directional_metrics(model, frame, feature_columns, ambiguous_return_r):
    selected = actionable(frame, ambiguous_return_r)
    target = selected["fixed_1r_direction"].astype(int)
    probabilities = model.predict_proba(selected[feature_columns])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    majority = float(target.value_counts(normalize=True).max())
    return {
        "rows": len(selected),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "majority_baseline_accuracy": majority,
        "accuracy_lift_over_majority": float(accuracy_score(target, predicted) - majority),
        "long_share": float(target.mean()),
    }


def parse_thresholds(raw):
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0.5 or value >= 1 for value in values):
        raise ValueError("Threshold candidates must be between 0.5 and 1.0.")
    return values


def timestamp_iso(value):
    return pd.to_datetime(int(value), unit="ms", utc=True).isoformat()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ML dataset does not exist: {dataset_path}")

    raw = pd.read_parquet(dataset_path).sort_values("timestamp").reset_index(drop=True)
    dataset, feature_columns = add_causal_features(raw)
    verify_causal_features(raw, dataset, feature_columns)
    split = chronological_split(dataset, args.train_fraction, args.validation_fraction)
    thresholds = parse_thresholds(args.threshold_candidates)

    validation_model, validation_train_rows = train_model(
        split["train"], feature_columns, args.ambiguous_return_r, args.random_state
    )
    selected_threshold, threshold_results = select_threshold(
        validation_model,
        split["validation"],
        feature_columns,
        thresholds,
        args.minimum_validation_trades,
        args.symbol,
        args.market,
        args.timeframe,
    )

    final_model, final_train_rows = train_model(
        split["train_validation"],
        feature_columns,
        args.ambiguous_return_r,
        args.random_state,
    )
    test_metrics = directional_metrics(
        final_model, split["test"], feature_columns, args.ambiguous_return_r
    )
    predictions = prediction_frame(
        final_model,
        split["test"],
        feature_columns,
        selected_threshold["threshold"],
    )

    model_dir = Path(args.model_folder)
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.market == "spot" else f"_{safe_value(args.market).upper()}"
    stem = f"selective_direction_model_{safe_value(args.symbol)}{suffix}_{safe_value(args.timeframe)}"
    model_path = model_dir / f"{stem}.json"
    metadata_path = model_dir / f"{stem}_metadata.json"
    final_model.save_model(model_path)

    output_prediction_path = prediction_path(
        args.prediction_folder, args.symbol, args.market, args.timeframe
    )
    output_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_prediction_path, index=False)

    metadata = {
        "symbol": safe_value(args.symbol),
        "market": args.market,
        "timeframe": args.timeframe,
        "dataset_file": str(dataset_path),
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_causality_check": "PASS",
        "target_definition": "binary direction from fixed 1R net label; ambiguous rows excluded from training",
        "ambiguous_return_r": args.ambiguous_return_r,
        "source_rows": len(raw),
        "usable_rows": len(dataset),
        "train_rows_before_action_filter": len(split["train"]),
        "validation_train_actionable_rows": validation_train_rows,
        "validation_rows": len(split["validation"]),
        "final_train_rows_before_action_filter": len(split["train_validation"]),
        "final_train_actionable_rows": final_train_rows,
        "test_rows": len(split["test"]),
        "train_purged_rows": split["train_purged_rows"],
        "validation_purged_rows": split["validation_purged_rows"],
        "final_purged_rows": split["final_purged_rows"],
        "label_horizon_purge_minutes": split["horizon_minutes"],
        "validation_start_utc": timestamp_iso(split["validation_start_timestamp"]),
        "test_start_utc": timestamp_iso(split["test_start_timestamp"]),
        "threshold_selection_rule": "validation return minus 0.5 times absolute drawdown; minimum trade count required",
        "minimum_validation_trades": args.minimum_validation_trades,
        "threshold_candidates": threshold_results,
        "selected_probability_threshold": selected_threshold["threshold"],
        "selected_validation_result": selected_threshold,
        "test_directional_metrics": test_metrics,
        "prediction_file": str(output_prediction_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_warning": "This is an offline research candidate, not a live-trading approval.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Source rows: {len(raw):,}; usable rows: {len(dataset):,}")
    print(f"Causal feature verification: PASS ({len(feature_columns)} features)")
    print(
        "Partitions: "
        f"train={len(split['train']):,}, validation={len(split['validation']):,}, "
        f"test={len(split['test']):,}"
    )
    print(
        f"Selected threshold: {selected_threshold['threshold']:.2f}; "
        f"validation trades={selected_threshold['trades']}, "
        f"return={selected_threshold['total_return_pct']:.2f}%"
    )
    print(
        f"Test directional accuracy={test_metrics['accuracy'] * 100:.2f}%, "
        f"baseline={test_metrics['majority_baseline_accuracy'] * 100:.2f}%, "
        f"AUC={test_metrics['roc_auc']:.4f}"
    )
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Predictions: {output_prediction_path}")


if __name__ == "__main__":
    main()
