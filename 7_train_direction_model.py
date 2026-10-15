import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


ACTION_MAP = {0: "no_trade", 1: "long", 2: "short"}
FEATURE_SCHEMA_VERSION = "ohlcv_basic_v2_boundary4"
BASE_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "return_1m",
    "hour_utc",
    "minute_utc",
    "day_of_week_utc",
]
FUTURES_CONTEXT_FEATURE_COLUMNS = [
    "mark_price",
    "mark_basis_pct",
    "funding_rate",
    "funding_rate_age_minutes",
    "open_interest",
    "open_interest_value",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a first direction model from the ML dataset.")
    parser.add_argument("--dataset-folder", default="ml_dataset_v2")
    parser.add_argument("--model-folder", default="models_v2")
    parser.add_argument("--prediction-folder", default="predictions_v2")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="spot")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def market_root(folder, market, symbol, timeframe):
    root = Path(folder)
    if market != "spot":
        root = root / f"market={safe_value(market)}"
    return root / f"symbol={safe_value(symbol)}" / f"timeframe={safe_value(timeframe)}"


def dataset_path(dataset_folder, symbol, market, timeframe):
    market_suffix = "" if market == "spot" else f"_{safe_value(market).upper()}"
    path = market_root(dataset_folder, market, symbol, timeframe) / f"ml_dataset_{safe_value(symbol)}{market_suffix}_{safe_value(timeframe)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ML 데이터셋 파일이 없습니다: {path}")
    return path


def add_time_features(df):
    dt = pd.to_datetime(df["datetime_utc"], utc=True)
    df["hour_utc"] = dt.dt.hour
    df["minute_utc"] = dt.dt.minute
    df["day_of_week_utc"] = dt.dt.dayofweek
    return df


def load_dataset(args):
    path = dataset_path(args.dataset_folder, args.symbol, args.market, args.timeframe)
    df = pd.read_parquet(path)
    df = add_time_features(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [
        column
        for column in BASE_FEATURE_COLUMNS
        + [
            "direction_target",
            "max_holding_minutes",
            "label_horizon_complete",
            "entry_timestamp",
            "entry_price",
            "entry_rule",
            "feature_schema_version",
        ]
        if column not in df.columns
    ]
    if missing:
        raise RuntimeError(f"학습에 필요한 컬럼이 없습니다: {missing}")
    if not df["label_horizon_complete"].eq(True).all():
        raise RuntimeError("불완전한 미래 구간 라벨이 ML 데이터셋에 포함되어 있습니다.")
    if not df["entry_rule"].eq("next_bar_open").all():
        raise RuntimeError("지원하지 않는 진입 규칙입니다. next_bar_open 데이터셋을 사용하세요.")
    if set(df["feature_schema_version"].astype(str)) != {FEATURE_SCHEMA_VERSION}:
        raise RuntimeError("지원하지 않는 Feature Store 스키마 버전입니다.")

    feature_columns = list(BASE_FEATURE_COLUMNS)
    for column in FUTURES_CONTEXT_FEATURE_COLUMNS:
        if column in df.columns and df[column].notna().any():
            feature_columns.append(column)
    return df, path, feature_columns


def split_time_ordered(df, test_size):
    if not 0 < test_size < 1:
        raise ValueError("--test-size는 0과 1 사이여야 합니다.")

    split_index = int(len(df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("train/test split 결과가 비정상입니다.")

    initial_train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    test_start_timestamp = int(test["timestamp"].min())
    max_holding_minutes = int(df["max_holding_minutes"].max())
    train = initial_train[
        initial_train["timestamp"] + max_holding_minutes * 60_000 < test_start_timestamp
    ].copy()
    purged_rows = len(initial_train) - len(train)
    if train.empty:
        raise ValueError("Label-horizon purge removed every training row.")
    return train, test, purged_rows, max_holding_minutes, test_start_timestamp


def class_weights(y):
    counts = y.value_counts().to_dict()
    total = len(y)
    classes = len(counts)
    return y.map(lambda value: total / (classes * counts[value])).astype(float)


def train_model(train, feature_columns):
    observed_classes = sorted(train["direction_target"].astype(int).unique().tolist())
    if observed_classes != [0, 1, 2]:
        raise RuntimeError(f"Training data must contain classes 0, 1, and 2: {observed_classes}")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        max_bin=256,
        n_jobs=4,
        random_state=42,
    )

    x_train = train[feature_columns]
    y_train = train["direction_target"].astype(int)
    model.fit(x_train, y_train, sample_weight=class_weights(y_train))
    return model


def make_predictions(model, test, feature_columns):
    probabilities = model.predict_proba(test[feature_columns])
    predicted = probabilities.argmax(axis=1)

    predictions = test.copy()
    predictions["pred_direction_target"] = predicted
    predictions["pred_action"] = [ACTION_MAP[int(value)] for value in predicted]
    predictions["pred_confidence"] = probabilities.max(axis=1)
    predictions["prob_no_trade"] = probabilities[:, 0]
    predictions["prob_long"] = probabilities[:, 1]
    predictions["prob_short"] = probabilities[:, 2]
    return predictions


def save_outputs(args, model, metadata, predictions):
    model_dir = Path(args.model_folder)
    model_dir.mkdir(parents=True, exist_ok=True)
    market_suffix = "" if args.market == "spot" else f"_{safe_value(args.market).upper()}"
    model_path = model_dir / f"direction_model_{safe_value(args.symbol)}{market_suffix}_{safe_value(args.timeframe)}.json"
    metadata_path = model_dir / f"direction_model_{safe_value(args.symbol)}{market_suffix}_{safe_value(args.timeframe)}_metadata.json"
    model.save_model(model_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_dir = market_root(args.prediction_folder, args.market, args.symbol, args.timeframe)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"direction_predictions_{safe_value(args.symbol)}{market_suffix}_{safe_value(args.timeframe)}.parquet"
    predictions.to_parquet(prediction_path, index=False)

    return model_path, metadata_path, prediction_path


def main():
    args = parse_args()
    dataset, dataset_file, feature_columns = load_dataset(args)
    train, test, purged_rows, purge_minutes, test_start_timestamp = split_time_ordered(
        dataset,
        args.test_size,
    )

    print(f"ML 데이터셋 로드: {dataset_file}")
    print(f"전체 rows={len(dataset):,}, train={len(train):,}, test={len(test):,}")
    print(f"테스트 경계 라벨 누수 방지 제거: {purged_rows:,} rows ({purge_minutes}분)")

    model = train_model(train, feature_columns)
    predictions = make_predictions(model, test, feature_columns)

    y_true = predictions["direction_target"].astype(int)
    y_pred = predictions["pred_direction_target"].astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    majority_baseline_accuracy = float(y_true.value_counts(normalize=True).max())
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=["no_trade", "long", "short"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    metadata = {
        "symbol": safe_value(args.symbol),
        "market": args.market,
        "timeframe": args.timeframe,
        "dataset_file": str(dataset_file),
        "feature_columns": feature_columns,
        "target_column": "direction_target",
        "entry_rule": "next_bar_open",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_map": ACTION_MAP,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_size": args.test_size,
        "purged_train_rows": purged_rows,
        "label_horizon_purge_minutes": purge_minutes,
        "test_start_timestamp": test_start_timestamp,
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "accuracy_lift_over_majority": accuracy - majority_baseline_accuracy,
        "classification_report": report,
        "confusion_matrix_labels": ["no_trade", "long", "short"],
        "confusion_matrix": matrix,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Do not use future label columns as features. This model uses only current price/volume/time features.",
    }

    model_path, metadata_path, prediction_path = save_outputs(args, model, metadata, predictions)

    print(f"모델 저장: {model_path}")
    print(f"메타데이터 저장: {metadata_path}")
    print(f"예측 결과 저장: {prediction_path}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Majority baseline: {majority_baseline_accuracy * 100:.2f}%")
    print("Confusion matrix labels: no_trade, long, short")
    print(matrix)


if __name__ == "__main__":
    main()
