"""Legacy binary-direction smoke model, retained only for compatibility tests."""

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Run the legacy single-market smoke model.")
    parser.add_argument("--feature-folder", default="feature_store_legacy")
    parser.add_argument("--model-folder", default="models_legacy")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="spot")
    parser.add_argument("--timeframe", default="1m")
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def load_feature_store(feature_store):
    feature_paths = sorted(Path(feature_store).glob("**/*.parquet"))
    if not feature_paths:
        raise RuntimeError(f"Feature Store에 학습할 parquet 파일이 없습니다: {feature_store}")

    frames = []
    for path in feature_paths:
        frames.append(pd.read_parquet(path))

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError("Feature Store에 학습할 데이터가 없습니다.")
    return df


def main():
    args = parse_args()
    print("주의: 3_ml_training.py는 연결 확인용 레거시 모델입니다. 전략 평가에는 4~8번 파이프라인을 사용합니다.")
    print("Feature Store에서 가공 데이터를 로드합니다.")
    df = load_feature_store(args.feature_folder)

    if "market" not in df.columns:
        df["market"] = "spot"
    if "symbol" not in df.columns or "timeframe" not in df.columns:
        raise RuntimeError("Feature Store에 symbol과 timeframe 컬럼이 필요합니다.")
    df = df[
        (df["symbol"].astype(str).str.replace("/", "", regex=False) == safe_value(args.symbol))
        & (df["market"].astype(str) == args.market)
        & (df["timeframe"].astype(str) == args.timeframe)
    ].copy()
    if df.empty:
        raise RuntimeError(
            f"선택한 데이터가 없습니다: symbol={args.symbol}, market={args.market}, timeframe={args.timeframe}"
        )
    dedupe_columns = [column for column in ["symbol", "market", "timeframe", "timestamp"] if column in df.columns]
    if dedupe_columns:
        before = len(df)
        df = df.drop_duplicates(subset=dedupe_columns, keep="last")
        removed = before - len(df)
        if removed:
            print(f"중복 Feature row 제거: {removed:,} rows")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["next_close"] = df["close"].shift(-1)
    df = df.dropna(subset=["next_close"]).copy()
    df["target"] = (df["next_close"] > df["close"]).astype(int)

    feature_columns = ["close", "ma_5", "return_1m"]
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"학습에 필요한 피처가 없습니다: {missing}")

    x = df[feature_columns]
    y = df["target"]

    if len(df) < 20:
        raise RuntimeError(f"학습 데이터가 너무 적습니다: {len(df)} rows")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    print("AI 모델 학습 시작")
    model = XGBClassifier(eval_metric="logloss")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    os.makedirs(args.model_folder, exist_ok=True)
    market_suffix = "" if args.market == "spot" else f"_{args.market.upper()}"
    model_path = os.path.join(
        args.model_folder,
        f"trading_ai_model_{safe_value(args.symbol)}{market_suffix}_{safe_value(args.timeframe)}.json",
    )
    model.save_model(model_path)
    print(f"모델 학습 완료. 예측 정확도: {accuracy * 100:.2f}%")
    print(f"모델 저장 완료: {model_path}")


if __name__ == "__main__":
    main()
