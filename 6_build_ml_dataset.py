import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RETURN_PATTERN = re.compile(r"^(long|short)_tp_(.+)r_return_r_net$")
FEATURE_SCHEMA_VERSION = "ohlcv_basic_v2_boundary4"
FUTURES_CONTEXT_SCHEMA_VERSION = "usdm_context_v2_complete_mark"
FEATURE_READ_COLUMNS = [
    "timestamp",
    "datetime_utc",
    "symbol",
    "market",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "return_1m",
    "feature_schema_version",
]
CONTEXT_READ_COLUMNS = [
    "timestamp",
    "symbol",
    "market",
    "timeframe",
    "mark_price",
    "funding_rate",
    "funding_rate_timestamp",
    "open_interest",
    "open_interest_value",
    "open_interest_timestamp",
    "context_schema_version",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Join Feature Store and Label Store into an ML dataset.")
    parser.add_argument("--feature-folder", default="feature_store_v2")
    parser.add_argument("--label-folder", default="label_store_v2")
    parser.add_argument("--context-folder", default="futures_context_store_v2")
    parser.add_argument("--dataset-folder", default="ml_dataset_v2")
    parser.add_argument("--label-file")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="spot")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--min-expected-r", type=float, default=0.0, help="Best candidate must exceed this R to trade.")
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def market_root(folder, market, symbol, timeframe):
    root = Path(folder)
    if market != "spot":
        root = root / f"market={safe_value(market)}"
    return root / f"symbol={safe_value(symbol)}" / f"timeframe={safe_value(timeframe)}"


def load_features(feature_folder, symbol, market, timeframe):
    root = market_root(feature_folder, market, symbol, timeframe)
    if root.exists():
        paths = sorted(root.glob("**/*.parquet"))
    else:
        paths = sorted(Path(feature_folder).glob("**/*.parquet"))

    if not paths:
        raise FileNotFoundError(f"Feature parquet 파일이 없습니다: {root}")

    df = pd.concat(
        [pd.read_parquet(path, columns=FEATURE_READ_COLUMNS) for path in paths],
        ignore_index=True,
    )
    if "market" not in df.columns:
        df["market"] = "spot"
    df = df[df["symbol"].astype(str).str.replace("/", "", regex=False) == safe_value(symbol)]
    df = df[df["market"].astype(str) == market]
    df = df[df["timeframe"].astype(str) == timeframe]
    df = df.drop_duplicates(subset=["symbol", "market", "timeframe", "timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"Feature 데이터가 비었습니다: symbol={symbol}, timeframe={timeframe}")
    if "feature_schema_version" not in df.columns:
        raise RuntimeError("구형 Feature Store입니다. feature_store_v2를 사용하세요.")
    if set(df["feature_schema_version"].astype(str)) != {FEATURE_SCHEMA_VERSION}:
        raise RuntimeError("지원하지 않는 Feature Store 스키마 버전입니다.")
    for column in ["symbol", "market", "timeframe", "feature_schema_version"]:
        df[column] = df[column].astype("category")
    return df


def find_label_file(label_folder, label_file, symbol, market, timeframe):
    if label_file:
        path = Path(label_file)
        if not path.exists():
            raise FileNotFoundError(f"라벨 파일이 없습니다: {path}")
        return path

    root = market_root(label_folder, market, symbol, timeframe)
    paths = sorted(root.glob("**/labels_*.parquet"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"라벨 파일이 없습니다: {root}")
    return paths[0]


def load_labels(label_folder, label_file, symbol, market, timeframe):
    path = find_label_file(label_folder, label_file, symbol, market, timeframe)
    available_columns = pq.read_schema(path).names
    label_columns = [
        "timestamp",
        "symbol",
        "market",
        "timeframe",
        "stop_distance_pct",
        "max_holding_minutes",
        "round_trip_cost_bps",
        "entry_timestamp",
        "entry_price",
        "entry_rule",
        "label_horizon_complete",
        "entry_datetime_utc",
    ]
    label_columns.extend(
        column
        for column in available_columns
        if column.startswith(("long_tp_", "short_tp_"))
    )
    df = pd.read_parquet(path, columns=label_columns)
    if "market" not in df.columns:
        df["market"] = "spot"
    df = df[df["symbol"].astype(str).str.replace("/", "", regex=False) == safe_value(symbol)]
    df = df[df["market"].astype(str) == market]
    df = df[df["timeframe"].astype(str) == timeframe]
    df = df.drop_duplicates(subset=["symbol", "market", "timeframe", "timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"Label 데이터가 비었습니다: symbol={symbol}, timeframe={timeframe}")
    for column in ["symbol", "market", "timeframe", "entry_rule"]:
        df[column] = df[column].astype("category")
    for column in df.columns:
        if column.endswith("_outcome"):
            df[column] = df[column].astype("category")
    return df, path


def load_futures_context(context_folder, symbol, market, timeframe):
    if market != "usdm":
        return None

    root = market_root(context_folder, market, symbol, timeframe)
    paths = sorted(root.glob("context_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            "USDT-M ML 데이터셋에는 선물 컨텍스트가 필요합니다. "
            f"먼저 9_futures_context_collector.py를 실행하세요: {root}"
        )

    context_frames = []
    numeric_context_columns = [
        "mark_price",
        "funding_rate",
        "funding_rate_timestamp",
        "open_interest",
        "open_interest_value",
        "open_interest_timestamp",
    ]
    for path in paths:
        frame = pd.read_parquet(path, columns=CONTEXT_READ_COLUMNS)
        for column in numeric_context_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        context_frames.append(frame)
    context = pd.concat(context_frames, ignore_index=True)
    context = context[context["symbol"].astype(str).str.replace("/", "", regex=False) == safe_value(symbol)]
    context = context[context["market"].astype(str) == market]
    context = context[context["timeframe"].astype(str) == timeframe]
    context = context.drop_duplicates(subset=["timestamp", "symbol", "market", "timeframe"], keep="last")
    context = context.sort_values("timestamp").reset_index(drop=True)
    if context.empty:
        raise RuntimeError(f"선물 컨텍스트 데이터가 비었습니다: {root}")
    if "context_schema_version" not in context.columns:
        raise RuntimeError("구형 선물 컨텍스트입니다. futures_context_store_v2를 사용하세요.")
    if set(context["context_schema_version"].astype(str)) != {FUTURES_CONTEXT_SCHEMA_VERSION}:
        raise RuntimeError("지원하지 않는 선물 컨텍스트 스키마 버전입니다.")
    for column in ["symbol", "market", "timeframe", "context_schema_version"]:
        context[column] = context[column].astype("category")
    return context


def return_cases(columns):
    cases = []
    for column in columns:
        match = RETURN_PATTERN.match(column)
        if match:
            side, r_key = match.groups()
            cases.append((column, side, r_key))
    return sorted(cases)


def display_r(r_key):
    return float(r_key.replace("_", "."))


def add_targets(dataset, min_expected_r):
    cases = return_cases(dataset.columns)
    if not cases:
        raise RuntimeError("라벨 return_r_net 컬럼을 찾지 못했습니다.")

    return_columns = [case[0] for case in cases]
    return_values = dataset[return_columns].to_numpy(dtype=float)
    best_indexes = np.argmax(return_values, axis=1)
    best_returns = return_values[np.arange(len(dataset)), best_indexes]
    best_columns = np.asarray(return_columns, dtype=object)[best_indexes]

    side_by_column = {column: side for column, side, _ in cases}
    take_profit_by_column = {
        column: display_r(r_key) for column, _, r_key in cases
    }
    trade_mask = best_returns > min_expected_r
    best_actions = np.asarray(
        [side_by_column[column] for column in best_columns], dtype=object
    )
    best_take_profit_values = np.asarray(
        [take_profit_by_column[column] for column in best_columns], dtype=float
    )

    dataset["best_case"] = np.where(
        trade_mask,
        np.char.replace(best_columns.astype(str), "_return_r_net", ""),
        "no_trade",
    )
    dataset["best_action"] = np.where(trade_mask, best_actions, "no_trade")
    dataset["best_take_profit_r"] = np.where(
        trade_mask, best_take_profit_values, 0.0
    )
    dataset["best_return_r_net"] = best_returns
    dataset["trade_target"] = (dataset["best_action"] != "no_trade").astype(int)
    dataset["direction_target"] = dataset["best_action"].map({"no_trade": 0, "long": 1, "short": 2}).astype(int)

    for action in ["long", "short"]:
        action_columns = [column for column, side, _ in cases if side == action]
        dataset[f"{action}_best_return_r_net"] = dataset[action_columns].max(axis=1)

    dataset["long_trade_target"] = (dataset["long_best_return_r_net"] > min_expected_r).astype(int)
    dataset["short_trade_target"] = (dataset["short_best_return_r_net"] > min_expected_r).astype(int)
    return dataset


def add_futures_context(features, context):
    if context is None:
        return features

    join_keys = ["timestamp"]
    context_columns = [
        "mark_price",
        "funding_rate",
        "funding_rate_timestamp",
        "open_interest",
        "open_interest_value",
        "open_interest_timestamp",
    ]
    available_columns = [column for column in context_columns if column in context.columns]
    context_for_join = context[join_keys + available_columns]
    enriched = features.merge(context_for_join, on=join_keys, how="left", validate="one_to_one")
    if enriched["mark_price"].isna().any():
        missing = int(enriched["mark_price"].isna().sum())
        raise RuntimeError(f"선물 mark price 결합 실패: {missing:,} rows")

    enriched["mark_basis_pct"] = (enriched["mark_price"] - enriched["close"]) / enriched["close"]
    enriched["funding_rate_age_minutes"] = (enriched["timestamp"] - enriched["funding_rate_timestamp"]) / 60_000
    return enriched


def build_dataset(features, labels, context, min_expected_r):
    required_label_metadata = {
        "label_horizon_complete",
        "entry_timestamp",
        "entry_datetime_utc",
        "entry_price",
        "entry_rule",
    }
    missing_label_metadata = sorted(required_label_metadata - set(labels.columns))
    if missing_label_metadata:
        raise RuntimeError(
            "구형 라벨이거나 실행 시점 정보가 부족합니다. "
            f"누락 컬럼={missing_label_metadata}. 4_triple_barrier_labeler.py로 다시 생성하세요."
        )
    labels = labels[labels["label_horizon_complete"].eq(True)].copy()
    if not labels["entry_rule"].eq("next_bar_open").all():
        raise RuntimeError("지원하지 않는 라벨 진입 규칙입니다. next_bar_open 라벨을 사용하세요.")
    features = add_futures_context(features, context)
    label_drop_columns = ["datetime_utc", "close", "symbol", "market", "timeframe"]
    labels_for_join = labels.drop(
        columns=[column for column in label_drop_columns if column in labels.columns]
    )
    dataset = features.merge(
        labels_for_join,
        on=["timestamp"],
        how="inner",
        validate="one_to_one",
    )

    if dataset.empty:
        raise RuntimeError("Feature와 Label 조인 결과가 비었습니다.")

    dataset = add_targets(dataset, min_expected_r)
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    return dataset


def write_dataset(dataset, dataset_folder, symbol, market, timeframe, label_path, min_expected_r):
    output_root = Path(dataset_folder)
    if market != "spot":
        output_root = output_root / f"market={safe_value(market)}"
    output_dir = output_root / f"symbol={safe_value(symbol)}" / f"timeframe={safe_value(timeframe)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    market_suffix = "" if market == "spot" else f"_{safe_value(market).upper()}"
    output_path = output_dir / f"ml_dataset_{safe_value(symbol)}{market_suffix}_{safe_value(timeframe)}.parquet"
    dataset.to_parquet(output_path, index=False)

    stored_rows = pq.ParquetFile(output_path).metadata.num_rows
    if stored_rows != len(dataset):
        raise RuntimeError(
            f"ML 데이터셋 저장 검증 실패: expected={len(dataset)}, actual={stored_rows}"
        )

    marker_path = output_dir / "_ML_DATASET_SUCCESS.json"
    marker = {
        "symbol": safe_value(symbol),
        "market": market,
        "timeframe": timeframe,
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "label_file": str(label_path),
        "dataset_file": str(output_path),
        "min_expected_r": min_expected_r,
        "entry_rule": "next_bar_open",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_columns": [
            "best_case",
            "best_action",
            "best_take_profit_r",
            "best_return_r_net",
            "trade_target",
            "direction_target",
            "long_trade_target",
            "short_trade_target",
        ],
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, marker_path


def print_summary(dataset):
    print(f"ML dataset rows: {len(dataset):,}")
    print(f"ML dataset columns: {len(dataset.columns):,}")
    print("best_action 분포:")
    for action, count in dataset["best_action"].value_counts().items():
        print(f"  - {action}: {count:,}")
    print("best_take_profit_r 분포:")
    for value, count in dataset["best_take_profit_r"].value_counts().sort_index().items():
        print(f"  - {value}: {count:,}")
    print("평균 best_return_r_net:", f"{dataset['best_return_r_net'].mean():.4f}")


def main():
    args = parse_args()

    print(f"Feature 로드: {args.market} {args.symbol} {args.timeframe}")
    features = load_features(args.feature_folder, args.symbol, args.market, args.timeframe)
    print(f"Feature rows: {len(features):,}")

    print("Label 로드")
    labels, label_path = load_labels(args.label_folder, args.label_file, args.symbol, args.market, args.timeframe)
    print(f"Label rows: {len(labels):,}")
    print(f"Label file: {label_path}")

    context = load_futures_context(args.context_folder, args.symbol, args.market, args.timeframe)
    if context is not None:
        print(f"선물 컨텍스트 rows: {len(context):,}")

    dataset = build_dataset(features, labels, context, args.min_expected_r)
    output_path, marker_path = write_dataset(
        dataset,
        args.dataset_folder,
        args.symbol,
        args.market,
        args.timeframe,
        label_path,
        args.min_expected_r,
    )

    print_summary(dataset)
    print(f"ML 데이터셋 저장 완료: {output_path}")
    print(f"성공 마커: {marker_path}")


if __name__ == "__main__":
    main()
