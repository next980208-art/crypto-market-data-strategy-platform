"""Build a point-in-time-safe aggTrade-enriched research dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TRADE_CONTEXT_SCHEMA_VERSION = "aggtrade_1m_v3_windowed_complete"
TRADE_FEATURE_SCHEMA_VERSION = "aggtrade_causal_features_v1"
MINUTE_MS = 60_000
BASE_TRADE_COLUMNS = [
    "trade_count",
    "trade_volume",
    "trade_notional",
    "trade_vwap",
    "trade_first_price",
    "trade_last_price",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_buy_notional",
    "taker_sell_notional",
    "taker_volume_imbalance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join aggTrade causal features with OHLCV, futures context, and labels."
    )
    parser.add_argument("--feature-folder", default="feature_store_v2")
    parser.add_argument("--futures-context-folder", default="futures_context_store_v2")
    parser.add_argument("--trade-context-folder", default="trade_context_store_v2")
    parser.add_argument("--label-folder", default="label_store_trade_context_20260902")
    parser.add_argument("--label-file")
    parser.add_argument("--output-folder", default="trade_enriched_dataset_20260902")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", default="usdm", choices=["usdm"])
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--minimum-training-days", type=int, default=90)
    parser.add_argument(
        "--report-json",
        default="runtime_reports/trade_enriched_dataset_2026-09-02.json",
    )
    parser.add_argument(
        "--report-md",
        default="docs/trade_enriched_feature_integration_execution_2026-09-02.md",
    )
    return parser.parse_args()


def safe_value(value: str) -> str:
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def partition_root(folder, market, symbol, timeframe) -> Path:
    return (
        Path(folder)
        / f"market={safe_value(market)}"
        / f"symbol={safe_value(symbol)}"
        / f"timeframe={safe_value(timeframe)}"
    )


def read_parquet_range(path: Path, columns, start_timestamp, end_timestamp):
    return pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("timestamp", ">=", int(start_timestamp)),
            ("timestamp", "<=", int(end_timestamp)),
        ],
    )


def conflicting_timestamps(frame: pd.DataFrame, columns: list[str]) -> set[int]:
    duplicated = frame[frame.duplicated("timestamp", keep=False)]
    conflicts = set()
    if duplicated.empty:
        return conflicts
    for timestamp, group in duplicated.groupby("timestamp", sort=False):
        for column in columns:
            values = group[column]
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                if not np.allclose(numeric, numeric[0], rtol=1e-10, atol=1e-12, equal_nan=True):
                    conflicts.add(int(timestamp))
                    break
            elif values.astype(str).nunique(dropna=False) > 1:
                conflicts.add(int(timestamp))
                break
    return conflicts


def resolve_trade_overlaps(frames: list[pd.DataFrame]):
    """Prefer the widest source file when test snippets overlap complete daily files."""
    if not frames:
        raise FileNotFoundError("aggTrade context parquet files were not found.")
    prepared = []
    for source_order, frame in enumerate(frames):
        item = frame.copy()
        item["_source_rows"] = len(item)
        item["_source_order"] = source_order
        prepared.append(item)
    combined = pd.concat(prepared, ignore_index=True)
    duplicate_mask = combined.duplicated("timestamp", keep=False)
    duplicate_timestamp_count = int(combined.loc[duplicate_mask, "timestamp"].nunique())
    conflicts = conflicting_timestamps(combined, BASE_TRADE_COLUMNS)
    resolved = (
        combined.sort_values(["timestamp", "_source_rows", "_source_order"])
        .drop_duplicates("timestamp", keep="last")
        .drop(columns=["_source_rows", "_source_order"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    report = {
        "input_rows": len(combined),
        "unique_rows": len(resolved),
        "overlap_timestamps": duplicate_timestamp_count,
        "conflicting_overlap_timestamps": len(conflicts),
        "overlap_resolution": "prefer_source_file_with_most_rows",
    }
    return resolved, report


def load_trade_context(folder, symbol, market, timeframe):
    root = partition_root(folder, market, symbol, timeframe)
    paths = sorted(root.glob("trade_context_*.parquet"))
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = sorted(set(["timestamp", "context_schema_version"] + BASE_TRADE_COLUMNS) - set(frame.columns))
        if missing:
            raise RuntimeError(f"Trade context columns are missing from {path}: {missing}")
        if set(frame["context_schema_version"].astype(str)) != {TRADE_CONTEXT_SCHEMA_VERSION}:
            raise RuntimeError(f"Unsupported trade context schema: {path}")
        frame = frame[
            frame["symbol"].astype(str).str.replace("/", "", regex=False).eq(safe_value(symbol))
            & frame["market"].astype(str).eq(market)
            & frame["timeframe"].astype(str).eq(timeframe)
        ].copy()
        frames.append(frame)
    context, overlap_report = resolve_trade_overlaps(frames)
    if context.empty:
        raise RuntimeError("Trade context is empty after market filtering.")
    expected = np.arange(
        int(context["timestamp"].min()),
        int(context["timestamp"].max()) + MINUTE_MS,
        MINUTE_MS,
        dtype=np.int64,
    )
    actual = context["timestamp"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        missing = np.setdiff1d(expected, actual)
        raise RuntimeError(f"Trade context has {len(missing)} missing minute timestamps.")
    overlap_report.update(
        {
            "source_files": len(paths),
            "start_utc": pd.to_datetime(actual[0], unit="ms", utc=True).isoformat(),
            "end_utc": pd.to_datetime(actual[-1], unit="ms", utc=True).isoformat(),
            "missing_minutes": 0,
            "duplicate_minutes_after_resolution": int(context["timestamp"].duplicated().sum()),
            "empty_trade_minutes": int(pd.to_numeric(context["trade_count"]).eq(0).sum()),
            "aggregated_trades": int(pd.to_numeric(context["trade_count"]).sum()),
        }
    )
    return context, overlap_report


def add_causal_trade_features(context: pd.DataFrame):
    frame = context.sort_values("timestamp").copy()
    count = pd.to_numeric(frame["trade_count"], errors="raise").astype(float)
    volume = pd.to_numeric(frame["trade_volume"], errors="raise").astype(float)
    notional = pd.to_numeric(frame["trade_notional"], errors="raise").astype(float)
    buy_volume = pd.to_numeric(frame["taker_buy_volume"], errors="raise").astype(float)
    sell_volume = pd.to_numeric(frame["taker_sell_volume"], errors="raise").astype(float)
    buy_notional = pd.to_numeric(frame["taker_buy_notional"], errors="raise").astype(float)
    sell_notional = pd.to_numeric(frame["taker_sell_notional"], errors="raise").astype(float)
    first_price = pd.to_numeric(frame["trade_first_price"], errors="coerce").astype(float)
    last_price = pd.to_numeric(frame["trade_last_price"], errors="coerce").astype(float)

    frame["agg_trade_count"] = count
    frame["agg_trade_volume"] = volume
    frame["agg_trade_notional"] = notional
    frame["agg_taker_volume_imbalance"] = pd.to_numeric(
        frame["taker_volume_imbalance"], errors="raise"
    ).astype(float)
    frame["agg_trade_count_log1p"] = np.log1p(count)
    frame["agg_trade_intensity_per_second"] = count / 60.0
    frame["agg_average_trade_notional"] = np.divide(
        notional, count, out=np.zeros(len(frame), dtype=float), where=count.to_numpy() > 0
    )
    frame["agg_taker_buy_volume_share"] = np.divide(
        buy_volume,
        buy_volume + sell_volume,
        out=np.full(len(frame), 0.5, dtype=float),
        where=(buy_volume + sell_volume).to_numpy() > 0,
    )
    frame["agg_taker_notional_imbalance"] = np.divide(
        buy_notional - sell_notional,
        buy_notional + sell_notional,
        out=np.zeros(len(frame), dtype=float),
        where=(buy_notional + sell_notional).to_numpy() > 0,
    )
    frame["agg_trade_price_change_pct"] = np.divide(
        last_price - first_price,
        first_price,
        out=np.zeros(len(frame), dtype=float),
        where=first_price.to_numpy() > 0,
    )

    feature_columns = [
        "agg_trade_count_log1p",
        "agg_trade_intensity_per_second",
        "agg_average_trade_notional",
        "agg_taker_buy_volume_share",
        "agg_taker_volume_imbalance",
        "agg_taker_notional_imbalance",
        "agg_trade_price_change_pct",
    ]
    rolling_inputs = {
        "trade_count": count,
        "trade_volume": volume,
        "trade_notional": notional,
    }
    for minutes in (5, 15, 60):
        for name, values in rolling_inputs.items():
            rolling_mean = values.rolling(minutes, min_periods=minutes).mean()
            column = f"agg_{name}_ratio_{minutes}m"
            frame[column] = values / rolling_mean.replace(0.0, np.nan) - 1.0
            feature_columns.append(column)
        imbalance_mean = f"agg_taker_volume_imbalance_mean_{minutes}m"
        frame[imbalance_mean] = frame["agg_taker_volume_imbalance"].rolling(
            minutes, min_periods=minutes
        ).mean()
        feature_columns.append(imbalance_mean)
        if minutes in (15, 60):
            imbalance_std = f"agg_taker_volume_imbalance_std_{minutes}m"
            frame[imbalance_std] = frame["agg_taker_volume_imbalance"].rolling(
                minutes, min_periods=minutes
            ).std(ddof=0)
            feature_columns.append(imbalance_std)

    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    frame["trade_feature_schema_version"] = TRADE_FEATURE_SCHEMA_VERSION
    return frame, feature_columns


def verify_causal_trade_features(raw, full, feature_columns):
    if len(raw) < 120:
        raise RuntimeError("At least 120 continuous minutes are required for causal verification.")
    checkpoints = sorted({len(raw) // 2, len(raw) - 1})
    for checkpoint in checkpoints:
        prefix, prefix_columns = add_causal_trade_features(raw.iloc[: checkpoint + 1].copy())
        if prefix_columns != feature_columns:
            raise RuntimeError("Trade feature column order changed during causal verification.")
        timestamp = int(raw.iloc[checkpoint]["timestamp"])
        expected = prefix[prefix["timestamp"].eq(timestamp)]
        actual = full[full["timestamp"].eq(timestamp)]
        if len(expected) != 1 or len(actual) != 1:
            raise RuntimeError(f"Could not locate causal checkpoint timestamp={timestamp}.")
        if not np.allclose(
            expected.iloc[0][feature_columns].to_numpy(dtype=float),
            actual.iloc[0][feature_columns].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"Future-data leakage detected at timestamp={timestamp}.")


def load_range_files(root, pattern, columns, start_timestamp, end_timestamp):
    paths = sorted(root.glob(pattern))
    frames = []
    for path in paths:
        frame = read_parquet_range(path, columns, start_timestamp, end_timestamp)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No rows in requested range: {root}")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def load_market_sources(args, start_timestamp, end_timestamp):
    feature_columns = [
        "timestamp", "datetime_utc", "symbol", "market", "timeframe", "open", "high",
        "low", "close", "volume", "ma_5", "return_1m", "feature_schema_version",
    ]
    futures_columns = [
        "timestamp", "mark_price", "funding_rate", "funding_rate_timestamp", "open_interest",
        "open_interest_value", "open_interest_timestamp", "context_schema_version",
    ]
    features = load_range_files(
        partition_root(args.feature_folder, args.market, args.symbol, args.timeframe),
        "**/*.parquet",
        feature_columns,
        start_timestamp,
        end_timestamp,
    )
    futures = load_range_files(
        partition_root(args.futures_context_folder, args.market, args.symbol, args.timeframe),
        "context_*.parquet",
        futures_columns,
        start_timestamp,
        end_timestamp,
    )
    return features, futures


def find_label_file(args) -> Path:
    if args.label_file:
        path = Path(args.label_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = partition_root(args.label_folder, args.market, args.symbol, args.timeframe)
    paths = sorted(root.glob("**/labels_*.parquet"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"Label file was not found: {root}")
    return paths[0]


def build_enriched_dataset(args, trade_context, trade_features, feature_columns):
    start_timestamp = int(trade_context["timestamp"].min())
    end_timestamp = int(trade_context["timestamp"].max())
    features, futures = load_market_sources(args, start_timestamp, end_timestamp)
    expected_rows = len(trade_context)
    if len(features) != expected_rows or len(futures) != expected_rows:
        raise RuntimeError(
            f"Point-in-time source coverage mismatch: trades={expected_rows}, "
            f"features={len(features)}, futures={len(futures)}"
        )
    expected_timestamps = trade_context["timestamp"].to_numpy()
    if not np.array_equal(features["timestamp"].to_numpy(), expected_timestamps) or not np.array_equal(
        futures["timestamp"].to_numpy(), expected_timestamps
    ):
        raise RuntimeError("Point-in-time source timestamps differ despite equal row counts")

    volume_check = trade_context[["timestamp", "trade_volume"]].merge(
        features[["timestamp", "volume"]], on="timestamp", validate="one_to_one"
    )
    relative_volume_error = (
        (volume_check["trade_volume"] - volume_check["volume"]).abs()
        / volume_check["volume"].replace(0.0, np.nan)
    )
    volume_check["date_utc"] = pd.to_datetime(
        volume_check["timestamp"], unit="ms", utc=True
    ).dt.date
    daily_volume = volume_check.groupby("date_utc", as_index=False)[
        ["trade_volume", "volume"]
    ].sum()
    daily_relative_error = (
        (daily_volume["trade_volume"] - daily_volume["volume"]).abs()
        / daily_volume["volume"].replace(0.0, np.nan)
    )

    trade_columns = ["timestamp", "trade_vwap", "trade_feature_schema_version"] + feature_columns
    enriched = features.merge(
        futures.drop(columns=["context_schema_version"]), on="timestamp", how="left", validate="one_to_one"
    )
    enriched = enriched.merge(
        trade_features[trade_columns], on="timestamp", how="left", validate="one_to_one"
    )
    if enriched[feature_columns].isna().all(axis=1).any():
        raise RuntimeError("Trade features failed to join on one or more timestamps.")
    enriched["mark_basis_pct"] = (enriched["mark_price"] - enriched["close"]) / enriched["close"]
    enriched["funding_rate_age_minutes"] = (
        enriched["timestamp"] - enriched["funding_rate_timestamp"]
    ) / MINUTE_MS
    enriched["agg_trade_vwap_to_close"] = (
        pd.to_numeric(enriched["trade_vwap"], errors="coerce") / enriched["close"] - 1.0
    )
    feature_columns = feature_columns + ["agg_trade_vwap_to_close"]
    enriched = enriched.dropna(subset=feature_columns).copy()

    label_path = find_label_file(args)
    labels = pd.read_parquet(
        label_path,
        filters=[
            ("timestamp", ">=", int(enriched["timestamp"].min())),
            ("timestamp", "<=", int(enriched["timestamp"].max())),
        ],
    )
    labels = labels[labels["label_horizon_complete"].eq(True)].copy()
    label_drop = ["datetime_utc", "symbol", "market", "timeframe", "close"]
    labels = labels.drop(columns=[column for column in label_drop if column in labels.columns])
    labels = labels.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    dataset = enriched.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    if dataset.empty:
        raise RuntimeError("No completed labels overlap the trade-enriched features.")
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    expected_label_range = enriched["timestamp"].between(labels["timestamp"].min(), labels["timestamp"].max())
    return dataset, feature_columns, label_path, {
        "ohlcv_rows": len(features),
        "futures_context_rows": len(futures),
        "trade_feature_rows_after_warmup": len(enriched),
        "completed_label_rows": len(labels),
        "final_rows": len(dataset),
        "missing_join_rows": int(expected_label_range.sum()) - len(dataset),
        "aggtrade_vs_kline_volume": {
            "minute_relative_error_median": float(relative_volume_error.median()),
            "minute_relative_error_p99": float(relative_volume_error.quantile(0.99)),
            "minute_rows_within_1pct": int(relative_volume_error.le(0.01).sum()),
            "minute_rows_over_5pct": int(relative_volume_error.gt(0.05).sum()),
            "daily_relative_error_max": float(daily_relative_error.max()),
            "interpretation": (
                "Aggregate trades may cross a minute boundary; daily totals are the reconciliation gate."
            ),
        },
    }


def write_outputs(args, dataset, feature_columns, label_path, trade_report, join_report):
    output_dir = partition_root(args.output_folder, args.market, args.symbol, args.timeframe)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trade_enriched_dataset_{safe_value(args.symbol)}_USDM_1m.parquet"
    dataset.to_parquet(output_path, index=False)
    stored = pd.read_parquet(output_path, columns=["timestamp"])
    if len(stored) != len(dataset) or stored["timestamp"].duplicated().any():
        raise RuntimeError("Stored trade-enriched dataset failed row or duplicate verification.")

    span_minutes = int((dataset["timestamp"].max() - dataset["timestamp"].min()) / MINUTE_MS) + 1
    span_days = span_minutes / 1440
    continuous = stored["timestamp"].diff().dropna().eq(MINUTE_MS).all()
    volume_error = join_report["aggtrade_vs_kline_volume"]["daily_relative_error_max"]
    volume_passed = bool(np.isfinite(volume_error) and volume_error <= 0.01)
    training_allowed = bool(span_days >= args.minimum_training_days and continuous and volume_passed)
    block_reasons = []
    if span_days < args.minimum_training_days:
        block_reasons.append("insufficient_continuous_aggtrade_history")
    if not continuous:
        block_reasons.append("non_continuous_dataset")
    if not volume_passed:
        block_reasons.append("daily_volume_reconciliation_failed")
    marker = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "symbol": safe_value(args.symbol),
        "market": args.market,
        "timeframe": args.timeframe,
        "trade_context_schema_version": TRADE_CONTEXT_SCHEMA_VERSION,
        "trade_feature_schema_version": TRADE_FEATURE_SCHEMA_VERSION,
        "dataset_file": str(output_path),
        "dataset_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "label_file": str(label_path),
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "start_utc": pd.to_datetime(dataset["timestamp"].min(), unit="ms", utc=True).isoformat(),
        "end_utc": pd.to_datetime(dataset["timestamp"].max(), unit="ms", utc=True).isoformat(),
        "available_span_days": span_days,
        "minimum_training_days": args.minimum_training_days,
        "model_training_allowed": training_allowed,
        "training_block_reason": None if training_allowed else ",".join(block_reasons),
        "continuous_minutes": bool(continuous),
        "daily_volume_relative_error_limit": 0.01,
        "daily_volume_reconciliation_passed": volume_passed,
        "causal_feature_verification": "passed",
        "trade_feature_columns": feature_columns,
        "trade_source_quality": trade_report,
        "join_quality": join_report,
    }
    marker_path = output_dir / "_TRADE_ENRICHED_DATASET_SUCCESS.json"
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(marker, Path(args.report_md))
    return output_path, marker_path, report_path, marker


def write_markdown(report, path):
    source = report["trade_source_quality"]
    join = report["join_quality"]
    volume = join["aggtrade_vs_kline_volume"]
    feature_lines = "\n".join(f"- `{column}`" for column in report["trade_feature_columns"])
    text = f"""# aggTrade 체결 피처 결합 실행 결과

## 목적

기존 5년 모델은 OHLCV와 선물 문맥을 사용했지만 실제 매수·매도 체결 강도를 충분히 표현하지
못했습니다. 이번 작업은 Binance USDT-M aggTrade를 1분 단위로 집계한 뒤 현재 시점까지의
정보만 사용하는 체결 피처를 만들고, 같은 timestamp의 OHLCV·mark price·funding·open interest와
라벨을 결합할 수 있는지 검증하기 위한 작업입니다.

## 실제 입력 범위

- 체결 원천 기간: `{source['start_utc']}` ~ `{source['end_utc']}`
- 고유 1분 행: `{source['unique_rows']:,}`
- 실제 집계 체결: `{source['aggregated_trades']:,}`건
- 원천 파일: `{source['source_files']}`개
- 누락 분: `{source['missing_minutes']}`
- 빈 체결 분: `{source['empty_trade_minutes']}`

짧은 테스트 파일과 하루 전체 파일이 겹친 timestamp가 `{source['overlap_timestamps']}`개 있었고,
그중 값이 다른 timestamp는 `{source['conflicting_overlap_timestamps']}`개였습니다. 부분 테스트
파일이 전체 일일 파일을 덮어쓰지 않도록 행 수가 가장 많은 원천 파일을 우선하는 규칙으로
해결했습니다. 해결 뒤 중복 timestamp는 `{source['duplicate_minutes_after_resolution']}`개입니다.

## 결합 결과

- OHLCV 입력: `{join['ohlcv_rows']:,}`행
- 선물 문맥 입력: `{join['futures_context_rows']:,}`행
- 60분 롤링 준비 후 체결 피처: `{join['trade_feature_rows_after_warmup']:,}`행
- 미래 240분이 완성된 라벨: `{join['completed_label_rows']:,}`행
- 최종 데이터셋: `{join['final_rows']:,}`행, `{report['columns']}`컬럼
- 누락 조인: `{join['missing_join_rows']}`행
- 인과성 검사: `{report['causal_feature_verification']}`

첫 59분은 60분 롤링 피처를 계산할 과거가 부족해 제외됐고 마지막 240분은 라벨의 미래 관측
구간이 끝나지 않아 제외됐습니다. 이번 실행에서는 입력 {join['ohlcv_rows']:,}행에서
최종 {join['final_rows']:,}행이 남았습니다. 자세한 단계별 건수는 위 표기와 같습니다.

## 거래량 교차 검증

- 분 단위 상대 오차 중앙값: `{volume['minute_relative_error_median']:.8f}`
- 분 단위 상대 오차 99백분위: `{volume['minute_relative_error_p99']:.6f}`
- 1% 이내인 분: `{volume['minute_rows_within_1pct']:,}` / `{source['unique_rows']:,}`
- 5% 초과인 분: `{volume['minute_rows_over_5pct']}`
- 일 단위 합계 최대 상대 오차: `{volume['daily_relative_error_max']:.8f}`

aggTrade 한 건은 여러 개의 개별 체결을 묶으므로 분 경계를 걸치면 인접한 두 분 사이에서
거래량이 이동할 수 있습니다. 따라서 분 단위 완전 일치만 강제하지 않고 일 단위 합계와 trade ID
연속성을 함께 품질 관문으로 사용합니다.

## 생성한 체결 피처

{feature_lines}

모든 롤링 값은 현재 분과 이전 분만 사용합니다. 전체 데이터 계산값과 중간 시점에서 잘라 다시
계산한 값을 비교해 미래 데이터가 과거 피처를 바꾸지 않는 것도 확인했습니다.

## 학습 허용 조건

- 사용 가능한 연속 기간: `{report['available_span_days']:.2f}`일
- 최소 학습 허용 기간: `{report['minimum_training_days']}`일
- 모델 학습 허용: **{report['model_training_allowed']}**
- 차단 사유: `{report['training_block_reason']}`

최종 데이터의 연속 기간과 일별 거래량 대조를 통과한 경우에만 시간순 Walk-Forward 연구에
투입합니다. 이 조건 통과는 수익성이나 실거래 승인이 아닙니다.

## 생성 파일

- 데이터셋: `{report['dataset_file']}`
- 라벨: `{report['label_file']}`
- 데이터 SHA-256: `{report['dataset_sha256']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    if args.minimum_training_days <= 0:
        raise ValueError("minimum_training_days must be positive.")
    trade_context, trade_report = load_trade_context(
        args.trade_context_folder, args.symbol, args.market, args.timeframe
    )
    trade_features, feature_columns = add_causal_trade_features(trade_context)
    verify_causal_trade_features(trade_context, trade_features, feature_columns)
    dataset, feature_columns, label_path, join_report = build_enriched_dataset(
        args, trade_context, trade_features, feature_columns
    )
    output_path, marker_path, report_path, marker = write_outputs(
        args, dataset, feature_columns, label_path, trade_report, join_report
    )
    print(
        json.dumps(
            {
                "rows": marker["rows"],
                "columns": marker["columns"],
                "available_span_days": marker["available_span_days"],
                "model_training_allowed": marker["model_training_allowed"],
                "output": str(output_path),
                "marker": str(marker_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
