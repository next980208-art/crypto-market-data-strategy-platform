"""Legacy local Pandas processor.

Despite the historical filename, this module does not submit an Apache Flink job.
Use flink_batch_submitter.py for the current PyFlink batch path.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "timestamp",
    "datetime_utc",
    "symbol",
    "market",
    "timeframe",
    "run_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "return_1m",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy Pandas processor; this does not run an Apache Flink job."
    )
    parser.add_argument("--temp-folder", default="temp_raw_data")
    parser.add_argument("--feature-folder", default="feature_store_legacy")
    parser.add_argument("--raw-file", help="Process one specific raw CSV file.")
    parser.add_argument("--keep-raw", action="store_true", help="Do not delete raw chunks after verified save.")
    parser.add_argument(
        "--source",
        choices=["files", "kafka"],
        default="files",
        help="files is the reliable offline test path; kafka requires a matching Flink Kafka connector.",
    )
    return parser.parse_args()


def find_raw_files(temp_folder, raw_file):
    if raw_file:
        paths = [Path(raw_file)]
    else:
        paths = sorted(Path(temp_folder).glob("raw_*.csv"))

    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(f"처리할 원본 청크가 없습니다: {temp_folder}")
    return existing


def clean_market_data(df):
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"원본 데이터에 필수 컬럼이 없습니다: {missing}")

    for column in RAW_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    df = df.dropna(subset=RAW_COLUMNS)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    removed = before - len(df)
    if df.empty:
        raise ValueError("정제 후 남은 데이터가 없습니다.")

    return df, removed


def build_features(df):
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["return_1m"] = df["close"].pct_change().fillna(0)

    for column, default in [("symbol", "UNKNOWN"), ("market", "spot"), ("timeframe", "UNKNOWN"), ("run_id", "manual")]:
        if column not in df.columns:
            df[column] = default

    return df[FEATURE_COLUMNS]


def safe_partition_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def write_partition(features, feature_folder, run_id, symbol, market, timeframe, year, month, partition):
    output_root = Path(feature_folder)
    if market != "spot":
        output_root = output_root / f"market={safe_partition_value(market)}"

    output_dir = (
        output_root / f"symbol={safe_partition_value(symbol)}"
        / f"timeframe={safe_partition_value(timeframe)}"
        / f"year={year:04d}"
        / f"month={month:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"features_{run_id}_{year:04d}{month:02d}.parquet"

    partition.to_parquet(output_path, index=False)
    verified = pd.read_parquet(output_path)

    if len(verified) != len(partition):
        raise RuntimeError(f"Feature Store 검증 실패: expected={len(partition)}, actual={len(verified)}")
    if verified[["timestamp", "close", "ma_5"]].isna().any().any():
        raise RuntimeError("Feature Store 검증 실패: 핵심 피처에 결측치가 있습니다.")

    return output_path, len(verified)


def write_and_verify(features, feature_folder, run_id):
    output_root = Path(feature_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    symbol = str(features["symbol"].iloc[0])
    market = str(features["market"].iloc[0])
    timeframe = str(features["timeframe"].iloc[0])
    partition_keys = features["datetime_utc"].dt.strftime("%Y-%m")

    output_files = []
    verified_rows = 0
    for period, partition in features.groupby(partition_keys):
        year_text, month_text = str(period).split("-")
        year = int(year_text)
        month = int(month_text)
        output_path, rows = write_partition(features, feature_folder, run_id, symbol, market, timeframe, year, month, partition)
        output_files.append(str(output_path))
        verified_rows += rows

    if verified_rows != len(features):
        raise RuntimeError(f"Feature Store 전체 검증 실패: expected={len(features)}, actual={verified_rows}")

    marker_dir = output_root / "_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"_SUCCESS_{run_id}.json"
    marker = {
        "run_id": run_id,
        "processor": "legacy_pandas",
        "market": market,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": verified_rows,
        "feature_files": output_files,
        "partition_layout": "feature_store/[market=<market>/]symbol=<symbol>/timeframe=<timeframe>/year=<YYYY>/month=<MM>/",
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_files, marker_path


def process_files(args):
    raw_paths = find_raw_files(args.temp_folder, args.raw_file)
    results = []

    for raw_path in raw_paths:
        print(f"원본 청크 로드: {raw_path}")
        raw_df = pd.read_csv(raw_path)
        cleaned_df, removed = clean_market_data(raw_df)
        features = build_features(cleaned_df)
        run_id = str(features["run_id"].iloc[0])

        output_files, marker_path = write_and_verify(features, args.feature_folder, run_id)
        print(f"Feature Store 저장 및 검증 완료: {len(output_files)} files ({len(features):,} rows, removed={removed:,})")
        for output_file in output_files:
            print(f"  - {output_file}")
        print(f"성공 마커 생성: {marker_path}")

        if args.keep_raw:
            print(f"원본 보존: {raw_path}")
        else:
            raw_path.unlink()
            print(f"검증 완료 후 원본 삭제: {raw_path}")

        results.append((raw_path, output_files, len(features)))

    return results


def explain_kafka_blocker():
    raise RuntimeError(
        "Kafka/Flink 스트리밍 모드는 아직 connector, Docker volume mount, flink run 제출 방식을 정리해야 합니다. "
        "현재 2주 백필 검증은 기본 files 모드로 진행하세요: python 2_flink_processor.py"
    )


def main():
    args = parse_args()
    print("주의: 2_flink_processor.py는 Apache Flink가 아니라 레거시 Pandas 처리기입니다.")
    if args.source == "kafka":
        explain_kafka_blocker()

    process_files(args)


if __name__ == "__main__":
    main()
