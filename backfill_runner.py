import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


EXPECTED_FLINK_SCHEMA_VERSION = "ohlcv_basic_v2_boundary4"
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated BTCUSDT backfill chunks.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="spot")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, exclusive")
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--temp-folder", default="temp_raw_data")
    parser.add_argument(
        "--feature-folder",
        help="Defaults to feature_store_v2 for Flink or feature_store_legacy for Pandas.",
    )
    parser.add_argument(
        "--processor",
        choices=["flink", "pandas"],
        default="flink",
        help="flink is the production backfill path; pandas is the legacy local compatibility path.",
    )
    parser.add_argument("--no-kafka", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument(
        "--recover-stale-staging",
        action="store_true",
        help="Allow the Flink submitter to replace an unfinished staging directory after an interrupted run.",
    )
    parser.add_argument(
        "--train-after",
        action="store_true",
        help="Legacy pandas path only: run the simple 3_ml_training.py smoke model.",
    )
    return parser.parse_args()


def utc_midnight(date_text):
    return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def safe_symbol(symbol):
    return symbol.replace("/", "").replace(":", "")


def run_id(symbol, timeframe, start_dt, end_dt):
    return f"{safe_symbol(symbol)}_{timeframe}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}"


def market_run_id(symbol, market, timeframe, start_dt, end_dt):
    base = run_id(symbol, timeframe, start_dt, end_dt)
    if market == "spot":
        return base
    symbol_part, remainder = base.split("_", 1)
    return f"{symbol_part}_{market.upper()}_{remainder}"


def marker_path(feature_folder, chunk_run_id):
    return Path(feature_folder) / "_markers" / f"_SUCCESS_{chunk_run_id}.json"


def raw_path(temp_folder, chunk_run_id):
    return Path(temp_folder) / f"raw_{chunk_run_id}.csv"


def marker_matches_processor(path, processor):
    expected = "apache_flink_pyflink_batch" if processor == "flink" else "legacy_pandas"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("processor") != expected:
        return False
    if processor == "flink":
        return payload.get("feature_schema_version") == EXPECTED_FLINK_SCHEMA_VERSION
    return True


def run_command(command):
    print(f"실행: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"명령 실패({completed.returncode}): {' '.join(command)}")


def build_chunks(start_dt, end_dt, chunk_days):
    if end_dt <= start_dt:
        raise ValueError("--end-date는 --start-date보다 뒤여야 합니다.")
    if chunk_days <= 0:
        raise ValueError("--chunk-days는 1 이상이어야 합니다.")

    chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def process_chunk(args, start_dt, end_dt):
    days = (end_dt - start_dt).days
    chunk_run_id = market_run_id(args.symbol, args.market, args.timeframe, start_dt, end_dt)
    success_marker = marker_path(args.feature_folder, chunk_run_id)

    if success_marker.exists() and marker_matches_processor(success_marker, args.processor):
        print(f"이미 처리된 청크 건너뜀: {chunk_run_id}")
        return {"run_id": chunk_run_id, "status": "skipped", "days": days}
    if success_marker.exists():
        print(
            f"기존 성공 마커의 처리기가 {args.processor}와 다르거나 기록되지 않아 재처리합니다: "
            f"{success_marker}"
        )

    downloader = [
        sys.executable,
        "1_chunk_downloader.py",
        "--symbol",
        args.symbol,
        "--market",
        args.market,
        "--timeframe",
        args.timeframe,
        "--start-date",
        f"{start_dt:%Y-%m-%d}",
        "--days",
        str(days),
        "--temp-folder",
        args.temp_folder,
    ]
    if args.processor == "flink":
        downloader.extend(["--no-kafka", "--no-header"])
    elif args.no_kafka:
        downloader.append("--no-kafka")

    run_command(downloader)

    if args.processor == "flink":
        processor = [
            sys.executable,
            "flink_batch_submitter.py",
            "--feature-folder",
            args.feature_folder,
            "--raw-file",
            str(raw_path(args.temp_folder, chunk_run_id)),
        ]
        if args.recover_stale_staging:
            processor.append("--reset-stale-staging")
    else:
        processor = [
            sys.executable,
            "2_flink_processor.py",
            "--temp-folder",
            args.temp_folder,
            "--feature-folder",
            args.feature_folder,
            "--raw-file",
            str(raw_path(args.temp_folder, chunk_run_id)),
        ]
    if args.keep_raw:
        processor.append("--keep-raw")

    run_command(processor)

    if not success_marker.exists():
        raise RuntimeError(f"처리 후 성공 마커가 없습니다: {success_marker}")

    return {"run_id": chunk_run_id, "status": "processed", "days": days}


def main():
    args = parse_args()
    if not args.feature_folder:
        args.feature_folder = (
            "feature_store_v2" if args.processor == "flink" else "feature_store_legacy"
        )
    if args.processor == "flink" and not Path("/opt/flink/bin/flink").exists():
        raise RuntimeError(
            "Actual PyFlink backfill must run inside the Airflow container. Example: "
            "docker compose exec -T airflow python backfill_runner.py ... --processor flink"
        )
    if args.train_after and args.processor != "pandas":
        raise ValueError(
            "--train-after runs only the legacy smoke model. For the Flink path, run labeling, "
            "ML dataset building, and time-ordered training explicitly after data validation."
        )
    start_dt = utc_midnight(args.start_date)
    end_dt = utc_midnight(args.end_date)
    chunks = build_chunks(start_dt, end_dt, args.chunk_days)

    print(
        f"백필 시작: processor={args.processor}, market={args.market}, {args.symbol} {args.timeframe} "
        f"{start_dt:%Y-%m-%d} ~ {end_dt:%Y-%m-%d}, chunks={len(chunks)}"
    )

    results = []
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"[{index}/{len(chunks)}] 청크 처리: {chunk_start:%Y-%m-%d} ~ {chunk_end:%Y-%m-%d}")
        results.append(process_chunk(args, chunk_start, chunk_end))

    if args.train_after:
        run_command(
            [
                sys.executable,
                "3_ml_training.py",
                "--symbol",
                args.symbol,
                "--market",
                args.market,
                "--timeframe",
                args.timeframe,
                "--feature-folder",
                args.feature_folder,
            ]
        )

    processed = sum(1 for result in results if result["status"] == "processed")
    skipped = sum(1 for result in results if result["status"] == "skipped")
    print(f"백필 완료: processed={processed}, skipped={skipped}, total={len(results)}")


if __name__ == "__main__":
    main()
