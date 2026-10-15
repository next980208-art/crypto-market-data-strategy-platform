"""Recover one completed UTC day of USD-M aggTrades from Binance public data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from market_metadata import normalize_symbol


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
ARCHIVE_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)


def load_context_module():
    path = PROJECT_ROOT / "10_aggtrade_collector.py"
    spec = importlib.util.spec_from_file_location("aggtrade_context_writer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"집계 모듈을 읽을 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover one completed UTC day from Binance USD-M daily aggTrades archive."
    )
    parser.add_argument("--data-date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--context-folder", default="trade_context_store_v2")
    parser.add_argument("--cache-folder", default="temp_raw_data/aggtrade_archive")
    parser.add_argument("--archive-base-url", default=DEFAULT_ARCHIVE_BASE_URL)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def parse_data_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def archive_urls(base_url: str, symbol: str, data_date: datetime) -> tuple[str, str, str]:
    symbol_code = normalize_symbol(symbol)
    archive_name = f"{symbol_code}-aggTrades-{data_date:%Y-%m-%d}.zip"
    archive_url = f"{base_url.rstrip('/')}/{symbol_code}/{archive_name}"
    return archive_name, archive_url, f"{archive_url}.CHECKSUM"


def download_file(url: str, destination: Path, timeout_seconds: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    request = Request(url, headers={"User-Agent": "coin-project-aggtrade-recovery/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"공식 aggTrade 아카이브 다운로드 실패: {url}: {error}") from error


def read_checksum(url: str, timeout_seconds: int) -> str:
    request = Request(url, headers={"User-Agent": "coin-project-aggtrade-recovery/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            text = response.read(1024).decode("utf-8").strip()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"공식 체크섬 다운로드 실패: {url}: {error}") from error
    checksum = text.split()[0].lower() if text else ""
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise RuntimeError(f"올바르지 않은 SHA-256 체크섬 응답입니다: {text!r}")
    return checksum


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_trade(row: dict[str, str]) -> dict[str, object]:
    buyer_is_maker = row["is_buyer_maker"].strip().lower()
    if buyer_is_maker not in {"true", "false"}:
        raise ValueError(f"is_buyer_maker 값이 올바르지 않습니다: {row['is_buyer_maker']!r}")
    return {
        "id": str(int(row["agg_trade_id"])),
        "timestamp": int(row["transact_time"]),
        "price": float(row["price"]),
        "amount": float(row["quantity"]),
        "side": "sell" if buyer_is_maker == "true" else "buy",
        "info": {"m": buyer_is_maker == "true"},
    }


def aggregate_rows(rows, start_ms: int, end_ms: int, context_module):
    aggregates = defaultdict(context_module.empty_aggregate)
    first_trade_id = None
    last_trade_id = None
    first_timestamp = None
    last_timestamp = None
    total_trades = 0

    for row in rows:
        trade = archive_trade(row)
        trade_id = int(trade["id"])
        timestamp = int(trade["timestamp"])
        if not start_ms <= timestamp < end_ms:
            raise RuntimeError(
                f"아카이브에 요청 날짜 밖의 체결이 있습니다: id={trade_id}, timestamp={timestamp}"
            )
        if last_trade_id is not None and trade_id != last_trade_id + 1:
            raise RuntimeError(
                "공식 아카이브 aggTrade ID가 연속적이지 않습니다: "
                f"previous={last_trade_id}, current={trade_id}"
            )
        first_trade_id = trade_id if first_trade_id is None else first_trade_id
        last_trade_id = trade_id
        first_timestamp = timestamp if first_timestamp is None else first_timestamp
        last_timestamp = timestamp
        context_module.add_trade(aggregates, trade)
        total_trades += 1

    if total_trades == 0:
        raise RuntimeError("공식 아카이브에 aggTrade 행이 없습니다.")
    if last_trade_id - first_trade_id + 1 != total_trades:
        raise RuntimeError("공식 아카이브의 행 수와 aggTrade ID 범위가 일치하지 않습니다.")

    stats = {
        "start_utc": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
        "end_utc": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
        "pages": 0,
        "aggregated_trades": total_trades,
        "duplicate_api_records": 0,
        "first_trade_id": first_trade_id,
        "last_trade_id": last_trade_id,
        "first_trade_timestamp": first_timestamp,
        "last_trade_timestamp": last_timestamp,
    }
    return aggregates, stats, total_trades


def aggregate_archive(path: Path, start_ms: int, end_ms: int, context_module):
    try:
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise RuntimeError(f"ZIP 안의 CSV가 정확히 하나여야 합니다: {csv_names}")
            with archive.open(csv_names[0]) as binary:
                text_stream = (line.decode("utf-8-sig") for line in binary)
                reader = csv.DictReader(text_stream)
                missing = sorted(set(ARCHIVE_COLUMNS) - set(reader.fieldnames or []))
                if missing:
                    raise RuntimeError(f"공식 aggTrade CSV 필드가 부족합니다: {missing}")
                return aggregate_rows(reader, start_ms, end_ms, context_module)
    except (zipfile.BadZipFile, UnicodeDecodeError, csv.Error, OSError) as error:
        raise RuntimeError(f"공식 aggTrade ZIP을 읽지 못했습니다: {path}: {error}") from error


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds는 1 이상이어야 합니다.")
    start_dt = parse_data_date(args.data_date)
    end_dt = start_dt + timedelta(days=1)
    if end_dt > datetime.now(timezone.utc):
        raise ValueError("완료된 UTC 날짜만 복구할 수 있습니다.")

    archive_name, archive_url, checksum_url = archive_urls(
        args.archive_base_url, args.symbol, start_dt
    )
    cache_path = PROJECT_ROOT / args.cache_folder / archive_name
    expected_checksum = read_checksum(checksum_url, args.timeout_seconds)
    if not cache_path.exists() or sha256_file(cache_path) != expected_checksum:
        download_file(archive_url, cache_path, args.timeout_seconds)
    actual_checksum = sha256_file(cache_path)
    if actual_checksum != expected_checksum:
        cache_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"공식 ZIP SHA-256 불일치: expected={expected_checksum}, actual={actual_checksum}"
        )

    context_module = load_context_module()
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    aggregates, stats, total_trades = aggregate_archive(
        cache_path, start_ms, end_ms, context_module
    )
    context = context_module.build_context(aggregates, args.symbol, start_ms, end_ms)
    validation = context_module.validate_context(context, start_ms, end_ms)
    writer_args = SimpleNamespace(
        symbol=args.symbol,
        context_folder=args.context_folder,
        window_minutes=1440,
        max_pages=0,
        max_total_pages=0,
    )
    output_path, marker_path = context_module.write_context(
        context,
        writer_args,
        start_dt,
        end_dt,
        [stats],
        0,
        total_trades,
        validation,
        marker_overrides={
            "source_type": "binance_public_data_daily_archive",
            "archive_url": archive_url,
            "archive_checksum_sha256": actual_checksum,
            "archive_bytes": cache_path.stat().st_size,
            "pagination": "not_applicable_daily_archive",
        },
    )
    if not args.keep_archive:
        cache_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": "completed",
                "source_type": "binance_public_data_daily_archive",
                "data_date": args.data_date,
                "aggregated_trades": total_trades,
                "minute_rows": len(context),
                "empty_trade_minutes": validation["empty_trade_minutes"],
                "context_file": str(output_path),
                "marker_file": str(marker_path),
                "raw_archive_deleted": not args.keep_archive,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
