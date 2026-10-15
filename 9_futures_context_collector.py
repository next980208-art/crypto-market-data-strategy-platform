import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

from market_metadata import (
    METADATA_SCHEMA_VERSION,
    TIMESTAMP_UNIT,
    normalize_symbol,
    to_ccxt_symbol,
)


FUTURES_CONTEXT_SCHEMA_VERSION = "usdm_context_v2_complete_mark"
CONTEXT_RUN_ID_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)_(?P<market>[A-Z0-9]+)_CONTEXT_(?P<timeframe>[A-Za-z0-9]+)_(?P<start>\d{8})_(?P<end>\d{8})$"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Binance USDT-M funding, open interest, and mark-price context.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1m", help="Mark-price OHLCV timeframe.")
    parser.add_argument("--open-interest-timeframe", default="5m")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, inclusive UTC")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, exclusive UTC")
    parser.add_argument("--context-folder", default="futures_context_store_v2")
    parser.add_argument(
        "--max-imputed-mark-price-rows",
        type=int,
        default=5,
        help="Maximum isolated missing mark-price minutes to fill with the prior observed value.",
    )
    return parser.parse_args()


def utc_midnight(date_text):
    return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def safe_value(value):
    return normalize_symbol(value)


def exchange_symbol(symbol):
    return to_ccxt_symbol(symbol, "usdm")


def fetch_paginated(fetch_page, start_ms, end_ms, rate_limit):
    rows_by_timestamp = {}
    cursor = start_ms
    while cursor < end_ms:
        batch = fetch_page(cursor)
        if not batch:
            break

        timestamps = []
        for row in batch:
            timestamp = int(row[0] if isinstance(row, (list, tuple)) else row["timestamp"])
            timestamps.append(timestamp)
            if start_ms <= timestamp < end_ms:
                rows_by_timestamp[timestamp] = row

        next_cursor = max(timestamps) + 1 if timestamps else end_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(rate_limit / 1000)

    return [rows_by_timestamp[key] for key in sorted(rows_by_timestamp)]


def funding_frame(rows):
    records = []
    for row in rows:
        records.append(
            {
                "funding_rate_timestamp": int(row["timestamp"]),
                "funding_rate": pd.to_numeric(row.get("fundingRate"), errors="coerce"),
            }
        )
    return pd.DataFrame(records, columns=["funding_rate_timestamp", "funding_rate"])


def open_interest_frame(rows):
    records = []
    for row in rows:
        records.append(
            {
                "open_interest_timestamp": int(row["timestamp"]),
                "open_interest": pd.to_numeric(row.get("openInterestAmount"), errors="coerce"),
                "open_interest_value": pd.to_numeric(row.get("openInterestValue"), errors="coerce"),
            }
        )
    return pd.DataFrame(
        records,
        columns=["open_interest_timestamp", "open_interest", "open_interest_value"],
    )


def mark_frame(rows):
    frame = pd.DataFrame(rows, columns=["timestamp", "mark_open", "mark_high", "mark_low", "mark_price", "mark_volume"])
    frame = frame[["timestamp", "mark_open", "mark_high", "mark_low", "mark_price"]]
    frame["mark_price_status"] = "observed"
    return frame


def complete_mark_price_minutes(rows, expected_timestamps, max_imputed_rows):
    """Fill tiny source gaps while retaining an explicit downstream audit flag."""
    if max_imputed_rows < 0:
        raise ValueError("--max-imputed-mark-price-rows must be zero or greater.")

    observed = mark_frame(rows).drop_duplicates("timestamp", keep="last").set_index("timestamp")
    expected_index = pd.Index(sorted(expected_timestamps), name="timestamp")
    frame = observed.reindex(expected_index)
    missing = frame["mark_price"].isna()
    missing_count = int(missing.sum())
    if missing_count > max_imputed_rows:
        raise RuntimeError(
            "mark price OHLCV has too many missing minutes for the configured imputation limit. "
            f"missing={missing_count:,}, limit={max_imputed_rows:,}"
        )
    if missing_count:
        price_columns = ["mark_open", "mark_high", "mark_low", "mark_price"]
        frame[price_columns] = frame[price_columns].ffill().bfill()
        frame["mark_price_status"] = "observed"
        frame.loc[missing, "mark_price_status"] = "imputed_previous_mark"
    else:
        frame["mark_price_status"] = "observed"
    return frame.reset_index(), missing_count


def merge_context(mark_prices, funding_rates, open_interest, symbol, timeframe, funding_status, open_interest_status):
    context = mark_prices.sort_values("timestamp").reset_index(drop=True)
    if not funding_rates.empty:
        context = pd.merge_asof(
            context,
            funding_rates.sort_values("funding_rate_timestamp"),
            left_on="timestamp",
            right_on="funding_rate_timestamp",
            direction="backward",
        )
    else:
        context["funding_rate_timestamp"] = pd.NA
        context["funding_rate"] = pd.NA

    if not open_interest.empty:
        context = pd.merge_asof(
            context,
            open_interest.sort_values("open_interest_timestamp"),
            left_on="timestamp",
            right_on="open_interest_timestamp",
            direction="backward",
        )
    else:
        context["open_interest_timestamp"] = pd.NA
        context["open_interest"] = pd.NA
        context["open_interest_value"] = pd.NA

    context["datetime_utc"] = pd.to_datetime(context["timestamp"], unit="ms", utc=True)
    context["event_time_ms"] = context["timestamp"].astype("int64")
    context["timestamp_unit"] = TIMESTAMP_UNIT
    context["metadata_schema_version"] = METADATA_SCHEMA_VERSION
    context["symbol"] = normalize_symbol(symbol)
    context["market"] = "usdm"
    context["timeframe"] = timeframe
    context["context_schema_version"] = FUTURES_CONTEXT_SCHEMA_VERSION
    context["funding_rate_status"] = funding_status
    context["open_interest_status"] = open_interest_status
    return context


def replace_fully_covered_context(context_folder, symbol, timeframe, start_dt, end_dt):
    """Replace older files only when the new range covers them completely.

    Daily collection can create one-day files before historical backfill creates
    a larger contiguous range. Keeping both would duplicate timestamps. Partial
    overlaps are refused because removing them could discard data outside the
    incoming range.
    """
    marker_dir = Path(context_folder) / "_markers"
    if not marker_dir.exists():
        return []

    symbol_code = safe_value(symbol)
    replacements = []
    for marker_path in marker_dir.glob("_SUCCESS_*_CONTEXT_*.json"):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("context_schema_version") != FUTURES_CONTEXT_SCHEMA_VERSION:
                continue
            match = CONTEXT_RUN_ID_PATTERN.match(str(marker.get("run_id", "")))
            if not match:
                continue
            if (
                match.group("symbol") != symbol_code
                or match.group("market") != "USDM"
                or marker.get("timeframe", match.group("timeframe")) != timeframe
            ):
                continue
            existing_start = datetime.strptime(match.group("start"), "%Y%m%d").replace(tzinfo=timezone.utc)
            existing_end = datetime.strptime(match.group("end"), "%Y%m%d").replace(tzinfo=timezone.utc)
            overlaps = existing_start < end_dt and start_dt < existing_end
            if not overlaps:
                continue
            if not (start_dt <= existing_start and existing_end <= end_dt):
                raise RuntimeError(
                    "기존 선물 컨텍스트 파일과 부분 중복됩니다. 자동 삭제하지 않습니다: "
                    f"existing={existing_start:%Y-%m-%d}~{existing_end:%Y-%m-%d}, "
                    f"incoming={start_dt:%Y-%m-%d}~{end_dt:%Y-%m-%d}"
                )
            context_path = Path(str(marker.get("context_file", "")))
            if not context_path.exists():
                raise RuntimeError(f"성공 마커의 컨텍스트 파일을 찾지 못했습니다: {context_path}")
            replacements.append((context_path, marker_path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"기존 선물 컨텍스트 마커를 안전하게 확인하지 못했습니다: {marker_path}") from error

    for context_path, marker_path in replacements:
        context_path.unlink()
        marker_path.unlink()
    return [str(context_path) for context_path, _ in replacements]


def write_context(context, context_folder, symbol, timeframe, start_dt, end_dt):
    if not re.fullmatch(r"[1-9]\d*[mhd]", timeframe):
        raise ValueError("Context timeframe must use canonical minute/hour/day units")
    run_id = f"{safe_value(symbol)}_USDM_CONTEXT_{timeframe}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}"
    root = Path(context_folder) / "market=usdm" / f"symbol={safe_value(symbol)}" / f"timeframe={timeframe}"
    root.mkdir(parents=True, exist_ok=True)
    replaced_paths = replace_fully_covered_context(context_folder, symbol, timeframe, start_dt, end_dt)
    output_path = root / f"context_{run_id}.parquet"
    context.to_parquet(output_path, index=False)

    verified = pd.read_parquet(output_path)
    if len(verified) != len(context):
        raise RuntimeError(f"선물 컨텍스트 저장 검증 실패: expected={len(context)}, actual={len(verified)}")
    if verified["mark_price"].isna().any():
        raise RuntimeError("선물 컨텍스트 저장 검증 실패: mark_price 결측치가 있습니다.")

    marker_dir = Path(context_folder) / "_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"_SUCCESS_{run_id}.json"
    marker = {
        "run_id": run_id,
        "symbol": safe_value(symbol),
        "market": "usdm",
        "timeframe": timeframe,
        "rows": len(context),
        "context_schema_version": FUTURES_CONTEXT_SCHEMA_VERSION,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "timestamp_field": "event_time_ms",
        "timestamp_unit": TIMESTAMP_UNIT,
        "funding_rate_status": str(context["funding_rate_status"].iloc[0]),
        "open_interest_status": str(context["open_interest_status"].iloc[0]),
        "mark_price_imputed_rows": int((context["mark_price_status"] != "observed").sum()),
        "context_file": str(output_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, marker_path, replaced_paths


def fetch_optional(name, fetcher):
    try:
        return fetcher(), "available"
    except ccxt.BaseError as error:
        status = f"unavailable: {type(error).__name__}: {error}"
        print(f"경고: {name} 수집 불가. 빈 값으로 저장합니다. {status}")
        return [], status


def main():
    args = parse_args()
    start_dt = utc_midnight(args.start_date)
    end_dt = utc_midnight(args.end_date)
    if end_dt <= start_dt:
        raise ValueError("--end-date는 --start-date보다 뒤여야 합니다.")

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    market_symbol = exchange_symbol(args.symbol)
    exchange = ccxt.binanceusdm({"enableRateLimit": True})

    print(f"선물 컨텍스트 수집 시작: {market_symbol} {start_dt:%Y-%m-%d} ~ {end_dt:%Y-%m-%d}")
    mark_rows = fetch_paginated(
        lambda since: exchange.fetch_mark_ohlcv(market_symbol, args.timeframe, since=since, limit=1000),
        start_ms,
        end_ms,
        exchange.rateLimit,
    )
    funding_rows, funding_status = fetch_optional(
        "funding rate",
        lambda: fetch_paginated(
            lambda since: exchange.fetch_funding_rate_history(market_symbol, since=since, limit=1000),
            start_ms,
            end_ms,
            exchange.rateLimit,
        ),
    )
    open_interest_rows, open_interest_status = fetch_optional(
        "open interest",
        lambda: fetch_paginated(
            lambda since: exchange.fetch_open_interest_history(
                market_symbol,
                args.open_interest_timeframe,
                since=since,
                limit=500,
            ),
            start_ms,
            end_ms,
            exchange.rateLimit,
        ),
    )

    if not mark_rows:
        raise RuntimeError("mark price OHLCV 데이터를 가져오지 못했습니다.")

    step_ms = int(exchange.parse_timeframe(args.timeframe) * 1000)
    expected_timestamps = set(range(start_ms, end_ms, step_ms))
    actual_timestamps = {int(row[0]) for row in mark_rows}
    missing_timestamps = expected_timestamps - actual_timestamps
    unexpected_timestamps = actual_timestamps - expected_timestamps
    mark_prices, imputed_mark_price_rows = complete_mark_price_minutes(
        mark_rows,
        expected_timestamps,
        args.max_imputed_mark_price_rows,
    )
    if imputed_mark_price_rows:
        print(
            "Warning: isolated missing mark-price minutes were filled from the prior observed value. "
            f"imputed_rows={imputed_mark_price_rows:,}"
        )
    if unexpected_timestamps:
        missing = len(expected_timestamps - actual_timestamps)
        unexpected = len(actual_timestamps - expected_timestamps)
        raise RuntimeError(
            "mark price OHLCV 구간이 완전하지 않습니다. "
            f"expected={len(expected_timestamps):,}, actual={len(actual_timestamps):,}, "
            f"missing={missing:,}, unexpected={unexpected:,}"
        )

    context = merge_context(
        mark_prices,
        funding_frame(funding_rows),
        open_interest_frame(open_interest_rows),
        args.symbol,
        args.timeframe,
        funding_status,
        open_interest_status,
    )
    output_path, marker_path, replaced_paths = write_context(
        context,
        args.context_folder,
        args.symbol,
        args.timeframe,
        start_dt,
        end_dt,
    )

    print(f"mark price rows: {len(mark_rows):,}")
    print(f"mark price imputed rows: {imputed_mark_price_rows:,}")
    print(f"funding rate rows: {len(funding_rows):,}")
    print(f"open interest rows: {len(open_interest_rows):,}")
    print(f"선물 컨텍스트 저장 완료: {output_path} ({len(context):,} rows)")
    print(f"성공 마커: {marker_path}")
    if replaced_paths:
        print(f"중복 방지를 위해 교체한 기존 컨텍스트 파일: {len(replaced_paths):,}개")


if __name__ == "__main__":
    main()
