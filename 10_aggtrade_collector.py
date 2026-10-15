import argparse
import json
from collections import defaultdict
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


TRADE_CONTEXT_SCHEMA_VERSION = "aggtrade_1m_v3_windowed_complete"
MINUTE_MS = 60_000


class PageLimitExceeded(RuntimeError):
    """Raised before a partial API window can be accepted as complete."""


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Binance USDT-M aggregated trades and build 1-minute trade features.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--start-datetime", help="UTC ISO time, e.g. 2024-01-01T00:00")
    parser.add_argument("--end-datetime", help="UTC ISO time, exclusive")
    parser.add_argument("--recent-minutes", type=int, help="Collect this many minutes ending at Binance server time.")
    parser.add_argument("--context-folder", default="trade_context_store_v2")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help="Split the requested range into this many minutes per API window.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Safety cap for 1,000-trade API pages per window.",
    )
    parser.add_argument(
        "--max-total-pages",
        type=int,
        default=10_000,
        help="Safety cap for all API pages in the complete request.",
    )
    return parser.parse_args()


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def safe_value(value):
    return normalize_symbol(value)


def exchange_symbol(symbol):
    return to_ccxt_symbol(symbol, "usdm")


def taker_side(trade):
    side = trade.get("side")
    if side in {"buy", "sell"}:
        return side
    buyer_is_maker = trade.get("info", {}).get("m")
    if buyer_is_maker is True:
        return "sell"
    return "buy"


def empty_aggregate():
    return {
        "trade_count": 0,
        "trade_volume": 0.0,
        "trade_notional": 0.0,
        "taker_buy_volume": 0.0,
        "taker_sell_volume": 0.0,
        "taker_buy_notional": 0.0,
        "taker_sell_notional": 0.0,
        "first_price": None,
        "last_price": None,
    }


def add_trade(aggregates, trade):
    timestamp = int(trade["timestamp"])
    minute = timestamp - (timestamp % 60_000)
    price = float(trade["price"])
    amount = float(trade["amount"])
    if price <= 0 or amount <= 0:
        return

    record = aggregates[minute]
    notional = price * amount
    side = taker_side(trade)
    record["trade_count"] += 1
    record["trade_volume"] += amount
    record["trade_notional"] += notional
    record["first_price"] = price if record["first_price"] is None else record["first_price"]
    record["last_price"] = price
    if side == "buy":
        record["taker_buy_volume"] += amount
        record["taker_buy_notional"] += notional
    else:
        record["taker_sell_volume"] += amount
        record["taker_sell_notional"] += notional


def collect_trade_window(exchange, symbol, start_ms, end_ms, max_pages):
    aggregates = defaultdict(empty_aggregate)
    cursor = start_ms
    next_from_id = None
    seen_trade_ids = set()
    numeric_in_range_ids = set()
    pages = 0
    total_trades = 0
    duplicate_trades = 0
    first_trade_timestamp = None
    last_trade_timestamp = None

    while cursor < end_ms:
        if pages >= max_pages:
            raise PageLimitExceeded(
                f"창별 --max-pages={max_pages}에 도달했습니다: "
                f"{datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()} ~ "
                f"{datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat()}. "
                "--window-minutes를 더 작게 하거나 페이지 상한을 높이세요. "
                "불완전한 체결 데이터는 저장하지 않습니다."
            )
        try:
            params = {"fromId": next_from_id} if next_from_id is not None else {}
            batch = exchange.fetch_trades(
                symbol,
                since=cursor if next_from_id is None else None,
                limit=1000,
                params=params,
            )
        except ccxt.ExchangeError as error:
            if "-4166" in str(error) or "recent 2 days" in str(error):
                raise RuntimeError(
                    "Binance USDT-M aggTrade 공개 API는 최근 2일만 과거 검색을 지원합니다. "
                    "5년 체결 백필에는 외부 과거 데이터 공급자 또는 지금부터의 자체 실시간 적재가 필요합니다."
                ) from error
            raise
        pages += 1
        if not batch:
            break

        latest_timestamp = cursor - 1
        numeric_trade_ids = []
        for trade in batch:
            timestamp = int(trade["timestamp"])
            latest_timestamp = max(latest_timestamp, timestamp)
            trade_id = trade.get("id")
            if trade_id is not None:
                try:
                    numeric_trade_ids.append(int(trade_id))
                except (TypeError, ValueError):
                    pass
            dedupe_key = (
                f"id:{trade_id}"
                if trade_id is not None
                else f"fallback:{timestamp}:{trade.get('price')}:{trade.get('amount')}:{trade.get('side')}"
            )
            if dedupe_key in seen_trade_ids:
                duplicate_trades += 1
                continue
            seen_trade_ids.add(dedupe_key)
            if start_ms <= timestamp < end_ms:
                if trade_id is None:
                    raise RuntimeError("Binance aggTrade에 trade ID가 없어 연속성을 검증할 수 없습니다.")
                try:
                    numeric_trade_id = int(trade_id)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(f"숫자가 아닌 Binance aggTrade ID입니다: {trade_id}") from error
                numeric_in_range_ids.add(numeric_trade_id)
                add_trade(aggregates, trade)
                total_trades += 1
                first_trade_timestamp = (
                    timestamp if first_trade_timestamp is None else min(first_trade_timestamp, timestamp)
                )
                last_trade_timestamp = (
                    timestamp if last_trade_timestamp is None else max(last_trade_timestamp, timestamp)
                )

        if latest_timestamp >= end_ms:
            cursor = end_ms
            break
        if numeric_trade_ids:
            candidate_from_id = max(numeric_trade_ids) + 1
            if next_from_id is not None and candidate_from_id <= next_from_id:
                break
            next_from_id = candidate_from_id
            cursor = max(cursor, latest_timestamp)
        elif latest_timestamp >= cursor:
            cursor = latest_timestamp + 1
            next_from_id = None
        else:
            break
    if cursor < end_ms:
        raise RuntimeError(
            "요청 구간 끝까지 체결 데이터를 받지 못했습니다. "
            "이 API의 과거 접근 범위 또는 페이지 제한을 확인해야 합니다."
        )

    first_trade_id = min(numeric_in_range_ids) if numeric_in_range_ids else None
    last_trade_id = max(numeric_in_range_ids) if numeric_in_range_ids else None
    if numeric_in_range_ids and last_trade_id - first_trade_id + 1 != len(numeric_in_range_ids):
        raise RuntimeError(
            "aggTrade ID 공백을 발견했습니다: "
            f"first={first_trade_id}, last={last_trade_id}, unique={len(numeric_in_range_ids)}"
        )
    if len(numeric_in_range_ids) != total_trades:
        raise RuntimeError(
            "집계한 체결 수와 고유 trade ID 수가 다릅니다: "
            f"trades={total_trades}, ids={len(numeric_in_range_ids)}"
        )

    stats = {
        "start_utc": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
        "end_utc": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
        "pages": pages,
        "aggregated_trades": total_trades,
        "duplicate_api_records": duplicate_trades,
        "first_trade_id": first_trade_id,
        "last_trade_id": last_trade_id,
        "first_trade_timestamp": first_trade_timestamp,
        "last_trade_timestamp": last_trade_timestamp,
    }
    return aggregates, stats


def merge_aggregates(target, source):
    for minute, source_item in source.items():
        target_item = target[minute]
        if target_item["first_price"] is None:
            target_item["first_price"] = source_item["first_price"]
        if source_item["last_price"] is not None:
            target_item["last_price"] = source_item["last_price"]
        for name in (
            "trade_count",
            "trade_volume",
            "trade_notional",
            "taker_buy_volume",
            "taker_sell_volume",
            "taker_buy_notional",
            "taker_sell_notional",
        ):
            target_item[name] += source_item[name]


def validate_trade_id_windows(window_stats):
    nonempty = [item for item in window_stats if item["first_trade_id"] is not None]
    for previous, current in zip(nonempty, nonempty[1:]):
        if current["first_trade_id"] != previous["last_trade_id"] + 1:
            raise RuntimeError(
                "시간 창 경계에서 aggTrade ID 공백 또는 중복을 발견했습니다: "
                f"previous_last={previous['last_trade_id']}, "
                f"current_first={current['first_trade_id']}, "
                f"current_window={current['start_utc']}"
            )


def collect_trades(
    exchange,
    symbol,
    start_ms,
    end_ms,
    max_pages,
    *,
    window_minutes=15,
    max_total_pages=10_000,
    progress_callback=None,
):
    aggregates = defaultdict(empty_aggregate)
    window_stats = []
    window_ms = window_minutes * MINUTE_MS
    window_start = start_ms
    total_pages = 0
    total_trades = 0

    while window_start < end_ms:
        window_end = min(window_start + window_ms, end_ms)
        remaining_pages = max_total_pages - total_pages
        if remaining_pages <= 0:
            raise PageLimitExceeded(
                f"전체 --max-total-pages={max_total_pages}에 도달했습니다. "
                "불완전한 체결 데이터는 저장하지 않습니다."
            )
        effective_page_limit = min(max_pages, remaining_pages)
        window_aggregates, stats = collect_trade_window(
            exchange,
            symbol,
            window_start,
            window_end,
            effective_page_limit,
        )
        if stats["pages"] >= remaining_pages and window_end < end_ms:
            raise PageLimitExceeded(
                f"전체 --max-total-pages={max_total_pages}로 다음 창을 처리할 수 없습니다. "
                "불완전한 체결 데이터는 저장하지 않습니다."
            )
        merge_aggregates(aggregates, window_aggregates)
        window_stats.append(stats)
        total_pages += stats["pages"]
        total_trades += stats["aggregated_trades"]
        window_start = window_end
        if progress_callback is not None:
            progress_callback(len(window_stats), total_pages, total_trades, stats)

    validate_trade_id_windows(window_stats)
    return aggregates, window_stats, total_pages, total_trades


def expected_minute_timestamps(start_ms, end_ms):
    first_minute = start_ms - (start_ms % MINUTE_MS)
    last_minute = (end_ms - 1) - ((end_ms - 1) % MINUTE_MS)
    return list(range(first_minute, last_minute + MINUTE_MS, MINUTE_MS))


def build_context(aggregates, symbol, start_ms, end_ms):
    records = []
    for timestamp in expected_minute_timestamps(start_ms, end_ms):
        item = aggregates.get(timestamp, empty_aggregate())
        volume = item["trade_volume"]
        records.append(
            {
                "timestamp": timestamp,
                "datetime_utc": pd.to_datetime(timestamp, unit="ms", utc=True),
                "event_time_ms": timestamp,
                "timestamp_unit": TIMESTAMP_UNIT,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "symbol": normalize_symbol(symbol),
                "market": "usdm",
                "timeframe": "1m",
                "context_schema_version": TRADE_CONTEXT_SCHEMA_VERSION,
                "trade_count": item["trade_count"],
                "trade_volume": volume,
                "trade_notional": item["trade_notional"],
                "trade_vwap": item["trade_notional"] / volume if volume else None,
                "trade_first_price": item["first_price"],
                "trade_last_price": item["last_price"],
                "taker_buy_volume": item["taker_buy_volume"],
                "taker_sell_volume": item["taker_sell_volume"],
                "taker_buy_notional": item["taker_buy_notional"],
                "taker_sell_notional": item["taker_sell_notional"],
                "taker_volume_imbalance": (item["taker_buy_volume"] - item["taker_sell_volume"]) / volume if volume else 0.0,
            }
        )
    return pd.DataFrame(records)


def validate_context(context, start_ms, end_ms):
    expected_timestamps = expected_minute_timestamps(start_ms, end_ms)
    actual_timestamps = context["timestamp"].astype("int64").tolist()
    if actual_timestamps != expected_timestamps:
        missing = sorted(set(expected_timestamps) - set(actual_timestamps))
        unexpected = sorted(set(actual_timestamps) - set(expected_timestamps))
        raise RuntimeError(
            "1분 체결 컨텍스트 범위 검증 실패: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    duplicate_timestamps = int(context["timestamp"].duplicated().sum())
    if duplicate_timestamps:
        raise RuntimeError(f"1분 체결 컨텍스트 timestamp 중복: {duplicate_timestamps}")
    empty_trade_minutes = int((context["trade_count"] == 0).sum())
    return {
        "expected_minute_rows": len(expected_timestamps),
        "actual_minute_rows": len(context),
        "missing_minute_rows": 0,
        "duplicate_minute_rows": duplicate_timestamps,
        "empty_trade_minutes": empty_trade_minutes,
        "complete_minute_boundaries": start_ms % MINUTE_MS == 0 and end_ms % MINUTE_MS == 0,
    }


def write_context(
    context,
    args,
    start_dt,
    end_dt,
    window_stats,
    pages,
    total_trades,
    validation,
    *,
    marker_overrides=None,
):
    run_id = f"{safe_value(args.symbol)}_USDM_AGGTRADE_1m_{start_dt:%Y%m%dT%H%M}_{end_dt:%Y%m%dT%H%M}"
    output_dir = Path(args.context_folder) / "market=usdm" / f"symbol={safe_value(args.symbol)}" / "timeframe=1m"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trade_context_{run_id}.parquet"
    temporary_output_path = output_path.with_name(f".{output_path.name}.tmp")
    context.to_parquet(temporary_output_path, index=False)
    try:
        verified = pd.read_parquet(temporary_output_path)
        if len(verified) != len(context):
            raise RuntimeError(
                f"체결 컨텍스트 저장 검증 실패: expected={len(context)}, actual={len(verified)}"
            )
        verified_validation = validate_context(
            verified,
            int(start_dt.timestamp() * 1000),
            int(end_dt.timestamp() * 1000),
        )
        if verified_validation != validation:
            raise RuntimeError("저장 전후 체결 컨텍스트 품질 값이 다릅니다.")
        temporary_output_path.replace(output_path)
    finally:
        temporary_output_path.unlink(missing_ok=True)

    marker_dir = Path(args.context_folder) / "_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"_SUCCESS_{run_id}.json"
    marker = {
        "run_id": run_id,
        "symbol": safe_value(args.symbol),
        "market": "usdm",
        "timeframe": "1m",
        "rows": len(context),
        "api_pages": pages,
        "api_windows": len(window_stats),
        "window_minutes": args.window_minutes,
        "max_pages_per_window": args.max_pages,
        "max_total_pages": args.max_total_pages,
        "pagination": "fixed_time_windows_then_startTime_and_aggregate_trade_fromId",
        "context_schema_version": TRADE_CONTEXT_SCHEMA_VERSION,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "timestamp_field": "event_time_ms",
        "timestamp_unit": TIMESTAMP_UNIT,
        "aggregated_trades": total_trades,
        "trade_id_sequence_status": "continuous",
        **validation,
        "windows": window_stats,
        "context_file": str(output_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if marker_overrides:
        marker.update(marker_overrides)
    temporary_marker_path = marker_path.with_name(f".{marker_path.name}.tmp")
    temporary_marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_marker_path.replace(marker_path)
    return output_path, marker_path


def main():
    args = parse_args()
    if args.recent_minutes is not None:
        if args.start_datetime or args.end_datetime:
            raise ValueError("--recent-minutes는 --start-datetime/--end-datetime과 함께 사용할 수 없습니다.")
        if args.recent_minutes <= 0:
            raise ValueError("--recent-minutes는 1 이상이어야 합니다.")
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        end_dt = datetime.fromtimestamp(exchange.fetch_time() / 1000, tz=timezone.utc)
        start_dt = end_dt - pd.Timedelta(minutes=args.recent_minutes)
    else:
        if not args.start_datetime or not args.end_datetime:
            raise ValueError("--start-datetime/--end-datetime 또는 --recent-minutes가 필요합니다.")
        start_dt = parse_utc(args.start_datetime)
        end_dt = parse_utc(args.end_datetime)
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
    if end_dt <= start_dt:
        raise ValueError("--end-datetime은 --start-datetime보다 뒤여야 합니다.")
    if args.window_minutes <= 0:
        raise ValueError("--window-minutes는 1 이상이어야 합니다.")
    if args.max_pages <= 0:
        raise ValueError("--max-pages는 1 이상이어야 합니다.")
    if args.max_total_pages <= 0:
        raise ValueError("--max-total-pages는 1 이상이어야 합니다.")

    market_symbol = exchange_symbol(args.symbol)
    print(f"aggTrade 수집 시작: {market_symbol} {start_dt.isoformat()} ~ {end_dt.isoformat()}")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    window_ms = args.window_minutes * MINUTE_MS
    expected_windows = (end_ms - start_ms + window_ms - 1) // window_ms

    def report_progress(completed_windows, total_pages, total_trades, _latest_window):
        if completed_windows % 8 == 0 or completed_windows == expected_windows:
            print(
                f"진행: windows={completed_windows}/{expected_windows}, "
                f"pages={total_pages:,}, trades={total_trades:,}",
                flush=True,
            )

    aggregates, window_stats, pages, total_trades = collect_trades(
        exchange,
        market_symbol,
        start_ms,
        end_ms,
        args.max_pages,
        window_minutes=args.window_minutes,
        max_total_pages=args.max_total_pages,
        progress_callback=report_progress,
    )
    context = build_context(aggregates, args.symbol, start_ms, end_ms)
    if context.empty:
        raise RuntimeError("집계된 체결 데이터가 없습니다.")
    validation = validate_context(context, start_ms, end_ms)
    output_path, marker_path = write_context(
        context,
        args,
        start_dt,
        end_dt,
        window_stats,
        pages,
        total_trades,
        validation,
    )
    print(
        f"API windows: {len(window_stats):,}, pages: {pages:,}, "
        f"aggregated trades: {total_trades:,}, minute rows: {len(context):,}, "
        f"empty trade minutes: {validation['empty_trade_minutes']:,}"
    )
    print(f"체결 컨텍스트 저장 완료: {output_path}")
    print(f"성공 마커: {marker_path}")


if __name__ == "__main__":
    main()
