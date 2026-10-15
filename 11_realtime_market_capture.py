import argparse
import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
import websockets

from market_metadata import (
    METADATA_SCHEMA_VERSION,
    TIMESTAMP_UNIT,
    normalize_symbol,
    to_ccxt_symbol,
)


LIVE_CAPTURE_SCHEMA_VERSION = "live_capture_v2_partial_window"


def parse_args():
    parser = argparse.ArgumentParser(description="Capture live Binance USDT-M trades, book data, and open interest into 1-minute features.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--output-folder", default="live_context_store_v2")
    return parser.parse_args()


def safe_value(value):
    return normalize_symbol(value)


def bucket():
    return {
        "trade_count": 0,
        "trade_volume": 0.0,
        "taker_buy_volume": 0.0,
        "taker_sell_volume": 0.0,
        "book_mid": None,
        "book_spread": None,
        "book_imbalance_top5": None,
        "open_interest": None,
        "open_interest_value": None,
        "open_interest_timestamp": None,
    }


def minute_of(timestamp):
    return int(timestamp) - (int(timestamp) % 60_000)


def fetch_open_interest(symbol):
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    market_symbol = to_ccxt_symbol(symbol, "usdm")
    data = exchange.fetch_open_interest(market_symbol)
    return {
        "open_interest": pd.to_numeric(data.get("openInterestAmount"), errors="coerce"),
        "open_interest_value": pd.to_numeric(data.get("openInterestValue"), errors="coerce"),
        "open_interest_timestamp": int(data.get("timestamp") or exchange.milliseconds()),
    }


def apply_open_interest(records, value):
    for record in records.values():
        record.update(value)


async def capture(args):
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds는 1 이상이어야 합니다.")
    stream_symbol = safe_value(args.symbol).lower()
    url = (
        "wss://fstream.binance.com/stream?streams="
        f"{stream_symbol}@aggTrade/{stream_symbol}@bookTicker/{stream_symbol}@depth5@100ms"
    )
    records = defaultdict(bucket)
    started = time.monotonic()
    try:
        latest_oi = await asyncio.to_thread(fetch_open_interest, args.symbol)
    except ccxt.BaseError as error:
        print(f"경고: open interest 초기 조회 실패: {error}")
        latest_oi = None

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as socket:
        while time.monotonic() - started < args.duration_seconds:
            remaining = max(1, args.duration_seconds - (time.monotonic() - started))
            try:
                message = json.loads(await asyncio.wait_for(socket.recv(), timeout=remaining))
            except asyncio.TimeoutError:
                break
            data = message.get("data", message)
            event_type = data.get("e")
            event_time = int(data.get("E") or data.get("T") or int(time.time() * 1000))
            record = records[minute_of(event_time)]
            if latest_oi:
                record.update(latest_oi)

            if event_type == "aggTrade":
                amount = float(data["q"])
                record["trade_count"] += 1
                record["trade_volume"] += amount
                if data.get("m"):
                    record["taker_sell_volume"] += amount
                else:
                    record["taker_buy_volume"] += amount
            elif event_type == "bookTicker":
                bid = float(data["b"])
                ask = float(data["a"])
                record["book_mid"] = (bid + ask) / 2
                record["book_spread"] = ask - bid
            elif event_type == "depthUpdate":
                bid_qty = sum(float(level[1]) for level in data.get("b", [])[:5])
                ask_qty = sum(float(level[1]) for level in data.get("a", [])[:5])
                total = bid_qty + ask_qty
                record["book_imbalance_top5"] = (bid_qty - ask_qty) / total if total else 0.0

    if not records:
        raise RuntimeError("실시간 메시지를 받지 못했습니다.")
    return records


def write_records(records, args):
    rows = []
    for timestamp, record in sorted(records.items()):
        total = record["trade_volume"]
        rows.append(
            {
                "timestamp": timestamp,
                "event_time_ms": timestamp,
                "timestamp_unit": TIMESTAMP_UNIT,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "datetime_utc": pd.to_datetime(timestamp, unit="ms", utc=True),
                "symbol": normalize_symbol(args.symbol),
                "market": "usdm",
                "timeframe": "1m",
                "context_schema_version": LIVE_CAPTURE_SCHEMA_VERSION,
                **record,
                "taker_volume_imbalance": (record["taker_buy_volume"] - record["taker_sell_volume"]) / total if total else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    started = pd.to_datetime(frame["timestamp"].min(), unit="ms", utc=True)
    ended = pd.to_datetime(frame["timestamp"].max(), unit="ms", utc=True)
    captured_at = datetime.now(timezone.utc)
    run_id = (
        f"{safe_value(args.symbol)}_USDM_LIVE_{started:%Y%m%dT%H%M}_"
        f"{ended:%Y%m%dT%H%M}_{captured_at:%Y%m%dT%H%M%S%f}"
    )
    output_dir = Path(args.output_folder) / "market=usdm" / f"symbol={safe_value(args.symbol)}" / "timeframe=1m"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"live_context_{run_id}.parquet"
    frame.to_parquet(output_path, index=False)
    verified = pd.read_parquet(output_path)
    if len(verified) != len(frame):
        raise RuntimeError(
            f"실시간 캡처 저장 검증 실패: expected={len(frame)}, actual={len(verified)}"
        )
    marker_dir = Path(args.output_folder) / "_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"_SUCCESS_{run_id}.json"
    marker = {
        "run_id": run_id,
        "rows": len(frame),
        "context_schema_version": LIVE_CAPTURE_SCHEMA_VERSION,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "timestamp_field": "event_time_ms",
        "timestamp_unit": TIMESTAMP_UNIT,
        "capture_status": "partial_window_test",
        "file": str(output_path),
        "saved_at_utc": captured_at.isoformat(),
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, marker_path, len(frame)


def main():
    args = parse_args()
    records = asyncio.run(capture(args))
    output_path, marker_path, rows = write_records(records, args)
    print(f"실시간 1분 피처 저장 완료: {output_path} ({rows} rows)")
    print(f"성공 마커: {marker_path}")


if __name__ == "__main__":
    main()
