import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from market_metadata import (
    METADATA_SCHEMA_VERSION,
    TIMESTAMP_UNIT,
    normalize_market,
    normalize_symbol,
    to_ccxt_symbol,
)


COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download one historical OHLCV chunk.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument(
        "--market",
        choices=["spot", "usdm"],
        default="spot",
        help="spot for Binance Spot, usdm for Binance USDT-M perpetual futures.",
    )
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start-date", default="2024-01-01", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--temp-folder", default="temp_raw_data")
    parser.add_argument("--topic", default="raw-market-data")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--no-kafka", action="store_true", help="Only save the raw chunk locally.")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Write CSV data without a header row for the PyFlink batch source.",
    )
    return parser.parse_args()


def utc_midnight(date_text):
    return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def safe_symbol(symbol):
    return normalize_symbol(symbol)


def make_run_id(symbol, market, timeframe, start_dt, end_dt):
    market_suffix = "" if market == "spot" else f"_{market.upper()}"
    return f"{safe_symbol(symbol)}{market_suffix}_{timeframe}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}"


def exchange_symbol(symbol, market):
    return to_ccxt_symbol(symbol, market)


def create_exchange(market):
    if market == "usdm":
        return ccxt.binanceusdm({"enableRateLimit": True})
    return ccxt.binance({"enableRateLimit": True})


def fetch_ohlcv_chunk(exchange, symbol, timeframe, start_dt, end_dt):
    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    rows = []
    seen_timestamps = set()

    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not batch:
            break

        progressed = False
        for row in batch:
            timestamp = int(row[0])
            if timestamp >= end_ms:
                continue
            if timestamp not in seen_timestamps:
                rows.append(row)
                seen_timestamps.add(timestamp)
            if timestamp >= since_ms:
                since_ms = timestamp + 1
                progressed = True

        if not progressed:
            break

        time.sleep(exchange.rateLimit / 1000)

    return sorted(rows, key=lambda item: item[0])


def make_producer(bootstrap_servers):
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=[server.strip() for server in bootstrap_servers.split(",")],
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
    )


def send_to_kafka(records, args, run_id):
    producer = make_producer(args.bootstrap_servers)
    for record in records:
        payload = dict(record)
        payload["event_time_ms"] = int(payload["timestamp"])
        payload["timestamp_unit"] = TIMESTAMP_UNIT
        payload["metadata_schema_version"] = METADATA_SCHEMA_VERSION
        payload["run_id"] = run_id
        payload["symbol"] = normalize_symbol(args.symbol)
        payload["market"] = normalize_market(args.market)
        payload["timeframe"] = args.timeframe
        producer.send(args.topic, value=payload)
    producer.flush()
    producer.close()


def main():
    args = parse_args()
    start_dt = utc_midnight(args.start_date)
    end_dt = start_dt + timedelta(days=args.days)
    run_id = make_run_id(args.symbol, args.market, args.timeframe, start_dt, end_dt)
    requested_symbol = exchange_symbol(args.symbol, args.market)

    os.makedirs(args.temp_folder, exist_ok=True)

    print(
        f"청크 수집 시작: market={args.market}, symbol={args.symbol}, "
        f"exchange_symbol={requested_symbol}, {start_dt:%Y-%m-%d} ~ {end_dt:%Y-%m-%d}"
    )

    exchange = create_exchange(args.market)
    ohlcv = fetch_ohlcv_chunk(exchange, requested_symbol, args.timeframe, start_dt, end_dt)
    if not ohlcv:
        raise RuntimeError("거래소에서 OHLCV 데이터를 가져오지 못했습니다.")

    df = pd.DataFrame(ohlcv, columns=COLUMNS)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= int(start_dt.timestamp() * 1000)) & (df["timestamp"] < int(end_dt.timestamp() * 1000))]
    step_ms = int(exchange.parse_timeframe(args.timeframe) * 1000)
    if step_ms <= 0:
        raise RuntimeError(f"지원하지 않는 timeframe입니다: {args.timeframe}")
    expected_timestamps = set(
        range(int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000), step_ms)
    )
    actual_timestamps = set(pd.to_numeric(df["timestamp"], errors="raise").astype("int64"))
    if actual_timestamps != expected_timestamps:
        missing = sorted(expected_timestamps - actual_timestamps)
        unexpected = sorted(actual_timestamps - expected_timestamps)
        raise RuntimeError(
            "OHLCV 구간 완전성 검사 실패: "
            f"expected={len(expected_timestamps)}, actual={len(actual_timestamps)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"first_missing={missing[:3]}"
        )
    df["run_id"] = run_id
    df["symbol"] = normalize_symbol(args.symbol)
    df["market"] = normalize_market(args.market)
    df["timeframe"] = args.timeframe

    raw_path = os.path.join(args.temp_folder, f"raw_{run_id}.csv")
    df.to_csv(raw_path, index=False, header=not args.no_header)
    print(f"원본 청크 저장 완료: {raw_path} ({len(df):,} rows)")

    if args.no_kafka:
        print(
            "Kafka 전송 건너뜀 (--no-kafka). 실제 배치는 flink_batch_submitter.py, "
            "로컬 호환 시험은 2_flink_processor.py를 사용합니다."
        )
    else:
        print(f"Kafka 전송 시작: topic={args.topic}")
        send_to_kafka(df[COLUMNS].to_dict("records"), args, run_id)
        print("Kafka 전송 완료")

    print("수집 완료. 원본 삭제는 Feature Store 저장 검증 후 처리기가 수행합니다.")


if __name__ == "__main__":
    main()
