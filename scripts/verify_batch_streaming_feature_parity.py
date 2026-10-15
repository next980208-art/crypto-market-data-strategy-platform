"""Compare actual PyFlink batch output with realtime feature calculations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flink_jobs.realtime_kafka_feature_job import OneMinuteFeatureFunction, apply_event, fresh_state


RAW_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "run_id",
    "symbol",
    "market",
    "timeframe",
]
PARITY_COLUMNS = ["open", "high", "low", "close", "volume", "ma_5", "return_1m"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify offline/online feature parity on fixed OHLCV input.")
    parser.add_argument(
        "--raw-file",
        help="Optional headerless OHLCV CSV. If omitted, the batch Parquet OHLCV columns are fixed input.",
    )
    parser.add_argument("--batch-parquet", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--output", default="runtime_reports/batch_streaming_feature_parity_latest.json")
    return parser.parse_args()


def trade_event(event_time_ms: int, price: float, quantity: float, event_id: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": "agg_trade",
        "event_time_ms": event_time_ms,
        "payload": {
            "price": str(price),
            "quantity": str(quantity),
            "buyer_is_maker": False,
        },
    }


def realtime_features_from_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    closes: deque[float] = deque(maxlen=5)
    features: list[dict[str, Any]] = []
    for row in raw.sort_values("timestamp").itertuples(index=False):
        minute = int(row.timestamp)
        state = fresh_state(minute)
        quantities = [float(row.volume) / 4.0] * 4
        prices = [float(row.open), float(row.high), float(row.low), float(row.close)]
        offsets = [1_000, 15_000, 30_000, 50_000]
        for index, (offset, price, quantity) in enumerate(zip(offsets, prices, quantities)):
            apply_event(
                state,
                trade_event(minute + offset, price, quantity, f"{minute}:{index}"),
            )
        state["closes"] = list(closes)
        feature = OneMinuteFeatureFunction.finish_feature(symbol, state)
        if feature is None:
            raise RuntimeError(f"Realtime feature was not created for minute={minute}")
        features.append(feature)
        closes.append(float(row.close))
    return pd.DataFrame(features)


def compare_frames(
    batch: pd.DataFrame,
    realtime: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    ready = realtime[realtime["feature_ready"]].copy()
    compared = batch.merge(
        ready[["event_time_ms", *PARITY_COLUMNS]],
        left_on="timestamp",
        right_on="event_time_ms",
        how="inner",
        suffixes=("_batch", "_realtime"),
        validate="one_to_one",
    )
    mismatches: list[dict[str, Any]] = []
    for row in compared.itertuples(index=False):
        timestamp = int(row.timestamp)
        for column in PARITY_COLUMNS:
            batch_value = float(getattr(row, f"{column}_batch"))
            realtime_value = float(getattr(row, f"{column}_realtime"))
            difference = abs(batch_value - realtime_value)
            tolerance = atol + rtol * abs(realtime_value)
            if difference > tolerance:
                mismatches.append(
                    {
                        "timestamp": timestamp,
                        "column": column,
                        "batch": batch_value,
                        "realtime": realtime_value,
                        "absolute_difference": difference,
                        "tolerance": tolerance,
                    }
                )
    expected_ready_rows = int(ready.shape[0])
    return {
        "batch_rows": int(batch.shape[0]),
        "realtime_rows": int(realtime.shape[0]),
        "warmup_rows": int((~realtime["feature_ready"]).sum()),
        "expected_ready_rows": expected_ready_rows,
        "compared_rows": int(compared.shape[0]),
        "compared_columns": PARITY_COLUMNS,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "healthy": expected_ready_rows > 0 and len(compared) == expected_ready_rows and not mismatches,
    }


def main() -> None:
    args = parse_args()
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("Tolerances must not be negative.")
    batch_path = Path(args.batch_parquet)
    batch = pd.read_parquet(batch_path)
    if args.raw_file:
        raw_path = Path(args.raw_file)
        raw = pd.read_csv(raw_path, header=None, names=RAW_COLUMNS)
        raw_source = str(raw_path)
    else:
        required_raw = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [column for column in required_raw if column not in batch.columns]
        if missing:
            raise RuntimeError(f"Batch Parquet cannot provide fixed OHLCV input; missing={missing}")
        raw = batch[required_raw].copy()
        raw_source = "batch_parquet_ohlcv_columns"
    realtime = realtime_features_from_ohlcv(raw, args.symbol)
    report = compare_frames(batch, realtime, rtol=args.rtol, atol=args.atol)
    report.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_file": raw_source,
            "batch_parquet": str(batch_path),
            "symbol": args.symbol,
            "rtol": args.rtol,
            "atol": args.atol,
            "warmup_policy": "Compare only realtime feature_ready=true rows after five closes.",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
