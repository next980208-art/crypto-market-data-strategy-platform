"""Convert Kafka-consumed JSONL events into the CSV schema used by the PyFlink batch job."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kafka event JSONL for the existing PyFlink job")
    parser.add_argument(
        "--input-path",
        default="examples/kafka_flink/data/consumed_market_events.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="examples/kafka_flink/data/flink_input.csv",
    )
    parser.add_argument(
        "--report-path",
        default="examples/kafka_flink/results/flink_input_report.json",
    )
    return parser.parse_args()


def event_timestamp_ms(value: object) -> int:
    text = str(value).replace("Z", "+00:00")
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def is_valid_ohlcv(event: dict[str, object]) -> bool:
    try:
        open_price = float(event["open"])
        high_price = float(event["high"])
        low_price = float(event["low"])
        close_price = float(event["close"])
        volume = float(event["volume"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        event.get("symbol") == "BTCUSDT"
        and event.get("market") == "USDT-M"
        and open_price > 0
        and high_price >= max(open_price, close_price)
        and low_price <= min(open_price, close_price)
        and low_price > 0
        and volume >= 0
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    report_path = Path(args.report_path)

    events: list[dict[str, object]] = []
    seen_event_ids: set[str] = set()
    raw_count = 0
    invalid_count = 0
    duplicate_count = 0

    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            raw_count += 1
            event = json.loads(line)
            event_id = str(event.get("event_id", ""))
            if not event_id or event_id in seen_event_ids:
                duplicate_count += 1
                continue
            seen_event_ids.add(event_id)
            if not is_valid_ohlcv(event):
                invalid_count += 1
                continue
            events.append(event)

    if not events:
        raise RuntimeError("Flink input preparation produced no valid events.")

    run_ids = {str(event["run_id"]) for event in events}
    if len(run_ids) != 1:
        raise RuntimeError(f"One Flink batch must use one run_id: {sorted(run_ids)}")

    events.sort(key=lambda event: (event_timestamp_ms(event["event_time"]), int(event["event_sequence"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for event in events:
            row = [
                event_timestamp_ms(event["event_time"]),
                float(event["open"]),
                float(event["high"]),
                float(event["low"]),
                float(event["close"]),
                float(event["volume"]),
                event["run_id"],
                event["symbol"],
                event["market"],
                "1m",
            ]
            output_file.write(",".join(str(value) for value in row) + "\n")

    report = {
        "status": "success",
        "raw_jsonl_count": raw_count,
        "valid_flink_input_count": len(events),
        "invalid_ohlcv_count": invalid_count,
        "duplicate_event_id_count": duplicate_count,
        "run_id": next(iter(run_ids)),
        "output_csv_path": str(output_path),
        "flink_input_columns": RAW_COLUMNS,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
