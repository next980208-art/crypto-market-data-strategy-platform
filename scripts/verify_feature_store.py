"""Validate one partitioned Parquet Feature Store and write a JSON report."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


RUN_ID_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)_(?P<market>[A-Z0-9]+)_(?P<timeframe>[A-Za-z0-9]+)_(?P<start>\d{8})_(?P<end>\d{8})$"
)
REQUIRED_COLUMNS = {
    "timestamp",
    "datetime_utc",
    "symbol",
    "market",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "return_1m",
    "feature_schema_version",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a BTCUSDT-style Parquet Feature Store.")
    parser.add_argument("--feature-folder", default="feature_store_v2")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", default="usdm")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--expected-schema", default="ohlcv_basic_v2_boundary4")
    parser.add_argument("--report-dir", default="runtime_reports/feature_store_quality")
    parser.add_argument("--run-context", default="manual")
    parser.add_argument(
        "--chain-end-date",
        help="Optional UTC date (YYYY-MM-DD) used to validate one contiguous historical chain.",
    )
    return parser.parse_args()


def safe_value(value: str):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def timeframe_milliseconds(timeframe: str):
    match = re.fullmatch(r"(\d+)([mhd])", timeframe)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    amount = int(match.group(1))
    return amount * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[match.group(2)]


def feature_root(args):
    root = Path(args.feature_folder)
    if args.market.lower() != "spot":
        root = root / f"market={safe_value(args.market)}"
    return root / f"symbol={safe_value(args.symbol)}" / f"timeframe={safe_value(args.timeframe)}"


def read_matching_markers(args):
    marker_dir = Path(args.feature_folder) / "_markers"
    ranges = []
    errors = []
    if not marker_dir.exists():
        return ranges, [f"Success marker directory does not exist: {marker_dir}"]

    for path in sorted(marker_dir.glob("_SUCCESS_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("processor") != "apache_flink_pyflink_batch":
                continue
            match = RUN_ID_PATTERN.match(str(payload.get("run_id", "")))
            if not match:
                continue
            if (
                match.group("symbol") != safe_value(args.symbol).upper()
                or match.group("market").lower() != args.market.lower()
                or match.group("timeframe") != args.timeframe
            ):
                continue
            start = datetime.strptime(match.group("start"), "%Y%m%d").date()
            end = datetime.strptime(match.group("end"), "%Y%m%d").date()
            rows = int(payload.get("rows", -1))
            if start >= end or rows < 0:
                errors.append(f"Invalid marker: {path.name}")
                continue
            ranges.append({"path": str(path), "start": start, "end": end, "rows": rows})
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"Unreadable marker {path.name}: {error}")
    return ranges, errors


def marker_chain(ranges, chain_end_date=None):
    if not ranges:
        return None, None, [], ["No matching PyFlink success markers were found."]

    starts_by_end = {}
    for item in ranges:
        end = item["end"]
        if end in starts_by_end:
            return None, None, [], [f"Multiple markers end at {end.isoformat()}."]
        starts_by_end[end] = item

    if chain_end_date:
        try:
            latest_end = date.fromisoformat(chain_end_date)
        except ValueError:
            return None, None, [], [f"Invalid --chain-end-date: {chain_end_date}"]
        if latest_end not in starts_by_end:
            return None, None, [], [f"No success marker ends at requested chain end: {latest_end.isoformat()}"]
    else:
        latest_end = max(item["end"] for item in ranges)

    boundary = latest_end
    chain = []
    while boundary in starts_by_end:
        item = starts_by_end[boundary]
        chain.append(item)
        boundary = item["start"]

    chain_paths = {item["path"] for item in chain}
    if chain_end_date:
        orphan_paths = [
            item["path"]
            for item in ranges
            if item["path"] not in chain_paths and item["end"] <= latest_end
        ]
    else:
        orphan_paths = [item["path"] for item in ranges if item["path"] not in chain_paths]
    errors = []
    if orphan_paths:
        errors.append("Markers outside the latest contiguous chain: " + ", ".join(orphan_paths))
    return boundary, latest_end, chain, errors


def validate_files(args, root, range_start=None, range_end=None):
    errors = []
    parquet_paths = sorted(root.rglob("*.parquet")) if root.exists() else []
    timestamp_parts = []
    total_rows = 0
    null_counts = {column: 0 for column in REQUIRED_COLUMNS}

    for path in parquet_paths:
        try:
            frame = pd.read_parquet(path)
        except Exception as error:  # parquet engine errors should appear in the report.
            errors.append(f"Unreadable Parquet {path}: {error}")
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            errors.append(f"Missing columns in {path}: {sorted(missing)}")
            continue

        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
        if range_start and range_end:
            start_ms = int(datetime.combine(range_start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
            end_ms = int(datetime.combine(range_end, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
            frame = frame.loc[(timestamps >= start_ms) & (timestamps < end_ms)].copy()
        if frame.empty:
            continue

        total_rows += len(frame)
        for column in REQUIRED_COLUMNS:
            null_counts[column] += int(frame[column].isna().sum())
        normalized_symbols = {safe_value(value).upper() for value in frame["symbol"].dropna().astype(str)}
        if normalized_symbols != {safe_value(args.symbol).upper()}:
            errors.append(f"Unexpected symbol values in {path}")
        if set(frame["market"].astype(str).str.lower()) != {args.market.lower()}:
            errors.append(f"Unexpected market values in {path}")
        if set(frame["timeframe"].astype(str)) != {args.timeframe}:
            errors.append(f"Unexpected timeframe values in {path}")
        if set(frame["feature_schema_version"].astype(str)) != {args.expected_schema}:
            errors.append(f"Unexpected feature schema version in {path}")
        timestamp_parts.append(pd.to_numeric(frame["timestamp"], errors="coerce"))

    if not parquet_paths:
        errors.append(f"No Parquet files found under {root}")
    if any(value for value in null_counts.values()):
        errors.append("Required feature columns contain null values.")
    if not timestamp_parts:
        if range_start and range_end:
            errors.append("No Parquet rows found in the selected marker-chain range.")
        return parquet_paths, total_rows, null_counts, None, errors

    timestamps = pd.concat(timestamp_parts, ignore_index=True).dropna().astype("int64").sort_values()
    unique_timestamps = timestamps.drop_duplicates().reset_index(drop=True)
    step_ms = timeframe_milliseconds(args.timeframe)
    return parquet_paths, total_rows, null_counts, (timestamps, unique_timestamps, step_ms), errors


def build_report(args):
    root = feature_root(args)
    marker_ranges, errors = read_matching_markers(args)
    chain_start, chain_end, chain, chain_errors = marker_chain(marker_ranges, args.chain_end_date)
    errors.extend(chain_errors)
    parquet_paths, total_rows, null_counts, timestamp_data, file_errors = validate_files(
        args,
        root,
        chain_start,
        chain_end,
    )
    errors.extend(file_errors)

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_context": args.run_context,
        "feature_folder": args.feature_folder,
        "feature_root": str(root),
        "symbol": safe_value(args.symbol),
        "market": args.market.lower(),
        "timeframe": args.timeframe,
        "expected_schema": args.expected_schema,
        "requested_chain_end_date": args.chain_end_date,
        "parquet_files": len(parquet_paths),
        "parquet_rows": total_rows,
        "success_markers": len(marker_ranges),
        "contiguous_markers": len(chain),
        "required_column_null_counts": null_counts,
        "errors": errors,
    }

    if timestamp_data is not None:
        timestamps, unique_timestamps, step_ms = timestamp_data
        duplicate_count = len(timestamps) - len(unique_timestamps)
        gap_count = int((unique_timestamps.diff().dropna() != step_ms).sum())
        report.update(
            {
                "unique_timestamps": len(unique_timestamps),
                "duplicate_timestamps": duplicate_count,
                "time_gap_count": gap_count,
                "start_utc": pd.to_datetime(unique_timestamps.iloc[0], unit="ms", utc=True).isoformat(),
                "end_utc": pd.to_datetime(unique_timestamps.iloc[-1], unit="ms", utc=True).isoformat(),
            }
        )
        if duplicate_count:
            errors.append(f"Duplicate timestamps: {duplicate_count}")
        if gap_count:
            errors.append(f"Timestamp gaps larger than {args.timeframe}: {gap_count}")

        if chain_start and chain_end:
            expected_rows = int((datetime.combine(chain_end, datetime.min.time()) - datetime.combine(chain_start, datetime.min.time())).total_seconds() * 1000 / step_ms)
            marker_rows = sum(item["rows"] for item in chain)
            report["contiguous_range_start"] = chain_start.isoformat()
            report["contiguous_range_end"] = chain_end.isoformat()
            report["expected_rows_from_marker_range"] = expected_rows
            report["marker_rows"] = marker_rows
            if expected_rows != len(unique_timestamps):
                errors.append("Timestamp count does not match the contiguous marker range.")
            if marker_rows != total_rows:
                errors.append("Marker row total does not match Parquet row total.")

    report["healthy"] = not errors
    return report


def main():
    args = parse_args()
    report = build_report(args)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"feature_store_quality_{timestamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), **report}, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise RuntimeError(f"Feature Store quality validation failed. See {report_path}")


if __name__ == "__main__":
    main()
