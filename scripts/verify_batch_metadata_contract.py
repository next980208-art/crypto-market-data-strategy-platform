"""Verify canonical metadata columns in a finalized batch Parquet and marker."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_metadata import METADATA_SCHEMA_VERSION, TIMESTAMP_UNIT, normalize_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify one finalized batch metadata contract.")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--output", default="runtime_reports/batch_metadata_contract_latest.json")
    return parser.parse_args()


def evaluate_batch_contract(frame: pd.DataFrame, marker: dict[str, Any], symbol: str) -> dict[str, Any]:
    expected_symbol = normalize_symbol(symbol)
    required_columns = {
        "timestamp",
        "event_time_ms",
        "timestamp_unit",
        "metadata_schema_version",
        "symbol",
        "market",
        "timeframe",
        "feature_schema_version",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    timestamp_mismatches = (
        int((frame["timestamp"].astype("int64") != frame["event_time_ms"].astype("int64")).sum())
        if not missing_columns and not frame.empty
        else None
    )
    field_checks = {
        "symbol": not frame.empty and set(frame.get("symbol", pd.Series(dtype=str)).astype(str)) == {expected_symbol},
        "market": not frame.empty and set(frame.get("market", pd.Series(dtype=str)).astype(str)) == {"usdm"},
        "timeframe": not frame.empty and set(frame.get("timeframe", pd.Series(dtype=str)).astype(str)) == {"1m"},
        "metadata_schema_version": not frame.empty
        and set(frame.get("metadata_schema_version", pd.Series(dtype=str)).astype(str)) == {METADATA_SCHEMA_VERSION},
        "timestamp_unit": not frame.empty
        and set(frame.get("timestamp_unit", pd.Series(dtype=str)).astype(str)) == {TIMESTAMP_UNIT},
    }
    marker_checks = {
        "metadata_schema_version": marker.get("metadata_schema_version") == METADATA_SCHEMA_VERSION,
        "timestamp_field": marker.get("timestamp_field") == "event_time_ms",
        "timestamp_unit": marker.get("timestamp_unit") == TIMESTAMP_UNIT,
        "rows": int(marker.get("rows", -1)) == len(frame),
    }
    healthy = (
        not frame.empty
        and not missing_columns
        and timestamp_mismatches == 0
        and all(field_checks.values())
        and all(marker_checks.values())
    )
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing_columns": missing_columns,
        "timestamp_alias_mismatches": timestamp_mismatches,
        "field_checks": field_checks,
        "marker_checks": marker_checks,
        "flink_job_id": marker.get("job_id"),
        "feature_schema_version": marker.get("feature_schema_version"),
        "metadata_schema_version": marker.get("metadata_schema_version"),
        "healthy": healthy,
    }


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet)
    marker_path = Path(args.marker)
    frame = pd.read_parquet(parquet_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    report = evaluate_batch_contract(frame, marker, args.symbol)
    report.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "parquet": str(parquet_path),
            "marker": str(marker_path),
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
