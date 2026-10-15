import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from market_metadata import (
    METADATA_SCHEMA_VERSION,
    TIMESTAMP_UNIT,
    normalize_market,
    normalize_symbol,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "timestamp",
    "datetime_utc",
    "symbol",
    "market",
    "timeframe",
    "run_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "return_1m",
]
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
FEATURE_LOOKBACK_ROWS = 4
FEATURE_SCHEMA_VERSION = "ohlcv_basic_v2_boundary4"


def parse_args():
    parser = argparse.ArgumentParser(description="Submit and finalize one PyFlink batch feature job.")
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--feature-folder", default="feature_store_v2")
    parser.add_argument(
        "--staging-root",
        default="/opt/flink/staging",
        help="Docker shared volume used only for temporary Flink output.",
    )
    parser.add_argument("--jobmanager-host", default="jobmanager")
    parser.add_argument("--jobmanager-port", type=int, default=8081)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument(
        "--reset-stale-staging",
        action="store_true",
        help="Replace this run's unfinished staging directory before submitting a new job.",
    )
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def apply_canonical_feature_metadata(frame):
    target = frame.copy()
    target["symbol"] = target["symbol"].map(normalize_symbol)
    target["market"] = target["market"].map(normalize_market)
    target["event_time_ms"] = pd.to_numeric(target["timestamp"], errors="raise").astype("int64")
    target["timestamp_unit"] = TIMESTAMP_UNIT
    target["metadata_schema_version"] = METADATA_SCHEMA_VERSION
    return target


def timeframe_milliseconds(timeframe):
    match = re.fullmatch(r"(\d+)([mhd])", str(timeframe))
    if not match:
        return None
    value = int(match.group(1))
    unit_ms = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return value * unit_ms[match.group(2)]


def feature_partition_root(feature_folder, symbol, market, timeframe):
    root = Path(feature_folder)
    if market != "spot":
        root = root / f"market={safe_value(market)}"
    return root / f"symbol={safe_value(symbol)}" / f"timeframe={safe_value(timeframe)}"


def load_boundary_context(feature_folder, raw):
    metadata = {}
    for column in ["run_id", "symbol", "market", "timeframe"]:
        values = raw[column].dropna().astype(str).unique()
        if len(values) != 1:
            raise RuntimeError(f"Raw input must contain exactly one {column}: {values.tolist()}")
        metadata[column] = values[0]

    step_ms = timeframe_milliseconds(metadata["timeframe"])
    if step_ms is None:
        return pd.DataFrame(columns=RAW_COLUMNS), "unsupported_timeframe"

    target_start = int(pd.to_numeric(raw["timestamp"], errors="raise").min())
    root = feature_partition_root(
        feature_folder,
        metadata["symbol"],
        metadata["market"],
        metadata["timeframe"],
    )
    paths = sorted(root.rglob("*.parquet"), reverse=True) if root.exists() else []
    candidates = []
    base_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    for path in paths:
        frame = pd.read_parquet(path)
        required_columns = base_columns + ["feature_schema_version"]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise RuntimeError(
                f"Prior feature file has no compatible schema version: {path}, "
                f"missing={missing_columns}"
            )
        if set(frame["feature_schema_version"].astype(str)) != {FEATURE_SCHEMA_VERSION}:
            raise RuntimeError(f"Prior feature schema version mismatch: {path}")
        frame = frame[base_columns]
        frame = frame[pd.to_numeric(frame["timestamp"], errors="coerce") < target_start]
        if not frame.empty:
            candidates.append(frame.tail(FEATURE_LOOKBACK_ROWS))
        if sum(len(item) for item in candidates) >= FEATURE_LOOKBACK_ROWS * 2:
            break

    if not candidates:
        return pd.DataFrame(columns=RAW_COLUMNS), "unavailable"

    context = (
        pd.concat(candidates, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )
    context = context[context["timestamp"] < target_start].tail(FEATURE_LOOKBACK_ROWS)
    expected_timestamp = target_start - step_ms
    contiguous_rows = []
    for _, row in context.sort_values("timestamp", ascending=False).iterrows():
        if int(row["timestamp"]) != expected_timestamp:
            break
        contiguous_rows.append(row)
        expected_timestamp -= step_ms

    if not contiguous_rows:
        return pd.DataFrame(columns=RAW_COLUMNS), "unavailable_or_gap"

    context = pd.DataFrame(reversed(contiguous_rows))
    context["timestamp"] = pd.to_numeric(context["timestamp"], errors="raise").astype("int64")
    for column, value in metadata.items():
        context[column] = value
    status = "applied" if len(context) == FEATURE_LOOKBACK_ROWS else "partial"
    return context[RAW_COLUMNS], f"{status}_{len(context)}_rows"


def prepare_flink_input(args, raw_path, staging_dir):
    raw = pd.read_csv(raw_path, header=None, names=RAW_COLUMNS)
    if raw.empty:
        raise RuntimeError(f"Raw input is empty: {raw_path}")
    raw["symbol"] = raw["symbol"].map(normalize_symbol)
    raw["market"] = raw["market"].map(normalize_market)

    context, context_status = load_boundary_context(args.feature_folder, raw)
    input_frames = [raw] if context.empty else [context, raw]
    combined = (
        pd.concat(input_frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined["timestamp"] = pd.to_numeric(combined["timestamp"], errors="raise").astype("int64")
    flink_input_path = staging_dir / "input_with_boundary_context.csv"
    combined.to_csv(flink_input_path, index=False, header=False)
    return raw, combined, flink_input_path, len(context), context_status


def submit_job(args, cluster_raw_file, cluster_staging_dir):
    job_file = PROJECT_ROOT / "flink_jobs" / "batch_feature_job.py"
    flink_binary = Path("/opt/flink/bin/flink")
    if not flink_binary.exists():
        raise RuntimeError("Flink CLI is unavailable. Run this submitter from the Airflow container.")
    command = [
        str(flink_binary),
        "run",
        "--detached",
        "--jobmanager",
        f"{args.jobmanager_host}:{args.jobmanager_port}",
        "--python",
        str(job_file),
        "--",
        "--raw-file",
        cluster_raw_file,
        "--output-dir",
        cluster_staging_dir,
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Flink submission failed:\n{completed.stdout}\n{completed.stderr}")
    match = re.search(r"JobID\s+([0-9a-fA-F]{32})", completed.stdout + completed.stderr)
    if not match:
        raise RuntimeError(f"Flink JobID was not found in submission output:\n{completed.stdout}\n{completed.stderr}")
    return match.group(1), command


def wait_for_job(args, job_id):
    endpoint = f"http://{args.jobmanager_host}:{args.jobmanager_port}/jobs/{job_id}"
    deadline = time.monotonic() + args.timeout_seconds
    last_state = None
    while time.monotonic() < deadline:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        last_state = response.json().get("state")
        if last_state == "FINISHED":
            return
        if last_state in {"FAILED", "CANCELED", "SUSPENDED"}:
            raise RuntimeError(f"Flink job {job_id} ended with state={last_state}")
        time.sleep(3)
    raise TimeoutError(f"Flink job {job_id} did not finish within {args.timeout_seconds}s; last_state={last_state}")


def write_final_partitions(frame, feature_folder, run_id):
    symbol = str(frame["symbol"].iloc[0])
    market = str(frame["market"].iloc[0])
    timeframe = str(frame["timeframe"].iloc[0])
    root = Path(feature_folder)
    if market != "spot":
        root = root / f"market={safe_value(market)}"
    output_files = []
    for period, partition in frame.groupby(frame["datetime_utc"].dt.strftime("%Y-%m")):
        year, month = period.split("-")
        output_dir = root / f"symbol={safe_value(symbol)}" / f"timeframe={safe_value(timeframe)}" / f"year={year}" / f"month={month}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"features_{run_id}_{year}{month}.parquet"
        partition.to_parquet(output_path, index=False)
        verified = pd.read_parquet(output_path)
        if len(verified) != len(partition) or verified[["timestamp", "close", "ma_5"]].isna().any().any():
            raise RuntimeError(f"Final Feature Store verification failed: {output_path}")
        if set(verified["feature_schema_version"].astype(str)) != {FEATURE_SCHEMA_VERSION}:
            raise RuntimeError(f"Feature schema version verification failed: {output_path}")
        output_files.append(str(output_path))
    return output_files


def verify_feature_values(staged, flink_input, target_timestamps):
    valid = flink_input.copy()
    for column in ["timestamp", "open", "high", "low", "close", "volume"]:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    valid = valid[
        (valid["open"] > 0)
        & (valid["high"] > 0)
        & (valid["low"] > 0)
        & (valid["close"] > 0)
        & (valid["volume"] >= 0)
    ]
    valid = valid.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    valid["expected_ma_5"] = valid["close"].rolling(5, min_periods=1).mean()
    valid["expected_return_1m"] = valid["close"].pct_change().fillna(0.0)
    expected = valid[valid["timestamp"].isin(target_timestamps)][
        ["timestamp", "expected_ma_5", "expected_return_1m"]
    ]
    compared = staged.merge(expected, on="timestamp", how="inner", validate="one_to_one")
    if len(compared) != len(target_timestamps):
        raise RuntimeError("Flink feature-value verification could not match every target timestamp.")
    if not np.allclose(compared["ma_5"], compared["expected_ma_5"], rtol=1e-9, atol=1e-9):
        raise RuntimeError("Flink ma_5 verification failed, including the chunk boundary.")
    if not np.allclose(
        compared["return_1m"],
        compared["expected_return_1m"],
        rtol=1e-9,
        atol=1e-9,
    ):
        raise RuntimeError("Flink return_1m verification failed, including the chunk boundary.")


def finalize(
    args,
    raw_path,
    raw,
    flink_input,
    staging_dir,
    output_dir,
    context_rows,
    context_status,
    job_id,
    command,
):
    staged_paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name.startswith("part-"))
    if not staged_paths:
        raise RuntimeError(f"Flink job finished without staged part files: {staging_dir}")
    staged = pd.concat(
        [pd.read_csv(path, header=None, names=FEATURE_COLUMNS) for path in staged_paths],
        ignore_index=True,
    )
    staged["datetime_utc"] = pd.to_datetime(staged["datetime_utc"], utc=True)
    staged = staged[FEATURE_COLUMNS].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    expected = raw[(raw["open"] > 0) & (raw["high"] > 0) & (raw["low"] > 0) & (raw["close"] > 0) & (raw["volume"] >= 0)]
    target_timestamps = set(expected["timestamp"])
    target = staged[staged["timestamp"].isin(target_timestamps)].reset_index(drop=True)
    if len(target) != len(expected) or set(target["timestamp"]) != target_timestamps:
        raise RuntimeError(f"Flink output row verification failed: expected={len(expected)}, actual={len(target)}")
    if target[["timestamp", "close", "ma_5", "return_1m"]].isna().any().any():
        raise RuntimeError("Flink output contains missing required feature values.")
    verify_feature_values(target, flink_input, target_timestamps)
    target["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    target = apply_canonical_feature_metadata(target)

    run_id = str(raw["run_id"].iloc[0])
    output_files = write_final_partitions(target, args.feature_folder, run_id)
    marker_dir = Path(args.feature_folder) / "_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"_SUCCESS_{run_id}.json"
    marker_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "processor": "apache_flink_pyflink_batch",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "timestamp_field": "event_time_ms",
                "timestamp_unit": TIMESTAMP_UNIT,
                "job_id": job_id,
                "rows": len(target),
                "boundary_context_rows": context_rows,
                "boundary_context_status": context_status,
                "feature_files": output_files,
                "raw_file": str(raw_path),
                "flink_command": command,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.keep_raw:
        raw_path.unlink()
    shutil.rmtree(staging_dir)
    return marker_path, len(target), output_files


def main():
    args = parse_args()
    raw_path = Path(args.raw_file).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw input does not exist: {raw_path}")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    run_id = str(pd.read_csv(raw_path, header=None, names=RAW_COLUMNS, nrows=1)["run_id"].iloc[0])
    staging_dir = Path(args.staging_root).resolve() / run_id
    if staging_dir.exists():
        if not args.reset_stale_staging:
            raise RuntimeError(
                f"Staging directory already exists; inspect it before retrying: {staging_dir}"
            )
        print(f"중단된 Flink staging 재설정: {staging_dir}")
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    staging_dir.chmod(0o777)
    raw, flink_input, flink_input_path, context_rows, context_status = prepare_flink_input(
        args,
        raw_path,
        staging_dir,
    )
    output_dir = staging_dir / "output"

    job_id, command = submit_job(
        args,
        str(flink_input_path),
        str(output_dir),
    )
    wait_for_job(args, job_id)
    marker_path, rows, output_files = finalize(
        args,
        raw_path,
        raw,
        flink_input,
        staging_dir,
        output_dir,
        context_rows,
        context_status,
        job_id,
        command,
    )
    print(f"Flink batch completed: job_id={job_id}, rows={rows}")
    print(f"Boundary context: {context_status}")
    print(f"Feature files: {output_files}")
    print(f"Success marker: {marker_path}")


if __name__ == "__main__":
    main()
