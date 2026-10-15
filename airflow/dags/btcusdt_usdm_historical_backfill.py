"""Scheduled, checkpoint-based historical BTCUSDT USDT-M backfill DAG."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")
RUN_ID_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)_(?P<market>[A-Z0-9]+)_(?P<timeframe>[A-Za-z0-9]+)_(?P<start>\d{8})_(?P<end>\d{8})$"
)
CONTEXT_RUN_ID_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)_(?P<market>[A-Z0-9]+)_CONTEXT_(?P<timeframe>[A-Za-z0-9]+)_(?P<start>\d{8})_(?P<end>\d{8})$"
)
RUNTIME_PARAM_KEYS = {
    "symbol",
    "market",
    "timeframe",
    "feature_folder",
    "context_folder",
    "temp_folder",
    "target_start_date",
    "initial_end_date",
    "chunk_days",
    "chunks_per_trigger",
    "start_date",
    "end_date",
}


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def get_runtime_params(context):
    """Combine DAG defaults with explicit Trigger DAG JSON inputs.

    Airflow installations can be configured either to merge dag_run.conf into
    params or to keep them separate. Reading dag_run.conf here makes the
    assignment inputs reliable in both configurations.
    """
    params = dict(context["params"])
    dag_run = context.get("dag_run")
    run_conf = getattr(dag_run, "conf", None) or {}
    if not isinstance(run_conf, dict):
        raise ValueError("DAG 실행 입력값은 JSON 객체여야 합니다.")
    for key, value in run_conf.items():
        if key in RUNTIME_PARAM_KEYS:
            params[key] = value

    symbol = str(params["symbol"]).upper().strip()
    if not re.fullmatch(r"[A-Z0-9]+/[A-Z0-9]+", symbol):
        raise ValueError("symbol은 BTC/USDT 또는 ETH/USDT 형식이어야 합니다.")
    if str(params["market"]).lower() != "usdm":
        raise ValueError("이 DAG는 Binance USDT-M 선물 시장(usdm)만 지원합니다.")
    params["symbol"] = symbol
    params["market"] = "usdm"
    return params


def find_contiguous_completed_start(
    feature_folder: str,
    symbol: str,
    market: str,
    timeframe: str,
    initial_end,
):
    """Return the oldest boundary in the contiguous completed range.

    A failed multi-chunk run can leave an older successful marker behind a gap.
    We must never use that isolated marker as the next boundary, otherwise later
    scheduled runs would move further into the past and permanently skip the gap.
    """
    marker_dir = PROJECT_ROOT / feature_folder / "_markers"
    symbol_code = symbol.replace("/", "").replace(":", "")
    starts_by_end = {}
    if not marker_dir.exists():
        return initial_end
    for marker_path in marker_dir.glob("_SUCCESS_*.json"):
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            if payload.get("processor") != "apache_flink_pyflink_batch":
                continue
            match = RUN_ID_PATTERN.match(str(payload.get("run_id", "")))
            if not match:
                continue
            if (
                match.group("symbol") != symbol_code
                or match.group("market") != market.upper()
                or match.group("timeframe") != timeframe
            ):
                continue
            start_date = datetime.strptime(match.group("start"), "%Y%m%d").date()
            end_date = datetime.strptime(match.group("end"), "%Y%m%d").date()
            if start_date < end_date:
                starts_by_end[end_date] = min(starts_by_end.get(end_date, start_date), start_date)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    boundary = initial_end
    visited = set()
    while boundary in starts_by_end and boundary not in visited:
        visited.add(boundary)
        boundary = starts_by_end[boundary]
    return boundary


def find_context_contiguous_completed_start(
    context_folder: str,
    symbol: str,
    market: str,
    timeframe: str,
    initial_end,
):
    """Return the oldest boundary covered by contiguous validated context markers."""
    marker_dir = PROJECT_ROOT / context_folder / "_markers"
    symbol_code = symbol.replace("/", "").replace(":", "")
    starts_by_end = {}
    if not marker_dir.exists():
        return initial_end
    for marker_path in marker_dir.glob("_SUCCESS_*_CONTEXT_*.json"):
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            if payload.get("context_schema_version") != "usdm_context_v2_complete_mark":
                continue
            match = CONTEXT_RUN_ID_PATTERN.match(str(payload.get("run_id", "")))
            if not match:
                continue
            if (
                match.group("symbol") != symbol_code
                or match.group("market") != market.upper()
                or match.group("timeframe") != timeframe
            ):
                continue
            start_date = datetime.strptime(match.group("start"), "%Y%m%d").date()
            end_date = datetime.strptime(match.group("end"), "%Y%m%d").date()
            if start_date < end_date:
                starts_by_end[end_date] = min(starts_by_end.get(end_date, start_date), start_date)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    boundary = initial_end
    visited = set()
    while boundary in starts_by_end and boundary not in visited:
        visited.add(boundary)
        boundary = starts_by_end[boundary]
    return boundary


def plan_next_backfill(**context):
    params = get_runtime_params(context)
    chunk_days = int(params["chunk_days"])
    chunks_per_trigger = int(params["chunks_per_trigger"])
    if chunk_days <= 0 or chunks_per_trigger <= 0 or chunks_per_trigger > 8:
        raise ValueError("chunk_days must be positive and chunks_per_trigger must be between 1 and 8.")

    manual_start = params.get("start_date")
    manual_end = params.get("end_date")
    if bool(manual_start) != bool(manual_end):
        raise ValueError("수동 실행은 start_date와 end_date를 함께 입력해야 합니다.")

    if manual_start and manual_end:
        range_start = parse_date(str(manual_start))
        range_end = parse_date(str(manual_end))
        if range_end <= range_start:
            raise ValueError("end_date는 start_date보다 뒤여야 합니다.")
        return {
            "status": "ready",
            "mode": "manual_range",
            "source": "dag_run.conf",
            "start_date": range_start.isoformat(),
            "end_date": range_end.isoformat(),
            "chunk_days": chunk_days,
            "symbol": params["symbol"],
            "market": params["market"],
            "timeframe": params["timeframe"],
        }

    target_start = parse_date(params["target_start_date"])
    initial_end = parse_date(params["initial_end_date"])
    feature_contiguous_start = find_contiguous_completed_start(
        params["feature_folder"],
        params["symbol"],
        params["market"],
        params["timeframe"],
        initial_end,
    )
    context_contiguous_start = find_context_contiguous_completed_start(
        params["context_folder"],
        params["symbol"],
        params["market"],
        params["timeframe"],
        initial_end,
    )
    if feature_contiguous_start <= target_start and context_contiguous_start <= target_start:
        return {
            "status": "complete",
            "message": "Target historical range has already been processed.",
            "target_start_date": target_start.isoformat(),
        }

    max_window = timedelta(days=chunk_days * chunks_per_trigger)
    if context_contiguous_start > feature_contiguous_start:
        # Existing OHLCV reaches farther into the past than context. Fill only
        # the matching context range first; running OHLCV here would create a
        # feature gap between the old feature boundary and the context boundary.
        mode = "context_only"
        range_end = context_contiguous_start
        range_start = max(feature_contiguous_start, range_end - max_window)
    elif feature_contiguous_start > context_contiguous_start:
        # Context was saved before the matching feature job finished. Catch up
        # only OHLCV, then return to paired backfill once boundaries match.
        mode = "feature_only"
        range_end = feature_contiguous_start
        range_start = max(context_contiguous_start, range_end - max_window)
    else:
        mode = "paired_backfill"
        range_end = feature_contiguous_start
        range_start = max(target_start, range_end - max_window)

    return {
        "status": "ready",
        "mode": mode,
        "start_date": range_start.isoformat(),
        "end_date": range_end.isoformat(),
        "chunk_days": chunk_days,
        "feature_contiguous_start": feature_contiguous_start.isoformat(),
        "context_contiguous_start": context_contiguous_start.isoformat(),
    }


def run_next_backfill(**context):
    ti = context["ti"]
    plan = ti.xcom_pull(task_ids="plan_next_backfill")
    if not plan or plan["status"] == "complete":
        return plan

    params = get_runtime_params(context)
    context_command = None
    feature_command = None
    if plan["mode"] in {"context_only", "paired_backfill", "manual_range"}:
        context_command = [
            sys.executable,
            "9_futures_context_collector.py",
            "--symbol",
            params["symbol"],
            "--timeframe",
            params["timeframe"],
            "--start-date",
            plan["start_date"],
            "--end-date",
            plan["end_date"],
            "--context-folder",
            params["context_folder"],
        ]
    if plan["mode"] in {"feature_only", "paired_backfill", "manual_range"}:
        feature_command = [
            sys.executable,
            "backfill_runner.py",
            "--market",
            params["market"],
            "--symbol",
            params["symbol"],
            "--timeframe",
            params["timeframe"],
            "--start-date",
            plan["start_date"],
            "--end-date",
            plan["end_date"],
            "--chunk-days",
            str(plan["chunk_days"]),
            "--temp-folder",
            params["temp_folder"],
            "--feature-folder",
            params["feature_folder"],
            "--processor",
            "flink",
            # A PC or container restart can leave an unfinished Flink staging directory.
            # The submitter verifies it is stale before replacing it, allowing the next
            # scheduled run to resume without manual cleanup.
            "--recover-stale-staging",
        ]
    # During paired backfill context runs first. A context failure leaves the
    # matching feature range untouched, so the next run retries the same range.
    if context_command:
        subprocess.run(context_command, cwd=PROJECT_ROOT, check=True)
    if feature_command:
        subprocess.run(feature_command, cwd=PROJECT_ROOT, check=True)

    report_dir = PROJECT_ROOT / "runtime_reports" / "historical_backfill_airflow"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"backfill_{plan['start_date']}_{plan['end_date']}.json"
    report_path.write_text(
        json.dumps(
            {
                "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                "dag_id": context["dag"].dag_id,
                "run_id": context["run_id"],
                "effective_params": {
                    "symbol": params["symbol"],
                    "market": params["market"],
                    "timeframe": params["timeframe"],
                    "start_date": plan["start_date"],
                    "end_date": plan["end_date"],
                    "chunk_days": plan["chunk_days"],
                },
                "plan": plan,
                "futures_context_command": context_command,
                "feature_command": feature_command,
                "result": "completed",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {**plan, "report_path": str(report_path), "result": "completed"}


def verify_feature_store(**context):
    params = get_runtime_params(context)
    plan = context["ti"].xcom_pull(task_ids="plan_next_backfill") or {}
    chain_end_date = (
        plan["end_date"]
        if plan.get("mode") == "manual_range"
        else params["initial_end_date"]
    )
    command = [
        sys.executable,
        "scripts/verify_feature_store.py",
        "--feature-folder",
        params["feature_folder"],
        "--symbol",
        params["symbol"].replace("/", ""),
        "--market",
        params["market"],
        "--timeframe",
        params["timeframe"],
        "--chain-end-date",
        chain_end_date,
        "--run-context",
        context["run_id"],
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def verify_futures_context_store(**context):
    params = get_runtime_params(context)
    plan = context["ti"].xcom_pull(task_ids="plan_next_backfill") or {}
    chain_end_date = (
        plan["end_date"]
        if plan.get("mode") == "manual_range"
        else params["initial_end_date"]
    )
    command = [
        sys.executable,
        "scripts/verify_futures_context_store.py",
        "--context-folder",
        params["context_folder"],
        "--symbol",
        params["symbol"].replace("/", ""),
        "--market",
        params["market"],
        "--timeframe",
        params["timeframe"],
        "--chain-end-date",
        chain_end_date,
        "--run-context",
        context["run_id"],
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


with DAG(
    dag_id="btcusdt_usdm_historical_backfill",
    description="USDT-M historical backfill with optional symbol and date-range trigger inputs.",
    start_date=pendulum.datetime(2026, 8, 25, tz="UTC"),
    # Every run discovers the earliest successful marker, so missed calendar runs
    # do not create a data hole. The next run simply continues from that marker.
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    params={
        "symbol": "BTC/USDT",
        "market": "usdm",
        "timeframe": "1m",
        "feature_folder": "feature_store_v2",
        "context_folder": "futures_context_store_v2",
        "temp_folder": "temp_raw_data_v3",
        "target_start_date": "2021-08-25",
        "initial_end_date": "2026-08-22",
        "chunk_days": 14,
        # Optional one-off Trigger DAG inputs. When both are supplied, the DAG
        # processes exactly that range without changing the scheduled checkpoint.
        "start_date": None,
        "end_date": None,
        # 4 x 14 days = up to 56 days per run. Keep max_active_runs=1 to avoid overlap.
        "chunks_per_trigger": 4,
    },
    default_args={"retries": 0},
    tags=["btc", "usdm", "historical", "backfill", "automatic"],
) as dag:
    plan_task = PythonOperator(task_id="plan_next_backfill", python_callable=plan_next_backfill)
    run_task = PythonOperator(task_id="run_next_backfill", python_callable=run_next_backfill)
    quality_task = PythonOperator(task_id="verify_feature_store", python_callable=verify_feature_store)
    context_quality_task = PythonOperator(
        task_id="verify_futures_context_store",
        python_callable=verify_futures_context_store,
    )
    plan_task >> run_task >> quality_task >> context_quality_task
