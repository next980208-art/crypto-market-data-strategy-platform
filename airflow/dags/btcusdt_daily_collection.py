import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")


def collect_completed_utc_day(**context):
    data_interval_start = context["data_interval_start"].in_timezone("UTC")
    data_date = data_interval_start.date().isoformat()
    command = [sys.executable, "daily_collection_runner.py", "--data-date", data_date]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


with DAG(
    dag_id="btcusdt_usdm_daily_collection",
    description="Collect one completed UTC day of BTCUSDT USDT-M batch research data.",
    start_date=pendulum.datetime(2026, 8, 18, 0, 15, tz="UTC"),
    schedule="15 0 * * *",
    catchup=True,
    max_active_runs=1,
    default_args={"retries": 2},
    tags=["btc", "usdm", "research-data"],
) as dag:
    PythonOperator(
        task_id="collect_completed_utc_day",
        python_callable=collect_completed_utc_day,
    )
