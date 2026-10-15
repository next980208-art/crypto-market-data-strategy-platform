"""Incrementally recover BTCUSDT aggTrade history from official daily archives."""

import subprocess
import sys
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")


def backfill_one_archive_day():
    command = [
        sys.executable,
        "aggtrade_archive_backfill_runner.py",
        "--target-start-date",
        "2026-06-06",
        "--max-days-per-run",
        "1",
        "--minimum-training-days",
        "90",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


with DAG(
    dag_id="btcusdt_aggtrade_archive_backfill",
    description="Fill one missing official BTCUSDT aggTrade day per run toward 90 continuous days.",
    start_date=pendulum.datetime(2026, 9, 4, 0, 0, tz="UTC"),
    schedule="45 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2},
    tags=["btc", "usdm", "aggtrade", "archive-backfill"],
) as dag:
    PythonOperator(
        task_id="backfill_one_archive_day",
        python_callable=backfill_one_archive_day,
    )
