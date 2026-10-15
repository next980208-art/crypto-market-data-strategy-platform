"""Manual Airflow DAG for the post-feature ML research lifecycle."""

import subprocess
import sys
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")


def run_offline_research(**context):
    params = context["params"]
    command = [
        sys.executable,
        "research_lifecycle_runner.py",
        "--feature-folder",
        params["feature_folder"],
        "--context-folder",
        params["context_folder"],
        "--label-folder",
        params["label_folder"],
        "--dataset-folder",
        params["dataset_folder"],
        "--model-folder",
        params["model_folder"],
        "--prediction-folder",
        params["prediction_folder"],
        "--report-path",
        params["report_path"],
        "--symbol",
        params["symbol"],
        "--market",
        params["market"],
        "--timeframe",
        params["timeframe"],
        "--risk-per-trade-pct",
        str(params["risk_per_trade_pct"]),
        "--start-step",
        params["start_step"],
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


with DAG(
    dag_id="btcusdt_offline_ml_research",
    description="Manually turn validated BTCUSDT features into a backtest report.",
    start_date=pendulum.datetime(2026, 8, 25, tz="UTC"),
    schedule=None,
    catchup=False,
    params={
        "feature_folder": "feature_store_v2",
        "context_folder": "futures_context_store_v2",
        "label_folder": "label_store_v2",
        "dataset_folder": "ml_dataset_v2",
        "model_folder": "models_v2",
        "prediction_folder": "predictions_v2",
        "report_path": "runtime_reports/latest_research_backtest.md",
        "symbol": "BTCUSDT",
        "market": "usdm",
        "timeframe": "1m",
        "risk_per_trade_pct": 0.02,
        "start_step": "label",
    },
    default_args={"retries": 0},
    tags=["btc", "ml", "research", "manual"],
) as dag:
    PythonOperator(task_id="label_train_backtest", python_callable=run_offline_research)
