"""Run the offline ML research lifecycle after validated features are available.

This runner is intentionally separate from data collection.  It turns an already
validated Feature Store into labels, an ML dataset, a candidate model, and a
cost-aware backtest report.  A successful run is research evidence only; it
does not approve a model for paper or live trading.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run label, dataset, training, and backtest from a validated Feature Store."
    )
    parser.add_argument("--feature-folder", default="feature_store_v2")
    parser.add_argument("--context-folder", default="futures_context_store_v2")
    parser.add_argument("--label-folder", default="label_store_v2")
    parser.add_argument("--dataset-folder", default="ml_dataset_v2")
    parser.add_argument("--model-folder", default="models_v2")
    parser.add_argument("--prediction-folder", default="predictions_v2")
    parser.add_argument("--report-path", default="runtime_reports/latest_research_backtest.md")
    parser.add_argument("--manifest-folder", default="runtime_reports/research_runs")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="usdm")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--stop-distance-pct", type=float, default=0.005)
    parser.add_argument("--take-profit-r", type=float, default=1.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    parser.add_argument(
        "--start-step",
        choices=["label", "ml_dataset", "train", "backtest"],
        default="label",
        help="Resume the lifecycle from this step after validated outputs already exist.",
    )
    return parser.parse_args()


def run_step(name, command):
    started_at = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    result = {
        "name": name,
        "command": command,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
        "stdout": completed.stdout[-10_000:],
        "stderr": completed.stderr[-10_000:],
        "status": "processed" if completed.returncode == 0 else "failed",
    }
    return result


def write_manifest(args, results, status, error=None):
    folder = PROJECT_ROOT / args.manifest_folder
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{args.symbol}_{args.market}_{args.timeframe}_{stamp}.json"
    payload = {
        "pipeline": "offline_ml_research_v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "market": args.market,
        "timeframe": args.timeframe,
        "feature_folder": args.feature_folder,
        "context_folder": args.context_folder,
        "label_folder": args.label_folder,
        "dataset_folder": args.dataset_folder,
        "model_folder": args.model_folder,
        "prediction_folder": args.prediction_folder,
        "report_path": args.report_path,
        "risk_per_trade_pct": args.risk_per_trade_pct,
        "start_step": args.start_step,
        "steps": results,
        "approval_note": "A completed research run does not approve paper or live trading.",
        "error": error,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    args = parse_args()
    if not 0 < args.risk_per_trade_pct <= 0.02:
        raise ValueError("--risk-per-trade-pct must be greater than 0 and at most 0.02.")
    if args.take_profit_r <= 0 or args.stop_distance_pct <= 0:
        raise ValueError("Stop distance and take-profit R must be positive.")

    common = ["--symbol", args.symbol, "--market", args.market, "--timeframe", args.timeframe]
    commands = [
        (
            "label",
            [
                sys.executable,
                "4_triple_barrier_labeler.py",
                "--feature-folder",
                args.feature_folder,
                "--label-folder",
                args.label_folder,
                "--stop-distance-pct",
                str(args.stop_distance_pct),
                "--take-profit-r",
                str(args.take_profit_r),
                *common,
            ],
        ),
        (
            "ml_dataset",
            [
                sys.executable,
                "6_build_ml_dataset.py",
                "--feature-folder",
                args.feature_folder,
                "--label-folder",
                args.label_folder,
                "--context-folder",
                args.context_folder,
                "--dataset-folder",
                args.dataset_folder,
                *common,
            ],
        ),
        (
            "train",
            [
                sys.executable,
                "7_train_direction_model.py",
                "--dataset-folder",
                args.dataset_folder,
                "--model-folder",
                args.model_folder,
                "--prediction-folder",
                args.prediction_folder,
                *common,
            ],
        ),
        (
            "backtest",
            [
                sys.executable,
                "8_model_signal_backtest.py",
                "--prediction-folder",
                args.prediction_folder,
                "--take-profit-r",
                str(args.take_profit_r),
                "--risk-per-trade-pct",
                str(args.risk_per_trade_pct),
                "--report-path",
                args.report_path,
                *common,
            ],
        ),
    ]
    step_names = [name for name, _ in commands]
    commands = commands[step_names.index(args.start_step) :]

    results = []
    try:
        for name, command in commands:
            result = run_step(name, command)
            results.append(result)
            if result["return_code"] != 0:
                raise RuntimeError(f"Research step failed: {name}")
            print(f"{name}: processed")
    except Exception as exc:
        manifest = write_manifest(args, results, "failed", str(exc))
        print(f"Research lifecycle failed. Manifest: {manifest}")
        raise

    manifest = write_manifest(args, results, "completed")
    print(f"Research lifecycle completed. Manifest: {manifest}")


if __name__ == "__main__":
    main()
