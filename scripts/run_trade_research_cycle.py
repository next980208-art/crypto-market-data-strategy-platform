"""Build an isolated trade dataset and compare fixed, matched research folds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def drop_incomplete_optional_context(dataset):
    excluded = {}
    for column in ("open_interest", "open_interest_value"):
        if column in dataset:
            missing = int((~np.isfinite(pd.to_numeric(dataset[column], errors="coerce"))).sum())
            if missing:
                excluded[column] = {"unavailable_rows": missing, "reason": "not_available_over_full_comparison_period"}
    return dataset.drop(columns=list(excluded)), excluded


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "stages": [], "real_trading_enabled": False,
        "registry_modified": False,
        "protocol": {"train_days": 45, "validation_days": 15, "test_days": 10,
                     "step_days": 10, "max_folds": 3, "n_estimators": 100,
                     "policy_selection": "validation_only", "comparison": "same_dataset_same_periods"},
    }
    report_path = output / "cycle.json"

    def checkpoint(stage):
        report["stages"].append({"stage": stage, "at_utc": datetime.now(timezone.utc).isoformat()})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(stage, flush=True)

    try:
        builder = module("17_build_trade_enriched_dataset.py", "cycle_builder")
        labeler = module("4_triple_barrier_labeler.py", "cycle_labeler")
        targets = module("6_build_ml_dataset.py", "cycle_targets")
        args = SimpleNamespace(
            feature_folder=str(ROOT / "feature_store_v2"),
            futures_context_folder=str(ROOT / "futures_context_store_v2"),
            trade_context_folder=str(ROOT / "trade_context_store_v2"),
            symbol="BTC/USDT", market="usdm", timeframe="1m",
            label_file=str(output / "labels.parquet"),
            output_folder=str(output / "dataset"), minimum_training_days=90,
            report_json=str(output / "dataset.json"), report_md=str(output / "dataset.md"),
        )
        checkpoint("loading_trade_context")
        raw, source = builder.load_trade_context(args.trade_context_folder, args.symbol, args.market, args.timeframe)
        report["trade_source"] = source
        if len(raw) < 91 * 1440:
            raise RuntimeError("At least 91 raw continuous days are required before label warmup/horizon exclusions")
        trade, columns = builder.add_causal_trade_features(raw)
        builder.verify_causal_trade_features(raw, trade, columns)
        features, _ = builder.load_market_sources(args, int(raw.timestamp.min()), int(raw.timestamp.max()))
        if len(features) != len(raw) or not features.timestamp.diff().dropna().eq(60_000).all():
            raise RuntimeError("OHLCV source coverage is incomplete")
        checkpoint("generating_and_verifying_labels")
        labels = labeler.add_labels(features, [1.0, 1.5, 3.0], 0.005, 240, 10.0, 5000, 100)
        targets.add_targets(labels, 0.0).to_parquet(args.label_file, index=False)
        checkpoint("joining_and_verifying_dataset")
        dataset, columns, label_path, joins = builder.build_enriched_dataset(args, raw, trade, columns)
        dataset, excluded = drop_incomplete_optional_context(dataset)
        report["excluded_optional_context"] = excluded
        path, _, _, marker = builder.write_outputs(args, dataset, columns, label_path, source, joins)
        report["dataset"] = marker
        if not marker["model_training_allowed"]:
            raise RuntimeError(f"Dataset training gate blocked: {marker['training_block_reason']}")
        for variant in ("baseline", "trade_enriched"):
            checkpoint(f"training_{variant}")
            command = [sys.executable, "-u", str(ROOT / "16_action_profitability_walk_forward_model.py"),
                       "--dataset-file", str(path), "--train-days", "45", "--validation-days", "15",
                       "--test-days", "10", "--step-days", "10", "--max-folds", "3",
                       "--n-estimators", "100", "--n-jobs", "2",
                       "--model-folder", str(output / variant / "models"),
                       "--prediction-folder", str(output / variant / "predictions"),
                       "--report-json", str(output / f"{variant}.json"),
                       "--report-md", str(output / f"{variant}.md")]
            if variant == "trade_enriched":
                command += ["--trade-feature-manifest", args.report_json]
            with (output / f"{variant}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{variant} failed with exit {result.returncode}; inspect its log")
            report[variant] = json.loads((output / f"{variant}.json").read_text(encoding="utf-8"))
        baseline, enriched = report["baseline"], report["trade_enriched"]
        for left, right in zip(baseline["folds"], enriched["folds"], strict=True):
            if left["periods"] != right["periods"] or left["rows"] != right["rows"]:
                raise RuntimeError("Comparison folds or row counts differ")
        report["comparison"] = {
            "matched_folds": True,
            "baseline": baseline["aggregate_test_backtest"],
            "trade_enriched": enriched["aggregate_test_backtest"],
            "paper_candidate": enriched["approved_for_paper_connection"],
            "live_policy": "unchanged_no_trade",
        }
        report["status"] = "completed"
        checkpoint("comparison_completed")
    except Exception as exc:
        report["status"] = "blocked_or_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-folder", type=Path, default=ROOT / "runtime_reports" / datetime.now(timezone.utc).strftime("trade_research_%Y%m%dT%H%M%SZ"))
    options = parser.parse_args()
    run(options.output_folder.resolve())
