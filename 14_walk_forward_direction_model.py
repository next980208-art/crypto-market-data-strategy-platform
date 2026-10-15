import argparse
import gc
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DAY_MS = 86_400_000
WARMUP_MINUTES = 60
DEFAULT_DATASET = (
    "ml_dataset_5y_20260828/market=usdm/symbol=BTCUSDT/timeframe=1m/"
    "ml_dataset_BTCUSDT_USDM_1m.parquet"
)

REQUIRED_SOURCE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "max_holding_minutes",
    "label_horizon_complete",
    "entry_timestamp",
    "entry_price",
    "stop_distance_pct",
    "round_trip_cost_bps",
    "long_tp_1_0r_return_r_net",
    "short_tp_1_0r_return_r_net",
    "long_tp_1_0r_holding_minutes",
    "short_tp_1_0r_holding_minutes",
    "direction_target",
    "best_action",
]
OPTIONAL_SOURCE_COLUMNS = [
    "mark_basis_pct",
    "funding_rate",
    "funding_rate_age_minutes",
    "open_interest",
    "open_interest_value",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leakage-safe rolling walk-forward validation on the BTCUSDT dataset."
    )
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="usdm")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--window-mode", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=365)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--ambiguous-return-r", type=float, default=0.20)
    parser.add_argument(
        "--threshold-candidates", default="0.52,0.55,0.58,0.60,0.62,0.65"
    )
    parser.add_argument("--minimum-validation-trades", type=int, default=20)
    parser.add_argument("--minimum-test-trades", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-folder", default="models_walk_forward_5y")
    parser.add_argument("--prediction-folder", default="predictions_walk_forward_5y")
    parser.add_argument(
        "--report-json",
        default="runtime_reports/walk_forward_5y_report_2026-08-29.json",
    )
    parser.add_argument(
        "--report-md",
        default="runtime_reports/walk_forward_5y_report_2026-08-29.md",
    )
    return parser.parse_args()


def load_numbered_module(filename, module_name):
    path = PROJECT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_args(args):
    for name in ("train_days", "validation_days", "test_days", "step_days", "max_folds"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.n_estimators <= 0 or args.n_jobs <= 0:
        raise ValueError("n_estimators and n_jobs must be positive.")


def dataset_bounds(dataset_path):
    timestamps = pd.read_parquet(dataset_path, columns=["timestamp"])["timestamp"]
    timestamps = pd.to_numeric(timestamps, errors="raise")
    if timestamps.empty:
        raise RuntimeError("The dataset is empty.")
    return int(timestamps.min()), int(timestamps.max()), int(len(timestamps))


def build_fold_specs(
    minimum_timestamp,
    maximum_timestamp,
    train_days,
    validation_days,
    test_days,
    step_days,
    max_folds,
    window_mode,
):
    data_end_exclusive = maximum_timestamp + 60_000
    first_train_start = minimum_timestamp
    train_duration = train_days * DAY_MS
    validation_duration = validation_days * DAY_MS
    test_duration = test_days * DAY_MS
    step_duration = step_days * DAY_MS
    candidates = []
    fold_offset = 0

    while True:
        rolling_train_start = first_train_start + fold_offset
        train_start = first_train_start if window_mode == "expanding" else rolling_train_start
        validation_start = rolling_train_start + train_duration
        test_start = validation_start + validation_duration
        test_end = test_start + test_duration
        if test_end > data_end_exclusive:
            break
        candidates.append(
            {
                "train_start": train_start,
                "validation_start": validation_start,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        fold_offset += step_duration

    if not candidates:
        raise RuntimeError("The dataset is too short for the requested walk-forward windows.")
    if len(candidates) > max_folds:
        indices = np.linspace(0, len(candidates) - 1, max_folds).round().astype(int)
        candidates = [candidates[index] for index in sorted(set(indices.tolist()))]
    for index, candidate in enumerate(candidates, start=1):
        candidate["fold"] = index
    return candidates


def available_columns(dataset_path):
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(dataset_path).schema.names)


def load_fold_source(dataset_path, fold, minimum_timestamp, extra_columns=()):
    available = available_columns(dataset_path)
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - available)
    if missing:
        raise RuntimeError(f"Dataset is missing walk-forward columns: {missing}")
    columns = REQUIRED_SOURCE_COLUMNS + [
        column for column in OPTIONAL_SOURCE_COLUMNS if column in available
    ]
    missing_extra = sorted(set(extra_columns) - available)
    if missing_extra:
        raise RuntimeError(f"Dataset is missing requested features: {missing_extra}")
    columns += [column for column in extra_columns if column not in columns]
    read_start = max(
        minimum_timestamp, fold["train_start"] - WARMUP_MINUTES * 60_000
    )
    frame = pd.read_parquet(
        dataset_path,
        columns=columns,
        filters=[
            ("timestamp", ">=", read_start),
            ("timestamp", "<", fold["test_end"]),
        ],
    )
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    frame["datetime_utc"] = pd.to_datetime(
        frame["timestamp"], unit="ms", utc=True
    ).astype(str)
    frame["entry_rule"] = "next_bar_open"
    return frame


def split_fold(dataset, fold):
    horizon_ms = int(dataset["max_holding_minutes"].max()) * 60_000
    initial_train = dataset[
        (dataset["timestamp"] >= fold["train_start"])
        & (dataset["timestamp"] < fold["validation_start"])
    ].copy()
    initial_validation = dataset[
        (dataset["timestamp"] >= fold["validation_start"])
        & (dataset["timestamp"] < fold["test_start"])
    ].copy()
    test = dataset[
        (dataset["timestamp"] >= fold["test_start"])
        & (dataset["timestamp"] < fold["test_end"])
    ].copy()
    train = initial_train[
        initial_train["timestamp"] + horizon_ms < fold["validation_start"]
    ].copy()
    validation = initial_validation[
        initial_validation["timestamp"] + horizon_ms < fold["test_start"]
    ].copy()
    initial_train_validation = dataset[
        (dataset["timestamp"] >= fold["train_start"])
        & (dataset["timestamp"] < fold["test_start"])
    ].copy()
    train_validation = initial_train_validation[
        initial_train_validation["timestamp"] + horizon_ms < fold["test_start"]
    ].copy()
    if any(frame.empty for frame in (train, validation, test, train_validation)):
        raise RuntimeError(f"Fold {fold['fold']} produced an empty partition.")
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "train_validation": train_validation,
        "train_purged_rows": len(initial_train) - len(train),
        "validation_purged_rows": len(initial_validation) - len(validation),
        "final_purged_rows": len(initial_train_validation) - len(train_validation),
        "horizon_minutes": horizon_ms // 60_000,
    }


def train_model(training, frame, feature_columns, args, seed_offset=0):
    selected = training.actionable(frame, args.ambiguous_return_r)
    target = selected["fixed_1r_direction"].astype(int)
    model = training.make_model(args.random_state + seed_offset)
    model.set_params(n_estimators=args.n_estimators, n_jobs=args.n_jobs)
    model.fit(
        selected[feature_columns],
        target,
        sample_weight=training.sample_weights(target),
    )
    return model, len(selected)


def timestamp_iso(value):
    return pd.to_datetime(int(value), unit="ms", utc=True).isoformat()


def fold_periods(fold):
    return {
        "train": [timestamp_iso(fold["train_start"]), timestamp_iso(fold["validation_start"])],
        "validation": [
            timestamp_iso(fold["validation_start"]),
            timestamp_iso(fold["test_start"]),
        ],
        "test": [timestamp_iso(fold["test_start"]), timestamp_iso(fold["test_end"])],
    }


def prediction_output_path(args):
    suffix = "" if args.market == "spot" else f"_{args.market.upper()}"
    return (
        Path(args.prediction_folder)
        / f"market={args.market}"
        / f"symbol={args.symbol}"
        / f"timeframe={args.timeframe}"
        / f"walk_forward_predictions_{args.symbol}{suffix}_{args.timeframe}.parquet"
    )


def approval_result(folds, aggregate, minimum_test_trades):
    profitable_folds = sum(fold["test_backtest"]["total_return_pct"] > 0 for fold in folds)
    required_profitable = math.ceil(len(folds) * 0.75)
    checks = {
        "aggregate_total_return_positive": aggregate["total_return_pct"] > 0,
        "aggregate_expectancy_positive": aggregate["avg_return_r"] > 0,
        "aggregate_drawdown_within_20pct": aggregate["max_drawdown_pct"] >= -20.0,
        "at_least_75pct_folds_profitable": profitable_folds >= required_profitable,
        "every_fold_has_minimum_test_trades": all(
            fold["test_backtest"]["trades"] >= minimum_test_trades for fold in folds
        ),
        "average_balanced_accuracy_above_random": float(
            np.mean([fold["test_direction_metrics"]["balanced_accuracy"] for fold in folds])
        )
        > 0.5,
    }
    return {
        "approved_for_paper_connection": all(checks.values()),
        "checks": checks,
        "profitable_folds": profitable_folds,
        "required_profitable_folds": required_profitable,
        "decision": "paper_candidate" if all(checks.values()) else "research_only",
    }


def write_markdown(report, path):
    lines = [
        "# 5년 데이터 Walk-Forward 검증 결과",
        "",
        f"- 실행 시각: `{report['created_at_utc']}`",
        f"- 데이터: `{report['dataset']['path']}`",
        f"- 전체 행 수: `{report['dataset']['rows']:,}`",
        f"- 검증 방식: `{report['configuration']['window_mode']}` window",
        f"- 모델 상태: **{report['approval']['decision']}**",
        "",
        "## 왜 이 검증을 했는가",
        "",
        "한 번의 80/20 분할은 특정 시장 시기에 우연히 맞을 수 있습니다. 이 실행은 과거로 학습하고,",
        "그 다음 검증 구간에서 확률 기준을 고른 뒤, 더 미래의 테스트 구간에서만 성능을 측정합니다.",
        "라벨이 경계를 넘어 미래 정보를 전달하지 않도록 각 경계 앞의 행도 제거했습니다.",
        "",
        "## Fold별 결과",
        "",
        "| Fold | 학습 | 검증 | 테스트 | 선택 기준 | 거래 | 평균 R | 수익률 | MDD | Balanced accuracy |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in report["folds"]:
        periods = fold["periods"]
        backtest = fold["test_backtest"]
        metrics = fold["test_direction_metrics"]
        lines.append(
            "| {fold} | {train} | {validation} | {test} | {threshold:.2f} | {trades} | "
            "{avg_r:.4f} | {ret:.2f}% | {mdd:.2f}% | {balanced:.4f} |".format(
                fold=fold["fold"],
                train=periods["train"][0][:10] + " ~ " + periods["train"][1][:10],
                validation=periods["validation"][0][:10] + " ~ " + periods["validation"][1][:10],
                test=periods["test"][0][:10] + " ~ " + periods["test"][1][:10],
                threshold=fold["selected_threshold"]["threshold"],
                trades=backtest["trades"],
                avg_r=backtest["avg_return_r"],
                ret=backtest["total_return_pct"],
                mdd=backtest["max_drawdown_pct"],
                balanced=metrics["balanced_accuracy"],
            )
        )
    aggregate = report["aggregate_test_backtest"]
    lines.extend(
        [
            "",
            "## 통합 테스트 결과",
            "",
            f"- 거래 수: `{aggregate['trades']}`",
            f"- 승률: `{aggregate['win_rate_pct']:.2f}%`",
            f"- 평균 손익: `{aggregate['avg_return_r']:.4f}R`",
            f"- 누적 계좌 수익률: `{aggregate['total_return_pct']:.2f}%`",
            f"- 최대 낙폭: `{aggregate['max_drawdown_pct']:.2f}%`",
            "",
            "## 승인 관문",
            "",
        ]
    )
    for name, passed in report["approval"]["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "이 결과는 연구 검증 결과이며 실거래 승인과 다릅니다. 승인 관문을 모두 통과하지 못하면",
            "실시간 추론 서비스나 주문 API에 연결하지 않습니다.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    dataset_path = Path(args.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ML dataset does not exist: {dataset_path}")

    training = load_numbered_module(
        "13_train_selective_direction_model.py", "selective_direction_training"
    )
    backtest = load_numbered_module("8_model_signal_backtest.py", "model_signal_backtest")
    thresholds = training.parse_thresholds(args.threshold_candidates)
    minimum_timestamp, maximum_timestamp, row_count = dataset_bounds(dataset_path)
    folds = build_fold_specs(
        minimum_timestamp,
        maximum_timestamp,
        args.train_days,
        args.validation_days,
        args.test_days,
        args.step_days,
        args.max_folds,
        args.window_mode,
    )

    fold_reports = []
    out_of_sample_predictions = []
    feature_columns = None
    causal_verification_passed = False
    model_root = Path(args.model_folder)
    model_root.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        print(f"[fold {fold['fold']}/{len(folds)}] loading {fold_periods(fold)}", flush=True)
        raw = load_fold_source(dataset_path, fold, minimum_timestamp)
        dataset, current_features = training.add_causal_features(raw)
        if feature_columns is None:
            feature_columns = current_features
            training.verify_causal_features(raw, dataset, feature_columns)
            causal_verification_passed = True
        elif current_features != feature_columns:
            raise RuntimeError("Feature columns changed between walk-forward folds.")
        dataset = dataset[
            (dataset["timestamp"] >= fold["train_start"])
            & (dataset["timestamp"] < fold["test_end"])
        ].copy()
        split = split_fold(dataset, fold)

        validation_model, validation_training_rows = train_model(
            training, split["train"], feature_columns, args, fold["fold"] * 2
        )
        selected_threshold, threshold_candidates = training.select_threshold(
            validation_model,
            split["validation"],
            feature_columns,
            thresholds,
            args.minimum_validation_trades,
            args.symbol,
            args.market,
            args.timeframe,
        )
        del validation_model
        gc.collect()

        final_model, final_training_rows = train_model(
            training,
            split["train_validation"],
            feature_columns,
            args,
            fold["fold"] * 2 + 1,
        )
        threshold = selected_threshold["threshold"]
        test_predictions = training.prediction_frame(
            final_model, split["test"], feature_columns, threshold
        )
        test_predictions["walk_forward_fold"] = fold["fold"]
        test_predictions["walk_forward_threshold"] = threshold
        test_args = training.backtest_args(
            threshold, args.symbol, args.market, args.timeframe
        )
        trades, equity_curve, skipped = backtest.simulate(test_predictions, test_args)
        test_summary = backtest.summarize(
            trades, equity_curve, skipped, test_predictions, test_args
        )
        metrics = training.directional_metrics(
            final_model,
            split["test"],
            feature_columns,
            args.ambiguous_return_r,
        )
        model_path = model_root / f"walk_forward_fold_{fold['fold']:02d}.json"
        final_model.save_model(model_path)
        fold_reports.append(
            {
                "fold": fold["fold"],
                "periods": fold_periods(fold),
                "rows": {
                    "train": len(split["train"]),
                    "validation": len(split["validation"]),
                    "test": len(split["test"]),
                    "train_validation": len(split["train_validation"]),
                    "validation_actionable_training": validation_training_rows,
                    "final_actionable_training": final_training_rows,
                },
                "purge": {
                    "horizon_minutes": split["horizon_minutes"],
                    "train_purged_rows": split["train_purged_rows"],
                    "validation_purged_rows": split["validation_purged_rows"],
                    "final_purged_rows": split["final_purged_rows"],
                },
                "selected_threshold": selected_threshold,
                "threshold_candidates": threshold_candidates,
                "test_direction_metrics": metrics,
                "test_backtest": test_summary,
                "model_path": str(model_path),
            }
        )
        out_of_sample_predictions.append(test_predictions)
        print(
            f"[fold {fold['fold']}] threshold={threshold:.2f} "
            f"trades={test_summary['trades']} return={test_summary['total_return_pct']:.2f}%",
            flush=True,
        )
        del raw, dataset, split, final_model, test_predictions
        gc.collect()

    combined = pd.concat(out_of_sample_predictions, ignore_index=True).sort_values("timestamp")
    aggregate_args = training.backtest_args(0.5, args.symbol, args.market, args.timeframe)
    trades, equity_curve, skipped = backtest.simulate(combined, aggregate_args)
    aggregate = backtest.summarize(trades, equity_curve, skipped, combined, aggregate_args)
    prediction_path = prediction_output_path(args)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(prediction_path, index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dataset": {
            "path": str(dataset_path),
            "rows": row_count,
            "start": timestamp_iso(minimum_timestamp),
            "end": timestamp_iso(maximum_timestamp),
        },
        "configuration": {
            "window_mode": args.window_mode,
            "train_days": args.train_days,
            "validation_days": args.validation_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "folds": len(folds),
            "ambiguous_return_r": args.ambiguous_return_r,
            "threshold_candidates": thresholds,
            "n_estimators": args.n_estimators,
            "risk_per_trade_pct": 0.02,
        },
        "leakage_controls": {
            "time_ordered_partitions": True,
            "threshold_selected_on_validation_only": True,
            "test_used_once_after_selection": True,
            "label_boundary_purge": True,
            "causal_prefix_verification": causal_verification_passed,
        },
        "feature_set_version": training.FEATURE_SET_VERSION,
        "feature_columns": feature_columns,
        "folds": fold_reports,
        "aggregate_test_backtest": aggregate,
        "prediction_path": str(prediction_path),
    }
    report["approval"] = approval_result(
        fold_reports, aggregate, args.minimum_test_trades
    )
    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, Path(args.report_md))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
