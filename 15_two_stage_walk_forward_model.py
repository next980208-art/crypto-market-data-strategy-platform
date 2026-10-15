"""Run a leakage-safe two-stage opportunity and direction walk-forward experiment."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_SET_VERSION = "two_stage_causal_regime_v1"
DEFAULT_DATASET = (
    "ml_dataset_5y_20260828/market=usdm/symbol=BTCUSDT/timeframe=1m/"
    "ml_dataset_BTCUSDT_USDM_1m.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage BTCUSDT walk-forward research model")
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=365)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--gate-return-r", type=float, default=0.20)
    parser.add_argument("--gate-max-holding-minutes", type=int, default=60)
    parser.add_argument("--gate-thresholds", default="0.52,0.55,0.58,0.60,0.65")
    parser.add_argument("--direction-thresholds", default="0.52,0.55,0.58,0.60,0.65")
    parser.add_argument("--minimum-validation-trades", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=142)
    parser.add_argument("--model-folder", default="models_two_stage_walk_forward_5y")
    parser.add_argument("--prediction-folder", default="predictions_two_stage_walk_forward_5y")
    parser.add_argument(
        "--report-json",
        default="runtime_reports/two_stage_walk_forward_5y_2026-08-31.json",
    )
    parser.add_argument(
        "--report-md",
        default="docs/two_stage_walk_forward_execution_2026-08-31.md",
    )
    return parser.parse_args()


def load_module(filename: str, name: str):
    path = PROJECT_ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(item <= 0.5 or item >= 1.0 for item in thresholds):
        raise ValueError("Thresholds must be greater than 0.5 and below 1.0.")
    return thresholds


def add_two_stage_features(raw: pd.DataFrame, training) -> tuple[pd.DataFrame, list[str]]:
    base, base_features = training.add_causal_features(raw)
    ordered = raw.sort_values("timestamp").copy()
    close = pd.to_numeric(ordered["close"], errors="coerce")
    volume = pd.to_numeric(ordered["volume"], errors="coerce")
    high = pd.to_numeric(ordered["high"], errors="coerce")
    low = pd.to_numeric(ordered["low"], errors="coerce")
    return_1m = close.pct_change(1, fill_method=None)

    extra = pd.DataFrame({"timestamp": ordered["timestamp"]})
    for minutes in (120, 240):
        mean = close.rolling(minutes, min_periods=minutes).mean()
        extra[f"return_{minutes}m"] = close.pct_change(minutes, fill_method=None)
        extra[f"close_to_ma_{minutes}"] = close / mean - 1.0
        extra[f"volatility_{minutes}m"] = return_1m.rolling(
            minutes, min_periods=minutes
        ).std(ddof=0)
        extra[f"volume_ratio_{minutes}"] = (
            volume / volume.rolling(minutes, min_periods=minutes).mean() - 1.0
        )

    ma_60 = close.rolling(60, min_periods=60).mean()
    ma_240 = close.rolling(240, min_periods=240).mean()
    volatility_15 = return_1m.rolling(15, min_periods=15).std(ddof=0)
    volatility_240 = return_1m.rolling(240, min_periods=240).std(ddof=0)
    extra["trend_ma_60_240"] = ma_60 / ma_240 - 1.0
    extra["volatility_ratio_15_240"] = volatility_15 / volatility_240.replace(0.0, np.nan)
    rolling_low = low.rolling(240, min_periods=240).min()
    rolling_high = high.rolling(240, min_periods=240).max()
    extra["range_position_240"] = (close - rolling_low) / (
        rolling_high - rolling_low
    ).replace(0.0, np.nan)

    if "mark_basis_pct" in ordered.columns:
        basis = pd.to_numeric(ordered["mark_basis_pct"], errors="coerce")
        basis_mean = basis.rolling(240, min_periods=60).mean()
        basis_std = basis.rolling(240, min_periods=60).std(ddof=0)
        extra["mark_basis_zscore_240"] = (basis - basis_mean) / basis_std.replace(0.0, np.nan)

    extra_features = [column for column in extra.columns if column != "timestamp"]
    merged = base.merge(extra, on="timestamp", how="left", validate="one_to_one")
    merged[extra_features] = merged[extra_features].replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=extra_features).reset_index(drop=True)

    long_return = pd.to_numeric(merged["long_tp_1_0r_return_r_net"], errors="raise")
    short_return = pd.to_numeric(merged["short_tp_1_0r_return_r_net"], errors="raise")
    best_is_long = long_return > short_return
    merged["two_stage_best_holding_minutes"] = np.where(
        best_is_long,
        merged["long_tp_1_0r_holding_minutes"],
        merged["short_tp_1_0r_holding_minutes"],
    ).astype(int)
    return merged, base_features + extra_features


def add_gate_target(frame: pd.DataFrame, return_r: float, max_holding: int) -> pd.DataFrame:
    output = frame.copy()
    output["opportunity_target"] = (
        (output["fixed_1r_best_return"] > return_r)
        & (output["two_stage_best_holding_minutes"] <= max_holding)
    ).astype(int)
    if output["opportunity_target"].nunique() != 2:
        raise RuntimeError("Opportunity gate target must contain both classes.")
    return output


def train_classifier(training, frame, features, target_column, args, seed: int):
    target = frame[target_column].astype(int)
    model = training.make_model(seed)
    model.set_params(n_estimators=args.n_estimators, n_jobs=args.n_jobs)
    model.fit(frame[features], target, sample_weight=training.sample_weights(target))
    return model


def train_pair(training, frame, features, args, seed: int):
    gate = train_classifier(training, frame, features, "opportunity_target", args, seed)
    directional = frame[frame["opportunity_target"].eq(1)].copy()
    if directional["fixed_1r_direction"].nunique() != 2:
        raise RuntimeError("Conditional direction target must contain both classes.")
    direction = train_classifier(
        training,
        directional,
        features,
        "fixed_1r_direction",
        args,
        seed + 1,
    )
    return gate, direction, {"gate_rows": len(frame), "direction_rows": len(directional)}


def prediction_frame(
    gate_model,
    direction_model,
    frame: pd.DataFrame,
    features: list[str],
    gate_threshold: float,
    direction_threshold: float,
    *,
    force_abstain: bool = False,
) -> pd.DataFrame:
    probability_opportunity = gate_model.predict_proba(frame[features])[:, 1]
    probability_long = direction_model.predict_proba(frame[features])[:, 1]
    probability_short = 1.0 - probability_long
    direction_confidence = np.maximum(probability_long, probability_short)
    if force_abstain:
        action = np.full(len(frame), "no_trade", dtype=object)
    else:
        direction = np.where(probability_long >= probability_short, "long", "short")
        action = np.where(
            (probability_opportunity >= gate_threshold)
            & (direction_confidence >= direction_threshold),
            direction,
            "no_trade",
        )

    output = frame.copy()
    output["original_direction_target"] = output["direction_target"]
    output["original_best_action"] = output["best_action"]
    output["direction_target"] = output["fixed_1r_direction"]
    output["best_action"] = output["fixed_1r_best_action"]
    output["pred_action"] = action
    output["pred_direction_target"] = np.where(
        action == "long", 1, np.where(action == "short", 0, -1)
    )
    output["prob_opportunity"] = probability_opportunity
    output["prob_long"] = probability_long
    output["prob_short"] = probability_short
    output["direction_confidence"] = direction_confidence
    output["pred_confidence"] = np.minimum(probability_opportunity, direction_confidence)
    output["prob_no_trade"] = 1.0 - output["pred_confidence"]
    output["model_feature_set_version"] = FEATURE_SET_VERSION
    return output


def backtest_args() -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTCUSDT",
        market="usdm",
        timeframe="1m",
        take_profit_r=1.0,
        risk_per_trade_pct=0.02,
        min_confidence=0.0,
        min_minutes_between_entries=15,
        max_new_entries_per_day=5,
        daily_stop_pct=0.04,
        hard_daily_stop_pct=0.06,
        max_consecutive_losses=3,
        extra_slippage_bps=5.0,
    )


def evaluate_candidate(backtest, predictions: pd.DataFrame) -> dict[str, float | int]:
    signals = predictions[predictions["pred_action"].ne("no_trade")].copy()
    args = backtest_args()
    trades, equity, skipped = backtest.simulate(signals, args)
    summary = backtest.summarize(trades, equity, skipped, signals, args)
    signal_count = len(signals)
    long_signals = int(signals["pred_action"].eq("long").sum())
    short_signals = int(signals["pred_action"].eq("short").sum())
    maximum_direction_share = (
        max(long_signals, short_signals) / signal_count if signal_count else 1.0
    )
    return {
        "signals": signal_count,
        "long_signals": long_signals,
        "short_signals": short_signals,
        "maximum_direction_share": maximum_direction_share,
        "trades": int(summary["trades"]),
        "win_rate_pct": float(summary["win_rate_pct"]),
        "avg_return_r": float(summary["avg_return_r"]),
        "total_return_pct": float(summary["total_return_pct"]),
        "max_drawdown_pct": float(summary["max_drawdown_pct"]),
    }


def select_threshold_pair(
    gate_model,
    direction_model,
    validation,
    features,
    gate_thresholds,
    direction_thresholds,
    minimum_trades,
    backtest,
):
    candidates = []
    for gate_threshold in gate_thresholds:
        for direction_threshold in direction_thresholds:
            predictions = prediction_frame(
                gate_model,
                direction_model,
                validation,
                features,
                gate_threshold,
                direction_threshold,
            )
            metrics = evaluate_candidate(backtest, predictions)
            metrics.update(
                {
                    "gate_threshold": gate_threshold,
                    "direction_threshold": direction_threshold,
                    "eligible": metrics["trades"] >= minimum_trades,
                }
            )
            metrics["selection_score"] = metrics["total_return_pct"] - 0.5 * abs(
                metrics["max_drawdown_pct"]
            )
            candidates.append(metrics)

    profitable = [
        item
        for item in candidates
        if item["eligible"] and item["avg_return_r"] > 0 and item["total_return_pct"] > 0
    ]
    if not profitable:
        eligible = [item for item in candidates if item["eligible"]]
        best_evaluated = max(
            eligible or candidates,
            key=lambda item: (item["selection_score"], item["avg_return_r"], item["trades"]),
        )
        return {
            "deployable": False,
            "gate_threshold": 1.0,
            "direction_threshold": 1.0,
            "abstain_reason": "no_positive_expectancy_two_stage_candidate",
            "best_evaluated_candidate": best_evaluated,
        }, candidates
    selected = max(
        profitable,
        key=lambda item: (item["selection_score"], item["avg_return_r"], item["trades"]),
    )
    selected = dict(selected)
    selected["deployable"] = True
    selected["abstain_reason"] = None
    return selected, candidates


def classifier_metrics(model, frame, features, target_column) -> dict[str, float | int]:
    target = frame[target_column].astype(int)
    probabilities = model.predict_proba(frame[features])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "rows": len(frame),
        "positive_share": float(target.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def write_markdown(report: dict, path: Path) -> None:
    aggregate = report["aggregate_test_backtest"]
    lines = [
        "# 2단계 Gate·방향 Walk-Forward 실행 결과",
        "",
        "## 모델 구조",
        "",
        "```text",
        "1단계: 60분 안에 비용 포함 0.2R 초과 기회가 있는가?",
        "2단계: 1단계를 통과한 경우 Long인가 Short인가?",
        "검증 기대값이 양수가 아니면: no_trade",
        "```",
        "",
        "## Fold 결과",
        "",
        "| Fold | Gate 기준 | 방향 기준 | 배포 가능 | 테스트 거래 | 평균 R | 수익률 | Gate AUC | 방향 AUC |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in report["folds"]:
        selected = fold["selected_thresholds"]
        result = fold["test_backtest"]
        lines.append(
            f"| {fold['fold']} | {selected['gate_threshold']:.2f} | "
            f"{selected['direction_threshold']:.2f} | {selected['deployable']} | "
            f"{result['trades']} | {result['avg_return_r']:.4f} | "
            f"{result['total_return_pct']:.2f}% | {fold['test_gate_metrics']['roc_auc']:.4f} | "
            f"{fold['test_direction_metrics']['roc_auc']:.4f} |"
        )
    lines += [
        "",
        "## 통합 결과",
        "",
        f"- 거래: `{aggregate['trades']}`건",
        f"- 평균 R: `{aggregate['avg_return_r']:.4f}R`",
        f"- 누적 수익률: `{aggregate['total_return_pct']:.2f}%`",
        f"- 최대 낙폭: `{aggregate['max_drawdown_pct']:.2f}%`",
        f"- 상태: **{report['decision']}**",
        "",
        "이 결과는 미관측 테스트 구간의 연구 결과다. `paper_candidate`가 아니면 실시간 추론이나",
        "주문 API에 연결하지 않는다.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    gate_thresholds = parse_thresholds(args.gate_thresholds)
    direction_thresholds = parse_thresholds(args.direction_thresholds)
    walk = load_module("14_walk_forward_direction_model.py", "two_stage_walk_helpers")
    training = load_module("13_train_selective_direction_model.py", "two_stage_training_helpers")
    backtest = load_module("8_model_signal_backtest.py", "two_stage_backtest")
    minimum_timestamp, maximum_timestamp, row_count = walk.dataset_bounds(dataset_path)
    folds = walk.build_fold_specs(
        minimum_timestamp,
        maximum_timestamp,
        args.train_days,
        args.validation_days,
        args.test_days,
        args.step_days,
        args.max_folds,
        "rolling",
    )
    model_root = Path(args.model_folder)
    model_root.mkdir(parents=True, exist_ok=True)
    fold_reports = []
    test_predictions = []
    feature_columns = None

    for fold in folds:
        print(f"[two-stage fold {fold['fold']}/{len(folds)}] loading", flush=True)
        raw = walk.load_fold_source(dataset_path, fold, minimum_timestamp)
        dataset, features = add_two_stage_features(raw, training)
        dataset = add_gate_target(
            dataset, args.gate_return_r, args.gate_max_holding_minutes
        )
        if feature_columns is None:
            feature_columns = features
        elif feature_columns != features:
            raise RuntimeError("Feature columns changed between folds.")
        dataset = dataset[
            (dataset["timestamp"] >= fold["train_start"])
            & (dataset["timestamp"] < fold["test_end"])
        ].copy()
        split = walk.split_fold(dataset, fold)

        gate_validation, direction_validation, validation_rows = train_pair(
            training,
            split["train"],
            features,
            args,
            args.random_state + fold["fold"] * 10,
        )
        selected, candidates = select_threshold_pair(
            gate_validation,
            direction_validation,
            split["validation"],
            features,
            gate_thresholds,
            direction_thresholds,
            args.minimum_validation_trades,
            backtest,
        )
        del gate_validation, direction_validation
        gc.collect()

        gate_final, direction_final, final_rows = train_pair(
            training,
            split["train_validation"],
            features,
            args,
            args.random_state + fold["fold"] * 10 + 2,
        )
        predictions = prediction_frame(
            gate_final,
            direction_final,
            split["test"],
            features,
            selected["gate_threshold"],
            selected["direction_threshold"],
            force_abstain=not selected["deployable"],
        )
        predictions["walk_forward_fold"] = fold["fold"]
        predictions["walk_forward_gate_threshold"] = selected["gate_threshold"]
        predictions["walk_forward_direction_threshold"] = selected["direction_threshold"]
        trades, equity, skipped = backtest.simulate(predictions, backtest_args())
        summary = backtest.summarize(trades, equity, skipped, predictions, backtest_args())
        gate_metrics = classifier_metrics(
            gate_final, split["test"], features, "opportunity_target"
        )
        directional_test = split["test"][split["test"]["opportunity_target"].eq(1)]
        direction_metrics = classifier_metrics(
            direction_final, directional_test, features, "fixed_1r_direction"
        )
        gate_path = model_root / f"fold_{fold['fold']:02d}_gate.json"
        direction_path = model_root / f"fold_{fold['fold']:02d}_direction.json"
        gate_final.save_model(gate_path)
        direction_final.save_model(direction_path)
        fold_reports.append(
            {
                "fold": fold["fold"],
                "periods": walk.fold_periods(fold),
                "rows": {
                    "train": len(split["train"]),
                    "validation": len(split["validation"]),
                    "test": len(split["test"]),
                    "validation_gate_training": validation_rows["gate_rows"],
                    "validation_direction_training": validation_rows["direction_rows"],
                    "final_gate_training": final_rows["gate_rows"],
                    "final_direction_training": final_rows["direction_rows"],
                },
                "selected_thresholds": selected,
                "threshold_candidates": candidates,
                "test_gate_metrics": gate_metrics,
                "test_direction_metrics": direction_metrics,
                "test_backtest": summary,
                "model_paths": {"gate": str(gate_path), "direction": str(direction_path)},
            }
        )
        test_predictions.append(predictions)
        print(
            f"[two-stage fold {fold['fold']}] deployable={selected['deployable']} "
            f"trades={summary['trades']} return={summary['total_return_pct']:.2f}%",
            flush=True,
        )
        del raw, dataset, split, gate_final, direction_final, predictions
        gc.collect()

    combined = pd.concat(test_predictions, ignore_index=True).sort_values("timestamp")
    trades, equity, skipped = backtest.simulate(combined, backtest_args())
    aggregate = backtest.summarize(trades, equity, skipped, combined, backtest_args())
    prediction_path = (
        Path(args.prediction_folder)
        / "market=usdm"
        / "symbol=BTCUSDT"
        / "timeframe=1m"
        / "two_stage_walk_forward_predictions_BTCUSDT_USDM_1m.parquet"
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(prediction_path, index=False)
    deployable_folds = sum(fold["selected_thresholds"]["deployable"] for fold in fold_reports)
    profitable_folds = sum(fold["test_backtest"]["total_return_pct"] > 0 for fold in fold_reports)
    approved = (
        deployable_folds >= 3
        and profitable_folds >= 3
        and aggregate["trades"] >= args.minimum_validation_trades * 3
        and aggregate["avg_return_r"] > 0
        and aggregate["total_return_pct"] > 0
        and aggregate["max_drawdown_pct"] >= -20
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dataset": {"path": str(dataset_path), "rows": row_count},
        "model_design": {
            "feature_set_version": FEATURE_SET_VERSION,
            "gate_target": (
                f"fixed_1r_best_return > {args.gate_return_r} and "
                f"best_holding_minutes <= {args.gate_max_holding_minutes}"
            ),
            "gate_thresholds": gate_thresholds,
            "direction_thresholds": direction_thresholds,
            "positive_validation_expectancy_required": True,
        },
        "leakage_controls": {
            "time_ordered_folds": True,
            "label_boundary_purge": True,
            "thresholds_selected_on_validation_only": True,
            "test_used_once": True,
        },
        "feature_columns": feature_columns,
        "folds": fold_reports,
        "aggregate_test_backtest": aggregate,
        "prediction_path": str(prediction_path),
        "deployable_folds": deployable_folds,
        "profitable_test_folds": profitable_folds,
        "approved_for_paper_connection": approved,
        "decision": "paper_candidate" if approved else "research_only",
        "real_trading_enabled": False,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.report_md))
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "deployable_folds": deployable_folds,
                "trades": aggregate["trades"],
                "avg_return_r": aggregate["avg_return_r"],
                "total_return_pct": aggregate["total_return_pct"],
                "report": str(report_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
