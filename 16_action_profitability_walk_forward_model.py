"""Run leakage-safe independent long/short profitability walk-forward research."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_SET_VERSION = "independent_action_profitability_v1"
DEFAULT_DATASET = (
    "ml_dataset_5y_20260828/market=usdm/symbol=BTCUSDT/timeframe=1m/"
    "ml_dataset_BTCUSDT_USDM_1m.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent long/short profitability walk-forward research model"
    )
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET)
    parser.add_argument("--trade-feature-manifest", help="Verified enriched dataset report; opt in to trade features")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=365)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument(
        "--probability-thresholds", default="0.52,0.55,0.58,0.60,0.65,0.70"
    )
    parser.add_argument("--probability-margins", default="0.00,0.03,0.05,0.10")
    parser.add_argument("--minimum-validation-trades", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=242)
    parser.add_argument("--extra-slippage-bps", type=float, default=5.0)
    parser.add_argument("--model-folder", default="models_action_profitability_walk_forward_5y")
    parser.add_argument(
        "--prediction-folder", default="predictions_action_profitability_walk_forward_5y"
    )
    parser.add_argument(
        "--report-json",
        default="runtime_reports/action_profitability_walk_forward_5y_2026-09-02.json",
    )
    parser.add_argument(
        "--report-md",
        default="docs/action_profitability_walk_forward_execution_2026-09-02.md",
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


def parse_probability_thresholds(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item <= 0.5 or item >= 1.0 for item in values):
        raise ValueError("Probability thresholds must be greater than 0.5 and below 1.0.")
    return values


def verified_trade_columns(manifest_path, dataset_path):
    if manifest_path is None:
        return []
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("model_training_allowed") is not True:
        raise RuntimeError("Trade dataset has not passed its training gate")
    if manifest.get("minimum_training_days", 0) < 90:
        raise RuntimeError("Trade research requires at least 90 continuous days")
    if Path(manifest["dataset_file"]).resolve() != dataset_path.resolve():
        raise RuntimeError("Trade manifest refers to another dataset")
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if digest != manifest.get("dataset_sha256"):
        raise RuntimeError("Trade dataset checksum mismatch")
    builder = load_module("17_build_trade_enriched_dataset.py", "action_trade_builder")
    empty = pd.DataFrame(columns=["timestamp"] + builder.BASE_TRADE_COLUMNS)
    _, allowed = builder.add_causal_trade_features(empty)
    allowed += ["agg_trade_vwap_to_close"]
    columns = manifest.get("trade_feature_columns", [])
    if columns != allowed:
        raise RuntimeError("Trade feature list differs from the causal feature contract")
    timestamps = pd.read_parquet(dataset_path, columns=["timestamp"])["timestamp"]
    if len(timestamps) < 90 * 1440 or not timestamps.diff().dropna().eq(60_000).all():
        raise RuntimeError("Trade dataset does not contain 90 continuous days")
    return columns


def parse_probability_margins(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 0.0 or item >= 1.0 for item in values):
        raise ValueError("Probability margins must be at least 0 and below 1.0.")
    return values


def add_profitability_targets(
    frame: pd.DataFrame, extra_slippage_bps: float
) -> pd.DataFrame:
    """Create action-specific targets that match the backtest cost assumptions."""
    output = frame.copy()
    stop_distance = pd.to_numeric(output["stop_distance_pct"], errors="raise")
    if (stop_distance <= 0).any():
        raise RuntimeError("stop_distance_pct must be positive.")
    extra_slippage_r = (extra_slippage_bps / 10_000) / stop_distance
    output["extra_slippage_r"] = extra_slippage_r

    for action in ("long", "short"):
        label_column = f"{action}_tp_1_0r_return_r_net"
        adjusted_column = f"{action}_profitability_return_r"
        target_column = f"{action}_profitability_target"
        output[adjusted_column] = (
            pd.to_numeric(output[label_column], errors="raise") - extra_slippage_r
        )
        output[target_column] = output[adjusted_column].gt(0).astype(int)
        if output[target_column].nunique() != 2:
            raise RuntimeError(f"{target_column} must contain both classes.")
    return output


def train_classifier(training, frame, features, target_column, args, seed):
    target = frame[target_column].astype(int)
    if target.nunique() != 2:
        raise RuntimeError(f"{target_column} must contain both classes for training.")
    model = training.make_model(seed)
    model.set_params(n_estimators=args.n_estimators, n_jobs=args.n_jobs)
    model.fit(frame[features], target, sample_weight=training.sample_weights(target))
    return model


def train_pair(training, frame, features, args, seed):
    long_model = train_classifier(
        training, frame, features, "long_profitability_target", args, seed
    )
    short_model = train_classifier(
        training, frame, features, "short_profitability_target", args, seed + 1
    )
    return long_model, short_model


def prediction_frame(
    long_model,
    short_model,
    frame: pd.DataFrame,
    features: list[str],
    probability_threshold: float,
    probability_margin: float,
    *,
    force_abstain: bool = False,
) -> pd.DataFrame:
    probability_long = long_model.predict_proba(frame[features])[:, 1]
    probability_short = short_model.predict_proba(frame[features])[:, 1]
    if force_abstain:
        action = np.full(len(frame), "no_trade", dtype=object)
    else:
        long_qualified = (
            (probability_long >= probability_threshold)
            & ((probability_long - probability_short) >= probability_margin)
            & (probability_long > probability_short)
        )
        short_qualified = (
            (probability_short >= probability_threshold)
            & ((probability_short - probability_long) >= probability_margin)
            & (probability_short > probability_long)
        )
        action = np.where(long_qualified, "long", np.where(short_qualified, "short", "no_trade"))

    output = frame.copy()
    output["original_direction_target"] = output["direction_target"]
    output["original_best_action"] = output["best_action"]
    output["direction_target"] = output["fixed_1r_direction"]
    output["best_action"] = output["fixed_1r_best_action"]
    output["pred_action"] = action
    output["pred_direction_target"] = np.where(
        action == "long", 1, np.where(action == "short", 0, -1)
    )
    output["prob_long_profit"] = probability_long
    output["prob_short_profit"] = probability_short
    output["probability_advantage"] = np.abs(probability_long - probability_short)
    output["pred_confidence"] = np.where(
        action == "long",
        probability_long,
        np.where(action == "short", probability_short, np.maximum(probability_long, probability_short)),
    )
    output["prob_no_trade"] = 1.0 - output["pred_confidence"]
    output["model_feature_set_version"] = FEATURE_SET_VERSION
    return output


def backtest_args(extra_slippage_bps: float = 5.0) -> SimpleNamespace:
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
        extra_slippage_bps=extra_slippage_bps,
    )


def evaluate_candidate(backtest, predictions, extra_slippage_bps=5.0, take_profit_r=1.0):
    signals = predictions[predictions["pred_action"].ne("no_trade")].copy()
    args = backtest_args(extra_slippage_bps)
    args.take_profit_r = take_profit_r
    trades, equity, skipped = backtest.simulate(signals, args)
    summary = backtest.summarize(trades, equity, skipped, signals, args)
    signal_count = len(signals)
    long_signals = int(signals["pred_action"].eq("long").sum())
    short_signals = int(signals["pred_action"].eq("short").sum())
    return {
        "signals": signal_count,
        "long_signals": long_signals,
        "short_signals": short_signals,
        "maximum_direction_share": (
            max(long_signals, short_signals) / signal_count if signal_count else 1.0
        ),
        "trades": int(summary["trades"]),
        "win_rate_pct": float(summary["win_rate_pct"]),
        "avg_return_r": float(summary["avg_return_r"]),
        "total_return_pct": float(summary["total_return_pct"]),
        "max_drawdown_pct": float(summary["max_drawdown_pct"]),
    }


def select_action_policy(
    long_model,
    short_model,
    validation,
    features,
    probability_thresholds,
    probability_margins,
    minimum_trades,
    backtest,
    extra_slippage_bps=5.0,
):
    candidates = []
    for threshold in probability_thresholds:
        for margin in probability_margins:
            predictions = prediction_frame(
                long_model, short_model, validation, features, threshold, margin
            )
            metrics = evaluate_candidate(backtest, predictions, extra_slippage_bps)
            metrics.update(
                {
                    "probability_threshold": threshold,
                    "probability_margin": margin,
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
            "probability_threshold": 1.0,
            "probability_margin": 1.0,
            "abstain_reason": "no_positive_expectancy_action_profitability_candidate",
            "best_evaluated_candidate": best_evaluated,
        }, candidates

    selected = dict(
        max(
            profitable,
            key=lambda item: (item["selection_score"], item["avg_return_r"], item["trades"]),
        )
    )
    selected["deployable"] = True
    selected["abstain_reason"] = None
    return selected, candidates


def classifier_metrics(model, frame, features, target_column):
    target = frame[target_column].astype(int)
    probabilities = model.predict_proba(frame[features])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    auc = float(roc_auc_score(target, probabilities)) if target.nunique() == 2 else None
    return {
        "rows": len(frame),
        "positive_share": float(target.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "roc_auc": auc,
    }


def metric_text(value):
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(report: dict, path: Path) -> None:
    aggregate = report["aggregate_test_backtest"]
    lines = [
        "# 롱·숏 독립 수익성 모델 Walk-Forward 실행 결과",
        "",
        f"- 실행 시각: `{report['created_at_utc']}`",
        f"- 데이터: `{report['dataset']['path']}`",
        f"- 전체 데이터 행: `{report['dataset']['rows']:,}`",
        f"- 최종 판단: **{report['decision']}**",
        f"- 실거래 연결: **{report['real_trading_enabled']}**",
        "",
        "## 이번 실험의 목적",
        "",
        "기존 모델은 먼저 거래 기회 또는 방향을 예측했습니다. 이번 모델은 질문을 바꿔",
        "`롱을 했을 때 설정한 비용을 빼고 수익인가?`와 `숏을 했을 때 수익인가?`를 각각",
        "독립적으로 학습했습니다. 둘 중 한쪽만 검증 기준을 통과할 때만 거래하고, 기준을",
        "통과하지 못하면 `no_trade`를 출력합니다.",
        "비용은 라벨의 고정 왕복 비용과 추가 슬리피지 가정입니다. 실제 스프레드·펀딩 현금흐름·지연을 모두 재현하지 않습니다.",
        "",
        "## 누수와 과최적화 방지",
        "",
        "- 시간순 rolling train / validation / test 분할",
        "- 각 경계에서 최대 라벨 보유 시간만큼 purge",
        "- 확률 기준과 롱·숏 확률 차이는 validation에서만 선택",
        "- validation 기대값이 양수가 아니면 test 전체를 `no_trade`로 강제",
        "- test 결과로 기준을 다시 고르지 않음",
        "",
        "## Fold별 결과",
        "",
        "| Fold | 확률 기준 | 최소 차이 | 배포 가능 | 거래 | 평균 R | 수익률 | MDD | Long AUC | Short AUC |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in report["folds"]:
        selected = fold["selected_policy"]
        result = fold["test_backtest"]
        lines.append(
            f"| {fold['fold']} | {selected['probability_threshold']:.2f} | "
            f"{selected['probability_margin']:.2f} | {selected['deployable']} | "
            f"{result['trades']} | {result['avg_return_r']:.4f} | "
            f"{result['total_return_pct']:.2f}% | {result['max_drawdown_pct']:.2f}% | "
            f"{metric_text(fold['test_long_metrics']['roc_auc'])} | "
            f"{metric_text(fold['test_short_metrics']['roc_auc'])} |"
        )
    lines += [
        "",
        "## Validation에서 가장 나았던 후보",
        "",
        "아래는 validation에서 선택한 후보입니다. 채택 여부는 위 표의 배포 가능 값으로 구분합니다.",
        "",
        "| Fold | 확률 기준 | 최소 차이 | 거래 | 평균 R | 수익률 | MDD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in report["folds"]:
        candidate = fold["selected_policy"].get("best_evaluated_candidate")
        if candidate is None:
            candidate = fold["selected_policy"]
        lines.append(
            f"| {fold['fold']} | {candidate['probability_threshold']:.2f} | "
            f"{candidate['probability_margin']:.2f} | {candidate['trades']} | "
            f"{candidate['avg_return_r']:.4f} | {candidate['total_return_pct']:.2f}% | "
            f"{candidate['max_drawdown_pct']:.2f}% |"
        )
    lines += [
        "",
        "## 통합 미관측 테스트 결과",
        "",
        f"- 거래 수: `{aggregate['trades']}`",
        f"- 승률: `{aggregate['win_rate_pct']:.2f}%`",
        f"- 평균 손익: `{aggregate['avg_return_r']:.4f}R`",
        f"- 누적 계좌 수익률: `{aggregate['total_return_pct']:.2f}%`",
        f"- 최대 낙폭: `{aggregate['max_drawdown_pct']:.2f}%`",
        f"- validation 기준을 통과한 fold: `{report['deployable_folds']}` / `{len(report['folds'])}`",
        f"- 수익인 test fold: `{report['profitable_test_folds']}` / `{len(report['folds'])}`",
        "",
        "## 해석과 안전 상태",
        "",
    ]
    if report["approved_for_paper_connection"]:
        lines.append("검증 관문을 통과해 장기 페이퍼 트레이딩 후보로만 분류했습니다.")
    else:
        lines.append(
            "검증 관문을 통과하지 못했으므로 연구 결과로만 보관합니다. 추론 서비스와 주문 API에는 연결하지 않습니다."
        )
    lines += [
        "",
        "실거래는 이 실험 결과와 관계없이 계속 차단되어 있습니다. 장기 페이퍼 검증과 별도 사람 승인을",
        "모두 통과하기 전에는 거래소 주문 API를 사용할 수 없습니다.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if args.minimum_validation_trades <= 0 or args.n_estimators <= 0 or args.n_jobs <= 0:
        raise ValueError("Trade, estimator, and job counts must be positive.")
    if args.extra_slippage_bps < 0:
        raise ValueError("extra_slippage_bps cannot be negative.")
    trade_columns = verified_trade_columns(args.trade_feature_manifest, dataset_path)

    thresholds = parse_probability_thresholds(args.probability_thresholds)
    margins = parse_probability_margins(args.probability_margins)
    walk = load_module("14_walk_forward_direction_model.py", "action_profit_walk_helpers")
    training = load_module("13_train_selective_direction_model.py", "action_profit_training")
    two_stage = load_module("15_two_stage_walk_forward_model.py", "action_profit_features")
    backtest = load_module("8_model_signal_backtest.py", "action_profit_backtest")
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
        print(f"[action-profit fold {fold['fold']}/{len(folds)}] loading", flush=True)
        raw = walk.load_fold_source(dataset_path, fold, minimum_timestamp, trade_columns)
        dataset, features = two_stage.add_two_stage_features(raw, training)
        if trade_columns:
            if not np.isfinite(dataset[trade_columns].to_numpy(dtype=float)).all():
                raise RuntimeError("Trade model input contains missing or nonfinite features")
            features = features + trade_columns
        dataset = add_profitability_targets(dataset, args.extra_slippage_bps)
        if feature_columns is None:
            feature_columns = features
        elif feature_columns != features:
            raise RuntimeError("Feature columns changed between folds.")
        dataset = dataset[
            (dataset["timestamp"] >= fold["train_start"])
            & (dataset["timestamp"] < fold["test_end"])
        ].copy()
        split = walk.split_fold(dataset, fold)

        long_validation, short_validation = train_pair(
            training,
            split["train"],
            features,
            args,
            args.random_state + fold["fold"] * 10,
        )
        selected, candidates = select_action_policy(
            long_validation,
            short_validation,
            split["validation"],
            features,
            thresholds,
            margins,
            args.minimum_validation_trades,
            backtest,
            args.extra_slippage_bps,
        )
        del long_validation, short_validation
        gc.collect()

        long_final, short_final = train_pair(
            training,
            split["train_validation"],
            features,
            args,
            args.random_state + fold["fold"] * 10 + 2,
        )
        predictions = prediction_frame(
            long_final,
            short_final,
            split["test"],
            features,
            selected["probability_threshold"],
            selected["probability_margin"],
            force_abstain=not selected["deployable"],
        )
        predictions["walk_forward_fold"] = fold["fold"]
        predictions["walk_forward_probability_threshold"] = selected[
            "probability_threshold"
        ]
        predictions["walk_forward_probability_margin"] = selected["probability_margin"]
        bt_args = backtest_args(args.extra_slippage_bps)
        trades, equity, skipped = backtest.simulate(predictions, bt_args)
        summary = backtest.summarize(trades, equity, skipped, predictions, bt_args)
        long_metrics = classifier_metrics(
            long_final, split["test"], features, "long_profitability_target"
        )
        short_metrics = classifier_metrics(
            short_final, split["test"], features, "short_profitability_target"
        )
        long_path = model_root / f"fold_{fold['fold']:02d}_long_profit.json"
        short_path = model_root / f"fold_{fold['fold']:02d}_short_profit.json"
        long_final.save_model(long_path)
        short_final.save_model(short_path)
        fold_reports.append(
            {
                "fold": fold["fold"],
                "periods": walk.fold_periods(fold),
                "rows": {
                    "train": len(split["train"]),
                    "validation": len(split["validation"]),
                    "test": len(split["test"]),
                    "train_purged_rows": split["train_purged_rows"],
                    "validation_purged_rows": split["validation_purged_rows"],
                    "final_purged_rows": split["final_purged_rows"],
                },
                "training_target_share": {
                    "long": float(split["train_validation"]["long_profitability_target"].mean()),
                    "short": float(split["train_validation"]["short_profitability_target"].mean()),
                },
                "selected_policy": selected,
                "policy_candidates": candidates,
                "test_long_metrics": long_metrics,
                "test_short_metrics": short_metrics,
                "test_backtest": summary,
                "model_paths": {"long": str(long_path), "short": str(short_path)},
            }
        )
        test_predictions.append(predictions)
        print(
            f"[action-profit fold {fold['fold']}] deployable={selected['deployable']} "
            f"trades={summary['trades']} return={summary['total_return_pct']:.2f}%",
            flush=True,
        )
        del raw, dataset, split, long_final, short_final, predictions
        gc.collect()

    combined = pd.concat(test_predictions, ignore_index=True).sort_values("timestamp")
    bt_args = backtest_args(args.extra_slippage_bps)
    trades, equity, skipped = backtest.simulate(combined, bt_args)
    aggregate = backtest.summarize(trades, equity, skipped, combined, bt_args)
    prediction_path = (
        Path(args.prediction_folder)
        / "market=usdm"
        / "symbol=BTCUSDT"
        / "timeframe=1m"
        / "action_profitability_walk_forward_predictions_BTCUSDT_USDM_1m.parquet"
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(prediction_path, index=False)

    deployable_folds = sum(fold["selected_policy"]["deployable"] for fold in fold_reports)
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
            "trade_feature_manifest": args.trade_feature_manifest,
            "trade_feature_count": len(trade_columns),
            "feature_count": len(feature_columns),
            "long_target": "long 1R label return minus extra slippage is positive",
            "short_target": "short 1R label return minus extra slippage is positive",
            "extra_slippage_bps": args.extra_slippage_bps,
            "probability_thresholds": thresholds,
            "probability_margins": margins,
            "positive_validation_expectancy_required": True,
        },
        "leakage_controls": {
            "time_ordered_folds": True,
            "label_boundary_purge": True,
            "policy_selected_on_validation_only": True,
            "test_used_once": True,
            "force_no_trade_without_positive_validation_expectancy": True,
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
