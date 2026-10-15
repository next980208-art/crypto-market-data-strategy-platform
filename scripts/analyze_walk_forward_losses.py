"""Explain why the current out-of-sample walk-forward strategy lost money."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "predictions_walk_forward_5y"
    / "market=usdm"
    / "symbol=BTCUSDT"
    / "timeframe=1m"
    / "walk_forward_predictions_BTCUSDT_USDM_1m.parquet"
)
DEFAULT_WALK_REPORT = PROJECT_ROOT / "runtime_reports" / "walk_forward_5y_report_2026-08-29.json"
DEFAULT_JSON = PROJECT_ROOT / "runtime_reports" / "walk_forward_loss_diagnostics_2026-08-31.json"
DEFAULT_MD = PROJECT_ROOT / "docs" / "walk_forward_loss_diagnostics_2026-08-31.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose current walk-forward losses.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--walk-report", type=Path, default=DEFAULT_WALK_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--actionable-return-r", type=float, default=0.20)
    parser.add_argument("--extra-slippage-bps", type=float, default=5.0)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> float:
    number = float(value)
    return number if np.isfinite(number) else 0.0


def percentage(mask: pd.Series) -> float:
    return finite(mask.mean() * 100) if len(mask) else 0.0


def summarize_group(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    output = []
    for value, group in frame.groupby(column, observed=True, dropna=False):
        output.append(
            {
                column: str(value),
                "trades": int(len(group)),
                "win_rate_pct": percentage(group["return_r"] > 0),
                "avg_return_r": finite(group["return_r"].mean()),
                "sum_return_r": finite(group["return_r"].sum()),
                "avg_confidence": finite(group["confidence"].mean()),
            }
        )
    return output


def build_backtest_args() -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTCUSDT",
        market="usdm",
        timeframe="1m",
        take_profit_r=1.0,
        risk_per_trade_pct=0.02,
        min_confidence=0.5,
        min_minutes_between_entries=15,
        max_new_entries_per_day=5,
        daily_stop_pct=0.04,
        hard_daily_stop_pct=0.06,
        max_consecutive_losses=3,
        extra_slippage_bps=5.0,
    )


def validation_diagnostics(walk_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fold in walk_report["folds"]:
        candidates = fold["threshold_candidates"]
        positive = [
            candidate
            for candidate in candidates
            if candidate["eligible"]
            and candidate["avg_return_r"] > 0
            and candidate["total_return_pct"] > 0
        ]
        selected = fold["selected_threshold"]
        rows.append(
            {
                "fold": int(fold["fold"]),
                "selected_threshold": finite(selected["threshold"]),
                "selected_validation_trades": int(selected["trades"]),
                "selected_validation_avg_return_r": finite(selected["avg_return_r"]),
                "selected_validation_return_pct": finite(selected["total_return_pct"]),
                "eligible_candidates": int(sum(candidate["eligible"] for candidate in candidates)),
                "profitable_eligible_candidates": int(len(positive)),
                "should_abstain": len(positive) == 0,
            }
        )
    return rows


def signal_diagnostics(predictions: pd.DataFrame, actionable_return_r: float) -> dict[str, Any]:
    selected = predictions[predictions["pred_action"].ne("no_trade")].copy()
    selected["selected_label_return_r_net"] = np.where(
        selected["pred_action"].eq("long"),
        selected["long_tp_1_0r_return_r_net"],
        selected["short_tp_1_0r_return_r_net"],
    )
    selected["opposite_label_return_r_net"] = np.where(
        selected["pred_action"].eq("long"),
        selected["short_tp_1_0r_return_r_net"],
        selected["long_tp_1_0r_return_r_net"],
    )
    selected["direction_correct"] = selected["pred_action"].eq(selected["fixed_1r_best_action"])
    selected["tradeable"] = selected["fixed_1r_best_return"] > actionable_return_r
    selected["both_directions_negative"] = (
        selected["long_tp_1_0r_return_r_net"].le(0)
        & selected["short_tp_1_0r_return_r_net"].le(0)
    )
    folds = []
    for fold, group in selected.groupby("walk_forward_fold"):
        folds.append(
            {
                "fold": int(fold),
                "signals": int(len(group)),
                "long_signals": int(group["pred_action"].eq("long").sum()),
                "short_signals": int(group["pred_action"].eq("short").sum()),
                "direction_accuracy_pct": percentage(group["direction_correct"]),
                "tradeable_share_pct": percentage(group["tradeable"]),
                "both_directions_negative_pct": percentage(group["both_directions_negative"]),
                "selected_action_positive_pct": percentage(group["selected_label_return_r_net"] > 0),
                "selected_action_avg_return_r_net": finite(group["selected_label_return_r_net"].mean()),
                "opposite_action_avg_return_r_net": finite(group["opposite_label_return_r_net"].mean()),
            }
        )

    confidence_bins = pd.cut(
        selected["pred_confidence"],
        bins=[0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.000001],
        right=False,
        include_lowest=True,
    )
    confidence = []
    for bucket, group in selected.groupby(confidence_bins, observed=True):
        confidence.append(
            {
                "confidence_bin": str(bucket),
                "signals": int(len(group)),
                "direction_accuracy_pct": percentage(group["direction_correct"]),
                "selected_action_positive_pct": percentage(group["selected_label_return_r_net"] > 0),
                "selected_action_avg_return_r_net": finite(group["selected_label_return_r_net"].mean()),
            }
        )
    return {
        "total_signals_before_risk_filters": int(len(selected)),
        "long_signals": int(selected["pred_action"].eq("long").sum()),
        "short_signals": int(selected["pred_action"].eq("short").sum()),
        "direction_accuracy_pct": percentage(selected["direction_correct"]),
        "tradeable_share_pct": percentage(selected["tradeable"]),
        "both_directions_negative_pct": percentage(selected["both_directions_negative"]),
        "selected_action_positive_pct": percentage(selected["selected_label_return_r_net"] > 0),
        "selected_action_avg_return_r_net": finite(selected["selected_label_return_r_net"].mean()),
        "opposite_action_avg_return_r_net": finite(selected["opposite_label_return_r_net"].mean()),
        "by_fold": folds,
        "by_confidence": confidence,
    }


def enrich_executed_trades(
    predictions: pd.DataFrame,
    trades: list[dict[str, Any]],
    extra_slippage_bps: float,
) -> pd.DataFrame:
    trade_frame = pd.DataFrame(trades)
    source_columns = [
        "timestamp",
        "walk_forward_fold",
        "close_to_ma_60",
        "volatility_60m",
        "funding_rate",
        "mark_basis_pct",
        "stop_distance_pct",
        "round_trip_cost_bps",
        "long_tp_1_0r_return_r_net",
        "short_tp_1_0r_return_r_net",
    ]
    merged = trade_frame.merge(
        predictions[source_columns],
        left_on="signal_timestamp",
        right_on="timestamp",
        how="left",
        validate="many_to_one",
    )
    if merged["walk_forward_fold"].isna().any():
        raise RuntimeError("Could not map every executed trade back to its prediction row.")
    merged["trend_regime"] = np.select(
        [merged["close_to_ma_60"] > 0.005, merged["close_to_ma_60"] < -0.005],
        ["uptrend", "downtrend"],
        default="sideways",
    )
    low_vol, high_vol = predictions["volatility_60m"].quantile([1 / 3, 2 / 3]).tolist()
    merged["volatility_regime"] = np.select(
        [merged["volatility_60m"] <= low_vol, merged["volatility_60m"] >= high_vol],
        ["low", "high"],
        default="medium",
    )
    merged["confidence_bin"] = pd.cut(
        merged["confidence"],
        bins=[0.50, 0.60, 0.65, 0.70, 0.80, 1.000001],
        right=False,
        include_lowest=True,
    ).astype(str)
    merged["label_cost_r"] = (
        merged["round_trip_cost_bps"] / 10_000
    ) / merged["stop_distance_pct"]
    merged["extra_slippage_r"] = (
        extra_slippage_bps / 10_000
    ) / merged["stop_distance_pct"]
    merged["approx_gross_return_r"] = merged["label_return_r_net"] + merged["label_cost_r"]
    merged["after_all_costs_before_risk_normalization_r"] = (
        merged["label_return_r_net"] - merged["extra_slippage_r"]
    )
    merged["costs_flipped_positive_to_nonpositive"] = (
        merged["approx_gross_return_r"] > 0
    ) & (merged["after_all_costs_before_risk_normalization_r"] <= 0)
    return merged


def executed_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "trades": int(len(frame)),
        "win_rate_pct": percentage(frame["return_r"] > 0),
        "avg_return_r": finite(frame["return_r"].mean()),
        "long_trades": int(frame["action"].eq("long").sum()),
        "short_trades": int(frame["action"].eq("short").sum()),
        "avg_approx_gross_return_r": finite(frame["approx_gross_return_r"].mean()),
        "avg_label_return_r_net": finite(frame["label_return_r_net"].mean()),
        "avg_extra_slippage_r": finite(frame["extra_slippage_r"].mean()),
        "avg_final_risk_normalized_return_r": finite(frame["return_r"].mean()),
        "cost_flip_trades": int(frame["costs_flipped_positive_to_nonpositive"].sum()),
        "by_fold": summarize_group(frame, "walk_forward_fold"),
        "by_action": summarize_group(frame, "action"),
        "by_trend_regime": summarize_group(frame, "trend_regime"),
        "by_volatility_regime": summarize_group(frame, "volatility_regime"),
        "by_confidence": summarize_group(frame, "confidence_bin"),
    }


def derive_findings(report: dict[str, Any]) -> list[dict[str, str]]:
    validation = report["validation_thresholds"]
    signals = report["signals_before_risk_filters"]
    executed = report["executed_trades"]
    findings = []
    if all(row["should_abstain"] for row in validation):
        findings.append(
            {
                "severity": "critical",
                "finding": "모든 Fold의 검증 후보가 손실인데도 기존 선택기가 반드시 거래 임곗값을 골랐다.",
                "action": "검증 기대값과 수익률이 양수가 아니면 해당 Fold를 no_trade로 만드는 안전 관문을 추가한다.",
            }
        )
    if signals["direction_accuracy_pct"] < 50:
        findings.append(
            {
                "severity": "critical",
                "finding": "신뢰도 통과 신호의 방향 정확도가 50%보다 낮아 방향 모델에 재현 가능한 우위가 없다.",
                "action": "바로 방향을 예측하기 전에 거래 가능 여부를 예측하는 1단계 gate를 추가한다.",
            }
        )
    if max(signals["long_signals"], signals["short_signals"]) / max(1, signals["total_signals_before_risk_filters"]) > 0.8:
        findings.append(
            {
                "severity": "high",
                "finding": "고신뢰 신호가 한 방향으로 심하게 쏠려 Fold 간 확률 보정이 불안정하다.",
                "action": "Fold별 확률 분포와 방향별 최소 표본을 검증하고 한 방향 신호만 발생하면 후보를 거절한다.",
            }
        )
    if (
        executed["avg_approx_gross_return_r"] > 0
        and executed["avg_label_return_r_net"] <= 0
    ):
        findings.append(
            {
                "severity": "high",
                "finding": "비용 전 근사 우위는 아주 작게 양수지만 왕복 비용과 슬리피지보다 작아 순손익이 음수가 됐다.",
                "action": "비용보다 충분히 큰 검증 기대값을 필수 승인 조건으로 두고 비용 전 성능만으로 후보를 승인하지 않는다.",
            }
        )
    elif executed["avg_approx_gross_return_r"] <= 0:
        findings.append(
            {
                "severity": "high",
                "finding": "비용을 되돌려 근사한 총손익도 음수여서 수수료만의 문제가 아니다.",
                "action": "비용 최적화보다 라벨, 진입 조건, 체결·호가·선물 문맥 피처 개선을 우선한다.",
            }
        )
    return findings


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(report: dict[str, Any], path: Path) -> None:
    aggregate = report["original_backtest"]
    signals = report["signals_before_risk_filters"]
    executed = report["executed_trades"]
    lines = [
        "# Walk-Forward 손실 원인 진단",
        "",
        "## 결론부터",
        "",
        "현재 손실의 핵심은 **검증 우위가 없는 방향 모델을 억지로 거래시킨 선택 구조**와",
        "**현실 비용보다 작은 미약한 총손익**이다. 네 Fold 모두 검증 구간에서 양의 기대값을 가진",
        "적격 임곗값이 0개였지만 기존 코드는 손실 후보 중 하나를 반드시 선택했다.",
        "",
        f"- Out-of-Sample 거래: `{aggregate['trades']}`건",
        f"- 평균 손익: `{aggregate['avg_return_r']:.4f}R`",
        f"- 누적 수익률: `{aggregate['total_return_pct']:.2f}%`",
        f"- 최대 낙폭: `{aggregate['max_drawdown_pct']:.2f}%`",
        f"- 위험 필터 전 고신뢰 신호 방향 정확도: `{signals['direction_accuracy_pct']:.2f}%`",
        f"- 실행 거래의 비용 복원 근사 총손익: `{executed['avg_approx_gross_return_r']:.4f}R`",
        "",
        "## 검증 임곗값 문제",
        "",
    ]
    lines += markdown_table(
        report["validation_thresholds"],
        [
            ("fold", "Fold"),
            ("selected_threshold", "선택 임곗값"),
            ("selected_validation_trades", "검증 거래"),
            ("selected_validation_avg_return_r", "검증 평균 R"),
            ("profitable_eligible_candidates", "수익 후보 수"),
            ("should_abstain", "거래 중단 필요"),
        ],
    )
    lines += [
        "",
        "`수익 후보 수=0`이면 0R인 `no_trade`가 모든 후보보다 낫다. 앞으로는 이 경우 테스트와",
        "페이퍼 경로에 신호를 내보내지 않는다.",
        "",
        "## 위험 필터 전 신호",
        "",
        f"- 전체 신호: `{signals['total_signals_before_risk_filters']:,}`건",
        f"- Long / Short: `{signals['long_signals']:,}` / `{signals['short_signals']:,}`",
        f"- 방향 정확도: `{signals['direction_accuracy_pct']:.2f}%`",
        f"- 선택 방향 평균 라벨 손익: `{signals['selected_action_avg_return_r_net']:.4f}R`",
        f"- 반대 방향 평균 라벨 손익: `{signals['opposite_action_avg_return_r_net']:.4f}R`",
        f"- 양 방향 모두 손실인 신호: `{signals['both_directions_negative_pct']:.2f}%`",
        "",
        "## 실제 실행 거래 분해",
        "",
    ]
    lines += markdown_table(
        executed["by_action"],
        [
            ("action", "방향"),
            ("trades", "거래"),
            ("win_rate_pct", "승률 %"),
            ("avg_return_r", "평균 R"),
            ("sum_return_r", "합계 R"),
        ],
    )
    lines += ["", "시장 추세별 결과:", ""]
    lines += markdown_table(
        executed["by_trend_regime"],
        [
            ("trend_regime", "추세"),
            ("trades", "거래"),
            ("win_rate_pct", "승률 %"),
            ("avg_return_r", "평균 R"),
            ("sum_return_r", "합계 R"),
        ],
    )
    lines += [
        "",
        "## 비용 영향",
        "",
        f"- 비용 복원 근사 평균: `{executed['avg_approx_gross_return_r']:.4f}R`",
        f"- 라벨 기본 비용 포함 평균: `{executed['avg_label_return_r_net']:.4f}R`",
        f"- 추가 슬리피지 평균: `{executed['avg_extra_slippage_r']:.4f}R`",
        f"- 최종 위험 정규화 평균: `{executed['avg_final_risk_normalized_return_r']:.4f}R`",
        f"- 비용 때문에 양수에서 0 이하로 바뀐 거래: `{executed['cost_flip_trades']}`건",
        "",
        "실행 거래의 비용 전 근사 평균은 아주 작은 양수지만 기본 왕복 비용과 추가 슬리피지보다",
        "훨씬 작다. 수수료를 일부 줄이는 것만으로는 충분하지 않고, 현실 비용을 넉넉히 넘는 검증",
        "기대값이 있는 모델만 통과시켜야 한다.",
        "",
        "## 발견 사항과 수정 방향",
        "",
    ]
    for finding in report["findings"]:
        lines.append(f"- **{finding['severity']}**: {finding['finding']} {finding['action']}")
    lines += [
        "",
        "## 다음 실험 설계",
        "",
        "1. 1단계 모델은 현재 시점이 비용 포함 양의 기대값을 가질 수 있는지 판정한다.",
        "2. 1단계를 통과한 행에서만 2단계 모델이 Long과 Short 방향을 선택한다.",
        "3. 검증 구간에서 기대값이 양수인 임곗값 조합이 없으면 해당 Fold는 no_trade다.",
        "4. 한 방향 신호 쏠림, 최소 거래 수, 비용 포함 기대값을 모두 승인 조건에 포함한다.",
        "5. 개선 모델도 같은 시간순 Train/Validation/Test와 Walk-Forward 기준으로만 비교한다.",
        "",
        "실거래 주문 API는 계속 차단하며, 이 진단 결과는 수익을 보장하지 않는다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    walk_report = json.loads(args.walk_report.read_text(encoding="utf-8"))
    backtest = load_module(PROJECT_ROOT / "8_model_signal_backtest.py", "loss_diagnostic_backtest")
    backtest_args = build_backtest_args()
    trades, equity, skipped = backtest.simulate(predictions, backtest_args)
    original_summary = backtest.summarize(trades, equity, skipped, predictions, backtest_args)
    expected = walk_report["aggregate_test_backtest"]
    if len(trades) != int(expected["trades"]):
        raise RuntimeError(f"Trade replay mismatch: expected={expected['trades']}, actual={len(trades)}")
    if not np.isclose(original_summary["total_return_pct"], expected["total_return_pct"]):
        raise RuntimeError("Backtest replay return does not match the original report.")

    executed = enrich_executed_trades(predictions, trades, args.extra_slippage_bps)
    report: dict[str, Any] = {
        "status": "completed",
        "predictions_path": str(args.predictions.relative_to(PROJECT_ROOT)),
        "prediction_rows": int(len(predictions)),
        "original_backtest": original_summary,
        "validation_thresholds": validation_diagnostics(walk_report),
        "signals_before_risk_filters": signal_diagnostics(
            predictions, args.actionable_return_r
        ),
        "executed_trades": executed_diagnostics(executed),
        "replay_matches_original": True,
        "real_trading_enabled": False,
    }
    report["findings"] = derive_findings(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.output_md)
    executed_output = args.output_json.with_name(args.output_json.stem + "_trades.parquet")
    executed.to_parquet(executed_output, index=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "trades": original_summary["trades"],
                "avg_return_r": original_summary["avg_return_r"],
                "all_validation_folds_should_abstain": all(
                    row["should_abstain"] for row in report["validation_thresholds"]
                ),
                "report": str(args.output_json.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
