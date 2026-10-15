import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build the selective-model comparison report.")
    parser.add_argument(
        "--baseline-metadata",
        default="models_v2_3m/direction_model_BTCUSDT_USDM_1m_metadata.json",
    )
    parser.add_argument(
        "--baseline-backtest",
        default="runtime_reports/model_signal_backtest_report_v2_3m_2026-08-23.json",
    )
    parser.add_argument(
        "--candidate-metadata",
        default=(
            "models_selective_v3/"
            "selective_direction_model_BTCUSDT_USDM_1m_metadata.json"
        ),
    )
    parser.add_argument(
        "--candidate-backtest",
        default="runtime_reports/model_signal_backtest_report_selective_v3_2026-08-23.json",
    )
    parser.add_argument(
        "--report-path",
        default="docs/selective_direction_model_experiment_2026-08-23.md",
    )
    return parser.parse_args()


def read_json(path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Required result does not exist: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def pct(value):
    return f"{float(value):.2f}%"


def main():
    args = parse_args()
    baseline_model = read_json(args.baseline_metadata)
    baseline_backtest = read_json(args.baseline_backtest)
    candidate_model = read_json(args.candidate_metadata)
    candidate_backtest = read_json(args.candidate_backtest)

    metrics = candidate_model["test_directional_metrics"]
    selected = candidate_model["selected_validation_result"]
    risk_cap_pass = abs(float(candidate_backtest["largest_realized_loss_pct"])) <= 2.000001
    research_pass = (
        int(candidate_backtest["trades"]) >= 20
        and float(candidate_backtest["avg_return_r"]) > 0
        and float(candidate_backtest["total_return_pct"]) > 0
        and risk_cap_pass
    )
    improved = float(candidate_backtest["total_return_pct"]) > float(
        baseline_backtest["total_return_pct"]
    )

    threshold_rows = []
    for row in candidate_model["threshold_candidates"]:
        selected_mark = "선택" if row["threshold"] == candidate_model["selected_probability_threshold"] else ""
        threshold_rows.append(
            "| {threshold:.2f} | {trades} | {win_rate_pct:.2f}% | {avg_return_r:.4f} | "
            "{total_return_pct:.2f}% | {max_drawdown_pct:.2f}% | {mark} |".format(
                mark=selected_mark,
                **row,
            )
        )

    skip_names = list(
        dict.fromkeys(
            list(baseline_backtest.get("skipped", {}))
            + list(candidate_backtest.get("skipped", {}))
        )
    )
    skip_rows = [
        f"| `{name}` | {int(baseline_backtest.get('skipped', {}).get(name, 0)):,} | "
        f"{int(candidate_backtest.get('skipped', {}).get(name, 0)):,} |"
        for name in skip_names
    ]

    lines = [
        "# 선택형 방향 모델 개선 실험 실행 기록",
        "",
        f"작성 시각: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. 이번 단계의 목적",
        "",
        "기존 3개월 모델은 데이터 파이프라인과 거래당 2% 손실 제한은 지켰지만,",
        "모델 정확도가 기준보다 낮고 비용 반영 백테스트가 음수였습니다. 이번 단계에서는",
        "5년 수집으로 바로 확장하지 않고 같은 실제 90일 데이터를 재사용해 모델 구조를 먼저 개선했습니다.",
        "",
        "## 2. 변경한 내용",
        "",
        "- 절대 가격 대신 과거 수익률, 이동평균 이격, 변동성, 캔들 모양, 상대 거래량, RSI, 시간 주기 피처를 사용했습니다.",
        "- 모든 피처는 현재 봉과 과거 봉만 사용하며, 여러 시점에서 잘라 다시 계산하는 인과성 검사를 통과했습니다.",
        "- 여러 목표가 중 가장 좋은 결과가 아니라 고정 1R 결과로 학습 목표와 백테스트 목표를 맞췄습니다.",
        "- 방향이 애매한 행은 학습에서 제외하고, 확률이 기준보다 낮으면 `no_trade`로 처리했습니다.",
        "- 시간순 학습 60%, 검증 20%, 평가 20%로 나누고 각 경계에서 미래 라벨 240분을 제거했습니다.",
        "- 진입 확률 기준은 검증 구간에서만 선택하고 마지막 평가 구간에는 고정했습니다.",
        "",
        "이 피처들은 빠른 연구 검증을 위해 Python에서 계산했습니다. 성능이 통과해야만 같은 정의를 Flink Feature Store로 옮깁니다.",
        "",
        "## 3. 데이터 분리와 인과성 검증",
        "",
        "| 항목 | 결과 |",
        "| --- | ---: |",
        f"| 원본 ML 행 | {candidate_model['source_rows']:,} |",
        f"| 피처 생성 후 사용 가능 행 | {candidate_model['usable_rows']:,} |",
        f"| 피처 개수 | {candidate_model['feature_count']} |",
        f"| 학습 구간 행 | {candidate_model['train_rows_before_action_filter']:,} |",
        f"| 검증 구간 행 | {candidate_model['validation_rows']:,} |",
        f"| 최종 평가 구간 행 | {candidate_model['test_rows']:,} |",
        f"| 라벨 경계 제거 | {candidate_model['label_horizon_purge_minutes']}분 |",
        f"| 인과성 검사 | {candidate_model['feature_causality_check']} |",
        "",
        "## 4. 검증 구간 진입 확률 선택",
        "",
        "검증 점수는 `복리 수익률 - 0.5 × 최대 낙폭 절댓값`으로 계산했고, 최소 20회 거래가 있는 후보만 허용했습니다.",
        "",
        "| 확률 기준 | 거래 | 승률 | 평균 R | 수익률 | MDD | 결과 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *threshold_rows,
        "",
        f"선택된 기준은 `{candidate_model['selected_probability_threshold']:.2f}`이며 검증 구간 수익률은 `{selected['total_return_pct']:.2f}%`입니다.",
        "",
        "## 5. 마지막 평가 구간 결과",
        "",
        "방향 정확도는 거래 실행 여부와 다른 지표입니다. 새 모델은 이진 방향 문제이므로 기존 3분류 정확도와 숫자를 직접 비교하지 않습니다.",
        "",
        "| 항목 | 결과 |",
        "| --- | ---: |",
        f"| 평가 가능한 방향 행 | {metrics['rows']:,} |",
        f"| 방향 정확도 | {metrics['accuracy'] * 100:.2f}% |",
        f"| 다수 방향 기준 | {metrics['majority_baseline_accuracy'] * 100:.2f}% |",
        f"| 기준 대비 | {metrics['accuracy_lift_over_majority'] * 100:+.2f}%p |",
        f"| 균형 정확도 | {metrics['balanced_accuracy'] * 100:.2f}% |",
        f"| ROC AUC | {metrics['roc_auc']:.4f} |",
        "",
        "## 6. 기존 모델과 비용 반영 백테스트 비교",
        "",
        "두 백테스트 모두 거래당 계좌 위험 2%, 추가 슬리피지 5bps, 동시 포지션 1개, 일일 위험 제한을 사용했습니다.",
        "",
        "| 항목 | 기존 v2 | 선택형 v3 |",
        "| --- | ---: | ---: |",
        f"| 거래 수 | {baseline_backtest['trades']} | {candidate_backtest['trades']} |",
        f"| 승률 | {pct(baseline_backtest['win_rate_pct'])} | {pct(candidate_backtest['win_rate_pct'])} |",
        f"| 평균 R | {baseline_backtest['avg_return_r']:.4f} | {candidate_backtest['avg_return_r']:.4f} |",
        f"| 복리 수익률 | {pct(baseline_backtest['total_return_pct'])} | {pct(candidate_backtest['total_return_pct'])} |",
        f"| 최대 낙폭 | {pct(baseline_backtest['max_drawdown_pct'])} | {pct(candidate_backtest['max_drawdown_pct'])} |",
        f"| 최장 연속 손실 | {baseline_backtest['max_consecutive_losses']} | {candidate_backtest['max_consecutive_losses']} |",
        f"| 거래 1회 최대 실현 손실 | {pct(baseline_backtest['largest_realized_loss_pct'])} | {pct(candidate_backtest['largest_realized_loss_pct'])} |",
        "",
        f"기존 대비 백테스트 개선 여부: **{'개선' if improved else '개선되지 않음'}**",
        "",
        f"연구 통과 판정: **{'통과' if research_pass else '미통과'}**",
        "",
        "통과 조건은 최소 20회 거래, 평균 R 양수, 비용 후 복리 수익률 양수, 거래당 최대 손실 2% 이내입니다.",
        "한 번 통과하더라도 바로 실거래하지 않고 더 긴 기간과 여러 시장 국면에서 재검증해야 합니다.",
        "",
        "## 7. 백테스트 실행 조건과 신호 제외 사유",
        "",
        "| 조건 | 기존 v2 | 선택형 v3 |",
        "| --- | ---: | ---: |",
        f"| 익절 기준 | {baseline_backtest['take_profit_r']:.1f}R | {candidate_backtest['take_profit_r']:.1f}R |",
        f"| 거래당 계좌 위험 | {baseline_backtest['risk_per_trade_pct'] * 100:.2f}% | {candidate_backtest['risk_per_trade_pct'] * 100:.2f}% |",
        f"| 최소 신뢰도 | {baseline_backtest['min_confidence']:.2f} | {candidate_backtest['min_confidence']:.2f} |",
        f"| 최소 진입 간격 | {baseline_backtest['min_minutes_between_entries']}분 | {candidate_backtest['min_minutes_between_entries']}분 |",
        f"| 하루 최대 신규 진입 | {baseline_backtest['max_new_entries_per_day']}회 | {candidate_backtest['max_new_entries_per_day']}회 |",
        f"| 하루 손실 중단 | -{baseline_backtest['daily_stop_pct'] * 100:.2f}% | -{candidate_backtest['daily_stop_pct'] * 100:.2f}% |",
        f"| 하루 하드 중단 | -{baseline_backtest['hard_daily_stop_pct'] * 100:.2f}% | -{candidate_backtest['hard_daily_stop_pct'] * 100:.2f}% |",
        f"| 추가 슬리피지 | {baseline_backtest['extra_slippage_bps']:.2f}bps | {candidate_backtest['extra_slippage_bps']:.2f}bps |",
        "",
        "| 제외 사유 | 기존 v2 | 선택형 v3 |",
        "| --- | ---: | ---: |",
        *skip_rows,
        "",
        "한 포지션이 끝나기 전 신호는 제외하며, 연속 손실 제한은 UTC 날짜가 바뀌면 초기화합니다.",
        "미래 캔들로 만든 Triple Barrier 결과를 사용한 오프라인 연구이며 실시간 체결 성과가 아닙니다.",
        "",
        "## 8. 다음 결정",
        "",
    ]

    if research_pass:
        lines.extend(
            [
                "이번 후보는 1차 연구 조건을 통과했습니다. 다음 단계는 이 피처 정의를 Flink에 옮긴 뒤",
                "6~12개월 워크포워드 검증을 수행하는 것입니다. 5년 전체 수집과 실거래는 그 이후입니다.",
            ]
        )
    else:
        lines.extend(
            [
                "이번 후보도 1차 연구 조건을 통과하지 못했습니다. 5년 수집이나 실거래로 넘어가지 않고",
                "라벨 보유 시간과 손절 거리, 시장 국면 분리, 추가 체결 데이터 피처를 순서대로 검증해야 합니다.",
            ]
        )

    lines.extend(
        [
            "",
            "## 9. 재실행",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File .\\scripts\\run_selective_model_experiment.ps1",
            "```",
            "",
            "이 명령은 기존 90일 Parquet을 읽기만 하며 Binance 원천 데이터를 다시 다운로드하지 않습니다.",
            "",
        ]
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Selective model report: {report_path}")
    print(f"Research verdict: {'PASS' if research_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
