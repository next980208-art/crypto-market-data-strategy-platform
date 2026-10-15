"""Validate the 90-day v2 research pipeline and write a Markdown report."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCHEMA = "ohlcv_basic_v2_boundary4"
CONTEXT_SCHEMA = "usdm_context_v2_complete_mark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the 90-day v2 research pipeline.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--report-path",
        default="docs/three_month_v2_research_pipeline_execution_2026-08-23.md",
    )
    return parser.parse_args()


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def read_json(path: Path, errors: list[str]) -> dict[str, object]:
    if not path.exists():
        errors.append(f"파일 없음: {path.relative_to(PROJECT_ROOT)}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"JSON 읽기 실패: {path.relative_to(PROJECT_ROOT)}: {exc}")
        return {}


def read_parquets(root: Path, errors: list[str]) -> pd.DataFrame:
    paths = sorted(root.rglob("*.parquet")) if root.exists() else []
    if not paths:
        errors.append(f"Parquet 없음: {root.relative_to(PROJECT_ROOT)}")
        return pd.DataFrame()
    try:
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    except Exception as exc:
        errors.append(f"Parquet 읽기 실패: {root.relative_to(PROJECT_ROOT)}: {exc}")
        return pd.DataFrame()


def expected_timestamps(start: datetime, end: datetime) -> list[int]:
    return list(range(int(start.timestamp() * 1000), int(end.timestamp() * 1000), 60_000))


def validate_time_rows(
    frame: pd.DataFrame,
    expected: list[int],
    name: str,
    errors: list[str],
) -> None:
    if frame.empty or "timestamp" not in frame.columns:
        errors.append(f"{name}: timestamp를 검증할 수 없습니다.")
        return
    actual = pd.to_numeric(frame["timestamp"], errors="coerce").dropna().astype("int64")
    if actual.duplicated().any():
        errors.append(f"{name}: 중복 timestamp {int(actual.duplicated().sum()):,}건")
    if sorted(actual.tolist()) != expected:
        errors.append(f"{name}: 1분 시각 불일치 expected={len(expected):,}, actual={len(actual):,}")


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end <= start:
        raise ValueError("--end-date는 --start-date보다 뒤여야 합니다.")
    days = (end - start).days
    expected = expected_timestamps(start, end)
    errors: list[str] = []

    feature_root = project_path("feature_store_v2_3m/market=usdm/symbol=BTCUSDT/timeframe=1m")
    features = read_parquets(feature_root, errors)
    if not features.empty:
        features = features.sort_values("timestamp").reset_index(drop=True)
        validate_time_rows(features, expected, "Feature Store", errors)
        if set(features.get("feature_schema_version", pd.Series(dtype=str)).astype(str)) != {FEATURE_SCHEMA}:
            errors.append("Feature Store 스키마 버전 불일치")
        if features[["ma_5", "return_1m"]].isna().any().any():
            errors.append("Feature Store 핵심 피처 결측")

    feature_markers = sorted(project_path("feature_store_v2_3m/_markers").glob("_SUCCESS_*.json"))
    expected_chunks = math.ceil(days / 14)
    if len(feature_markers) != expected_chunks:
        errors.append(f"Flink 성공 마커 수 불일치 expected={expected_chunks}, actual={len(feature_markers)}")
    boundary_rows = []
    flink_jobs = []
    for marker_path in feature_markers:
        marker = read_json(marker_path, errors)
        boundary_rows.append(int(marker.get("boundary_context_rows", -1)))
        flink_jobs.append(str(marker.get("job_id", "")))
        if marker.get("processor") != "apache_flink_pyflink_batch":
            errors.append(f"Flink 처리 엔진 불일치: {marker_path.name}")
    if boundary_rows and boundary_rows[0] != 0:
        errors.append("첫 청크 경계 문맥은 0행이어야 합니다.")
    if len(boundary_rows) > 1 and any(value != 4 for value in boundary_rows[1:]):
        errors.append(f"후속 청크 경계 문맥 4행 불일치: {boundary_rows}")

    context_root = project_path("futures_context_store_v2_3m/market=usdm/symbol=BTCUSDT/timeframe=1m")
    context = read_parquets(context_root, errors)
    if not context.empty:
        context = context.sort_values("timestamp").reset_index(drop=True)
        validate_time_rows(context, expected, "선물 문맥", errors)
        if set(context.get("context_schema_version", pd.Series(dtype=str)).astype(str)) != {CONTEXT_SCHEMA}:
            errors.append("선물 문맥 스키마 버전 불일치")
        if context["mark_price"].isna().any():
            errors.append("mark_price 결측")

    context_run_id = f"BTCUSDT_USDM_CONTEXT_1m_{start:%Y%m%d}_{end:%Y%m%d}"
    context_marker = read_json(
        project_path(f"futures_context_store_v2_3m/_markers/_SUCCESS_{context_run_id}.json"),
        errors,
    )

    label_marker_path = project_path(
        "label_store_v2_3m/market=usdm/symbol=BTCUSDT/timeframe=1m/"
        "stop=0.0050/hold=240m/_LABEL_SUCCESS.json"
    )
    label_marker = read_json(label_marker_path, errors)
    label_file = project_path(str(label_marker.get("label_file", ""))) if label_marker else Path()
    labels = pd.read_parquet(label_file) if label_marker and label_file.exists() else pd.DataFrame()
    if label_marker and labels.empty:
        errors.append("라벨 Parquet를 읽을 수 없습니다.")
    if not labels.empty:
        if not labels["label_horizon_complete"].eq(True).all():
            errors.append("불완전한 라벨 미래 구간 포함")
        if not labels["entry_rule"].eq("next_bar_open").all():
            errors.append("라벨 진입 규칙 불일치")

    dataset_marker = read_json(
        project_path(
            "ml_dataset_v2_3m/market=usdm/symbol=BTCUSDT/timeframe=1m/"
            "_ML_DATASET_SUCCESS.json"
        ),
        errors,
    )
    dataset_file = project_path(str(dataset_marker.get("dataset_file", ""))) if dataset_marker else Path()
    dataset = pd.read_parquet(dataset_file) if dataset_marker and dataset_file.exists() else pd.DataFrame()
    if dataset_marker and dataset.empty:
        errors.append("ML 데이터셋 Parquet를 읽을 수 없습니다.")
    if not dataset.empty and dataset["mark_price"].isna().any():
        errors.append("ML 데이터셋 mark_price 결측")

    model_metadata = read_json(
        project_path("models_v2_3m/direction_model_BTCUSDT_USDM_1m_metadata.json"),
        errors,
    )
    prediction_file = project_path(
        "predictions_v2_3m/market=usdm/symbol=BTCUSDT/timeframe=1m/"
        "direction_predictions_BTCUSDT_USDM_1m.parquet"
    )
    predictions = pd.read_parquet(prediction_file) if prediction_file.exists() else pd.DataFrame()
    if predictions.empty:
        errors.append("모델 예측 Parquet가 없거나 비었습니다.")
    if model_metadata and int(model_metadata.get("purged_train_rows", 0)) <= 0:
        errors.append("학습·테스트 경계의 라벨 구간 purge가 적용되지 않았습니다.")

    backtest_json_path = project_path(
        "runtime_reports/model_signal_backtest_report_v2_3m_2026-08-23.json"
    )
    backtest = read_json(backtest_json_path, errors)
    largest_loss = float(backtest.get("largest_realized_loss_pct", 0.0)) if backtest else 0.0
    if largest_loss < -2.000001:
        errors.append(f"거래 1회 최대 실현 손실 2% 초과: {largest_loss:.4f}%")

    raw_files = list(project_path("temp_raw_data_v2_3m").glob("**/*"))
    raw_file_count = sum(path.is_file() for path in raw_files)
    if raw_file_count:
        errors.append(f"성공 후 임시 원천 파일이 남아 있습니다: {raw_file_count}개")

    overall_pass = not errors
    funding_available = int(context["funding_rate"].notna().sum()) if not context.empty else 0
    open_interest_available = int(context["open_interest"].notna().sum()) if not context.empty else 0
    accuracy = float(model_metadata.get("accuracy", 0.0)) if model_metadata else 0.0
    baseline_accuracy = (
        float(model_metadata.get("majority_baseline_accuracy", 0.0))
        if model_metadata
        else 0.0
    )
    total_return_pct = float(backtest.get("total_return_pct", 0.0)) if backtest else 0.0
    research_pass = accuracy > baseline_accuracy and total_return_pct > 0
    skipped = backtest.get("skipped", {}) if backtest else {}
    skip_rows = [f"| `{name}` | {int(count):,} |" for name, count in skipped.items()]

    lines = [
        "# 3개월 Feature Store v2·ML·백테스트 통합 실행 기록",
        "",
        f"> 파이프라인 검증: **{'통과(PASS)' if overall_pass else '실패(FAIL)'}**  ",
        f"> 기록 생성: {datetime.now(timezone.utc).isoformat()}  ",
        "> 주의: 파이프라인 통과는 수익성 검증 통과를 의미하지 않습니다.",
        "",
        "## 1. 실행 범위",
        "",
        f"- 실제 데이터 구간: `{args.start_date}` 이상 `{args.end_date}` 미만 UTC, `{days}`일",
        "- 시장: Binance USDT-M `BTCUSDT`",
        "- 해상도: 완료된 `1m` 봉",
        f"- 예상·검증 Feature 행: `{len(expected):,}`건",
        f"- Flink 청크: `{len(feature_markers)}`개, Job ID: `{', '.join(flink_jobs)}`",
        "",
        "## 2. 전체 실행 순서",
        "",
        "```text",
        "실제 OHLCV 90일 -> 14일 청크 -> PyFlink -> Feature Store v2",
        "실제 mark price·funding·open interest -> Futures Context Store",
        "Feature -> 다음 봉 시가 Triple Barrier 라벨 -> ML 데이터셋",
        "시간 순서 train/test + 240분 purge -> XGBoost 방향 모델",
        "테스트 구간 예측 -> 비용·슬리피지·거래당 2% 위험 백테스트",
        "```",
        "",
        "## 3. 다시 실행하는 파일",
        "",
        "```powershell",
        f".\\scripts\\run_3m_research_pipeline.ps1 -StartDate {args.start_date} -EndDate {args.end_date}",
        "```",
        "",
        "## 4. 데이터와 학습 결과",
        "",
        "| 항목 | 결과 |",
        "| --- | ---: |",
        f"| Feature Store 행 | {len(features):,} |",
        f"| Flink 성공 마커 | {len(feature_markers)} |",
        f"| 후속 청크 경계 문맥 | {boundary_rows[1:] if len(boundary_rows) > 1 else []} |",
        f"| 선물 문맥 행 | {len(context):,} |",
        f"| funding 값이 있는 행 | {funding_available:,} |",
        f"| open interest 값이 있는 행 | {open_interest_available:,} |",
        f"| 완전한 라벨 행 | {len(labels):,} |",
        f"| ML 데이터셋 행 | {len(dataset):,} |",
        f"| 학습 행 | {int(model_metadata.get('train_rows', 0)) if model_metadata else 0:,} |",
        f"| 테스트 행 | {int(model_metadata.get('test_rows', 0)) if model_metadata else 0:,} |",
        f"| purge된 학습 경계 행 | {int(model_metadata.get('purged_train_rows', 0)) if model_metadata else 0:,} |",
        f"| 모델 테스트 정확도 | {accuracy * 100:.2f}% |",
        f"| 다수 클래스 기준 정확도 | {baseline_accuracy * 100:.2f}% |",
        f"| 기준 대비 정확도 차이 | {(accuracy - baseline_accuracy) * 100:.2f}%p |",
        "",
        "## 5. 백테스트 결과",
        "",
        "| 항목 | 결과 |",
        "| --- | ---: |",
        f"| 테스트 rows | {int(backtest.get('test_rows', 0)) if backtest else 0:,} |",
        f"| 거래 수 | {int(backtest.get('trades', 0)) if backtest else 0:,} |",
        f"| 승률 | {float(backtest.get('win_rate_pct', 0.0)) if backtest else 0.0:.2f}% |",
        f"| 평균 R | {float(backtest.get('avg_return_r', 0.0)) if backtest else 0.0:.4f} |",
        f"| 총 복리 수익률 | {total_return_pct:.2f}% |",
        f"| 최대 낙폭 | {float(backtest.get('max_drawdown_pct', 0.0)) if backtest else 0.0:.2f}% |",
        f"| 거래 1회 최대 실현 손실 | {largest_loss:.2f}% |",
        "",
        "이 결과는 테스트 구간의 오프라인 연구 결과이며 실시간 모의투자나 미래 수익 보장이 아닙니다.",
        f"연구 성능 판정: **{'통과' if research_pass else '미통과'}**. 정확도가 다수 클래스 기준보다 높고 비용 후 수익률이 양수일 때만 통과로 봅니다.",
        "",
        "## 6. 백테스트 실행 조건과 신호 제외 사유",
        "",
        "| 조건 | 값 |",
        "| --- | ---: |",
        f"| 익절 기준 | {float(backtest.get('take_profit_r', 0.0)) if backtest else 0.0:.1f}R |",
        f"| 거래당 계좌 위험 | {float(backtest.get('risk_per_trade_pct', 0.0)) * 100 if backtest else 0.0:.2f}% |",
        f"| 최소 신뢰도 | {float(backtest.get('min_confidence', 0.0)) if backtest else 0.0:.2f} |",
        f"| 최소 진입 간격 | {int(backtest.get('min_minutes_between_entries', 0)) if backtest else 0}분 |",
        f"| 하루 최대 신규 진입 | {int(backtest.get('max_new_entries_per_day', 0)) if backtest else 0}회 |",
        f"| 하루 손실 중단 | -{float(backtest.get('daily_stop_pct', 0.0)) * 100 if backtest else 0.0:.2f}% |",
        f"| 하루 하드 중단 | -{float(backtest.get('hard_daily_stop_pct', 0.0)) * 100 if backtest else 0.0:.2f}% |",
        f"| 추가 슬리피지 | {float(backtest.get('extra_slippage_bps', 0.0)) if backtest else 0.0:.2f}bps |",
        "",
        "| 제외 사유 | 행 수 |",
        "| --- | ---: |",
        *skip_rows,
        "",
        "한 포지션이 끝나기 전 신호는 제외하며, 연속 손실 제한은 UTC 날짜가 바뀌면 초기화합니다.",
        "미래 캔들로 만든 Triple Barrier 결과를 사용한 오프라인 연구이며 실시간 체결 성과가 아닙니다.",
        "",
        "## 7. 발견한 문제와 검증 결과",
        "",
        "- 최초 실행 중 PC가 종료됐지만 앞의 성공 청크 5개는 성공 마커로 보존됐습니다.",
        "- 성공 마커가 없는 여섯 번째 청크의 Flink 임시 staging만 재설정하고 같은 실행 파일로 이어서 처리했습니다.",
        "- 최초 백테스트에서 연속 손실 제한이 다음 날 초기화되지 않아 이후 거래를 영구 차단하는 문제를 발견했습니다.",
        "- 연속 손실 제한을 UTC 날짜별로 관리하도록 수정하고 모델 학습·백테스트를 다시 실행했습니다.",
    ]
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.extend(
            [
                "- 1분 시각 누락·중복, 핵심 피처 결측, 스키마 불일치가 없습니다.",
                "- 청크 경계 4행, 다음 봉 시가 라벨, 미래 구간 제거와 train/test purge가 적용됐습니다.",
                "- 거래 1회 최대 실현 손실이 계좌 기준 2%를 넘지 않았습니다.",
                "- 성공한 임시 원천 파일은 모두 삭제됐습니다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 8. 다음 판단",
            "",
            "수익률이 음수이거나 구간별 안정성이 부족하면 5년 수집이나 실거래로 넘어가지 않고",
            "피처·라벨·모델 기준을 수정해 다시 검증합니다. 이 3개월 결과가 데이터·위험·성능",
            "검증을 모두 만족할 때만 14일 청크를 5년 범위로 확장합니다.",
            "",
            "선물 open interest는 거래소의 과거 제공 기간 제한으로 일부 또는 전체가 없을 수 있습니다.",
            f"이번 수집 상태: funding=`{context_marker.get('funding_rate_status', '-')}`, "
            f"open_interest=`{context_marker.get('open_interest_status', '-')}`.",
            "",
        ]
    )

    report_path = project_path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"3개월 통합 보고서 저장: {report_path}")
    print(f"파이프라인 검증: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
