"""Validate Feature Store v2 chunks and write a Korean Markdown run report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "timestamp",
    "datetime_utc",
    "symbol",
    "market",
    "timeframe",
    "run_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "return_1m",
    "feature_schema_version",
}
EXPECTED_PROCESSOR = "apache_flink_pyflink_batch"
EXPECTED_SCHEMA = "ohlcv_basic_v2_boundary4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Feature Store v2 backfill run.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, exclusive")
    parser.add_argument("--feature-folder", default="feature_store_v2")
    parser.add_argument("--raw-folder", default="temp_raw_data_v2")
    parser.add_argument(
        "--report-path",
        default="runtime_reports/v2_batch_pipeline_execution.md",
    )
    return parser.parse_args()


def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def resolve_project_path(path_text: str) -> Path:
    normalized = str(path_text).replace("\\", "/")
    prefix = "/opt/airflow/project/"
    if normalized.startswith(prefix):
        return PROJECT_ROOT / normalized[len(prefix) :]
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def chunk_run_id(start: datetime, end: datetime) -> str:
    return f"BTCUSDT_USDM_1m_{start:%Y%m%d}_{end:%Y%m%d}"


def validate_chunk(
    feature_folder: Path,
    raw_folder: Path,
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    run_id = chunk_run_id(start, end)
    marker_path = feature_folder / "_markers" / f"_SUCCESS_{run_id}.json"
    raw_path = raw_folder / f"raw_{run_id}.csv"
    result: dict[str, object] = {
        "run_id": run_id,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "marker": marker_path,
        "raw_deleted": not raw_path.exists(),
        "errors": [],
    }

    errors: list[str] = result["errors"]  # type: ignore[assignment]
    if not marker_path.exists():
        errors.append(f"성공 마커 없음: {marker_path}")
        return result

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"성공 마커 읽기 실패: {exc}")
        return result

    result.update(
        {
            "processor": marker.get("processor"),
            "schema": marker.get("feature_schema_version"),
            "job_id": marker.get("job_id"),
            "marker_rows": int(marker.get("rows", -1)),
            "boundary_context_rows": int(marker.get("boundary_context_rows", -1)),
            "boundary_context_status": marker.get("boundary_context_status"),
        }
    )
    if result["processor"] != EXPECTED_PROCESSOR:
        errors.append(f"처리 엔진 불일치: {result['processor']}")
    if result["schema"] != EXPECTED_SCHEMA:
        errors.append(f"피처 스키마 불일치: {result['schema']}")

    feature_paths = [resolve_project_path(path) for path in marker.get("feature_files", [])]
    if not feature_paths:
        errors.append("성공 마커에 Parquet 경로가 없습니다.")
        return result

    frames: list[pd.DataFrame] = []
    for path in feature_paths:
        if not path.exists():
            errors.append(f"Parquet 파일 없음: {path}")
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:  # pandas exposes backend-specific exceptions here.
            errors.append(f"Parquet 읽기 실패: {path}: {exc}")

    if not frames:
        return result

    frame = pd.concat(frames, ignore_index=True)
    if "run_id" in frame.columns:
        frame = frame[frame["run_id"].astype(str) == run_id].copy()
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        errors.append(f"필수 컬럼 누락: {missing_columns}")
        return result

    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    expected_timestamps = list(
        range(int(start.timestamp() * 1000), int(end.timestamp() * 1000), 60_000)
    )
    actual_timestamps = (
        frame["timestamp"].dropna().astype("int64").sort_values().tolist()
    )
    result["parquet_rows"] = len(frame)
    result["feature_files"] = [str(path.relative_to(PROJECT_ROOT)) for path in feature_paths]
    result["timestamp_start_utc"] = (
        pd.to_datetime(actual_timestamps[0], unit="ms", utc=True).isoformat()
        if actual_timestamps
        else None
    )
    result["timestamp_end_utc"] = (
        pd.to_datetime(actual_timestamps[-1], unit="ms", utc=True).isoformat()
        if actual_timestamps
        else None
    )

    if actual_timestamps != expected_timestamps:
        errors.append(
            "1분 시각 연속성 불일치: "
            f"expected={len(expected_timestamps)}, actual={len(actual_timestamps)}"
        )
    if frame["timestamp"].duplicated().any():
        errors.append("중복 timestamp가 있습니다.")
    if frame[["timestamp", "close", "ma_5", "return_1m"]].isna().any().any():
        errors.append("핵심 피처에 결측값이 있습니다.")
    if set(frame["feature_schema_version"].astype(str)) != {EXPECTED_SCHEMA}:
        errors.append("Parquet 내부 피처 스키마 버전이 다릅니다.")
    if result["marker_rows"] != len(frame):
        errors.append(
            f"마커와 Parquet 행 수 불일치: marker={result['marker_rows']}, parquet={len(frame)}"
        )
    if not result["raw_deleted"]:
        errors.append(f"가공 성공 후 원천 CSV가 남아 있습니다: {raw_path}")

    return result


def markdown_report(
    args: argparse.Namespace,
    chunks: list[dict[str, object]],
    overall_pass: bool,
) -> str:
    total_rows = sum(int(chunk.get("parquet_rows", 0)) for chunk in chunks)
    raw_deleted = sum(bool(chunk.get("raw_deleted")) for chunk in chunks)
    status = "통과(PASS)" if overall_pass else "실패(FAIL)"
    lines = [
        "# Feature Store v2 실제 배치 파이프라인 실행 기록",
        "",
        f"> 실행 결과: **{status}**  ",
        f"> 실행 기록 생성 시각: {datetime.now(timezone.utc).isoformat()}  ",
        "> 데이터 출처: Binance USDT-M 공개 API의 실제 BTCUSDT 완료 1분봉",
        "",
        "## 1. 이번 작업을 실행한 이유",
        "",
        "Kafka·Flink 1,000건 재현 경로 다음 단계로, 실제 프로젝트의 표준 저장소인",
        "`feature_store_v2`에 작은 연속 구간을 다시 생성했습니다. 하루 청크 두 개를 사용해",
        "기본 피처 계산뿐 아니라 두 번째 청크가 이전 청크의 마지막 4개 봉을 문맥으로",
        "이어받는지도 함께 검사했습니다.",
        "",
        "## 2. 실행한 흐름",
        "",
        "```text",
        "Binance USDT-M 실제 1분봉",
        "  -> 하루 단위 청크 수집과 1분 시각 완전성 검사",
        "  -> Apache Flink / PyFlink 배치 피처 생성",
        "  -> ma_5, return_1m 값 재검증",
        "  -> feature_store_v2 Parquet 저장",
        "  -> 성공 마커와 행 수 검증",
        "  -> 성공한 원천 CSV 자동 삭제",
        "```",
        "",
        "## 3. 다시 실행하는 명령",
        "",
        "프로젝트 루트 PowerShell에서 실행합니다.",
        "",
        "```powershell",
        f".\\scripts\\run_v2_feature_validation.ps1 -StartDate {args.start_date} -Days {(parse_date(args.end_date) - parse_date(args.start_date)).days}",
        "```",
        "",
        "`StartDate`는 포함, 종료일은 `StartDate + Days`인 제외 경계입니다.",
        "",
        "## 4. 실제 실행 결과",
        "",
        f"- 검증 구간: `{args.start_date}` 이상 `{args.end_date}` 미만 UTC",
        f"- 생성된 청크: `{len(chunks)}`개",
        f"- Parquet 총 행 수: `{total_rows:,}`건",
        f"- 성공 후 원천 CSV 삭제: `{raw_deleted}/{len(chunks)}`개 청크",
        f"- 처리 엔진: `{EXPECTED_PROCESSOR}`",
        f"- 피처 스키마: `{EXPECTED_SCHEMA}`",
        "",
        "| 청크 | Flink Job ID | Parquet 행 | 경계 문맥 | 원천 삭제 | 결과 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for chunk in chunks:
        errors = chunk.get("errors", [])
        result_text = "통과" if not errors else "실패"
        lines.append(
            f"| `{chunk['start']} -> {chunk['end']}` "
            f"| `{chunk.get('job_id', '-')}` "
            f"| {int(chunk.get('parquet_rows', 0)):,} "
            f"| `{chunk.get('boundary_context_status', '-')}` "
            f"| {'예' if chunk.get('raw_deleted') else '아니오'} "
            f"| {result_text} |"
        )

    lines.extend(
        [
            "",
            "## 5. 생성·검증된 피처",
            "",
            "| 컬럼 | 의미 |",
            "| --- | --- |",
            "| `ma_5` | 현재 봉을 포함한 최근 최대 5개 종가 평균 |",
            "| `return_1m` | 직전 1분봉 종가 대비 현재 종가 수익률 |",
            "| `feature_schema_version` | 경계 문맥 규칙을 포함한 피처 정의 버전 |",
            "",
            "각 Parquet에서 1분 시각 연속성, 중복, 핵심 피처 결측, 스키마 버전과",
            "성공 마커의 행 수를 다시 읽어 확인했습니다.",
            "",
            "## 6. 오류와 해결 결과",
            "",
            "- 첫 실행에서 Airflow 컨테이너의 기본 작업 폴더가 `/opt/airflow`여서 하위 수집기 파일을 찾지 못했습니다.",
            "- `backfill_runner.py`가 자신의 프로젝트 경로를 작업 폴더로 사용하게 수정하고, Docker 실행에도 `/opt/airflow/project`를 명시해 해결했습니다.",
            "- 다음 실행에서는 `--no-kafka` 배치인데도 Kafka 라이브러리를 미리 불러오며 Airflow의 OpenSSL 의존성과 충돌했습니다.",
            "- Kafka Producer를 실제 Kafka 전송 시점에만 불러오도록 변경해, Kafka가 필요 없는 과거 배치 수집 경로를 분리했습니다.",
        ]
    )
    all_errors = [
        f"{chunk['run_id']}: {error}"
        for chunk in chunks
        for error in chunk.get("errors", [])
    ]
    if all_errors:
        lines.extend(f"- {error}" for error in all_errors)
    else:
        lines.append("- 수집 누락, 중복 timestamp, 피처 결측, 행 수 불일치가 발견되지 않았습니다.")
        lines.append("- 가공 성공 후 임시 원천 CSV가 자동 삭제되어 저장 공간 정리 규칙도 통과했습니다.")

    lines.extend(
        [
            "",
            "## 7. 다음 진행 순서",
            "",
            "이 작은 구간 검증이 통과한 뒤에는 같은 실행 파일의 날짜 범위를 늘려 먼저",
            "3개월 Feature Store v2를 생성합니다. 그 데이터에서 라벨 생성, ML 데이터셋 생성,",
            "시간 순서 모델 학습과 비용·2% 위험 한도를 적용한 백테스트를 차례로 다시 실행해야 합니다.",
            "5년 전체 수집과 실시간 주문은 이 검증보다 뒤의 단계입니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end <= start:
        raise ValueError("--end-date는 --start-date보다 뒤여야 합니다.")

    feature_folder = resolve_project_path(args.feature_folder)
    raw_folder = resolve_project_path(args.raw_folder)
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=1), end)
        chunks.append(validate_chunk(feature_folder, raw_folder, current, chunk_end))
        current = chunk_end

    overall_pass = all(not chunk["errors"] for chunk in chunks)
    if len(chunks) >= 2 and int(chunks[1].get("boundary_context_rows", -1)) != 4:
        chunks[1]["errors"].append("두 번째 청크에 이전 4개 봉 경계 문맥이 적용되지 않았습니다.")
        overall_pass = False

    report_path = resolve_project_path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(args, chunks, overall_pass), encoding="utf-8")
    print(f"검증 보고서 저장: {report_path}")
    print(f"검증 결과: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
