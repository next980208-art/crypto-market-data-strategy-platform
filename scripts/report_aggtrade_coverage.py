"""Report complete daily aggTrade coverage and the 90-day training gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_metadata import normalize_symbol


SCHEMA_VERSION = "aggtrade_1m_v3_windowed_complete"
RUN_ID_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)_USDM_AGGTRADE_1m_"
    r"(?P<start>\d{8}T\d{4})_(?P<end>\d{8}T\d{4})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report continuous daily aggTrade coverage.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--context-folder", default="trade_context_store_v2")
    parser.add_argument("--minimum-training-days", type=int, default=90)
    parser.add_argument(
        "--report-json", default="runtime_reports/aggtrade_coverage_latest.json"
    )
    parser.add_argument(
        "--report-md", default="runtime_reports/aggtrade_coverage_latest.md"
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_run_interval(run_id: str, symbol: str) -> tuple[datetime, datetime] | None:
    match = RUN_ID_PATTERN.match(run_id)
    if not match or match.group("symbol") != normalize_symbol(symbol):
        return None
    start = datetime.strptime(match.group("start"), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
    end = datetime.strptime(match.group("end"), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
    if start.time() != datetime.min.time() or end != start + timedelta(days=1):
        return None
    return start, end


def validate_daily_marker(marker_path: Path, symbol: str) -> tuple[date | None, dict[str, Any]]:
    result: dict[str, Any] = {"marker_file": str(marker_path), "valid": False, "errors": []}
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result["errors"].append(f"마커 읽기 실패: {error}")
        return None, result

    interval = parse_run_interval(str(marker.get("run_id", "")), symbol)
    if interval is None:
        result["ignored_reason"] = "not_a_complete_daily_marker"
        return None, result
    start, end = interval
    expected_rows = 1440
    checks = {
        "context_schema_version": marker.get("context_schema_version") == SCHEMA_VERSION,
        "trade_id_sequence_status": marker.get("trade_id_sequence_status") == "continuous",
        "complete_minute_boundaries": marker.get("complete_minute_boundaries") is True,
        "rows": int(marker.get("rows", -1)) == expected_rows,
        "expected_minute_rows": int(marker.get("expected_minute_rows", -1)) == expected_rows,
        "missing_minute_rows": int(marker.get("missing_minute_rows", -1)) == 0,
        "duplicate_minute_rows": int(marker.get("duplicate_minute_rows", -1)) == 0,
    }
    for name, passed in checks.items():
        if not passed:
            result["errors"].append(f"마커 검증 실패: {name}")

    context_path = resolve_project_path(str(marker.get("context_file", "")))
    if not context_path.is_file():
        result["errors"].append(f"Parquet 없음: {context_path}")
    else:
        try:
            timestamps = pd.read_parquet(context_path, columns=["timestamp"])["timestamp"].astype("int64")
            expected = pd.RangeIndex(
                int(start.timestamp() * 1000), int(end.timestamp() * 1000), 60_000
            )
            if len(timestamps) != expected_rows or timestamps.tolist() != expected.tolist():
                result["errors"].append("Parquet timestamp가 UTC 하루 1,440분과 일치하지 않음")
        except Exception as error:
            result["errors"].append(f"Parquet 검증 실패: {error}")

    result.update(
        {
            "data_date": start.date().isoformat(),
            "source_type": marker.get("source_type", "binance_usdm_rest_api"),
            "rows": marker.get("rows"),
            "aggregated_trades": marker.get("aggregated_trades"),
            "context_file": str(context_path),
            "valid": not result["errors"],
        }
    )
    return start.date(), result


def streaks(days: list[date]) -> list[list[date]]:
    groups: list[list[date]] = []
    for day in sorted(set(days)):
        if not groups or day != groups[-1][-1] + timedelta(days=1):
            groups.append([day])
        else:
            groups[-1].append(day)
    return groups


def build_report(context_folder: Path, symbol: str, minimum_days: int) -> dict[str, Any]:
    marker_dir = context_folder / "_markers"
    validations = []
    valid_days = []
    for marker_path in sorted(marker_dir.glob("_SUCCESS_*_AGGTRADE_1m_*.json")):
        data_date, result = validate_daily_marker(marker_path, symbol)
        if result.get("ignored_reason"):
            continue
        validations.append(result)
        if data_date is not None and result["valid"]:
            valid_days.append(data_date)

    groups = streaks(valid_days)
    longest = max(groups, key=len, default=[])
    earliest = min(valid_days, default=None)
    latest = max(valid_days, default=None)
    missing_days = []
    if earliest is not None and latest is not None:
        cursor = earliest
        available = set(valid_days)
        while cursor <= latest:
            if cursor not in available:
                missing_days.append(cursor.isoformat())
            cursor += timedelta(days=1)

    invalid = [item for item in validations if not item["valid"]]
    training_allowed = len(longest) >= minimum_days
    status = "healthy_ready" if training_allowed else "healthy_accumulating"
    if missing_days or invalid:
        status = "gaps_or_invalid_days_detected"
    source_counts: dict[str, int] = {}
    for item in validations:
        if item["valid"]:
            source = str(item["source_type"])
            source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "symbol": normalize_symbol(symbol),
        "context_folder": str(context_folder),
        "minimum_training_days": minimum_days,
        "valid_complete_days": len(set(valid_days)),
        "earliest_valid_date": earliest.isoformat() if earliest else None,
        "latest_valid_date": latest.isoformat() if latest else None,
        "longest_continuous_start": longest[0].isoformat() if longest else None,
        "longest_continuous_end": longest[-1].isoformat() if longest else None,
        "longest_continuous_days": len(longest),
        "remaining_days_to_training_gate": max(0, minimum_days - len(longest)),
        "next_expected_date": (latest + timedelta(days=1)).isoformat() if latest else None,
        "model_training_allowed": training_allowed,
        "missing_dates_inside_observed_range": missing_days,
        "invalid_daily_markers": invalid,
        "valid_days_by_source": source_counts,
        "daily_validations": validations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    source_rows = "\n".join(
        f"| `{name}` | {count}일 |" for name, count in report["valid_days_by_source"].items()
    ) or "| 없음 | 0일 |"
    missing = ", ".join(report["missing_dates_inside_observed_range"]) or "없음"
    text = f"""# aggTrade 장기 수집 진행률

## 현재 판정

- 상태: `{report['status']}`
- 유효한 완료 UTC 날짜: **{report['valid_complete_days']}일**
- 가장 긴 연속 구간: **{report['longest_continuous_days']}일**
- 연속 구간: `{report['longest_continuous_start']}` ~ `{report['longest_continuous_end']}`
- 90일 학습 관문까지 남은 날짜: **{report['remaining_days_to_training_gate']}일**
- 모델 학습 허용: **{report['model_training_allowed']}**
- 관측 범위 내부 누락 날짜: {missing}
- 다음 수집 대상 날짜: `{report['next_expected_date']}`

## 수집 경로별 완료 날짜

| 수집 경로 | 검증 완료 일수 |
| --- | ---: |
{source_rows}

## 이 보고서의 의미

각 날짜는 성공 마커만 존재한다고 완료로 세지 않습니다. Parquet 파일이 실제로 존재하고,
UTC 하루 1,440분의 timestamp가 순서대로 있으며, 중복·누락이 없고 aggTrade ID 연속성
검사를 통과한 날짜만 유효 날짜로 계산합니다. 가장 긴 연속 구간이 90일 이상일 때만
체결 피처 모델 학습을 허용합니다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.minimum_training_days <= 0:
        raise ValueError("--minimum-training-days는 1 이상이어야 합니다.")
    report = build_report(
        resolve_project_path(args.context_folder), args.symbol, args.minimum_training_days
    )
    json_path = resolve_project_path(args.report_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, resolve_project_path(args.report_md))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
