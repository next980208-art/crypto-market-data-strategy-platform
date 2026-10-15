"""Build quality checks and fixed weekly reports for forward observation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATION = PROJECT_ROOT / "runtime_reports" / "fixed_breakout_forward_20260922"
COST_CASES = ("normal", "stress", "severe")
STEP_MS = 15 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000
WEEK_MS = 7 * DAY_MS
MONTH_BLOCK_MS = 30 * DAY_MS
REQUIRED_BAR_COLUMNS = ("timestamp", "period_end_ms", "open", "high", "low", "close", "volume")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_iso(milliseconds: int) -> str:
    return pd.Timestamp(milliseconds, unit="ms", tz="UTC").isoformat()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def compound_return_pct(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Trade return contains a non-numeric value")
    return float((np.prod(1.0 + numeric.to_numpy(dtype=float)) - 1.0) * 100.0)


def load_trades(root: Path, cost: str) -> pd.DataFrame:
    path = root / f"current_{cost}_trades.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["exit_time_utc", "return_pct"])
    frame = pd.read_parquet(path)
    required = {"exit_time_utc", "return_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["exit_time_utc"] = pd.to_datetime(frame["exit_time_utc"], utc=True, errors="coerce")
    if frame["exit_time_utc"].isna().any():
        raise ValueError(f"{path.name} contains invalid exit timestamps")
    return frame.sort_values("exit_time_utc").reset_index(drop=True)


def period_result(frame: pd.DataFrame, start_ms: int, end_ms: int) -> dict[str, Any]:
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    selected = frame[(frame.exit_time_utc >= start) & (frame.exit_time_utc < end)]
    value = compound_return_pct(selected.return_pct)
    return {
        "trades": int(len(selected)),
        "return_pct": value,
        "positive": bool(value > 0.0),
    }


def build_weekly_periods(
    root: Path, protocol: dict[str, Any], report: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start_ms = int(protocol["observation_start_ms"])
    evaluation_ms = int(pd.Timestamp(report["exchange_server_utc"]).timestamp() * 1000)
    elapsed = max(0, evaluation_ms - start_ms)
    complete_count = elapsed // WEEK_MS
    has_partial = elapsed % WEEK_MS > 0
    total_count = int(complete_count + (1 if has_partial else 0))
    ledgers = {cost: load_trades(root, cost) for cost in COST_CASES}
    weeks: list[dict[str, Any]] = []
    for index in range(total_count):
        period_start = start_ms + index * WEEK_MS
        period_end = period_start + WEEK_MS
        complete = period_end <= evaluation_ms
        costs = {cost: period_result(ledgers[cost], period_start, period_end) for cost in COST_CASES}
        all_cost_positive = all(row["positive"] for row in costs.values())
        reasons: list[str] = []
        if complete:
            if costs["normal"]["trades"] == 0:
                reasons.append("no_completed_trades")
            nonpositive = [cost for cost, row in costs.items() if not row["positive"]]
            if nonpositive:
                reasons.append("nonpositive_cost_cases:" + ",".join(nonpositive))
        weeks.append({
            "week": index + 1,
            "start_utc": utc_iso(period_start),
            "end_utc": utc_iso(period_end),
            "complete": complete,
            "status": "pass" if complete and all_cost_positive else "fail" if complete else "collecting",
            "all_cost_positive": bool(complete and all_cost_positive),
            "reasons": reasons,
            "costs": costs,
        })

    blocks: list[dict[str, Any]] = []
    complete_weeks = [week for week in weeks if week["complete"]]
    for offset in range(0, len(complete_weeks), 4):
        selected = complete_weeks[offset:offset + 4]
        if len(selected) < 4:
            break
        positive_count = sum(week["all_cost_positive"] for week in selected)
        blocks.append({
            "block": len(blocks) + 1,
            "weeks": [week["week"] for week in selected],
            "positive_weeks": int(positive_count),
            "required_positive_weeks": 3,
            "status": "pass" if positive_count >= 3 else "fail",
        })
    return weeks, blocks


def build_monthly_periods(
    root: Path, protocol: dict[str, Any], report: dict[str, Any]
) -> list[dict[str, Any]]:
    start_ms = int(protocol["observation_start_ms"])
    evaluation_ms = int(pd.Timestamp(report["exchange_server_utc"]).timestamp() * 1000)
    elapsed = max(0, evaluation_ms - start_ms)
    complete_count = elapsed // MONTH_BLOCK_MS
    has_partial = elapsed % MONTH_BLOCK_MS > 0
    total_count = int(complete_count + (1 if has_partial else 0))
    minimum = int(protocol.get("minimum_monthly_trades", 0))
    trades = load_trades(root, "normal")
    periods = []
    for index in range(total_count):
        period_start = start_ms + index * MONTH_BLOCK_MS
        period_end = period_start + MONTH_BLOCK_MS
        complete = period_end <= evaluation_ms
        result = period_result(trades, period_start, period_end)
        periods.append({
            "period": index + 1,
            "start_utc": utc_iso(period_start),
            "end_utc": utc_iso(period_end),
            "complete": complete,
            "trades": result["trades"],
            "minimum_trades": minimum,
            "status": "pass" if complete and result["trades"] >= minimum else "fail" if complete else "collecting",
            "reasons": [] if not complete or result["trades"] >= minimum else ["insufficient_trades"],
        })
    return periods


def check_row(check_id: str, status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"id": check_id, "status": status, "message": message, "details": details}


def bar_checks(root: Path, filename: str, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / filename
    if not path.exists():
        return [check_row(f"{filename}:exists", "critical", f"{filename} is missing")], {}
    frame = pd.read_parquet(path)
    missing_columns = sorted(set(REQUIRED_BAR_COLUMNS).difference(frame.columns))
    if missing_columns:
        return [check_row(f"{filename}:schema", "critical", "Required columns are missing",
                          missing_columns=missing_columns)], {"rows": len(frame)}
    numeric = frame[list(REQUIRED_BAR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    timestamps = numeric.timestamp.sort_values()
    differences = timestamps.diff().dropna()
    duplicates = int(timestamps.duplicated().sum())
    irregular = differences[differences != STEP_MS]
    missing_bars = int(sum(max(0, math.floor(float(value) / STEP_MS) - 1) for value in irregular if value > 0))
    misaligned = int((numeric.timestamp % STEP_MS != 0).sum())
    null_values = int(numeric.isna().sum().sum())
    invalid_ohlcv = int((
        (numeric.low <= 0)
        | (numeric.low > numeric[["open", "close"]].min(axis=1))
        | (numeric[["open", "close"]].max(axis=1) > numeric.high)
        | (numeric.volume < 0)
    ).sum())
    rows = len(frame)
    checks = [
        check_row(f"{filename}:rows", "pass" if rows == expected_rows else "critical",
                  "Stored row count matches latest report" if rows == expected_rows else "Stored row count differs from latest report",
                  actual=rows, expected=expected_rows),
        check_row(f"{filename}:continuity", "pass" if not missing_bars and not duplicates and not misaligned else "critical",
                  "15-minute timestamps are contiguous" if not missing_bars and not duplicates and not misaligned else "Timestamp gap, duplicate, or alignment error detected",
                  missing_bars=missing_bars, duplicates=duplicates, misaligned=misaligned),
        check_row(f"{filename}:values", "pass" if not null_values and not invalid_ohlcv else "critical",
                  "OHLCV values are valid" if not null_values and not invalid_ohlcv else "Invalid or missing OHLCV values detected",
                  null_values=null_values, invalid_ohlcv=invalid_ohlcv),
    ]
    stats = {
        "rows": rows,
        "missing_bars": missing_bars,
        "duplicates": duplicates,
        "misaligned": misaligned,
        "null_values": null_values,
        "invalid_ohlcv": invalid_ohlcv,
        "first_timestamp": int(timestamps.iloc[0]) if rows else None,
        "last_timestamp": int(timestamps.iloc[-1]) if rows else None,
        "last_period_end_ms": int(numeric.period_end_ms.max()) if rows else None,
        "timestamps": set(numeric.timestamp.dropna().astype("int64")),
    }
    return checks, stats


def build_quality_checks(
    root: Path,
    protocol: dict[str, Any],
    state: dict[str, Any],
    report: dict[str, Any],
    observer_exit_code: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(check_row(
        "observer_exit", "pass" if observer_exit_code == 0 else "critical",
        "Observer completed successfully" if observer_exit_code == 0 else "Observer process failed",
        exit_code=observer_exit_code,
    ))
    protocol_hash = sha256(root / "protocol.json")
    expected_hash = str(state.get("protocol_sha256", ""))
    checks.append(check_row(
        "protocol_hash", "pass" if protocol_hash == expected_hash else "critical",
        "Frozen protocol hash matches state" if protocol_hash == expected_hash else "Frozen protocol changed",
        actual=protocol_hash, expected=expected_hash,
    ))
    orders = int(report.get("orders_sent", 0))
    credentials = bool(report.get("credentials_used", False))
    safety_ok = orders == 0 and not credentials and not protocol.get("orders_allowed", False)
    checks.append(check_row(
        "safety_lock", "pass" if safety_ok else "critical",
        "No credentials or orders were used" if safety_ok else "Trading safety invariant was violated",
        orders_sent=orders, credentials_used=credentials, orders_allowed=protocol.get("orders_allowed"),
    ))

    contract_checks, contract = bar_checks(root, "bars_15m.parquet", int(report.get("contract_bars", 0)))
    mark_checks, mark = bar_checks(root, "mark_15m.parquet", int(report.get("mark_bars", 0)))
    checks.extend(contract_checks)
    checks.extend(mark_checks)
    if contract and mark:
        unmatched = len(contract["timestamps"].symmetric_difference(mark["timestamps"]))
        checks.append(check_row(
            "contract_mark_alignment", "pass" if unmatched == 0 else "critical",
            "Contract and mark-price bars are aligned" if unmatched == 0 else "Contract and mark-price timestamps differ",
            unmatched_timestamps=unmatched,
        ))
        server_ms = int(pd.Timestamp(report["exchange_server_utc"]).timestamp() * 1000)
        lag_minutes = max(0.0, (server_ms - int(contract["last_period_end_ms"])) / 60000.0)
        lag_status = "pass" if lag_minutes <= 30 else "warning" if lag_minutes <= 24 * 60 else "critical"
        checks.append(check_row(
            "latest_bar_lag", lag_status,
            "Latest completed bar is current" if lag_status == "pass" else "Latest completed bar is delayed",
            lag_minutes=lag_minutes,
        ))

    funding_path = root / "funding.json"
    if funding_path.exists():
        funding = json.loads(funding_path.read_text(encoding="utf-8"))
        times = [int(row["fundingTime"]) for row in funding]
        invalid = sum(not np.isfinite(float(row["fundingRate"])) for row in funding)
        duplicate = len(times) - len(set(times))
        expected = int(report.get("funding_events", 0))
        ok = not invalid and not duplicate and len(funding) == expected
        checks.append(check_row(
            "funding_integrity", "pass" if ok else "critical",
            "Funding events are unique and valid" if ok else "Funding event integrity error detected",
            rows=len(funding), expected=expected, duplicates=duplicate, invalid_rates=invalid,
        ))
    else:
        checks.append(check_row("funding_integrity", "critical", "funding.json is missing"))

    storage_bytes = int(report.get("database_bytes", 0))
    storage_limit = int(protocol.get("storage_limit_bytes", 0))
    storage_ratio = storage_bytes / storage_limit if storage_limit else 1.0
    storage_status = "pass" if storage_ratio < 0.8 else "warning" if storage_ratio < 1.0 else "critical"
    checks.append(check_row(
        "storage_limit", storage_status,
        "Observation database is below the warning threshold" if storage_status == "pass" else "Observation database is near or over its limit",
        bytes=storage_bytes, limit_bytes=storage_limit, ratio=storage_ratio,
    ))
    statuses = {row["status"] for row in checks}
    overall = "critical" if "critical" in statuses else "warning" if "warning" in statuses else "healthy"
    return {
        "status": overall,
        "pass_count": sum(row["status"] == "pass" for row in checks),
        "warning_count": sum(row["status"] == "warning" for row in checks),
        "critical_count": sum(row["status"] == "critical" for row in checks),
        "checks": checks,
    }


def current_results(report: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for cost in COST_CASES:
        metrics = report.get("results", {}).get(cost, {}).get("metrics", {})
        output[cost] = {
            "trades": int(metrics.get("trades", 0)),
            "return_pct": float(metrics.get("total_return_pct", 0.0)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
        }
    return output


def review_status(
    protocol: dict[str, Any], report: dict[str, Any], quality: dict[str, Any],
    blocks: list[dict[str, Any]], months: list[dict[str, Any]], results: dict[str, Any],
) -> dict[str, Any]:
    observed = float(report.get("observed_days", 0.0))
    minimum = float(protocol.get("minimum_observation_days", 90))
    if observed < minimum:
        status = "collecting"
        reasons = ["minimum_observation_period_not_reached"]
    else:
        reasons = []
        if quality["status"] != "healthy":
            reasons.append("data_quality_not_healthy")
        if any(block["status"] == "fail" for block in blocks):
            reasons.append("weekly_gate_failed")
        if any(month["status"] == "fail" for month in months if month["complete"]):
            reasons.append("monthly_trade_gate_failed")
        nonpositive = [cost for cost, value in results.items() if value["return_pct"] <= 0]
        if nonpositive:
            reasons.append("nonpositive_total_return:" + ",".join(nonpositive))
        status = "manual_review_required" if not reasons else "precheck_failed"
    return {
        "status": status,
        "reasons": reasons,
        "decision": "no_trade",
        "deployment_allowed": False,
        "orders_allowed": False,
        "note": "This report never approves live trading automatically.",
    }


def build_evaluation(root: Path, observer_exit_code: int = 0) -> dict[str, Any]:
    protocol = read_json(root / "protocol.json")
    state = read_json(root / "state.json")
    report = read_json(root / "latest_report.json")
    quality = build_quality_checks(root, protocol, state, report, observer_exit_code)
    weeks, blocks = build_weekly_periods(root, protocol, report)
    months = build_monthly_periods(root, protocol, report)
    results = current_results(report)
    complete_weeks = [week for week in weeks if week["complete"]]
    current_week = next((week for week in reversed(weeks) if not week["complete"]), None)
    evaluation = {
        "schema_version": "fixed_observation_weekly_v1",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "observation": {
            "rule_id": protocol.get("rule_id"),
            "status": report.get("status"),
            "observed_days": float(report.get("observed_days", 0.0)),
            "minimum_observation_days": int(protocol.get("minimum_observation_days", 90)),
            "review_after_utc": protocol.get("review_after_utc"),
            "orders_sent": int(report.get("orders_sent", 0)),
        },
        "quality": quality,
        "weekly": {
            "definition": "Seven-day blocks anchored at observation_start_utc; a week passes only when all cost cases are positive.",
            "complete_weeks": len(complete_weeks),
            "positive_weeks": sum(week["all_cost_positive"] for week in complete_weeks),
            "failed_weeks": sum(week["status"] == "fail" for week in complete_weeks),
            "current_week": current_week,
            "weeks": weeks,
            "four_week_blocks": blocks,
        },
        "monthly": {
            "definition": "Thirty-day blocks anchored at observation_start_utc.",
            "periods": months,
        },
        "current_results": results,
    }
    evaluation["review"] = review_status(protocol, report, quality, blocks, months, results)
    return evaluation


def markdown_report(value: dict[str, Any]) -> str:
    observation = value["observation"]
    quality = value["quality"]
    weekly = value["weekly"]
    current = weekly.get("current_week")
    lines = [
        "# 고정 전략 주간 관찰 보고서",
        "",
        f"생성 시각(UTC): `{value['generated_utc']}`  ",
        f"상태: `{observation['status']}` / 검토 판정: `{value['review']['status']}` / 주문: `{observation['orders_sent']}건`",
        "",
        "## 현재 요약",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 관찰 경과 | {observation['observed_days']:.3f} / {observation['minimum_observation_days']}일 |",
        f"| 데이터 품질 | {quality['status']} (통과 {quality['pass_count']}, 경고 {quality['warning_count']}, 심각 {quality['critical_count']}) |",
        f"| 완료 주 | {weekly['complete_weeks']}주 |",
        f"| 모든 비용에서 양수인 주 | {weekly['positive_weeks']}주 |",
        f"| 실패한 완료 주 | {weekly['failed_weeks']}주 |",
        f"| 실제 주문 | {observation['orders_sent']}건 |",
        "",
        "완료되지 않은 주는 합격·실패 판정에서 제외한다. 주간 합격은 기본·스트레스·가혹 비용이 모두 양수일 때만 인정한다.",
        "",
        "## 현재 누적 결과",
        "",
        "| 비용 조건 | 거래 | 누적 수익률 | 최대 낙폭 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cost in COST_CASES:
        result = value["current_results"][cost]
        lines.append(f"| {cost} | {result['trades']} | {result['return_pct']:+.4f}% | {result['max_drawdown_pct']:+.4f}% |")
    lines.extend(["", "## 주별 기록", "", "| 주 | 기간(UTC) | 상태 | 기본 | 스트레스 | 가혹 | 거래 | 실패 이유 |", "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |"])
    if not weekly["weeks"]:
        lines.append("| - | 아직 관찰 구간 없음 | collecting | - | - | - | 0 | - |")
    for week in weekly["weeks"]:
        costs = week["costs"]
        reasons = ", ".join(week["reasons"]) or "-"
        lines.append(
            f"| {week['week']} | {week['start_utc']} ~ {week['end_utc']} | {week['status']} | "
            f"{costs['normal']['return_pct']:+.4f}% | {costs['stress']['return_pct']:+.4f}% | "
            f"{costs['severe']['return_pct']:+.4f}% | {costs['normal']['trades']} | {reasons} |"
        )
    lines.extend(["", "## 4주 기준", "", "| 블록 | 포함 주 | 양수 주 | 기준 | 상태 |", "| ---: | --- | ---: | ---: | --- |"])
    if not weekly["four_week_blocks"]:
        lines.append("| - | 완료된 4주 블록 없음 | 0 | 3 | collecting |")
    for block in weekly["four_week_blocks"]:
        lines.append(f"| {block['block']} | {', '.join(map(str, block['weeks']))} | {block['positive_weeks']} | {block['required_positive_weeks']} | {block['status']} |")
    lines.extend(["", "## 30일 거래 수 기준", "", "| 구간 | 기간(UTC) | 거래 | 최소 | 상태 |", "| ---: | --- | ---: | ---: | --- |"])
    for month in value["monthly"]["periods"]:
        lines.append(f"| {month['period']} | {month['start_utc']} ~ {month['end_utc']} | {month['trades']} | {month['minimum_trades']} | {month['status']} |")
    if not value["monthly"]["periods"]:
        lines.append("| - | 아직 관찰 구간 없음 | 0 | - | collecting |")
    lines.extend(["", "## 데이터 품질 검사", "", "| 검사 | 상태 | 설명 |", "| --- | --- | --- |"])
    for check in quality["checks"]:
        lines.append(f"| `{check['id']}` | {check['status']} | {check['message']} |")
    if current:
        lines.extend(["", f"다음 첫 주 확정 시각은 `{current['end_utc']}`이다."])
    lines.extend([
        "",
        "## 현재 결론",
        "",
        f"- 자동 사전 검토: `{value['review']['status']}`",
        f"- 사유: `{', '.join(value['review']['reasons']) or 'none'}`",
        "- 실제 거래 판단: `no_trade`",
        "- 이 보고서는 실거래를 자동 승인하지 않는다.",
        "",
    ])
    return "\n".join(lines)


def update_alert_history(root: Path, value: dict[str, Any]) -> None:
    active = [
        {"id": check["id"], "severity": check["status"], "message": check["message"]}
        for check in value["quality"]["checks"] if check["status"] != "pass"
    ]
    active.extend({
        "id": f"week_{week['week']}", "severity": "warning",
        "message": ";".join(week["reasons"]) or "weekly_gate_failed",
    } for week in value["weekly"]["weeks"] if week["status"] == "fail")
    status = "active" if active else "clear"
    fingerprint_source = json.dumps({"status": status, "alerts": active}, sort_keys=True)
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    path = root / "alert_history.jsonl"
    previous = None
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous = json.loads(lines[-1]).get("fingerprint")
    if fingerprint == previous:
        return
    record = {
        "created_utc": value["generated_utc"],
        "status": status,
        "fingerprint": fingerprint,
        "alerts": active,
        "delivery": "local_file",
        "external_notification_configured": False,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_outputs(root: Path, value: dict[str, Any]) -> None:
    atomic_json(root / "weekly_evaluation.json", value)
    atomic_text(root / "weekly_evaluation.md", markdown_report(value))
    update_alert_history(root, value)


def validate_root(path: Path) -> Path:
    root = path.resolve()
    if root == PROJECT_ROOT or not root.is_relative_to(PROJECT_ROOT):
        raise ValueError("Observation must be a project subdirectory")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--observer-exit-code", type=int, default=0)
    args = parser.parse_args()
    root = validate_root(args.output)
    value = build_evaluation(root, args.observer_exit_code)
    write_outputs(root, value)
    print(json.dumps({
        "quality": value["quality"]["status"],
        "complete_weeks": value["weekly"]["complete_weeks"],
        "failed_weeks": value["weekly"]["failed_weeks"],
        "review": value["review"]["status"],
        "decision": value["review"]["decision"],
    }, ensure_ascii=False))
    return 0 if value["quality"]["status"] != "critical" else 2


if __name__ == "__main__":
    raise SystemExit(main())
