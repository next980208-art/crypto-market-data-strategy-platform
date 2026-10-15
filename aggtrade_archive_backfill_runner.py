"""Incrementally backfill daily aggTrade archives toward a continuous target range."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import daily_collection_runner


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental Binance daily aggTrade archive backfill.")
    parser.add_argument("--target-start-date", default="2026-06-06", help="Inclusive UTC date")
    parser.add_argument("--target-end-date", help="Exclusive UTC date; defaults to today UTC")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--context-folder", default="trade_context_store_v2")
    parser.add_argument("--max-days-per-run", type=int, default=1)
    parser.add_argument("--archive-ready-delay-hours", type=int, default=6)
    parser.add_argument("--minimum-training-days", type=int, default=90)
    parser.add_argument("--state-folder", default="scheduler_state/aggtrade_archive_backfill")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def day_marker(context_folder: str, symbol: str, day: date) -> Path:
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    symbol_code = daily_collection_runner.safe_value(symbol)
    run_id = f"{symbol_code}_USDM_AGGTRADE_1m_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}"
    return PROJECT_ROOT / context_folder / "_markers" / f"_SUCCESS_{run_id}.json"


def available_target_end(requested_end: date, now: datetime, ready_delay_hours: int) -> date:
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=ready_delay_hours)
    return min(requested_end, cutoff.date())


def target_days(start: date, end: date) -> list[date]:
    if end <= start:
        return []
    return [start + timedelta(days=index) for index in range((end - start).days)]


def pending_days(
    start: date,
    end: date,
    context_folder: str,
    symbol: str,
    max_days: int,
) -> tuple[list[date], int]:
    missing = [
        day
        for day in target_days(start, end)
        if not daily_collection_runner.success_marker_is_valid(
            "aggtrade", day_marker(context_folder, symbol, day)
        )
    ]
    # Fill backward from the newest missing day so it joins the current continuous tail first.
    selected = sorted(missing, reverse=True)[:max_days]
    return selected, len(missing)


def archive_command(args: argparse.Namespace, day: date) -> list[str]:
    return [
        sys.executable,
        "aggtrade_archive_collector.py",
        "--data-date",
        day.isoformat(),
        "--symbol",
        args.symbol,
        "--context-folder",
        args.context_folder,
    ]


def run_command(command: list[str], dry_run: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "command": command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        result["status"] = "dry_run"
        return result
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    result.update(
        {
            "return_code": completed.returncode,
            "stdout": completed.stdout[-10_000:],
            "stderr": completed.stderr[-10_000:],
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "processed" if completed.returncode == 0 else "failed",
        }
    )
    return result


def write_manifest(args: argparse.Namespace, payload: dict[str, object]) -> Path:
    state_dir = PROJECT_ROOT / args.state_folder
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.max_days_per_run <= 0:
        raise ValueError("--max-days-per-run은 1 이상이어야 합니다.")
    if args.archive_ready_delay_hours < 0:
        raise ValueError("--archive-ready-delay-hours는 0 이상이어야 합니다.")
    if args.minimum_training_days <= 0:
        raise ValueError("--minimum-training-days는 1 이상이어야 합니다.")

    requested_start = parse_date(args.target_start_date)
    requested_end = (
        parse_date(args.target_end_date)
        if args.target_end_date
        else datetime.now(timezone.utc).date()
    )
    effective_end = available_target_end(
        requested_end, datetime.now(timezone.utc), args.archive_ready_delay_hours
    )
    if effective_end <= requested_start:
        raise ValueError("공식 아카이브 공개 지연을 반영한 대상 날짜 범위가 비어 있습니다.")

    selected, missing_before = pending_days(
        requested_start,
        effective_end,
        args.context_folder,
        args.symbol,
        args.max_days_per_run,
    )
    results = []
    for day in selected:
        result = run_command(archive_command(args, day), args.dry_run)
        result["data_date"] = day.isoformat()
        results.append(result)
        if result["status"] == "failed":
            break

    coverage_command = [
        sys.executable,
        "scripts/report_aggtrade_coverage.py",
        "--symbol",
        args.symbol,
        "--context-folder",
        args.context_folder,
        "--minimum-training-days",
        str(args.minimum_training_days),
    ]
    coverage = run_command(coverage_command, args.dry_run)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "target_start_date": requested_start.isoformat(),
        "requested_target_end_date_exclusive": requested_end.isoformat(),
        "effective_target_end_date_exclusive": effective_end.isoformat(),
        "archive_ready_delay_hours": args.archive_ready_delay_hours,
        "max_days_per_run": args.max_days_per_run,
        "missing_days_before_run": missing_before,
        "selected_days": [day.isoformat() for day in selected],
        "archive_results": results,
        "coverage_result": coverage,
        "status": "failed" if any(item["status"] == "failed" for item in results) else "completed",
    }
    manifest = None if args.dry_run else write_manifest(args, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if manifest:
        print(f"Manifest: {manifest}")
    if payload["status"] == "failed" or coverage["status"] == "failed":
        raise RuntimeError("aggTrade archive backfill run failed.")


if __name__ == "__main__":
    main()
