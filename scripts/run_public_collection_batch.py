"""One finite collection batch using existing public collectors; no services or orders."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_bounded_profitability_search import ROOT, utc_now, write_json


def commands(output, start, end):
    days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    if not 1 <= days <= 7 or datetime.fromisoformat(end).date() > datetime.now(timezone.utc).date():
        raise ValueError("Require 1..7 fully elapsed UTC days")
    return [
        ("futures_ohlcv", ["1_chunk_downloader.py", "--market", "usdm", "--start-date", start,
                           "--days", str(days), "--no-kafka", "--temp-folder", str(output / "ohlcv")]),
        ("futures_context", ["9_futures_context_collector.py", "--start-date", start, "--end-date", end,
                             "--max-imputed-mark-price-rows", "0", "--context-folder", str(output / "context")]),
        ("recent_aggregate_trades", ["10_aggtrade_collector.py", "--recent-minutes", "10",
                                     "--max-pages", "100", "--max-total-pages", "100",
                                     "--context-folder", str(output / "trades")]),
        ("public_raw_websocket", ["-m", "scripts.capture_public_raw_window", "--seconds", "120",
                                  "--output", str(output / "raw_websocket")]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    tasks = commands(output, args.start_date, args.end_date)
    output.mkdir(parents=True, exist_ok=False)
    report = {"started_utc": utc_now(), "status": "running", "scope": "BTCUSDT existing project public data",
              "task_timeout_seconds": 240, "max_batch_seconds": 960, "orders_sent": 0,
              "credentials_used": False, "docker_started": False, "tasks": []}
    started = time.monotonic()
    for name, arguments in tasks:
        if shutil.disk_usage(output).free < 5*1024**3:
            report["stop_reason"] = "free_disk_below_5GiB"
            break
        row = {"name": name, "command": [sys.executable, *arguments], "started_utc": utc_now()}
        print(f"Starting {name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(row["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=240, encoding="utf-8", env=None)
                row.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
            except subprocess.TimeoutExpired:
                row.update(status="timeout")
        row["finished_utc"] = utc_now()
        report["tasks"].append(row)
        write_json(output / "batch.json", report)
        print(f"Finished {name}: {row['status']}", flush=True)
    report.update(status="completed" if len(report["tasks"]) == len(tasks) and all(r["status"] == "completed" for r in report["tasks"]) else "partial",
                  finished_utc=utc_now(), elapsed_seconds=round(time.monotonic()-started, 3),
                  output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
    write_json(output / "batch.json", report)
    print(json.dumps(report))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
