"""Capture current local execution evidence without changing pipeline outputs."""

import argparse
import io
import json
import sqlite3
import subprocess
import unittest
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capture(destination):
    def read(relative):
        path = ROOT / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    cycle = read("runtime_reports/trade_research_20260908/cycle.json")
    coverage = read("runtime_reports/aggtrade_coverage_latest.json")
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2026-09-08 project resumption; snapshot, not a long-duration operational guarantee",
        "coverage": {key: coverage[key] for key in (
            "valid_complete_days", "earliest_valid_date", "latest_valid_date", "longest_continuous_days",
            "missing_dates_inside_observed_range", "invalid_daily_markers")},
        "aggregated_trades": sum(item.get("aggregated_trades", 0) for item in coverage["daily_validations"] if item["valid"]),
        "research_status": cycle["status"], "research_comparison": cycle.get("comparison"),
        "daily_runs": {},
    }
    for day in ("20260904", "20260905", "20260906", "20260907"):
        snapshot_relative = f"runtime_reports/project_resume_daily_20260908/BTCUSDT_USDM_1m_{day}.json"
        record = read(snapshot_relative)
        if record is None:
            record = read(f"scheduler_state/daily_runs/BTCUSDT_USDM_1m_{day}.json")
            if record is not None:
                snapshot = ROOT / snapshot_relative
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                with snapshot.open("x", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
        report["daily_runs"][day] = None if record is None else {
            "created_at_utc": record["created_at_utc"],
            "steps": [{"name": step["name"], "status": step["status"], "return_code": step.get("return_code")}
                      for step in record["steps"]],
        }
    for name, path in {
        "collector": "realtime_runtime/collector_metrics.json",
        "archive": "runtime_reports/market_archive_latest.json",
        "feature_worker": "realtime_runtime/feature_worker_metrics.json",
        "inference": "realtime_runtime/inference_metrics.json",
        "risk": "realtime_runtime/risk_metrics.json",
    }.items():
        report[name] = read(path)
    db = ROOT / "realtime_runtime/market_archive.sqlite3"
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
        report["archive_readback"] = []
        for (topic,) in connection.execute("SELECT DISTINCT topic FROM positions"):
            row = connection.execute(
                "SELECT partition_id, offset_id, value_zlib FROM records WHERE topic=? "
                "ORDER BY offset_id DESC LIMIT 1", (topic,),
            ).fetchone()
            report["archive_readback"].append({
                "topic": topic, "partition": row[0], "offset": row[1],
                "decoded_saved_record": json.loads(zlib.decompress(row[2])),
            })
    docker = subprocess.run(["docker", "compose", "--profile", "streaming", "--profile", "realtime",
                             "ps", "--format", "json"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    services = []
    if docker.returncode == 0:
        records = json.loads(docker.stdout) if docker.stdout.strip().startswith("[") else [json.loads(line) for line in docker.stdout.splitlines() if line.strip()]
        services = [{key: record.get(key) for key in ("Service", "Name", "Image", "State", "Status", "Health")}
                    for record in records]
    report["docker"] = {"return_code": docker.returncode, "services": services, "stderr": docker.stderr}
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream).run(suite)
    report["tests"] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                       "skipped": len(result.skipped), "successful": result.wasSuccessful(), "output": stream.getvalue()}
    report["real_orders_enabled_by_this_work"] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination), "days": report["coverage"]["valid_complete_days"],
                      "research_status": report["research_status"], "tests": report["tests"]["run"],
                      "tests_passed": result.wasSuccessful()}, ensure_ascii=True))
    if not result.wasSuccessful() or docker.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime_reports/project_resume_20260908.json")
    capture(parser.parse_args().output)
