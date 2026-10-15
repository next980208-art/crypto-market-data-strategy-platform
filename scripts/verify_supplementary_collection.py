"""Read back captured records and export a bounded, reproducible verification report."""

import argparse
import hashlib
import json
import math
import sqlite3
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def verify(database, manifest):
    results = {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        for family, window in manifest["history"].items():
            if window["status"] != "completed":
                results[family] = {"status": "not_verified", "reason": window}
                continue
            start, end = [int(datetime.fromisoformat(window[k]).timestamp() * 1000)
                          for k in ("start_utc", "end_exclusive_utc")]
            rows = db.execute("SELECT event_ms, period_end_ms, payload FROM observations "
                              "WHERE family=? AND event_ms>=? AND event_ms<? ORDER BY event_ms",
                              (family, start, end)).fetchall()
            invalid, gaps, previous = 0, 0, None
            digest = hashlib.sha256()
            for stamp, period_end, payload in rows:
                digest.update(f"{stamp}:{period_end}:{payload}\n".encode())
                row = json.loads(payload)
                try:
                    if family in ("spot_1m", "index_1m"):
                        o, h, low, close = [float(x) for x in row[1:5]]
                        valid = all(math.isfinite(x) and x > 0 for x in (o, h, low, close))
                        valid = valid and low <= min(o, close) <= max(o, close) <= h
                        valid = valid and period_end == stamp + 60000
                        if family == "spot_1m":
                            valid = valid and math.isfinite(float(row[5])) and float(row[5]) >= 0
                    elif family == "open_interest_5m":
                        valid = all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                                    for k in ("sumOpenInterest", "sumOpenInterestValue"))
                    else:
                        valid = math.isfinite(float(row["fundingRate"]))
                except (KeyError, TypeError, ValueError, IndexError):
                    valid = False
                invalid += not valid
                step = window["interval_ms"]
                if step and previous is not None and stamp - previous != step:
                    gaps += 1
                previous = stamp
            passed = len(rows) == window["rows_in_window"] and invalid == 0 and gaps == 0
            results[family] = {"status": "passed" if passed else "failed", "rows": len(rows),
                               "invalid_values": invalid, "irregular_intervals": gaps if window["interval_ms"] else None,
                               "canonical_records_sha256": digest.hexdigest()}
        requests = dict(zip(("count", "errors", "avg_rtt_ms", "max_rtt_ms"), db.execute(
            "SELECT COUNT(*), SUM(error IS NOT NULL), AVG(elapsed_ms), MAX(elapsed_ms) FROM requests"
        ).fetchone()))
    return {"sqlite_check": integrity, "history": results, "request_metrics_at_readback": requests,
            "passed": integrity == "ok" and all(r["status"] == "passed" for r in results.values())}


def archive_snapshot(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        counts = db.execute("SELECT topic, partition_id, COUNT(*) FROM records GROUP BY topic, partition_id").fetchall()
        partitions = []
        for topic, partition, count in counts:
            row = db.execute("SELECT offset_id, value_zlib FROM records WHERE topic=? AND partition_id=? "
                             "ORDER BY offset_id DESC LIMIT 1", (topic, partition)).fetchone()
            payload = json.loads(zlib.decompress(row[1]))
            partitions.append({"topic": topic, "partition": partition, "stored_rows": count,
                               "last_offset": row[0], "latest_event": payload})
    return {"stored_rows": sum(row[2] for row in counts), "partitions": partitions,
            "scope": "Archive readback counts include previous runs, not proof of continuous exchange capture."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("supplementary_market_store/market_context.sqlite3"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(), "manifest": str(args.manifest),
              **verify(args.database, manifest)}
    path = Path("realtime_runtime/market_archive.sqlite3")
    if path.exists():
        report["archive"] = archive_snapshot(path)
    for name in ("runtime_reports/market_archive_latest.json", "realtime_runtime/collector_metrics.json",
                 "runtime_reports/supplementary_market_latest.json"):
        path = Path(name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Preserve timestamps; these are service reports, not a liveness assertion.
            report.setdefault("service_reports", {})[name] = {k: v for k, v in data.items() if k != "stored"}
        except (OSError, ValueError) as exc:
            report.setdefault("service_report_errors", {})[name] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "history": report["history"],
                      "archive_rows": report.get("archive", {}).get("stored_rows"), "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
