"""Finite valuation data for predeclared longer-training development folds."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd

from scripts import collect_valuation_marks as marks
from scripts import validate_funding_event_execution as replay

search, prior, runner, public = marks.search, marks.prior, marks.runner, marks.public


def schedules():
    return [{"fold": n, "validation_start": int(pd.Timestamp(f"{year}-12-23", tz="UTC").timestamp()*1000),
             "validation_end": int((pd.Timestamp(f"{year}-12-23", tz="UTC")+pd.Timedelta(days=30)).timestamp()*1000)}
            for n, year in enumerate((2022, 2023, 2024), 1)]


def validate_marks(frame, start, end):
    np.testing.assert_array_equal(frame.event_ms, np.arange(start, end, search.MINUTE))
    if not np.isfinite(frame[["open", "high", "low", "close"]].to_numpy()).all():
        raise ValueError("Nonfinite mark prices")


def collect(root):
    baseline = search.ROOT / "runtime_reports/funding_event_validation_20260920_v2"
    replay.check_hashes(prior.read(baseline / "protocol.json"), prior.read(baseline / "state.json"), baseline)
    old_marks = search.ROOT / "runtime_reports/valuation_marks_20260919"
    old_context = search.ROOT / "runtime_reports/historical_context_20260918_v2"
    if shutil.disk_usage(search.ROOT).free < 5*1024**3:
        raise ValueError("Insufficient free space")
    root.mkdir(parents=True, exist_ok=False)
    plan = {"created_utc": search.utc_now(), "schedules": schedules(), "symbol": "BTCUSDT",
            "purpose": "Development valuation only; no independent holdout or trading",
            "limits": {"requests": 50, "seconds": 300, "database_bytes": 32*1024**2, "minimum_free_gib": 5},
            "reused": {str(p): runner.checksum(p) for p in [old_marks / "mark1m_2.parquet", old_marks / "mark1m_3.parquet", old_context / "context.sqlite3"]},
            "code_sha256": runner.checksum(Path(__file__)), "orders_allowed": False}
    search.write_json(root / "protocol.json", plan)
    report = {"status": "running", "protocol_sha256": runner.checksum(root / "protocol.json"), "tasks": [], "orders_sent": 0}
    store = public.Store(root / "new_raw.sqlite3", plan["limits"]["database_bytes"])
    client = public.PublicClient(store)
    started = time.monotonic()

    def budget():
        if time.monotonic()-started > 300 or store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] >= 48 or shutil.disk_usage(root).free < 5*1024**3:
            raise RuntimeError("Collection budget reached")

    try:
        for spec in plan["schedules"]:
            n, start, end = spec["fold"], spec["validation_start"], spec["validation_end"]
            task = {**spec, "status": "running", "reused": n < 3}
            report["tasks"].append(task)
            search.write_json(root / "collection.json", report)
            if n < 3:
                frame = pd.read_parquet(old_marks / f"mark1m_{n+1}.parquet")
                with closing(sqlite3.connect((old_context / "context.sqlite3").as_uri()+"?mode=ro", uri=True)) as db:
                    events = [json.loads(r[0]) for r in db.execute("SELECT payload FROM observations WHERE family=? AND event_ms>=? AND event_ms<? ORDER BY event_ms", (f"funding_{n+1}", start, end))]
            else:
                cursor = start
                while cursor < end:
                    budget()
                    page_end = min(end, cursor + 1000*search.MINUTE)
                    rows, received, url = client.get(public.FUTURES, "/fapi/v1/markPriceKlines",
                        {"symbol": "BTCUSDT", "interval": "1m", "startTime": cursor, "endTime": page_end-1, "limit": 1000})
                    marks.normalize(rows, cursor, page_end)
                    store.save("mark1m_3", [(int(r[0]), int(r[6])+1, r) for r in rows], received, url)
                    cursor = max(int(r[0]) for r in rows)+search.MINUTE
                rows = store.db.execute("SELECT payload FROM observations WHERE family='mark1m_3' ORDER BY event_ms").fetchall()
                frame = marks.normalize([json.loads(r[0]) for r in rows], start, end)
                budget()
                events, received, url = client.get(public.FUTURES, "/fapi/v1/fundingRate",
                    {"symbol": "BTCUSDT", "startTime": start, "endTime": end-1, "limit": 1000})
                replay.context_run.collection.normalize(events, "funding", start, end)
                store.save("funding_3", [(int(r["fundingTime"]), int(r["fundingTime"]), r) for r in events], received, url)
            validate_marks(frame, start, end)
            # All three historical BTCUSDT intervals used eight-hour funding. Milliseconds may vary.
            hours = sorted(int(r["fundingTime"]) // (8*3600000) for r in events)
            if hours != list(range(start // (8*3600000), end // (8*3600000))):
                raise ValueError("Missing or duplicate eight-hour funding bucket")
            frame.to_parquet(root / f"mark1m_{n}.parquet", index=False)
            search.write_json(root / f"funding_{n}.json", events)
            task.update(status="completed", mark_rows=len(frame), funding_rows=len(events),
                        hashes={name: runner.checksum(root / name) for name in (f"mark1m_{n}.parquet", f"funding_{n}.json")})
            search.write_json(root / "collection.json", report)
            print(f"fold{n}: marks={len(frame)} funding={len(events)} reused={n<3}", flush=True)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="partial", error=str(exc))
        report["tasks"][-1].update(status="error", error=str(exc))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, finished_utc=search.utc_now(),
                      requests=store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
        store.close()
        search.write_json(root / "collection.json", report)
    print(json.dumps(report))
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use a project subdirectory")
    with runner.exclusive_lock(root.parent):
        return collect(root)


if __name__ == "__main__":
    raise SystemExit(main())
