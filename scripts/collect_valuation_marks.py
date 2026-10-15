"""Bounded historical mark-price collection for valuation, never trading inputs."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from scripts import validate_market_context as previous

public = previous.collection.public
search, prior, runner = previous.search, previous.prior, previous.runner
MINUTE = search.MINUTE


def normalize(rows, start, end):
    items = []
    for row in rows:
        stamp, close_end = int(row[0]), int(row[6]) + 1
        o, high, low, close = map(float, row[1:5])
        if stamp % MINUTE or close_end != stamp + MINUTE or not start <= stamp < end:
            raise ValueError("Incomplete or out-of-range minute")
        if not np.isfinite([o, high, low, close]).all() or not 0 < low <= min(o, close) <= max(o, close) <= high:
            raise ValueError("Invalid mark OHLC")
        items.append({"event_ms": stamp, "open": o, "high": high, "low": low, "close": close})
    frame = pd.DataFrame(items)
    if frame.empty or frame.event_ms.duplicated().any():
        raise ValueError("Empty or duplicate mark source")
    return frame.sort_values("event_ms").reset_index(drop=True)


def collect(root):
    reference = search.ROOT / "runtime_reports/market_context_validation_20260918"
    _, _, source, _, _ = previous.inputs(reference)
    schedules = [row["schedule"] for row in prior.read(source / "training.json")]
    if shutil.disk_usage(root.parent).free < 5 * 1024**3:
        raise ValueError("Insufficient disk space")
    root.mkdir(exist_ok=False)
    plan = {"created_utc": search.utc_now(), "schedules": schedules,
            "purpose": "Historical valuation only; not training or independent holdout",
            "endpoint": "/fapi/v1/markPriceKlines", "symbol": "BTCUSDT", "interval": "1m",
            "limits": {"requests": 180, "seconds": 600, "database_bytes": 96 * 1024**2},
            "hashes": {str(p): runner.checksum(p) for p in (Path(__file__), Path(public.__file__), source / "training.json")},
            "orders_allowed": False}
    search.write_json(root / "protocol.json", plan)
    report = {"status": "running", "protocol_sha256": runner.checksum(root / "protocol.json"), "tasks": [], "orders_sent": 0}
    store = public.Store(root / "marks.sqlite3", plan["limits"]["database_bytes"])
    client = public.PublicClient(store)
    started = time.monotonic()
    try:
        for schedule in schedules:
            start, end = schedule["validation_start"], schedule["validation_end"]
            n = schedule["fold"]
            task = {"fold": n, "start_ms": start, "end_ms": end, "status": "running"}
            report["tasks"].append(task)
            search.write_json(root / "collection.json", report)
            cursor = start
            while cursor < end:
                count = store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
                if count >= 178 or time.monotonic() - started > 600 or shutil.disk_usage(root).free < 5 * 1024**3:
                    raise RuntimeError("Collection budget reached")
                page_end = min(end, cursor + 1000 * MINUTE)
                rows, received, url = client.get(public.FUTURES, plan["endpoint"],
                    {"symbol": "BTCUSDT", "interval": "1m", "startTime": cursor, "endTime": page_end - 1, "limit": 1000})
                normalize(rows, cursor, page_end)
                store.save(f"mark1m_{n}", [(int(row[0]), int(row[6]) + 1, row) for row in rows], received, url)
                cursor = max(int(row[0]) for row in rows) + MINUTE
            stored = store.db.execute("SELECT payload,first_received_ms FROM observations WHERE family=? ORDER BY event_ms", (f"mark1m_{n}",)).fetchall()
            frame = normalize([json.loads(row[0]) for row in stored], start, end)
            frame["first_received_ms"] = [row[1] for row in stored]
            path = root / f"mark1m_{n}.parquet"
            frame.to_parquet(path, index=False)
            task.update(status="completed", rows=len(frame), missing_rows=(end - start)//MINUTE - len(frame),
                        path=path.name, sha256=runner.checksum(path))
            search.write_json(root / "collection.json", report)
            print(f"fold{n}: {len(frame)} rows, missing={task['missing_rows']}", flush=True)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="partial", error=str(exc))
        if report["tasks"] and report["tasks"][-1]["status"] == "running":
            report["tasks"][-1].update(status="error", error=str(exc))
    finally:
        report.update(finished_utc=search.utc_now(), elapsed_seconds=time.monotonic() - started,
                      requests=store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
        search.write_json(root / "collection.json", report)
        store.close()
    print(json.dumps(report))
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT) or not root.parent.exists():
        raise ValueError("Use project subdirectory with existing parent")
    with runner.exclusive_lock(root.parent):
        return collect(root)


if __name__ == "__main__":
    raise SystemExit(main())
