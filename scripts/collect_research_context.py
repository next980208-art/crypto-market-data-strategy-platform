"""Finite public historical context collection for the existing three research folds."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from scripts import collect_supplementary_market_data as public
from scripts import search_risk_sizing as sizing

search, prior, runner = sizing.search, sizing.prior, sizing.runner
STEP = 15*search.MINUTE
SPECS = {
    "spot15": (public.SPOT, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "15m"}),
    "index15": (public.FUTURES, "/fapi/v1/indexPriceKlines", {"pair": "BTCUSDT", "interval": "15m"}),
    "mark15": (public.FUTURES, "/fapi/v1/markPriceKlines", {"symbol": "BTCUSDT", "interval": "15m"}),
    "funding": (public.FUTURES, "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}),
}


def normalize(rows, family, start, end):
    parsed = []
    for row in rows:
        if family == "funding":
            stamp, value = int(row["fundingTime"]), float(row["fundingRate"])
            if row["symbol"] != "BTCUSDT" or not np.isfinite(value):
                raise ValueError("Invalid funding payload")
            parsed.append({"event_ms": stamp, "period_end_ms": stamp, "value": value})
        else:
            stamp = int(row[0])
            o, h, low, close = map(float, row[1:5])
            close_end = int(row[6])+1
            if stamp % STEP or not stamp < close_end <= stamp+STEP or not np.isfinite([o, h, low, close]).all() or not 0 < low <= min(o, close) <= max(o, close) <= h:
                raise ValueError("Invalid candle payload")
            item = {"event_ms": stamp, "period_end_ms": close_end, "value": close,
                    "complete_candle": close_end == stamp+STEP}
            if family == "spot15":
                volume, taker = float(row[5]), float(row[9])
                if not np.isfinite([volume, taker]).all() or not 0 <= taker <= volume:
                    raise ValueError("Invalid spot volume")
                item.update(volume=volume, taker_ratio=taker/volume if volume else np.nan)
            parsed.append(item)
    frame = pd.DataFrame(parsed)
    if frame.empty or frame.event_ms.duplicated().any() or not frame.event_ms.between(start, end-1).all():
        raise ValueError("Empty, duplicate or out-of-range source")
    frame = frame.sort_values("event_ms").reset_index(drop=True)
    return frame


def collect(root, resume):
    reference = (search.ROOT/"runtime_reports/literature_sampling_search_20260917").resolve()
    source, _ = sizing.check_reference(reference)
    schedules = [r["schedule"] for r in prior.read(source/"training.json")]
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        plan = {"created_utc": search.utc_now(), "schedules": schedules, "warmup_days": 2,
                "scope": "BTCUSDT 15m spot/index/mark and funding; 90d train +30d validation per fold, 2d warmup",
                "limits": {"requests": 160, "seconds_per_invocation": 600, "store_bytes": 128*1024**2, "minimum_free_gib": 5},
                "source_training_sha256": runner.checksum(source/"training.json"), "code_sha256": runner.checksum(Path(__file__)),
                "collector_sha256": runner.checksum(Path(public.__file__)), "orders_sent": 0, "credentials_used": False}
        search.write_json(root/"protocol.json", plan)
        search.write_json(root/"collection.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json"), "tasks": []})
    plan, report = prior.read(root/"protocol.json"), prior.read(root/"collection.json")
    if runner.checksum(root/"protocol.json") != report["protocol_sha256"] or runner.checksum(Path(__file__)) != plan["code_sha256"] or runner.checksum(Path(public.__file__)) != plan["collector_sha256"] or runner.checksum(source/"training.json") != plan["source_training_sha256"]:
        raise ValueError("Frozen collection input changed")
    if shutil.disk_usage(root).free < 5*1024**3:
        raise ValueError("Insufficient disk space")
    started = time.monotonic()
    store = public.Store(root/"context.sqlite3", plan["limits"]["store_bytes"])
    client = public.PublicClient(store)
    try:
        for fold in schedules:
            n = fold["fold"]
            start, end = fold["validation_start"]-92*search.DAY, fold["validation_end"]
            for family, (base, endpoint, params) in SPECS.items():
                task_id = f"{family}_{n}"
                path = root/(task_id+".parquet")
                complete = next((r for r in report["tasks"] if r["id"] == task_id and r["status"] == "completed"), None)
                if complete:
                    if runner.checksum(path) != complete["sha256"]:
                        raise ValueError("Completed artifact changed")
                    continue
                task = {"id": task_id, "family": family, "fold": n, "start_ms": start, "end_ms": end, "started_utc": search.utc_now(), "status": "running"}
                report["tasks"].append(task)
                report["status"] = "running"
                search.write_json(root/"collection.json", report)
                try:
                    cursor = start
                    while cursor < end:
                        count = store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
                        if count >= 158 or time.monotonic()-started > 600 or shutil.disk_usage(root).free < 5*1024**3:
                            raise RuntimeError("Collection request/time/disk budget reached")
                        page_end = min(end, cursor+1000*STEP) if family != "funding" else end
                        rows, received, url = client.get(base, endpoint, {**params, "startTime": cursor, "endTime": page_end-1, "limit": 1000})
                        if not isinstance(rows, list) or not rows:
                            break
                        records = []
                        for row in rows:
                            stamp = int(row["fundingTime"] if family == "funding" else row[0])
                            period_end = stamp if family == "funding" else int(row[6])+1
                            if not cursor <= stamp < page_end or period_end > end:
                                raise ValueError("API returned out-of-window data")
                            records.append((stamp, period_end, row))
                        store.save(task_id, records, received, url)
                        cursor = max(r[0] for r in records)+(1 if family == "funding" else STEP)
                        if family == "funding" and len(rows) < 1000:
                            break
                    rows = store.db.execute("SELECT payload, first_received_ms FROM observations WHERE family=? ORDER BY event_ms", (task_id,)).fetchall()
                    frame = normalize([json.loads(r[0]) for r in rows], family, start, end)
                    frame["first_received_ms"] = [r[1] for r in rows]
                    frame.to_parquet(path, index=False)
                    task.update(status="completed", rows=len(frame), sha256=runner.checksum(path), path=path.name,
                                missing_rows=(end-start)//STEP-len(frame) if family != "funding" else None,
                                incomplete_candles=int((~frame.complete_candle).sum()) if family != "funding" else None)
                except Exception as exc:
                    task.update(status="error", error=str(exc))
                search.write_json(root/"collection.json", report)
                print(f"{task_id}: {task['status']} rows={task.get('rows')} error={task.get('error')}", flush=True)
                # Preserve first connectivity failure and require an explicit retry, not a loop across all sources.
                if task["status"] == "error":
                    report.update(status="partial", stopped_utc=search.utc_now())
                    search.write_json(root/"collection.json", report)
                    return 1
        report.update(status="completed", finished_utc=search.utc_now(), orders_sent=0,
                      requests=store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0], elapsed_seconds=time.monotonic()-started)
        search.write_json(root/"collection.json", report)
    finally:
        store.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use project subdirectory")
    with runner.exclusive_lock(root.parent):
        return collect(root, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
