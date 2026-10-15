"""Conservative funding/one-minute mark diagnostic on fixed historical trades.

This does not rerun trading guards or approve models. OHLC proxies are not exact
settlement marks; a one-minute boundary buffer excludes uncertain credits.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from scripts import search_context_ablation as research

search, prior, runner = research.search, research.prior, research.runner
MINUTE = search.MINUTE
RISK, STOP, COST = .005, .005, .0015
RATIO = RISK / (STOP + COST)


def funding_flows(side, entry, exit_ms, entry_price, events, marks):
    if side not in ("long", "short") or exit_ms < entry or not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("Invalid trade")
    sign = 1 if side == "long" else -1
    flows = []
    for event in events:
        stamp = int(event["fundingTime"])
        if not entry - MINUTE <= stamp <= exit_ms + MINUTE:
            continue
        rate = float(event["fundingRate"])
        if not np.isfinite(rate):
            raise ValueError("Nonfinite funding")
        uncertain = stamp <= entry + MINUTE or stamp >= exit_ms - MINUTE
        # Keep possible payments, never credit uncertain boundary receipts.
        if uncertain and sign * rate <= 0:
            continue
        raw_mark = event.get("markPrice")
        proxy = raw_mark in (None, "")
        if proxy:
            minute = stamp // MINUTE * MINUTE
            if minute not in marks.index:
                raise ValueError("Missing funding proxy minute")
            price = float(marks.loc[minute, "high" if sign * rate > 0 else "low"])
        else:
            price = float(raw_mark)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Invalid settlement price")
        fraction = -sign * RATIO * (price / entry_price) * rate
        flows.append({"funding_ms": stamp, "booked_ms": min(max(stamp, entry), exit_ms),
                      "return_fraction": fraction, "price_used": price, "ohlc_proxy": proxy,
                      "uncertain_timing": uncertain})
    return flows


def audit_trades(trades, events, marks):
    marks = marks.set_index("event_ms").sort_index()
    if marks.index.duplicated().any():
        raise ValueError("Duplicate marks")
    equity, original_equity, peak, mdd = 1., 1., 1., 0.
    rows, previous_exit = [], None

    def observe(value):
        nonlocal peak, mdd
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid or insolvent diagnostic equity")
        peak = max(peak, value)
        mdd = min(mdd, value / peak - 1)

    for trade in trades.to_dict("records"):
        entry = int(trade["entry_timestamp"])
        exit_ms = int(pd.Timestamp(trade["exit_time_utc"]).timestamp() * 1000)
        if previous_exit is not None and entry < previous_exit:
            raise ValueError("Overlapping or unordered trades")
        if entry % MINUTE or exit_ms % MINUTE:
            raise ValueError("Unexpected trade clock")
        np.testing.assert_allclose(trade["return_pct"], trade["return_r"] * RISK, atol=1e-12, rtol=0)
        flows = funding_flows(trade["action"], entry, exit_ms, trade["entry_price"], events, marks)
        timestamps = np.arange(entry, exit_ms - MINUTE, MINUTE, dtype=np.int64)
        # Exclude the exit minute: an intrabar stop may precede its closing mark.
        if not pd.Index(timestamps).isin(marks.index).all():
            raise ValueError("Missing open-position mark minute")
        sign = 1 if trade["action"] == "long" else -1
        booked = np.array([f["booked_ms"] for f in flows], dtype=np.int64)
        amounts = np.array([f["return_fraction"] for f in flows], dtype=float)
        observe(equity * (1 - RATIO * COST + amounts[booked <= entry].sum()))
        for stamp, price in zip(timestamps, marks.loc[timestamps, "close"].to_numpy()):
            elapsed_funding = amounts[booked <= stamp + MINUTE].sum()
            value = equity * (1 + sign * RATIO * (price / trade["entry_price"] - 1) - RATIO * COST + elapsed_funding)
            observe(value)
        funding = float(amounts.sum())
        equity *= 1 + trade["return_pct"] + funding
        original_equity *= 1 + trade["return_pct"]
        observe(equity)
        rows.append({"entry_timestamp": entry, "exit_ms": exit_ms, "side": trade["action"],
                     "original_return_fraction": trade["return_pct"], "funding_return_fraction": funding,
                     "diagnostic_equity": equity, "mark_observations": len(timestamps), "funding_flows": flows})
        previous_exit = exit_ms
    return {"trades": len(rows), "original_return_pct": (original_equity - 1) * 100,
            "funding_adjusted_fixed_schedule_return_pct": (equity - 1) * 100,
            "return_delta_pp": (equity - original_equity) * 100,
            "sampled_mark_drawdown_pct": mdd * 100,
            "funding_events_booked": sum(len(r["funding_flows"]) for r in rows),
            "proxy_events": sum(f["ohlc_proxy"] for r in rows for f in r["funding_flows"]),
            "uncertain_timing_events": sum(f["uncertain_timing"] for r in rows for f in r["funding_flows"]),
            "trade_rows": rows}


def build(plan):
    marks_root = Path(plan["marks"])
    collection = prior.read(marks_root / "collection.json")
    if collection["status"] != "completed":
        raise ValueError("Incomplete marks collection")
    marks = {}
    for task in collection["tasks"]:
        path = marks_root / task["path"]
        if runner.checksum(path) != task["sha256"] or task["missing_rows"]:
            raise ValueError("Mark coverage or hash failure")
        marks[task["fold"]] = pd.read_parquet(path)
    with closing(sqlite3.connect(Path(plan["funding_db"]).as_uri() + "?mode=ro", uri=True)) as db:
        events = {n: [json.loads(row[0]) for row in db.execute(
            "SELECT payload FROM observations WHERE family=? ORDER BY event_ms", (f"funding_{n}",))] for n in (1, 2, 3)}
    results = []
    for root_name in plan["research_roots"]:
        root = Path(root_name)
        report = research.verified_reference(root)
        for trial in report["trials"]:
            for fold in trial["folds"]:
                n, key = fold["fold"], trial["config"]["id"]
                item = {"research": root.name, "policy": key, "fold": n, "cost_case": "normal"}
                try:
                    trades = pd.read_parquet(root / "trades" / f"{key}_{n}_normal.parquet")
                    diagnostic = audit_trades(trades, events[n], marks[n])
                    np.testing.assert_allclose(diagnostic["original_return_pct"], fold["normal"]["total_return_pct"], atol=1e-10, rtol=0)
                    item.update(status="completed", **diagnostic)
                except Exception as exc:
                    item.update(status="error", error=str(exc))
                results.append(item)
    return {"status": "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors",
            "cases": results, "orders_sent": 0, "model_approved": False,
            "limitations": plan["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marks", type=Path, default=Path("runtime_reports/valuation_marks_20260919"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use a project subdirectory")
    if not args.verify:
        roots = [search.ROOT / ("runtime_reports/" + name) for name in
                 ("market_context_validation_20260918", "context_ablation_20260919")]
        _, _, source, _, context = research.context_run.inputs(roots[0])
        for n in (1, 2, 3):
            frame = pd.read_parquet(source / f"validation_{n}.parquet", columns=["stop_distance_pct", "round_trip_cost_bps"])
            if not frame.stop_distance_pct.eq(STOP).all() or not frame.round_trip_cost_bps.eq(10).all():
                raise ValueError("Unexpected sizing inputs")
        files = [Path(__file__).resolve(), Path(research.__file__).resolve(), args.marks.resolve() / "collection.json", context / "context.sqlite3"]
        files += [r / name for r in roots for name in ("results.json", "verification.json")]
        root.mkdir(parents=True, exist_ok=False)
        plan = {"marks": str(args.marks.resolve()), "funding_db": str(context / "context.sqlite3"),
                "research_roots": [str(r) for r in roots], "hashes": {str(p): runner.checksum(p) for p in files},
                "scope": "90 normal-cost fixed trade schedules, historical accounting diagnostic only",
                "limitations": "No guard rerun: fixed historical trade schedule. Roundtrip cost reserved at entry; funding booked at observed event clamped to trade bounds. +/-1m timing uncertainty: payments kept, credits excluded. Missing settlement mark uses minute high for payments and low for credits, not exact price. 1m close samples exclude exit minute; not tick drawdown, margin or liquidation. Not independent validation or approval."}
        search.write_json(root / "protocol.json", plan)
        (root / "code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    plan = prior.read(root / "protocol.json")
    for name, digest in plan["hashes"].items():
        if runner.checksum(Path(name)) != digest:
            raise ValueError("Frozen valuation input changed: " + name)
    results = build(plan)
    if args.verify:
        if results != prior.read(root / "results.json"):
            raise ValueError("Valuation replay mismatch")
        search.write_json(root / "verification.json", {"status": "passed", "cases": len(results["cases"]),
                          "results_sha256": runner.checksum(root / "results.json"), "orders_sent": 0})
    else:
        search.write_json(root / "results.json", results)
    print(json.dumps({"status": results["status"], "cases": len(results["cases"]),
                      "errors": [r for r in results["cases"] if r["status"] == "error"]}))
    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
