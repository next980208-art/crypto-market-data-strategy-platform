"""One-shot forward observation for the frozen 15m/24h breakout rule; no orders."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from scripts import collect_supplementary_market_data as public
from scripts import search_low_frequency_rules as rules

lowfreq = rules.lowfreq
prior, runner, search = rules.prior, rules.runner, rules.search

STEP = 15*search.MINUTE
DAY = search.DAY
WARMUP_BARS = 48
HORIZON_BARS = 96
OBSERVATION_DAYS = 90
SYMBOL = "BTCUSDT"
FUTURES = public.FUTURES


def next_boundary(now_ms):
    return (int(now_ms)//STEP+1)*STEP


def normalize_klines(rows, family):
    output = []
    for row in rows:
        stamp, period_end = int(row[0]), int(row[6])+1
        values = list(map(float, row[1:6]))
        if stamp % STEP or period_end != stamp+STEP or not np.isfinite(values).all():
            raise ValueError(f"Invalid {family} candle")
        open_, high, low, close, volume = values
        if not 0 < low <= min(open_, close) <= max(open_, close) <= high or volume < 0:
            raise ValueError(f"Invalid {family} OHLCV")
        output.append({"timestamp": stamp, "event_ms": stamp, "period_end_ms": period_end,
                       "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    frame = pd.DataFrame(output)
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if not frame.timestamp.diff().dropna().eq(STEP).all():
        raise ValueError(f"Non-contiguous {family} candles")
    return frame


class FifteenMinuteFundingMarket:
    """Funding valuation using the last completed 15m mark bar as an explicit proxy."""

    def __init__(self, events, marks):
        self.events = sorted(events, key=lambda row: int(row["fundingTime"]))
        self.times = np.asarray([int(row["fundingTime"]) for row in self.events], dtype=np.int64)
        if len(self.times) and (np.diff(self.times) <= 0).any():
            raise ValueError("Duplicate funding events")
        marks = marks.sort_values("event_ms")
        self.available = marks.period_end_ms.to_numpy(dtype=np.int64)
        self.prices = marks.close.to_numpy(dtype=float)
        if len(self.available) and ((np.diff(self.available) <= 0).any() or not np.isfinite(self.prices).all()):
            raise ValueError("Invalid mark proxy")

    def settlements(self, entry_ms, exit_ms):
        left = self.times.searchsorted(entry_ms, side="right")
        right = self.times.searchsorted(exit_ms, side="left")
        result = []
        for source in self.events[left:right]:
            stamp, rate = int(source["fundingTime"]), float(source["fundingRate"])
            index = self.available.searchsorted(stamp, side="right")-1
            if index < 0 or stamp-self.available[index] >= STEP or not np.isfinite(rate):
                raise ValueError("Missing causal 15m funding mark proxy")
            result.append({"timestamp": stamp, "rate": rate, "price": float(self.prices[index]),
                           "proxy": True, "price_available_ms": int(self.available[index]),
                           "boundary_uncertain": min(stamp-entry_ms, exit_ms-stamp) <= STEP})
        return result


def fetch_klines(store, client, family, endpoint, start, end):
    cursor = start
    while cursor < end:
        page_end = min(end, cursor+1000*STEP)
        rows, received, url = client.get(FUTURES, endpoint,
            {"symbol": SYMBOL, "interval": "15m", "startTime": cursor, "endTime": page_end-1, "limit": 1000})
        if not isinstance(rows, list) or not rows:
            break
        records = []
        for row in rows:
            stamp, period_end = int(row[0]), int(row[6])+1
            if cursor <= stamp < page_end and period_end <= end:
                records.append((stamp, period_end, row))
        store.save(family, records, received, url)
        cursor = max(int(row[0]) for row in rows)+STEP


def fetch_funding(store, client, start, end):
    rows, received, url = client.get(FUTURES, "/fapi/v1/fundingRate",
        {"symbol": SYMBOL, "startTime": start, "endTime": end-1, "limit": 1000})
    if not isinstance(rows, list):
        raise ValueError("Invalid funding payload")
    records = []
    for row in rows:
        stamp, rate = int(row["fundingTime"]), float(row["fundingRate"])
        if row.get("symbol") != SYMBOL or not start <= stamp < end or not np.isfinite(rate):
            raise ValueError("Invalid funding event")
        records.append((stamp, stamp, row))
    store.save("funding", records, received, url)


def read_payloads(store, family):
    return [json.loads(row[0]) for row in store.db.execute(
        "SELECT payload FROM observations WHERE family=? ORDER BY event_ms", (family,))]


def build_scored_frame(bars, start_ms):
    if bars.empty or len(bars) < WARMUP_BARS+HORIZON_BARS+1:
        return pd.DataFrame()
    raw = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    raw["datetime_utc"] = pd.to_datetime(raw.timestamp, unit="ms", utc=True)
    raw["symbol"], raw["market"], raw["timeframe"] = SYMBOL, "usdm", "15m"
    raw["feature_schema_version"] = "fixed_breakout_forward_v1"
    features, _ = lowfreq.add_features(raw)
    labeler = search.load("4_triple_barrier_labeler.py", "fixed_breakout_labels")
    labels = labeler.add_labels(raw, [1.5], .01, 1440, 10, 4096, min(10, len(raw)))
    columns = [column for column in labels if column not in ("datetime_utc", "symbol", "market", "timeframe", "close")]
    frame = features.merge(labels[columns], on="timestamp", how="inner", validate="one_to_one")
    frame = frame[frame.timestamp >= start_ms].reset_index(drop=True)
    return rules.apply_rule(frame, "breakout") if not frame.empty else frame


def evaluate(root, plan, bars, marks, funding):
    scored = build_scored_frame(bars, plan["observation_start_ms"])
    action = search.load("16_action_profitability_walk_forward_model.py", "fixed_breakout_action")
    backtest = search.load("8_model_signal_backtest.py", "fixed_breakout_backtest")
    market = FifteenMinuteFundingMarket(funding, marks)
    results, hashes = {}, {}
    for cost_case, extra_bps in rules.consensus.COSTS:
        if scored.empty:
            metrics = {"signals": 0, "long_signals": 0, "short_signals": 0, "maximum_direction_share": 1.,
                       "trades": 0, "win_rate_pct": 0., "avg_return_r": 0., "total_return_pct": 0., "max_drawdown_pct": 0.}
            summary = {}
            tables = {
                "trades": pd.DataFrame(columns=["signal_timestamp", "equity"]),
                "events": pd.DataFrame(columns=["timestamp", "exit_time_utc", "return_pct"]),
                "daily": pd.DataFrame(columns=["day_start_ms", "realized_pnl"]),
            }
        else:
            metrics, summary, tables = lowfreq.evaluate(scored, action, backtest, extra_bps, market)
        for name, table in tables.items():
            path = root/f"current_{cost_case}_{name}.parquet"
            table.to_parquet(path, index=False)
            hashes[path.name] = runner.checksum(path)
        results[cost_case] = {"metrics": metrics, "summary": summary}
    return scored, results, hashes


def initialize(root):
    if shutil.disk_usage(search.ROOT).free < 5*1024**3:
        raise ValueError("Insufficient free space")
    root.mkdir(parents=True, exist_ok=False)
    (root/"snapshots").mkdir()
    now_ms = time.time_ns()//1_000_000
    start = next_boundary(now_ms)
    source_trial = search.ROOT/"runtime_reports/low_frequency_rules_20260922/trials/15m_hold1440_stop1pct_breakout.json"
    plan = {"created_utc": search.utc_now(), "observation_start_ms": start,
            "observation_start_utc": pd.Timestamp(start, unit="ms", tz="UTC").isoformat(),
            "review_after_ms": start+OBSERVATION_DAYS*DAY,
            "review_after_utc": pd.Timestamp(start+OBSERVATION_DAYS*DAY, unit="ms", tz="UTC").isoformat(),
            "symbol": SYMBOL, "timeframe": "15m", "rule_id": "15m_hold1440_stop1pct_breakout",
            "rule": "range_position_48bar >=0.9 long; <=0.1 short", "stop_distance_pct": .01,
            "take_profit_r": 1.5, "max_holding_minutes": 1440, "risk_per_trade_pct": .005,
            "cost_cases_extra_bps": {name: bps for name, bps in rules.consensus.COSTS},
            "minimum_observation_days": OBSERVATION_DAYS, "minimum_monthly_trades": 20,
            "weekly_gate": "At least 3 positive complete weeks out of each 4-week review block",
            "storage_limit_bytes": 32*1024**2, "minimum_free_gib": 5,
            "source_trial": str(source_trial), "source_trial_sha256": runner.checksum(source_trial),
            "code_sha256": runner.checksum(Path(__file__)),
            "limitations": "15m mark close is a causal funding price proxy; public REST observation, not fill or order evidence; no independent result exists until future outcomes mature",
            "credentials_used": False, "orders_allowed": False, "deployment_allowed": False}
    search.write_json(root/"protocol.json", plan)
    search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json"),
        "orders_sent": 0, "runs": []})


def collect(root, resume):
    if not resume:
        initialize(root)
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    if runner.checksum(Path(__file__)) != plan["code_sha256"]:
        raise ValueError("Observer code changed; review before starting a new run")
    if runner.checksum(Path(plan["source_trial"])) != plan["source_trial_sha256"]:
        raise ValueError("Frozen source trial changed")
    if shutil.disk_usage(root).free < plan["minimum_free_gib"]*1024**3:
        raise ValueError("Insufficient free space")
    store = public.Store(root/"forward_observation.sqlite3", plan["storage_limit_bytes"])
    client = public.PublicClient(store)
    started = time.monotonic()
    try:
        server, _, _ = client.get(FUTURES, "/fapi/v1/time")
        now_ms = int(server["serverTime"])
        latest_end = now_ms//STEP*STEP
        warmup_start = plan["observation_start_ms"]-WARMUP_BARS*STEP
        for family, endpoint in (("contract15", "/fapi/v1/klines"), ("mark15", "/fapi/v1/markPriceKlines")):
            last = store.db.execute("SELECT MAX(event_ms) FROM observations WHERE family=?", (family,)).fetchone()[0]
            cursor = warmup_start if last is None else int(last)+STEP
            if cursor < latest_end:
                fetch_klines(store, client, family, endpoint, cursor, latest_end)
        if plan["observation_start_ms"] < now_ms:
            fetch_funding(store, client, plan["observation_start_ms"], now_ms)
        bars = normalize_klines(read_payloads(store, "contract15"), "contract15")
        marks = normalize_klines(read_payloads(store, "mark15"), "mark15")
        funding = read_payloads(store, "funding")
        bars.to_parquet(root/"bars_15m.parquet", index=False)
        marks.to_parquet(root/"mark_15m.parquet", index=False)
        search.write_json(root/"funding.json", funding)
        scored, results, derived_hashes = evaluate(root, plan, bars, marks, funding)
        observed_end = max(plan["observation_start_ms"], int(bars.period_end_ms.max()) if not bars.empty else plan["observation_start_ms"])
        observed_days = max(0., (observed_end-plan["observation_start_ms"])/DAY)
        status = "ready_for_review" if observed_days >= plan["minimum_observation_days"] else "collecting"
        report = {"status": status, "created_utc": search.utc_now(), "exchange_server_utc": pd.Timestamp(now_ms, unit="ms", tz="UTC").isoformat(),
            "observation_start_utc": plan["observation_start_utc"], "review_after_utc": plan["review_after_utc"],
            "observed_days": observed_days, "contract_bars": len(bars), "mark_bars": len(marks),
            "funding_events": len(funding), "mature_signal_rows": len(scored), "results": results,
            "hashes": {"bars_15m.parquet": runner.checksum(root/"bars_15m.parquet"),
                       "mark_15m.parquet": runner.checksum(root/"mark_15m.parquet"),
                       "funding.json": runner.checksum(root/"funding.json"), **derived_hashes},
            "database_bytes": sum(path.stat().st_size for path in (root/"forward_observation.sqlite3", Path(str(root/"forward_observation.sqlite3")+"-wal")) if path.exists()),
            "elapsed_seconds": time.monotonic()-started, "credentials_used": False, "orders_sent": 0,
            "decision": "no_trade" if status != "ready_for_review" else "manual_review_required"}
        snapshot = root/"snapshots"/(str(now_ms)+".json")
        if snapshot.exists() and prior.read(snapshot) != report:
            raise ValueError("Snapshot timestamp collision")
        search.write_json(snapshot, report)
        search.write_json(root/"latest_report.json", report)
        state.update(status=status, last_run_utc=search.utc_now(), orders_sent=0)
        state["runs"].append({"snapshot": snapshot.name, "exchange_server_ms": now_ms,
                              "observed_days": observed_days, "status": status})
        search.write_json(root/"state.json", state)
        return report
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Output must be a project subdirectory")
    if args.resume and not root.exists():
        raise ValueError("Cannot resume a missing observation folder")
    if not args.resume and root.exists():
        raise ValueError("Observation folder exists; use --resume")
    with runner.exclusive_lock(root.parent):
        report = collect(root, args.resume)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
