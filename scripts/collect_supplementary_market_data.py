"""Public, read-only Binance context capture; never loads keys or sends orders."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


MINUTE = 60_000
DAY = 1440 * MINUTE
FUTURES = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
SPECS = {
    "open_interest_5m": (FUTURES, "/futures/data/openInterestHist", {"symbol": "BTCUSDT", "period": "5m"}, 500, 5 * MINUTE, "timestamp"),
    "spot_1m": (SPOT, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m"}, 1000, MINUTE, None),
    "index_1m": (FUTURES, "/fapi/v1/indexPriceKlines", {"pair": "BTCUSDT", "interval": "1m"}, 1000, MINUTE, None),
    "funding": (FUTURES, "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000, None, "fundingTime"),
}
LIMITATIONS = {
    "old_open_interest": "Public OI statistics retain about one month; this capture conservatively requests 29 days. Older missing values are not fabricated.",
    "historical_depth": "Existing archive plus capture from now; disconnected source intervals cannot be reconstructed by current depth snapshots.",
    "account_commission": "Not collected: requires signed USER_DATA access and account-specific fee tier; no credentials loaded.",
    "actual_fills_and_order_latency": "Not collected: requires actual order/fill events. No orders sent, no testnet fills presented as mainnet fills.",
    "latency": "REST round-trip duration is measured, not one-way network delay or order execution latency.",
    "slippage": "Hypothetical depth VWAP, not a fill. Excludes fees, funding cashflows, queue position and market changes.",
    "training": "Separate raw research store; not merged into models or approved for trading. Historical collection time is retained to avoid pretending it was observed live.",
}


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


class Store:
    def __init__(self, path, max_bytes=1024 ** 3):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                family TEXT NOT NULL, event_ms INTEGER NOT NULL,
                period_end_ms INTEGER NOT NULL, first_received_ms INTEGER NOT NULL,
                source TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(family, event_ms));
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY, requested_ms INTEGER, received_ms INTEGER,
                elapsed_ms REAL, url TEXT, status INTEGER, error TEXT);
            CREATE TABLE IF NOT EXISTS coverage (
                family TEXT PRIMARY KEY, complete_end_ms INTEGER NOT NULL);
        """)

    def save(self, family, records, received, source):
        self.check_budget()
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?)",
                [(family, stamp, period_end, received, source, json.dumps(payload, separators=(",", ":")))
                 for stamp, period_end, payload in records],
            )

    def check_budget(self):
        size = sum(p.stat().st_size for p in (self.path, Path(str(self.path) + "-wal")) if p.exists())
        if size >= self.max_bytes:
            raise RuntimeError("Supplementary storage budget reached; no existing records deleted")

    def summary(self):
        result = {}
        for family, count, first, last in self.db.execute(
            "SELECT family, COUNT(*), MIN(event_ms), MAX(event_ms) FROM observations GROUP BY family"
        ):
            last_row = self.db.execute(
                "SELECT first_received_ms, payload FROM observations WHERE family=? AND event_ms=?", (family, last)
            ).fetchone()
            result[family] = {"rows": count, "first_utc": utc(first), "last_utc": utc(last),
                              "last_received_utc": utc(last_row[0]), "latest": json.loads(last_row[1])}
        return result

    def close(self):
        self.db.close()


class PublicClient:
    def __init__(self, store):
        self.store = store

    def get(self, base, path, params=None):
        self.store.check_budget()
        url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        for attempt in range(3):
            started_ms = time.time_ns() // 1_000_000
            started = time.monotonic()
            status, error, body = 0, None, None
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "coin-project-public-research/1"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    status = response.status
                    body = json.load(response)
                if isinstance(body, dict) and int(body.get("code", 0)) < 0:
                    raise ValueError(f"Binance API error: {body}")
            except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
                status = getattr(exc, "code", status)
                error = str(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    error += ": " + exc.read(2048).decode("utf-8", errors="replace")
            received = time.time_ns() // 1_000_000
            elapsed = round((time.monotonic() - started) * 1000, 3)
            with self.store.db:
                self.store.db.execute("INSERT INTO requests VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                                      (started_ms, received, elapsed, url, status, error))
            time.sleep(0.15)
            if error is None:
                return body, received, url
            # Never hammer a ban/rate limit or retry invalid historical requests.
            if status in (400, 401, 403, 418, 429, 451) or attempt == 2:
                raise RuntimeError(f"HTTP {status}: {url}: {error}")
            time.sleep(2 ** attempt)


def collect_history(store, client, family, now_ms, days):
    base, path, params, limit, step, key = SPECS[family]
    end = now_ms // (step or MINUTE) * (step or MINUTE)
    lookback = min(days, 29) if family == "open_interest_5m" else days
    start = end - lookback * DAY
    complete = store.db.execute("SELECT complete_end_ms FROM coverage WHERE family=?", (family,)).fetchone()
    first = store.db.execute("SELECT MIN(event_ms) FROM observations WHERE family=?", (family,)).fetchone()[0]
    if first is None or first > start:
        complete = None
    cursor = max(start, complete[0] - 2 * (step or DAY)) if complete else start
    requested_start = cursor
    while cursor < end:
        # OI can return the newest rows in the range; bound each request to one page.
        page_end = min(end, cursor + limit * step) if step else end
        rows, received, source = client.get(base, path, {**params, "startTime": cursor, "endTime": page_end - 1, "limit": limit})
        if not isinstance(rows, list):
            raise ValueError(f"{family}: unexpected API payload")
        if not rows:
            break
        records, stamps = [], []
        for row in rows:
            stamp = int(row[key] if key else row[0])
            period_end = stamp if key else int(row[6]) + 1
            stamps.append(stamp)
            if cursor <= stamp < page_end and period_end <= now_ms:
                records.append((stamp, period_end, row))
        store.save(family, records, received, source)
        next_cursor = max(stamps) + (step or 1)
        if next_cursor <= cursor:
            raise ValueError(f"{family}: pagination did not advance")
        cursor = next_cursor
    count = store.db.execute(
        "SELECT COUNT(*) FROM observations WHERE family=? AND event_ms>=? AND event_ms<?", (family, start, end)
    ).fetchone()[0]
    expected = (end - start) // step if step else None
    bad_grid = 0 if step is None else store.db.execute(
        "SELECT COUNT(*) FROM observations WHERE family=? AND event_ms>=? AND event_ms<? AND event_ms % ? != 0",
        (family, start, end, step),
    ).fetchone()[0]
    status = "completed" if count > 0 and (expected is None or (count == expected and bad_grid == 0)) else "partial"
    if status == "completed":
        with store.db:
            store.db.execute("INSERT INTO coverage VALUES (?, ?) ON CONFLICT(family) DO UPDATE SET complete_end_ms=excluded.complete_end_ms",
                             (family, end))
    return {"status": status, "start_utc": utc(start), "end_exclusive_utc": utc(end),
            "requested_start_utc": utc(requested_start), "rows_in_window": count,
            "expected_rows": expected, "missing_rows": None if expected is None else expected - count,
            "off_grid_rows": bad_grid, "interval_ms": step}


def depth_costs(book):
    bids = sorted([(float(p), float(q)) for p, q in book["bids"] if float(q) > 0], reverse=True)
    asks = sorted([(float(p), float(q)) for p, q in book["asks"] if float(q) > 0])
    if not bids or not asks:
        raise ValueError("Empty depth")
    if any(not math.isfinite(p) or not math.isfinite(q) or p <= 0 for p, q in bids + asks):
        raise ValueError("Invalid depth values")
    bid, ask = bids[0][0], asks[0][0]
    if ask < bid:
        raise ValueError("Crossed order book")
    mid = (bid + ask) / 2
    estimates = []
    for side, levels in (("buy", asks), ("sell", bids)):
        for size in (0.001, 0.01, 0.1):
            remaining, notional = size, 0.0
            for price, quantity in levels:
                taken = min(remaining, quantity)
                notional += taken * price
                remaining -= taken
                if remaining <= 1e-12:
                    break
            filled = remaining <= 1e-12
            vwap = notional / size if filled else None
            estimates.append({"side": side, "quantity_btc": size, "sufficient_visible_depth": filled,
                              "estimated_vwap": vwap,
                              "cost_vs_mid_bps": None if vwap is None else (vwap / mid - 1) * 10000 * (1 if side == "buy" else -1)})
    return {"spread": ask - bid, "spread_bps": (ask - bid) / mid * 10000,
            "bid": bid, "ask": ask, "estimates": estimates,
            "actual_execution": False, "includes_fees_or_funding": False}


def collect_snapshots(store, client):
    result = {}
    for family, path, params in (
        ("depth_snapshot", "/fapi/v1/depth", {"symbol": "BTCUSDT", "limit": 100}),
        ("mark_index_funding_snapshot", "/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"}),
        ("exchange_rules", "/fapi/v1/exchangeInfo", {}),
    ):
        try:
            body, received, source = client.get(FUTURES, path, params)
            if family == "exchange_rules":
                body = {"symbols": [row for row in body["symbols"] if row["symbol"] == "BTCUSDT"],
                        "rateLimits": body.get("rateLimits", [])}
            # Snapshots are keyed by capture time, not misleading historical availability.
            store.save(family, [(received, received, body)], received, source)
            if family == "depth_snapshot":
                costs = depth_costs(body)
                store.save("estimated_costs", [(received, received, costs)], received, source)
            result[family] = {"status": "completed", "received_utc": utc(received)}
        except Exception as exc:
            result[family] = {"status": "failed", "error": str(exc)}
    return result


def collect_cycle(args):
    store = Store(args.database, args.max_bytes)
    started = time.time_ns() // 1_000_000
    report = {"started_utc": utc(started), "status": "running", "database": str(args.database),
              "symbol": "BTCUSDT", "orders_sent": 0, "credentials_used": False,
              "limitations": LIMITATIONS, "history": {}}
    try:
        store.check_budget()
        client = PublicClient(store)
        clock, received, _ = client.get(FUTURES, "/fapi/v1/time")
        now_ms = int(clock["serverTime"])
        report["exchange_time_utc"] = utc(now_ms)
        report["local_receive_minus_server_ms"] = received - now_ms
        report["clock_warning"] = abs(received - now_ms) > 1000
        report["clock_note"] = "Cross-clock difference includes clock offset and network time; never treat it as one-way latency. RTT uses a monotonic clock."
        for family in SPECS:
            try:
                report["history"][family] = collect_history(store, client, family, now_ms, args.days)
            except Exception as exc:
                report["history"][family] = {"status": "failed", "error": str(exc)}
            print(json.dumps({"family": family, **report["history"][family]}), flush=True)
        report["snapshots"] = collect_snapshots(store, client)
        report["status"] = "completed" if all(
            item["status"] == "completed" for item in [*report["history"].values(), *report["snapshots"].values()]
        ) else "partial"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
    finally:
        report["stored"] = store.summary()
        report["sqlite_check"] = store.db.execute("PRAGMA quick_check").fetchone()[0]
        report["requests"] = dict(zip(("count", "failed", "average_rtt_ms", "max_rtt_ms"), store.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(error IS NOT NULL),0), AVG(elapsed_ms), MAX(elapsed_ms) FROM requests"
        ).fetchone()))
        report["finished_utc"] = utc(time.time_ns() // 1_000_000)
        store.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(args.report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--database", type=Path, default=Path("supplementary_market_store/market_context.sqlite3"))
    parser.add_argument("--report", type=Path, default=Path("runtime_reports/supplementary_market_latest.json"))
    parser.add_argument("--interval-seconds", type=int, default=0, help="0: one bounded collection; >=300: repeat")
    parser.add_argument("--max-bytes", type=int, default=1024 ** 3)
    args = parser.parse_args()
    if not 1 <= args.days <= 365 or args.max_bytes <= 0 or (args.interval_seconds != 0 and args.interval_seconds < 300):
        parser.error("days=1..365, positive max-bytes, interval=0 or >=300 required")
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopped:
        report = collect_cycle(args)
        print(json.dumps({"status": report["status"], "report": str(args.report)}), flush=True)
        if not args.interval_seconds:
            return 0 if report["status"] == "completed" else 1
        if report["status"] == "failed":
            return 1
        deadline = time.monotonic() + args.interval_seconds
        while not stopped and time.monotonic() < deadline:
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
