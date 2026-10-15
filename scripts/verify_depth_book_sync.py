"""Verify Binance public depth snapshot+delta reconstruction without Kafka."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.contracts import safe_symbol
from realtime.depth_book import DepthSequenceGap, LocalDepthBook, fetch_depth_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify public USD-M local order-book synchronization.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--duration-seconds", type=int, default=20)
    parser.add_argument("--required-events", type=int, default=20)
    parser.add_argument("--snapshot-limit", type=int, default=1000)
    parser.add_argument("--output-levels", type=int, default=5)
    parser.add_argument("--rest-base-url", default="https://fapi.binance.com")
    parser.add_argument("--websocket-base-url", default="wss://fstream.binance.com/public/stream")
    parser.add_argument("--report-path", default="runtime_reports/depth_book_sync_latest.json")
    return parser.parse_args()


async def verify(args: argparse.Namespace) -> dict[str, object]:
    symbol = safe_symbol(args.symbol)
    stream = f"{symbol.lower()}@depth@100ms"
    url = f"{args.websocket_base_url.rstrip('/')}?streams={stream}"
    book = LocalDepthBook(symbol, output_levels=args.output_levels)
    started = monotonic()
    synchronized_events = 0
    sequence_resets = 0
    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as socket:
        while monotonic() - started < args.duration_seconds and synchronized_events < args.required_events:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
            data = message.get("data", message)
            if data.get("e") != "depthUpdate":
                continue
            if book.snapshot_update_id is None:
                snapshot = await asyncio.to_thread(
                    fetch_depth_snapshot,
                    symbol,
                    limit=args.snapshot_limit,
                    rest_base_url=args.rest_base_url,
                )
                book.load_snapshot(snapshot)
            try:
                payload = book.apply_event(data)
            except DepthSequenceGap:
                sequence_resets += 1
                book.reset("sequence_gap")
                continue
            if payload is not None:
                synchronized_events += 1

    health = book.health()
    healthy = bool(
        health["synced"]
        and synchronized_events >= args.required_events
        and health["best_bid"] is not None
        and health["best_ask"] is not None
        and float(health["best_bid"]) < float(health["best_ask"])
    )
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "duration_seconds": round(monotonic() - started, 3),
        "required_events": args.required_events,
        "synchronized_events": synchronized_events,
        "sequence_resets": sequence_resets,
        "healthy": healthy,
        "book": health,
    }


def main() -> None:
    args = parse_args()
    if args.duration_seconds <= 0 or args.required_events <= 0:
        raise ValueError("duration-seconds and required-events must be positive.")
    report = asyncio.run(verify(args))
    path = Path(args.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(path), **report}, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise RuntimeError("Depth book synchronization did not reach the required healthy state.")


if __name__ == "__main__":
    main()
