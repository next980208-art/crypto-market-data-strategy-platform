"""Bounded direct WebSocket capture of public BTCUSDT messages, without Kafka/orders."""

import argparse
import asyncio
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import websockets

from scripts.run_bounded_profitability_search import utc_now, write_json

URL = "wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/btcusdt@bookTicker/btcusdt@depth5@100ms"


def encode_record(message, received_utc):
    if not isinstance(message, str):
        raise ValueError("Expected text WebSocket message")
    decoded = json.loads(message)
    event = decoded.get("data", decoded)
    kind = event.get("e", "unknown")
    encoded = (json.dumps({"received_utc": received_utc, "raw_message": message},
                          ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    return kind, encoded


async def capture(output, seconds=120, max_bytes=64 * 1024**2, max_messages=100000, trade_stream="aggTrade"):
    if not 1 <= seconds <= 120 or not 1 <= max_bytes <= 64 * 1024**2 or not 1 <= max_messages <= 100000:
        raise ValueError("Bounded duration/byte/message limits required")
    if trade_stream not in ("aggTrade", "trade"):
        raise ValueError("Unsupported public trade stream")
    url = URL.replace("@aggTrade", "@" + trade_stream)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "messages.jsonl.gz"
    started = time.monotonic()
    report = {"started_utc": utc_now(), "status": "running", "source": url,
              "duration_limit_seconds": seconds, "uncompressed_byte_limit": max_bytes,
              "message_limit": max_messages, "orders_sent": 0, "credentials_used": False,
              "kafka_used": False, "continuous_capture_guaranteed": False,
              "depth_scope": "partial top-5 stream, not full-depth delta archive",
              "complete_minute_windows_guaranteed": False}
    counts, raw_bytes = Counter(), 0
    try:
        with gzip.open(path, "xb", compresslevel=6) as destination:
            async with websockets.connect(url, open_timeout=15, close_timeout=3, max_size=1024**2,
                                          ping_interval=20, ping_timeout=20) as socket:
                while True:
                    remaining = seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        report["stop_reason"] = "duration_limit"
                        break
                    try:
                        message = await asyncio.wait_for(socket.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        report["stop_reason"] = "duration_limit"
                        break
                    kind, encoded = encode_record(message, utc_now())
                    if raw_bytes + len(encoded) > max_bytes:
                        report["stop_reason"] = "byte_limit"
                        break
                    destination.write(encoded)
                    raw_bytes += len(encoded)
                    counts[kind] += 1
                    if sum(counts.values()) >= max_messages:
                        report["stop_reason"] = "message_limit"
                        break
        report["status"] = "completed" if all(counts[k] for k in (trade_stream, "bookTicker", "depthUpdate")) else "partial"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        verified = Counter()
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    stored = json.loads(line)
                    event = json.loads(stored["raw_message"])
                    verified[event.get("data", event).get("e", "unknown")] += 1
            if verified != counts:
                report.update(status="failed", error="Readback counts differ")
        report.update(finished_utc=utc_now(), elapsed_seconds=round(time.monotonic()-started, 3),
                      counts=dict(counts), readback_counts=dict(verified), uncompressed_bytes=raw_bytes,
                      compressed_bytes=path.stat().st_size if path.exists() else 0,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
        write_json(output / "capture.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--trade-stream", choices=["aggTrade", "trade"], default="aggTrade")
    args = parser.parse_args()
    report = asyncio.run(capture(args.output, args.seconds, trade_stream=args.trade_stream))
    print(json.dumps(report))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
