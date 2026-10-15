"""Collect Binance USDT-M market events and publish them to Kafka."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import Counter
from pathlib import Path

import websockets
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError, KafkaTimeoutError, TopicAlreadyExistsError

from realtime.contracts import ALL_TOPICS, TOPIC_BOOK_TICKER, TOPIC_DEPTH, TOPIC_TRADES, build_raw_event, json_bytes, safe_symbol, utc_now_iso
from realtime.depth_book import DepthBookError, DepthSequenceGap, LocalDepthBook, fetch_depth_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Binance USDT-M websocket events to Kafka.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic-partitions", type=int, default=1)
    parser.add_argument("--duration-seconds", type=int, default=0, help="0 runs until interrupted.")
    parser.add_argument("--metrics-path", default="realtime_runtime/collector_metrics.json")
    parser.add_argument("--depth-snapshot-limit", type=int, default=1000)
    parser.add_argument("--depth-snapshot-timeout", type=float, default=10.0)
    parser.add_argument("--depth-output-levels", type=int, default=5)
    parser.add_argument("--depth-rest-base-url", default="https://fapi.binance.com")
    parser.add_argument("--websocket-base-url", default="wss://fstream.binance.com/public/stream")
    return parser.parse_args()


def ensure_topics(bootstrap_servers: str, partitions: int) -> None:
    if partitions <= 0:
        raise ValueError("--topic-partitions must be positive.")
    for attempt in range(1, 31):
        admin = None
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id="market-topic-bootstrap")
            existing = set(admin.list_topics())
            missing = [topic for topic in ALL_TOPICS if topic not in existing]
            if missing:
                try:
                    admin.create_topics(
                        [NewTopic(name=topic, num_partitions=partitions, replication_factor=1) for topic in missing],
                        validate_only=False,
                    )
                except TopicAlreadyExistsError:
                    # A concurrent restart created one or more requested topics first.
                    pass
            return
        except KafkaTimeoutError:
            if attempt == 30:
                raise
            print(f"Kafka not ready for topic setup; retrying ({attempt}/30).")
            time.sleep(2)
        finally:
            if admin is not None:
                admin.close()


def event_for_message(data: dict[str, object], symbol: str, session_id: str) -> tuple[str, dict[str, object]] | None:
    event_type = data.get("e")
    event_time_ms = int(data.get("E") or data.get("T") or 0)
    if event_time_ms <= 0:
        return None
    if event_type in {"aggTrade", "trade"}:
        trade_id = data.get("a") if event_type == "aggTrade" else data.get("t")
        event = build_raw_event(
            event_id=f"{safe_symbol(symbol)}:trade:{trade_id}",
            event_type="trade",
            event_time_ms=event_time_ms,
            symbol=symbol,
            session_id=session_id,
            payload={
                "aggregate_trade_id": trade_id,
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "buyer_is_maker": bool(data.get("m")),
            },
        )
        return TOPIC_TRADES, event
    if event_type == "bookTicker":
        update_id = data.get("u")
        event = build_raw_event(
            event_id=f"{safe_symbol(symbol)}:book:{update_id}",
            event_type="book_ticker",
            event_time_ms=event_time_ms,
            symbol=symbol,
            session_id=session_id,
            payload={
                "update_id": update_id,
                "bid_price": float(data["b"]),
                "bid_quantity": float(data["B"]),
                "ask_price": float(data["a"]),
                "ask_quantity": float(data["A"]),
            },
        )
        return TOPIC_BOOK_TICKER, event
    if event_type == "depthUpdate":
        update_id = data.get("u")
        event = build_raw_event(
            event_id=f"{safe_symbol(symbol)}:depth:{update_id}",
            event_type="depth_delta",
            event_time_ms=event_time_ms,
            symbol=symbol,
            session_id=session_id,
            payload={
                "first_update_id": data.get("U"),
                "final_update_id": update_id,
                "previous_final_update_id": data.get("pu"),
                "bids": data.get("b", []),
                "asks": data.get("a", []),
            },
        )
        return TOPIC_DEPTH, event
    return None


def depth_event(data: dict[str, object], symbol: str, session_id: str, payload: dict[str, object]) -> dict[str, object]:
    update_id = data.get("u")
    return build_raw_event(
        event_id=f"{safe_symbol(symbol)}:depth:{update_id}",
        event_type="depth_delta",
        event_time_ms=int(data.get("E") or data.get("T") or 0),
        symbol=symbol,
        session_id=session_id,
        payload=payload,
    )


def write_metrics(
    path: Path,
    session_id: str,
    counts: Counter[str],
    health_counts: Counter[str],
    depth_book: LocalDepthBook,
    started: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "captured_at_utc": utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "published_by_topic": dict(counts),
        "published_total": sum(counts.values()),
        "collector_health": dict(health_counts),
        "depth_book": depth_book.health(),
        "mode": "paper_only_market_capture",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def collect(args: argparse.Namespace) -> None:
    symbol = safe_symbol(args.symbol)
    stream_symbol = symbol.lower()
    session_id = f"{symbol}-{uuid.uuid4().hex[:12]}"
    ensure_topics(args.bootstrap_servers, args.topic_partitions)
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        acks="all",
        retries=5,
        key_serializer=lambda value: str(value).encode("utf-8"),
        value_serializer=json_bytes,
    )
    url = f"{args.websocket_base_url.rstrip('/')}?streams=" f"{stream_symbol}@trade/{stream_symbol}@bookTicker/{stream_symbol}@depth@100ms"
    started = time.monotonic()
    counts: Counter[str] = Counter()
    health_counts: Counter[str] = Counter()
    depth_book = LocalDepthBook(symbol, output_levels=args.depth_output_levels)
    reconnect_delay = 1.0
    try:
        while args.duration_seconds == 0 or time.monotonic() - started < args.duration_seconds:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as socket:
                    depth_book.reset("websocket_reconnect")
                    health_counts["websocket_connections"] += 1
                    reconnect_delay = 1.0
                    while args.duration_seconds == 0 or time.monotonic() - started < args.duration_seconds:
                        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
                        data = message.get("data", message)
                        if data.get("e") == "depthUpdate":
                            if depth_book.snapshot_update_id is None:
                                snapshot = await asyncio.to_thread(
                                    fetch_depth_snapshot,
                                    symbol,
                                    limit=args.depth_snapshot_limit,
                                    timeout_seconds=args.depth_snapshot_timeout,
                                    rest_base_url=args.depth_rest_base_url,
                                )
                                depth_book.load_snapshot(snapshot)
                                health_counts["depth_snapshots_loaded"] += 1
                            try:
                                synchronized_payload = depth_book.apply_event(data)
                            except DepthSequenceGap as error:
                                health_counts["depth_sequence_resyncs"] += 1
                                print(f"depth resync after sequence failure: {error}")
                                depth_book.reset("sequence_gap")
                                continue
                            if synchronized_payload is None:
                                health_counts["depth_events_ignored_before_bridge"] += 1
                                continue
                            parsed = TOPIC_DEPTH, depth_event(data, symbol, session_id, synchronized_payload)
                            health_counts["depth_events_synchronized"] += 1
                        else:
                            parsed = event_for_message(data, symbol, session_id)
                        if parsed is None:
                            continue
                        topic, event = parsed
                        producer.send(topic, key=symbol, value=event)
                        counts[topic] += 1
                        if sum(counts.values()) % 500 == 0:
                            producer.flush(timeout=10)
                            write_metrics(Path(args.metrics_path), session_id, counts, health_counts, depth_book, started)
            except (OSError, asyncio.TimeoutError, KafkaError, DepthBookError, websockets.WebSocketException) as error:
                print(f"collector reconnect after error: {error}")
                health_counts["collector_reconnects_after_error"] += 1
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)
    finally:
        producer.flush(timeout=20)
        producer.close(timeout=20)
        write_metrics(Path(args.metrics_path), session_id, counts, health_counts, depth_book, started)


def main() -> None:
    args = parse_args()
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
