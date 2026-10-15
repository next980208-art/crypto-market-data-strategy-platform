"""Shared message contracts for the paper-only realtime pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from market_metadata import (
    MARKET_USDM,
    canonical_event_metadata,
    normalize_symbol,
)


RAW_EVENT_SCHEMA_VERSION = "market_raw_v1"
FEATURE_SCHEMA_VERSION = "realtime_feature_v1"
SIGNAL_SCHEMA_VERSION = "trading_signal_v1"
PAPER_EVENT_SCHEMA_VERSION = "paper_event_v1"

TOPIC_TRADES = "market.trade.v1"
TOPIC_BOOK_TICKER = "market.book-ticker.v1"
TOPIC_DEPTH = "market.depth.delta.v1"
TOPIC_FEATURES = "market.features.v1"
TOPIC_SIGNALS = "signal.trading.v1"
TOPIC_PAPER_EXECUTIONS = "paper.execution.v1"
TOPIC_RISK_AUDIT = "risk.audit.v1"

RAW_TOPICS = (TOPIC_TRADES, TOPIC_BOOK_TICKER, TOPIC_DEPTH)
ALL_TOPICS = RAW_TOPICS + (
    TOPIC_FEATURES,
    TOPIC_SIGNALS,
    TOPIC_PAPER_EXECUTIONS,
    TOPIC_RISK_AUDIT,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_symbol(value: str) -> str:
    return normalize_symbol(value)


def json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_json(value: bytes | str) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("Kafka message must be a JSON object.")
    return decoded


def build_raw_event(
    *,
    event_id: str,
    event_type: str,
    event_time_ms: int,
    symbol: str,
    payload: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    metadata = canonical_event_metadata(
        symbol=symbol,
        market=MARKET_USDM,
        timeframe="1m",
        event_time_ms=event_time_ms,
    )
    return {
        "schema_version": RAW_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "event_type": event_type,
        "received_at_utc": utc_now_iso(),
        "source": "binance_usdm_websocket",
        "session_id": session_id,
        "payload": payload,
        **metadata,
    }
