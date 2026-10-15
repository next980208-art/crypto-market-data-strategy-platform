"""Canonical market identifiers and timestamp metadata shared by every pipeline."""

from __future__ import annotations

import re
from typing import Any, Mapping


METADATA_SCHEMA_VERSION = "market_metadata_v1"
EVENT_ID_SCHEMA_VERSION = "market_event_id_v1"
TIMESTAMP_UNIT = "milliseconds"
MARKET_SPOT = "spot"
MARKET_USDM = "usdm"

_MARKET_ALIASES = {
    "spot": MARKET_SPOT,
    "binance_spot": MARKET_SPOT,
    "usdm": MARKET_USDM,
    "usdt-m": MARKET_USDM,
    "usdt_m": MARKET_USDM,
    "binance_usdm": MARKET_USDM,
    "futures": MARKET_USDM,
}
_QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "FDUSD", "USD", "BTC", "ETH")


def normalize_source(value: str) -> str:
    source = str(value).strip().lower().replace("_", "-")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", source):
        raise ValueError(f"Unsupported event source identifier: {value!r}")
    return source


def normalize_market(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _MARKET_ALIASES[key]
    except KeyError as error:
        raise ValueError(f"Unsupported market identifier: {value!r}") from error


def normalize_symbol(value: str) -> str:
    text = str(value).strip().upper()
    if not text:
        raise ValueError("Symbol must not be empty.")
    contract = text.split(":", 1)[0]
    canonical = re.sub(r"[\s/_-]", "", contract)
    if not re.fullmatch(r"[A-Z0-9]+", canonical):
        raise ValueError(f"Unsupported symbol identifier: {value!r}")
    return canonical


def split_symbol(value: str) -> tuple[str, str]:
    canonical = normalize_symbol(value)
    for quote in _QUOTE_ASSETS:
        if canonical.endswith(quote) and len(canonical) > len(quote):
            return canonical[: -len(quote)], quote
    raise ValueError(f"Cannot identify quote asset from symbol: {value!r}")


def to_ccxt_symbol(value: str, market: str) -> str:
    base, quote = split_symbol(value)
    normalized_market = normalize_market(market)
    unified = f"{base}/{quote}"
    return f"{unified}:{quote}" if normalized_market == MARKET_USDM else unified


def canonical_event_metadata(
    *,
    symbol: str,
    market: str,
    timeframe: str,
    event_time_ms: int,
) -> dict[str, Any]:
    normalized_timeframe = str(timeframe).strip().lower()
    if not re.fullmatch(r"[1-9]\d*[mhd]", normalized_timeframe):
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    timestamp = int(event_time_ms)
    if timestamp < 0:
        raise ValueError("event_time_ms must not be negative.")
    return {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "symbol": normalize_symbol(symbol),
        "market": normalize_market(market),
        "timeframe": normalized_timeframe,
        "event_time_ms": timestamp,
        "timestamp_unit": TIMESTAMP_UNIT,
    }


def canonical_candle_event_id(
    *,
    source: str,
    symbol: str,
    market: str,
    timeframe: str,
    event_time_ms: int,
) -> str:
    """Build a stable candle ID that does not change between pipeline runs."""
    metadata = canonical_event_metadata(
        symbol=symbol,
        market=market,
        timeframe=timeframe,
        event_time_ms=event_time_ms,
    )
    return ":".join(
        [
            normalize_source(source),
            metadata["market"],
            metadata["symbol"],
            "candle",
            metadata["timeframe"],
            str(metadata["event_time_ms"]),
        ]
    )


def normalize_message_metadata(
    message: Mapping[str, Any],
    *,
    default_market: str = MARKET_USDM,
    default_timeframe: str = "1m",
) -> dict[str, Any]:
    version = message.get("metadata_schema_version")
    if version not in {None, METADATA_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported metadata schema version: {version!r}")
    timestamp = message.get("event_time_ms", message.get("timestamp"))
    if timestamp is None:
        raise ValueError("Message has no event_time_ms or compatible timestamp field.")
    normalized = dict(message)
    normalized.update(
        canonical_event_metadata(
            symbol=str(message.get("symbol", "")),
            market=str(message.get("market", default_market)),
            timeframe=str(message.get("timeframe", default_timeframe)),
            event_time_ms=int(timestamp),
        )
    )
    return normalized
