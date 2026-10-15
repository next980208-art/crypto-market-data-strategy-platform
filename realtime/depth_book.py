"""Binance USDT-M local order-book reconstruction utilities."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEPTH_PAYLOAD_SCHEMA_VERSION = "usdm_depth_book_v1"
VALID_SNAPSHOT_LIMITS = {5, 10, 20, 50, 100, 500, 1000}


class DepthBookError(RuntimeError):
    """Base class for snapshot and sequence failures."""


class DepthSnapshotError(DepthBookError):
    """Raised when a REST snapshot is missing required data."""


class DepthSequenceGap(DepthBookError):
    """Raised when a websocket update cannot follow the local book."""


def _update_id(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise DepthBookError(f"Invalid depth {field}: {value!r}") from error
    if parsed < 0:
        raise DepthBookError(f"Depth {field} must not be negative: {parsed}")
    return parsed


def _level(value: Any) -> tuple[Decimal, Decimal]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise DepthBookError(f"Invalid depth level: {value!r}")
    try:
        price = Decimal(str(value[0]))
        quantity = Decimal(str(value[1]))
    except (InvalidOperation, ValueError) as error:
        raise DepthBookError(f"Invalid depth level: {value!r}") from error
    if price <= 0 or quantity < 0:
        raise DepthBookError(f"Invalid depth price or quantity: {value!r}")
    return price, quantity


def fetch_depth_snapshot(
    symbol: str,
    *,
    limit: int = 1000,
    timeout_seconds: float = 10.0,
    rest_base_url: str = "https://fapi.binance.com",
) -> dict[str, Any]:
    """Fetch a public USD-M order-book snapshot without an API key."""
    if limit not in VALID_SNAPSHOT_LIMITS:
        raise ValueError(f"Snapshot limit must be one of {sorted(VALID_SNAPSHOT_LIMITS)}.")
    if timeout_seconds <= 0:
        raise ValueError("Snapshot timeout must be positive.")
    query = urlencode({"symbol": str(symbol).upper(), "limit": limit})
    url = f"{rest_base_url.rstrip('/')}/fapi/v1/depth?{query}"
    request = Request(url, headers={"User-Agent": "coin-project-depth-sync/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DepthSnapshotError(f"Could not fetch depth snapshot: {error}") from error
    if not isinstance(payload, dict):
        raise DepthSnapshotError("Depth snapshot response must be a JSON object.")
    return payload


class LocalDepthBook:
    """Maintain one synchronized USD-M order book from a snapshot and deltas."""

    def __init__(self, symbol: str, *, output_levels: int = 5):
        if output_levels <= 0:
            raise ValueError("output_levels must be positive.")
        self.symbol = str(symbol).upper()
        self.output_levels = int(output_levels)
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.snapshot_update_id: int | None = None
        self.last_update_id: int | None = None
        self.synced = False
        self.stats: Counter[str] = Counter()

    def reset(self, reason: str = "manual") -> None:
        self.bids.clear()
        self.asks.clear()
        self.snapshot_update_id = None
        self.last_update_id = None
        self.synced = False
        self.stats["resets"] += 1
        self.stats[f"reset_{reason}"] += 1

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not isinstance(snapshot, dict):
            raise DepthSnapshotError("Depth snapshot must be a JSON object.")
        snapshot_id = _update_id(snapshot.get("lastUpdateId"), "lastUpdateId")
        bids = self._levels_from_snapshot(snapshot.get("bids"), "bids")
        asks = self._levels_from_snapshot(snapshot.get("asks"), "asks")
        if not bids or not asks:
            raise DepthSnapshotError("Depth snapshot must contain bids and asks.")
        self.bids = bids
        self.asks = asks
        self.snapshot_update_id = snapshot_id
        self.last_update_id = snapshot_id
        self.synced = False
        self._validate_not_crossed()
        self.stats["snapshots_loaded"] += 1

    @staticmethod
    def _levels_from_snapshot(values: Any, name: str) -> dict[Decimal, Decimal]:
        if not isinstance(values, list):
            raise DepthSnapshotError(f"Depth snapshot {name} must be a list.")
        levels: dict[Decimal, Decimal] = {}
        for value in values:
            price, quantity = _level(value)
            if quantity:
                levels[price] = quantity
        return levels

    @staticmethod
    def _apply_levels(book: dict[Decimal, Decimal], values: Any) -> None:
        if not isinstance(values, list):
            raise DepthBookError("Websocket depth levels must be a list.")
        for value in values:
            price, quantity = _level(value)
            if quantity == 0:
                book.pop(price, None)
            else:
                book[price] = quantity

    def apply_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Apply one depthUpdate and return an enriched synchronized payload."""
        if self.snapshot_update_id is None or self.last_update_id is None:
            raise DepthSnapshotError("Load a snapshot before applying depth events.")
        if not isinstance(event, dict):
            raise DepthBookError("Depth event must be a JSON object.")

        first_update_id = _update_id(event.get("U"), "U")
        final_update_id = _update_id(event.get("u"), "u")
        previous_final_update_id = _update_id(event.get("pu"), "pu")
        if first_update_id > final_update_id:
            raise DepthBookError("Depth event U must not be greater than u.")

        if final_update_id < self.snapshot_update_id:
            self.stats["stale_before_bridge"] += 1
            return None

        if not self.synced:
            if not first_update_id <= self.snapshot_update_id <= final_update_id:
                self.stats["initial_bridge_gaps"] += 1
                raise DepthSequenceGap(
                    "First websocket event does not bridge the REST snapshot: "
                    f"U={first_update_id}, snapshot={self.snapshot_update_id}, u={final_update_id}"
                )
        else:
            if final_update_id <= self.last_update_id:
                self.stats["duplicate_or_stale_updates"] += 1
                return None
            if previous_final_update_id != self.last_update_id:
                self.stats["sequence_gaps"] += 1
                raise DepthSequenceGap(
                    "Depth pu does not match the previous u: "
                    f"pu={previous_final_update_id}, previous_u={self.last_update_id}"
                )

        delta_bids = event.get("b", [])
        delta_asks = event.get("a", [])
        self._apply_levels(self.bids, delta_bids)
        self._apply_levels(self.asks, delta_asks)
        self.last_update_id = final_update_id
        self.synced = True
        self._validate_not_crossed()
        self.stats["events_applied"] += 1
        return self.payload(
            first_update_id=first_update_id,
            final_update_id=final_update_id,
            previous_final_update_id=previous_final_update_id,
            delta_bids=delta_bids,
            delta_asks=delta_asks,
        )

    def _validate_not_crossed(self) -> None:
        if not self.bids or not self.asks:
            raise DepthBookError("Local order book lost all bids or asks.")
        if max(self.bids) >= min(self.asks):
            self.stats["crossed_books"] += 1
            raise DepthSequenceGap("Local order book is crossed after applying depth data.")

    @staticmethod
    def _json_levels(levels: list[tuple[Decimal, Decimal]]) -> list[list[str]]:
        return [[format(price, "f"), format(quantity, "f")] for price, quantity in levels]

    def top_levels(self) -> tuple[list[list[str]], list[list[str]]]:
        bids = sorted(self.bids.items(), key=lambda item: item[0], reverse=True)[: self.output_levels]
        asks = sorted(self.asks.items(), key=lambda item: item[0])[: self.output_levels]
        return self._json_levels(bids), self._json_levels(asks)

    def payload(
        self,
        *,
        first_update_id: int,
        final_update_id: int,
        previous_final_update_id: int,
        delta_bids: list[Any],
        delta_asks: list[Any],
    ) -> dict[str, Any]:
        top_bids, top_asks = self.top_levels()
        bid_volume = sum(Decimal(level[1]) for level in top_bids)
        ask_volume = sum(Decimal(level[1]) for level in top_asks)
        total = bid_volume + ask_volume
        return {
            "depth_schema_version": DEPTH_PAYLOAD_SCHEMA_VERSION,
            "sequence_status": "synchronized",
            "snapshot_update_id": self.snapshot_update_id,
            "first_update_id": first_update_id,
            "final_update_id": final_update_id,
            "previous_final_update_id": previous_final_update_id,
            "delta_bids": delta_bids,
            "delta_asks": delta_asks,
            "top_bids": top_bids,
            "top_asks": top_asks,
            "best_bid": top_bids[0][0],
            "best_ask": top_asks[0][0],
            "book_imbalance_top5": float((bid_volume - ask_volume) / total) if total else 0.0,
            "local_bid_levels": len(self.bids),
            "local_ask_levels": len(self.asks),
        }

    def health(self) -> dict[str, Any]:
        top_bids, top_asks = self.top_levels() if self.bids and self.asks else ([], [])
        return {
            "symbol": self.symbol,
            "synced": self.synced,
            "snapshot_update_id": self.snapshot_update_id,
            "last_update_id": self.last_update_id,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "best_bid": top_bids[0][0] if top_bids else None,
            "best_ask": top_asks[0][0] if top_asks else None,
            "stats": dict(self.stats),
        }
