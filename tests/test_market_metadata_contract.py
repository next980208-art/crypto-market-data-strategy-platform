import unittest

import pandas as pd

from flink_batch_submitter import apply_canonical_feature_metadata
from market_metadata import (
    METADATA_SCHEMA_VERSION,
    TIMESTAMP_UNIT,
    canonical_event_metadata,
    normalize_market,
    normalize_message_metadata,
    normalize_symbol,
    to_ccxt_symbol,
)
from realtime.contracts import build_raw_event
from realtime.inference_service import blocked_signal
from scripts.verify_market_metadata_contract import validate_contract_record
from scripts.verify_batch_metadata_contract import evaluate_batch_contract


class MarketMetadataContractTests(unittest.TestCase):
    def test_symbol_aliases_share_one_internal_identifier(self):
        aliases = ["BTCUSDT", "btc/usdt", "BTC/USDT:USDT"]
        self.assertEqual({"BTCUSDT"}, {normalize_symbol(value) for value in aliases})
        self.assertEqual("BTC/USDT:USDT", to_ccxt_symbol("BTCUSDT", "USDT-M"))

    def test_market_aliases_share_one_internal_identifier(self):
        aliases = ["usdm", "USDT-M", "binance_usdm", "futures"]
        self.assertEqual({"usdm"}, {normalize_market(value) for value in aliases})

    def test_legacy_message_is_normalized_without_losing_payload(self):
        normalized = normalize_message_metadata(
            {
                "symbol": "BTC/USDT:USDT",
                "market": "USDT-M",
                "timestamp": 1_780_000_000_000,
                "payload": {"price": "100"},
            }
        )
        self.assertEqual("BTCUSDT", normalized["symbol"])
        self.assertEqual("usdm", normalized["market"])
        self.assertEqual(1_780_000_000_000, normalized["event_time_ms"])
        self.assertEqual(TIMESTAMP_UNIT, normalized["timestamp_unit"])
        self.assertEqual({"price": "100"}, normalized["payload"])

    def test_raw_event_uses_canonical_contract(self):
        event = build_raw_event(
            event_id="test-1",
            event_type="trade",
            event_time_ms=1_780_000_000_000,
            symbol="BTC/USDT",
            payload={"price": 100.0},
            session_id="session-1",
        )
        self.assertEqual("BTCUSDT", event["symbol"])
        self.assertEqual("usdm", event["market"])
        self.assertEqual(METADATA_SCHEMA_VERSION, event["metadata_schema_version"])
        self.assertEqual(TIMESTAMP_UNIT, event["timestamp_unit"])

    def test_batch_frame_keeps_timestamp_alias_and_adds_event_time(self):
        frame = pd.DataFrame(
            {
                "timestamp": [1_780_000_000_000],
                "symbol": ["BTC/USDT"],
                "market": ["USDT-M"],
            }
        )
        normalized = apply_canonical_feature_metadata(frame)
        self.assertEqual(1_780_000_000_000, int(normalized.loc[0, "timestamp"]))
        self.assertEqual(1_780_000_000_000, int(normalized.loc[0, "event_time_ms"]))
        self.assertEqual("BTCUSDT", normalized.loc[0, "symbol"])
        self.assertEqual("usdm", normalized.loc[0, "market"])

    def test_invalid_contract_values_are_rejected(self):
        with self.assertRaises(ValueError):
            canonical_event_metadata(
                symbol="BTCUSDT",
                market="unknown",
                timeframe="1m",
                event_time_ms=1,
            )

    def test_signal_inherits_the_same_metadata_contract(self):
        signal = blocked_signal(
            {
                "feature_id": "BTCUSDT:1m:1780000000000",
                "event_time_ms": 1_780_000_000_000,
                "symbol": "BTC/USDT:USDT",
                "market": "USDT-M",
                "timeframe": "1m",
                "close": 100.0,
            },
            "test",
        )
        self.assertEqual("BTCUSDT", signal["symbol"])
        self.assertEqual("usdm", signal["market"])
        self.assertEqual(METADATA_SCHEMA_VERSION, signal["metadata_schema_version"])

    def test_live_contract_validator_rejects_display_aliases(self):
        record = canonical_event_metadata(
            symbol="BTCUSDT",
            market="usdm",
            timeframe="1m",
            event_time_ms=1_780_000_000_000,
        )
        self.assertEqual([], validate_contract_record(record, "BTCUSDT"))
        record["market"] = "USDT-M"
        self.assertIn("market must be canonical usdm", validate_contract_record(record, "BTCUSDT"))

    def test_finalized_batch_contract_requires_matching_timestamp_alias(self):
        frame = pd.DataFrame(
            {
                "timestamp": [1_780_000_000_000],
                "event_time_ms": [1_780_000_000_000],
                "timestamp_unit": [TIMESTAMP_UNIT],
                "metadata_schema_version": [METADATA_SCHEMA_VERSION],
                "symbol": ["BTCUSDT"],
                "market": ["usdm"],
                "timeframe": ["1m"],
                "feature_schema_version": ["ohlcv_basic_v2_boundary4"],
            }
        )
        marker = {
            "rows": 1,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "timestamp_field": "event_time_ms",
            "timestamp_unit": TIMESTAMP_UNIT,
        }
        self.assertTrue(evaluate_batch_contract(frame, marker, "BTCUSDT")["healthy"])
        frame.loc[0, "event_time_ms"] += 1
        self.assertFalse(evaluate_batch_contract(frame, marker, "BTCUSDT")["healthy"])
        with self.assertRaises(ValueError):
            normalize_message_metadata(
                {
                    "metadata_schema_version": "future_v99",
                    "symbol": "BTCUSDT",
                    "market": "usdm",
                    "event_time_ms": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
