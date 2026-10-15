import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from market_metadata import EVENT_ID_SCHEMA_VERSION, canonical_candle_event_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_PATH = PROJECT_ROOT / "examples" / "kafka_flink" / "kafka_market_event_producer.py"
SPEC = importlib.util.spec_from_file_location("kafka_market_event_producer", PRODUCER_PATH)
PRODUCER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PRODUCER)


class MarketEventIdentityTests(unittest.TestCase):
    def test_candle_id_is_canonical_across_aliases(self):
        first = canonical_candle_event_id(
            source="Binance",
            symbol="BTC/USDT:USDT",
            market="USDT-M",
            timeframe="1M",
            event_time_ms=1_788_134_400_000,
        )
        second = canonical_candle_event_id(
            source="binance",
            symbol="BTCUSDT",
            market="usdm",
            timeframe="1m",
            event_time_ms=1_788_134_400_000,
        )
        self.assertEqual(first, second)
        self.assertEqual("binance:usdm:BTCUSDT:candle:1m:1788134400000", first)

    def test_test_event_id_does_not_depend_on_run_id(self):
        event_time = datetime(2026, 8, 31, 0, 0, tzinfo=timezone.utc)
        first = PRODUCER.build_test_event(0, event_time, "run-a")
        second = PRODUCER.build_test_event(0, event_time, "run-b")
        self.assertEqual(first["event_id"], second["event_id"])
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(EVENT_ID_SCHEMA_VERSION, first["event_id_schema_version"])

    def test_local_replay_is_stable_between_runs(self):
        source_event = {
            "event_time": "2026-08-21T22:02:00Z",
            "event_id": "legacy-run-specific-id",
            "symbol": "BTCUSDT",
            "market": "USDT-M",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(json.dumps(source_event) + "\n", encoding="utf-8")
            first = PRODUCER.load_local_replay_events(path, 3, "run-a")
            second = PRODUCER.load_local_replay_events(path, 3, "run-b")

        self.assertEqual(
            [event["event_id"] for event in first],
            [event["event_id"] for event in second],
        )
        self.assertEqual(3, len({event["event_id"] for event in first}))
        self.assertEqual("run-a", first[0]["run_id"])
        self.assertEqual("run-b", second[0]["run_id"])

    def test_different_candle_dimensions_get_different_ids(self):
        base = {
            "source": "binance",
            "symbol": "BTCUSDT",
            "market": "usdm",
            "timeframe": "1m",
            "event_time_ms": 1_788_134_400_000,
        }
        ids = {
            canonical_candle_event_id(**base),
            canonical_candle_event_id(**{**base, "symbol": "ETHUSDT"}),
            canonical_candle_event_id(**{**base, "timeframe": "5m"}),
            canonical_candle_event_id(**{**base, "event_time_ms": base["event_time_ms"] + 60_000}),
        }
        self.assertEqual(4, len(ids))


if __name__ == "__main__":
    unittest.main()
