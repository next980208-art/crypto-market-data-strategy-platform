import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load_module("aggtrade_collector", "10_aggtrade_collector.py")
daily_runner = load_module("daily_collection_runner_for_test", "daily_collection_runner.py")


class FakeExchange:
    rateLimit = 0

    def __init__(self, trades):
        self.trades = sorted(trades, key=lambda trade: int(trade["id"]))

    def fetch_trades(self, _symbol, since=None, limit=1000, params=None):
        params = params or {}
        if "fromId" in params:
            from_id = int(params["fromId"])
            matching = [trade for trade in self.trades if int(trade["id"]) >= from_id]
        else:
            matching = [trade for trade in self.trades if int(trade["timestamp"]) >= int(since)]
        return matching[:limit]


def make_trades(counts_by_minute, *, missing_ids=None):
    missing_ids = set(missing_ids or [])
    trades = []
    trade_id = 0
    for minute, count in enumerate(counts_by_minute):
        for index in range(count):
            while trade_id in missing_ids:
                trade_id += 1
            timestamp = minute * collector.MINUTE_MS + min(index * 20, collector.MINUTE_MS - 1)
            trades.append(
                {
                    "id": str(trade_id),
                    "timestamp": timestamp,
                    "price": 100.0 + minute,
                    "amount": 0.01,
                    "side": "buy" if index % 2 == 0 else "sell",
                    "info": {},
                }
            )
            trade_id += 1
    return trades


class WindowedAggTradeCollectionTests(unittest.TestCase):
    def test_windowing_succeeds_when_whole_range_page_cap_would_fail(self):
        trades = make_trades([1500, 1500, 1500, 1500, 10])
        exchange = FakeExchange(trades)
        end_ms = 4 * collector.MINUTE_MS

        with self.assertRaises(collector.PageLimitExceeded):
            collector.collect_trade_window(exchange, "BTC/USDT:USDT", 0, end_ms, max_pages=2)

        aggregates, windows, pages, total_trades = collector.collect_trades(
            exchange,
            "BTC/USDT:USDT",
            0,
            end_ms,
            max_pages=2,
            window_minutes=1,
            max_total_pages=10,
        )

        self.assertEqual(4, len(windows))
        self.assertEqual(8, pages)
        self.assertEqual(6000, total_trades)
        self.assertEqual(1500, aggregates[0]["trade_count"])
        self.assertEqual(1500, aggregates[3 * collector.MINUTE_MS]["trade_count"])

        context = collector.build_context(aggregates, "BTC/USDT", 0, end_ms)
        validation = collector.validate_context(context, 0, end_ms)
        self.assertEqual(4, validation["expected_minute_rows"])
        self.assertEqual(0, validation["empty_trade_minutes"])

    def test_zero_trade_minute_is_preserved_as_a_complete_row(self):
        trades = make_trades([10, 0, 10, 5])
        exchange = FakeExchange(trades)
        end_ms = 3 * collector.MINUTE_MS

        aggregates, windows, _pages, total_trades = collector.collect_trades(
            exchange,
            "BTC/USDT:USDT",
            0,
            end_ms,
            max_pages=2,
            window_minutes=1,
            max_total_pages=10,
        )
        collector.validate_trade_id_windows(windows)
        self.assertEqual(20, total_trades)

        context = collector.build_context(aggregates, "BTC/USDT", 0, end_ms)
        validation = collector.validate_context(context, 0, end_ms)
        self.assertEqual([10, 0, 10], context["trade_count"].tolist())
        self.assertEqual(1, validation["empty_trade_minutes"])
        self.assertEqual(0, validation["missing_minute_rows"])

    def test_trade_id_gap_fails_before_context_can_be_saved(self):
        trades = make_trades([5, 5], missing_ids={3})
        exchange = FakeExchange(trades)

        with self.assertRaises(RuntimeError) as raised:
            collector.collect_trade_window(
                exchange,
                "BTC/USDT:USDT",
                0,
                collector.MINUTE_MS,
                max_pages=2,
            )
        self.assertIn("first=0", str(raised.exception))
        self.assertIn("last=5", str(raised.exception))

    def test_total_page_cap_stops_incomplete_day(self):
        trades = make_trades([1500, 1500, 1500, 1500, 10])
        exchange = FakeExchange(trades)

        with self.assertRaises(collector.PageLimitExceeded):
            collector.collect_trades(
                exchange,
                "BTC/USDT:USDT",
                0,
                4 * collector.MINUTE_MS,
                max_pages=2,
                window_minutes=1,
                max_total_pages=5,
            )

    def test_atomic_output_marker_and_daily_marker_validation(self):
        trades = make_trades([10, 10, 5])
        exchange = FakeExchange(trades)
        end_ms = 2 * collector.MINUTE_MS
        aggregates, windows, pages, total_trades = collector.collect_trades(
            exchange,
            "BTC/USDT:USDT",
            0,
            end_ms,
            max_pages=2,
            window_minutes=1,
            max_total_pages=10,
        )
        context = collector.build_context(aggregates, "BTC/USDT", 0, end_ms)
        validation = collector.validate_context(context, 0, end_ms)

        with tempfile.TemporaryDirectory() as temporary_directory:
            args = SimpleNamespace(
                symbol="BTC/USDT",
                context_folder=temporary_directory,
                window_minutes=1,
                max_pages=2,
                max_total_pages=10,
            )
            start_dt = collector.datetime.fromtimestamp(0, tz=collector.timezone.utc)
            end_dt = collector.datetime.fromtimestamp(end_ms / 1000, tz=collector.timezone.utc)
            output_path, marker_path = collector.write_context(
                context,
                args,
                start_dt,
                end_dt,
                windows,
                pages,
                total_trades,
                validation,
            )

            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertTrue(output_path.is_file())
            self.assertEqual(collector.TRADE_CONTEXT_SCHEMA_VERSION, marker["context_schema_version"])
            self.assertEqual(2, marker["expected_minute_rows"])
            self.assertEqual("continuous", marker["trade_id_sequence_status"])
            self.assertTrue(daily_runner.success_marker_is_valid("aggtrade", marker_path))

            marker["missing_minute_rows"] = 1
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            self.assertFalse(daily_runner.success_marker_is_valid("aggtrade", marker_path))


if __name__ == "__main__":
    unittest.main()
