import tempfile
import unittest
from pathlib import Path

from scripts.collect_supplementary_market_data import DAY, MINUTE, Store, collect_history, depth_costs
from scripts.verify_supplementary_collection import verify


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get(self, base, path, params):
        self.calls.append(params)
        rows = [row for row in self.rows if params["startTime"] <= row[0] <= params["endTime"]]
        return rows[:params["limit"]], 3 * DAY, base + path


class SupplementaryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.root.name) / "store.sqlite3")

    def tearDown(self):
        self.store.close()
        self.root.cleanup()

    def test_replay_keeps_original_observation_time(self):
        self.store.save("oi", [(100, 100, {"value": 1})], 200, "public")
        self.store.save("oi", [(100, 100, {"value": 2})], 300, "public")
        self.assertEqual(1, self.store.summary()["oi"]["rows"])
        self.assertEqual({"value": 1}, self.store.summary()["oi"]["latest"])

    def test_storage_budget_prevents_writes(self):
        self.store.max_bytes = 1
        with self.assertRaisesRegex(RuntimeError, "storage budget"):
            self.store.save("oi", [(100, 100, {})], 200, "public")
        self.assertEqual({}, self.store.summary())

    def test_pagination_complete_and_restart_no_duplicates(self):
        rows = [[stamp, "1", "1", "1", "1", "1", stamp + MINUTE - 1]
                for stamp in range(DAY, 2 * DAY, MINUTE)]
        client = FakeClient(rows)
        report = collect_history(self.store, client, "spot_1m", 2 * DAY, 1)
        self.assertEqual("completed", report["status"])
        self.assertEqual(1440, report["rows_in_window"])
        self.assertGreater(len(client.calls), 1)
        report = collect_history(self.store, client, "spot_1m", 2 * DAY, 1)
        self.assertEqual("completed", report["status"])
        self.assertEqual(1440, self.store.summary()["spot_1m"]["rows"])

    def test_missing_minutes_do_not_advance_coverage(self):
        client = FakeClient([[DAY, 1, 1, 1, 1, 1, DAY + MINUTE - 1]])
        report = collect_history(self.store, client, "spot_1m", 2 * DAY, 1)
        self.assertEqual("partial", report["status"])
        self.assertEqual(1439, report["missing_rows"])
        self.assertEqual([], self.store.db.execute("SELECT * FROM coverage").fetchall())

    def test_incomplete_candle_not_saved(self):
        client = FakeClient([[DAY, 1, 1, 1, 1, 1, 3 * DAY]])
        collect_history(self.store, client, "spot_1m", 2 * DAY, 1)
        self.assertEqual({}, self.store.summary())

    def test_larger_lookback_fetches_earlier_history(self):
        rows = [[stamp, 1, 1, 1, 1, 1, stamp + MINUTE - 1]
                for stamp in range(DAY, 3 * DAY, MINUTE)]
        client = FakeClient(rows)
        collect_history(self.store, client, "spot_1m", 3 * DAY, 1)
        result = collect_history(self.store, client, "spot_1m", 3 * DAY, 2)
        self.assertEqual("completed", result["status"])
        self.assertEqual(2880, result["rows_in_window"])

    def test_readback_rejects_bad_ohlc_even_with_matching_counts(self):
        rows = [[stamp, 1, 1, 1, 1, 1, stamp + MINUTE - 1]
                for stamp in range(DAY, 2 * DAY, MINUTE)]
        rows[0][2] = 0.5
        result = collect_history(self.store, FakeClient(rows), "spot_1m", 2 * DAY, 1)
        report = verify(self.store.path, {"history": {"spot_1m": result}})
        self.assertFalse(report["passed"])
        self.assertEqual(1, report["history"]["spot_1m"]["invalid_values"])

    def test_oi_stays_five_minute_and_29_day_limit(self):
        class OIClient:
            def get(self, base, path, params):
                start = (params["startTime"] + 5 * MINUTE - 1) // (5 * MINUTE) * (5 * MINUTE)
                rows = [{"timestamp": stamp, "sumOpenInterest": "2"}
                        for stamp in range(start, params["endTime"] + 1, 5 * MINUTE)]
                return rows[-params["limit"]:], 31 * DAY, base + path
        report = collect_history(self.store, OIClient(), "open_interest_5m", 30 * DAY, 90)
        self.assertEqual(29 * 288, report["rows_in_window"])
        self.assertEqual(5 * MINUTE, report["interval_ms"])
        self.assertEqual("completed", report["status"])

    def test_visible_depth_insufficient_is_not_a_fill(self):
        result = depth_costs({"bids": [[99, 0.005]], "asks": [[101, 0.005]]})
        self.assertEqual(200, result["spread_bps"])
        self.assertFalse(result["actual_execution"])
        self.assertIsNone(result["estimates"][1]["estimated_vwap"])
        self.assertAlmostEqual(100, result["estimates"][0]["cost_vs_mid_bps"])

    def test_crossed_book_rejected(self):
        with self.assertRaisesRegex(ValueError, "Crossed"):
            depth_costs({"bids": [[102, 1]], "asks": [[101, 1]]})

    def test_cost_uses_multiple_levels(self):
        result = depth_costs({"bids": [[99, 1]], "asks": [[101, 0.005], [103, 1]]})
        self.assertAlmostEqual(102, result["estimates"][1]["estimated_vwap"])


if __name__ == "__main__":
    unittest.main()
