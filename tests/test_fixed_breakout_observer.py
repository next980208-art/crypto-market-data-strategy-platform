import unittest

import pandas as pd

from scripts import observe_fixed_breakout as observer


class FixedBreakoutObserverTests(unittest.TestCase):
    def test_next_boundary_is_strictly_future_and_aligned(self):
        self.assertEqual(observer.next_boundary(0), observer.STEP)
        self.assertEqual(observer.next_boundary(observer.STEP), 2*observer.STEP)
        self.assertEqual(observer.next_boundary(observer.STEP+1), 2*observer.STEP)

    def test_kline_normalization(self):
        row = [0, "100", "102", "99", "101", "5", observer.STEP-1, "0", 1, "0", "0", "0"]
        frame = observer.normalize_klines([row], "test")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].period_end_ms, observer.STEP)
        self.assertEqual(frame.iloc[0].close, 101)

    def test_invalid_kline_is_rejected(self):
        row = [1, "100", "102", "99", "101", "5", observer.STEP, "0", 1, "0", "0", "0"]
        with self.assertRaises(ValueError):
            observer.normalize_klines([row], "test")

    def test_funding_uses_completed_15m_mark_proxy(self):
        marks = pd.DataFrame({"event_ms": [0, observer.STEP], "period_end_ms": [observer.STEP, 2*observer.STEP],
                              "close": [100., 101.]})
        event_time = 2*observer.STEP
        market = observer.FifteenMinuteFundingMarket(
            [{"fundingTime": event_time, "fundingRate": "0.0001"}], marks)
        rows = market.settlements(observer.STEP, 3*observer.STEP)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["price"], 101.)
        self.assertTrue(rows[0]["proxy"])


if __name__ == "__main__":
    unittest.main()
