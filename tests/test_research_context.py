import unittest

from scripts import collect_research_context as research


class ResearchContextTests(unittest.TestCase):
    def candle(self, stamp):
        return [stamp, "100", "102", "99", "101", "10", stamp+research.STEP-1, "1000", 10, "4", "400", "0"]

    def test_valid_candles(self):
        frame = research.normalize([self.candle(0), self.candle(research.STEP)], "spot15", 0, 2*research.STEP)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0].taker_ratio, .4)

    def test_gaps_are_not_filled_and_invalid_ohlc_fails(self):
        self.assertEqual(len(research.normalize([self.candle(0)], "spot15", 0, 2*research.STEP)), 1)
        with self.assertRaises(ValueError):
            research.normalize([self.candle(0)]*2, "spot15", 0, research.STEP)
        bad = self.candle(0)
        bad[4] = "200"
        with self.assertRaises(ValueError):
            research.normalize([bad], "mark15", 0, research.STEP)

    def test_partial_close_is_flagged(self):
        row = self.candle(0)
        row[6] -= 1000
        frame = research.normalize([row], "spot15", 0, research.STEP)
        self.assertFalse(frame.iloc[0].complete_candle)

    def test_funding_is_not_assumed_regular(self):
        frame = research.normalize([{"symbol": "BTCUSDT", "fundingTime": 1, "fundingRate": "-0.0001"}], "funding", 0, 100)
        self.assertAlmostEqual(frame.iloc[0].value, -.0001)
        with self.assertRaises(ValueError):
            research.normalize([{"symbol": "ETHUSDT", "fundingTime": 1, "fundingRate": "0"}], "funding", 0, 100)


if __name__ == "__main__":
    unittest.main()
