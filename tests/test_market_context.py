import unittest

import numpy as np
import pandas as pd

from scripts import validate_market_context as research


class MarketContextTests(unittest.TestCase):
    def sources(self):
        times = np.arange(8)*research.collection.STEP
        candles = pd.DataFrame({"event_ms": times, "period_end_ms": times+research.collection.STEP,
                                "value": np.arange(8)+100., "complete_candle": True, "taker_ratio": .6})
        return {"spot15": candles.copy(), "index15": candles.copy(), "mark15": candles.copy(),
                "funding": pd.DataFrame({"event_ms": [0], "period_end_ms": [0], "value": [.0001]})}

    def test_fixed_paired_grid(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({c["id"] for c in configs}), 12)
        self.assertEqual(len(research.FEATURES), 7)

    def test_candle_not_available_until_close_plus_lag(self):
        sources = self.sources()
        boundary = 5*research.collection.STEP+research.search.MINUTE
        frame = pd.DataFrame({"timestamp": [boundary-2*research.search.MINUTE, boundary-research.search.MINUTE],
                              "entry_timestamp": [boundary-research.search.MINUTE, boundary], "close": 105.})
        result, counts = research.enrich(frame, sources)
        self.assertEqual(counts["common_rows"], 1)
        self.assertEqual(result.iloc[0].entry_timestamp, boundary)
        self.assertAlmostEqual(result.iloc[0].public_spot_return60, 104/100-1)
        self.assertAlmostEqual(result.iloc[0].public_last_funding, .0001)
        sources["spot15"].loc[sources["spot15"].event_ms > 4*research.collection.STEP, "value"] = 10000
        other, _ = research.enrich(frame, sources)
        pd.testing.assert_frame_equal(result, other)

    def test_missing_and_incomplete_candles_are_not_filled(self):
        boundary = 5*research.collection.STEP+research.search.MINUTE
        frame = pd.DataFrame({"timestamp": [boundary-research.search.MINUTE], "entry_timestamp": [boundary], "close": [105.]})
        for mode in ("missing", "incomplete"):
            sources = self.sources()
            if mode == "missing":
                sources["spot15"] = sources["spot15"].drop(index=2)
            else:
                sources["spot15"].loc[2, "complete_candle"] = False
            result, counts = research.enrich(frame, sources)
            self.assertTrue(result.empty)
            self.assertEqual(counts["excluded_rows"], 1)

    def test_stale_context_rejected(self):
        boundary = 10*research.collection.STEP
        frame = pd.DataFrame({"timestamp": [boundary-research.search.MINUTE], "entry_timestamp": [boundary], "close": [105.]})
        self.assertTrue(research.enrich(frame, self.sources())[0].empty)


if __name__ == "__main__":
    unittest.main()
