import unittest

import numpy as np
import pandas as pd

from scripts import search_chart_ict_patterns as research


def sample_frame(rows=80):
    close = np.linspace(100., 108., rows)
    open_ = close-.1
    return pd.DataFrame({
        "timestamp": np.arange(rows, dtype=np.int64)*900000,
        "open": open_, "high": close+.2, "low": open_-.2, "close": close,
        "trend_ma_12_48": .01, "return_12bar": .01, "return_3bar": .001,
        "return_1bar": .001, "range_position_48bar": .8, "rsi_14bar": .55,
        "atr_14bar_pct": .005, "close_location": .75,
        "volume_ratio_12bar": .1, "upper_wick_pct": .001, "lower_wick_pct": .001,
    })


class ChartIctPatternTests(unittest.TestCase):
    def test_configuration_space(self):
        rows = research.configurations()
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({row["id"] for row in rows}), 28)

    def test_all_patterns_return_causal_boolean_masks(self):
        frame = sample_frame()
        masks = research.pattern_masks(frame)
        self.assertEqual(set(masks), set(research.PATTERNS))
        for long_mask, short_mask in masks.values():
            self.assertEqual(long_mask.dtype, bool)
            self.assertEqual(short_mask.dtype, bool)
            self.assertFalse((long_mask & short_mask).any())

    def test_future_change_does_not_change_prior_signals(self):
        frame = sample_frame()
        before = {name: tuple(mask.iloc[:-1].tolist() for mask in pair)
                  for name, pair in research.pattern_masks(frame).items()}
        changed = frame.copy()
        changed.loc[len(changed)-1, ["open", "high", "low", "close"]] = [50., 200., 40., 190.]
        after = {name: tuple(mask.iloc[:-1].tolist() for mask in pair)
                 for name, pair in research.pattern_masks(changed).items()}
        self.assertEqual(before, after)

    def test_liquidity_sweep_reclaims_prior_low(self):
        frame = sample_frame()
        index = len(frame)-1
        prior_low = frame.loc[index-20:index-1, "low"].min()
        frame.loc[index, ["open", "low", "close", "high", "close_location", "volume_ratio_12bar"]] = [
            prior_low+.05, prior_low-.2, prior_low+.3, prior_low+.4, .8, .5]
        long_mask, short_mask = research.pattern_masks(frame)["ict_liquidity_sweep"]
        self.assertTrue(long_mask.iloc[-1])
        self.assertFalse(short_mask.iloc[-1])


if __name__ == "__main__":
    unittest.main()
