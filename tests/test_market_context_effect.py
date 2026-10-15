import unittest

import numpy as np
import pandas as pd

from scripts.analyze_market_context_effect import paired_trials, trade_effect


def trades(rows):
    return pd.DataFrame(rows, columns=["entry_timestamp", "action", "return_pct"])


class ContextEffectTests(unittest.TestCase):
    def test_trade_set_accounting(self):
        b = trades([(1, "long", -.01), (2, "short", .02)])
        c = trades([(2, "short", .02), (3, "long", .03)])
        result = trade_effect(b, c)
        self.assertEqual([result[k] for k in ("common_trades", "removed_trades", "added_trades")], [1, 1, 1])
        self.assertAlmostEqual(result["removed_log_return_effect"], -np.log1p(-.01))
        self.assertAlmostEqual(result["total_log_return_delta"], np.log1p(.03) - np.log1p(-.01))

    def test_empty_and_changed_common_trade(self):
        empty = trades([])
        self.assertEqual(trade_effect(empty, empty)["total_log_return_delta"], 0)
        self.assertEqual(trade_effect(pd.DataFrame(), empty)["total_log_return_delta"], 0)
        result = trade_effect(trades([(1, "long", .01)]), trades([(1, "long", .02)]))
        self.assertAlmostEqual(result["common_log_return_delta"], result["total_log_return_delta"])

    def test_reject_duplicate_and_invalid_return(self):
        for bad in ([(1, "long", 0), (1, "long", 0)], [(1, "long", -1)], [(1, "long", float("nan"))]):
            with self.assertRaises(ValueError):
                trade_effect(trades(bad), trades([]))

    def test_requires_complete_pairs(self):
        row = {"status": "completed", "config": {"family": "ridge", "threshold_r": .1, "group": "baseline"}}
        with self.assertRaises(ValueError):
            list(paired_trials([row]))
        other = {"status": "completed", "config": {**row["config"], "group": "context"}}
        self.assertEqual(len(list(paired_trials([row, other]))), 1)
        with self.assertRaises(ValueError):
            list(paired_trials([row, row, other]))


if __name__ == "__main__":
    unittest.main()
