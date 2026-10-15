import unittest

import numpy as np
import pandas as pd

from scripts import search_trade_patterns as patterns


class PatternSearchTests(unittest.TestCase):
    def test_frozen_grid(self):
        configs = patterns.configurations()
        self.assertEqual(len(configs), 36)
        self.assertEqual(len({c["id"] for c in configs}), 36)

    def test_threshold_direction_and_source_preservation(self):
        frame = pd.DataFrame({"pred_action": ["long", "short", "long", "no_trade"]})
        original = frame.copy()
        result = patterns.filter_signals(frame, [.5, .9, .49, 1], {"threshold": .5, "direction": "long"})
        self.assertEqual(result.pred_action.tolist(), ["long", "no_trade", "no_trade", "no_trade"])
        pd.testing.assert_frame_equal(frame, original)

    def test_reject_invalid_probabilities(self):
        frame = pd.DataFrame({"pred_action": ["long"]})
        for probability in ([np.nan], [1.1], [-.1], [], [[.5]]):
            with self.assertRaises(ValueError):
                patterns.filter_signals(frame, probability, {"threshold": .5, "direction": "both"})

    def test_candidate_requires_all_folds_and_stress(self):
        metric = {"trades": 20, "avg_return_r": .1, "total_return_pct": 1, "max_drawdown_pct": -2}
        folds = [{"normal": metric.copy(), "stress": metric.copy()} for _ in range(3)]
        self.assertTrue(patterns.summarize({}, folds)["candidate"])
        folds[1]["stress"]["total_return_pct"] = -1
        self.assertFalse(patterns.summarize({}, folds)["candidate"])
        folds[1]["stress"]["total_return_pct"] = 1
        folds[0]["normal"]["trades"] = 0
        self.assertFalse(patterns.summarize({}, folds)["candidate"])

    def test_no_trade_is_not_best_diagnostic(self):
        def row(trades, r):
            return {"normal": {"folds_with_enough_trades": trades, "worst_fold_avg_r": r, "weighted_avg_r": r},
                    "stress": {"worst_fold_avg_r": r}}
        self.assertGreater(patterns.ranking(row(3, -.1)), patterns.ranking(row(0, 0)))


if __name__ == "__main__":
    unittest.main()
