import unittest

import numpy as np
import pandas as pd

from scripts import search_low_frequency_rules as rules
from scripts import search_low_frequency_targets as targets


def frame():
    return pd.DataFrame({
        "trend_ma_12_48": [1., -1., 0., 0.],
        "return_12bar": [.01, -.01, .006, 0.],
        "return_3bar": [.01, -.01, .001, 0.],
        "range_position_48bar": [.95, .05, .5, .5],
        "rsi_14bar": [.2, .8, .5, .5],
    })


class LowFrequencyRuleTests(unittest.TestCase):
    def test_grid_is_frozen(self):
        configs = rules.configurations(targets.target_specs())
        self.assertEqual(len(configs), 32)
        self.assertEqual(len({row["id"] for row in configs}), 32)

    def test_each_rule_has_expected_actions(self):
        expected = {
            "trend": ["long", "short", "no_trade", "no_trade"],
            "breakout": ["long", "short", "no_trade", "no_trade"],
            "momentum": ["long", "short", "long", "no_trade"],
            "rsi_reversal": ["long", "short", "no_trade", "no_trade"],
        }
        for name, actions in expected.items():
            self.assertEqual(rules.apply_rule(frame(), name).pred_action.tolist(), actions)

    def test_rule_does_not_mutate_input(self):
        source = frame()
        before = source.copy(deep=True)
        result = rules.apply_rule(source, "trend")
        pd.testing.assert_frame_equal(source, before)
        self.assertTrue(np.isfinite(result.pred_confidence).all())

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            rules.apply_rule(frame(), "unknown")


if __name__ == "__main__":
    unittest.main()
