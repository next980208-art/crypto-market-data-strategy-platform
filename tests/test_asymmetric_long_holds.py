import unittest

import pandas as pd

from scripts import search_asymmetric_long_holds as research


class AsymmetricLongHoldTests(unittest.TestCase):
    def test_configuration_space_is_fixed_and_unique(self):
        rows = research.configurations()
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["id"] for row in rows}), 96)
        self.assertEqual({row["target_r"] for row in rows}, {2., 3., 5., 8.})
        self.assertTrue(all(row["risk_per_trade_pct"] == .005 for row in rows))

    def test_pullback_requires_trend_extreme_and_confirmation(self):
        frame = pd.DataFrame({
            "trend_ma_12_48": [.01, -.01, .01],
            "return_12bar": [0., 0., 0.],
            "return_3bar": [.001, -.001, -.001],
            "return_1bar": [0., 0., 0.],
            "range_position_48bar": [.2, .8, .2],
            "rsi_14bar": [.5, .5, .5],
        })
        result = research.apply_rule(frame, "pullback")
        self.assertEqual(result.pred_action.tolist(), ["long", "short", "no_trade"])

    def test_reversal_requires_one_bar_confirmation(self):
        frame = pd.DataFrame({
            "trend_ma_12_48": [0., 0., 0.],
            "return_12bar": [0., 0., 0.],
            "return_3bar": [0., 0., 0.],
            "return_1bar": [.001, -.001, -.001],
            "range_position_48bar": [.05, .95, .05],
            "rsi_14bar": [.2, .8, .2],
        })
        result = research.apply_rule(frame, "reversal")
        self.assertEqual(result.pred_action.tolist(), ["long", "short", "no_trade"])

    def test_development_survivor_is_never_model_approval(self):
        config = research.configurations()[0]
        folds = []
        for fold in (1, 2, 3):
            metrics = {"total_return_pct": 1., "trades": 6, "win_rate_pct": 30., "avg_return_r": .2,
                       "max_drawdown_pct": -3., "funding_cash_delta_pct_of_initial": -.1}
            folds.append({"fold": fold, "normal": metrics, "stress": metrics, "severe": metrics})
        result = research.summarize(config, folds)
        self.assertTrue(result["development_survivor"])
        self.assertFalse(result["independent_test_evaluated"])
        self.assertFalse(result["model_approved"])
        self.assertEqual(result["orders_sent"], 0)


if __name__ == "__main__":
    unittest.main()
