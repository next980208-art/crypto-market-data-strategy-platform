import unittest

import numpy as np
import pandas as pd

from scripts import analyze_loss_causes as diagnosis
from scripts import run_bounded_profitability_search as search


class LossDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.action = search.load("16_action_profitability_walk_forward_model.py", "test_loss_action")
        cls.backtest = search.load("8_model_signal_backtest.py", "test_loss_backtest")

    def fixture(self):
        frame = pd.DataFrame({"timestamp": [0, 1200000], "datetime_utc": ["1970-01-01"]*2,
                              "entry_timestamp": [60000, 1260000], "entry_price": [100]*2,
                              "entry_rule": ["next_bar_open"]*2, "label_horizon_complete": [True]*2,
                              "pred_action": ["long"]*2, "pred_confidence": [0.0]*2,
                              "direction_target": [1]*2, "best_action": ["long"]*2,
                              "stop_distance_pct": [.005]*2, "round_trip_cost_bps": [10]*2,
                              "predicted_long_net_label_r": [.2]*2, "predicted_short_net_label_r": [.1]*2,
                              "volatility_group": ["low"]*2, "score_group": ["at_least_0.20"]*2})
        for side, net, holding, outcome in [("long", 1.3, 1, "take_profit"), ("short", -1.2, 60, "stop_loss")]:
            frame[f"{side}_tp_1_5r_return_r_net"] = net
            frame[f"{side}_tp_1_5r_holding_minutes"] = holding
            frame[f"{side}_tp_1_5r_outcome"] = outcome
        args = self.action.backtest_args(5)
        args.take_profit_r = 1.5
        return frame, args

    def test_flip_is_copy_and_preserves_abstention(self):
        frame = pd.DataFrame({"pred_action": ["long", "short", "no_trade"]})
        self.assertEqual(diagnosis.flip_predictions(frame).pred_action.tolist(), ["short", "long", "no_trade"])
        self.assertEqual(frame.pred_action.tolist(), ["long", "short", "no_trade"])
        with self.assertRaises(ValueError):
            diagnosis.flip_predictions(pd.DataFrame({"pred_action": ["unknown"]}))

    def test_fixed_exposure_costs_reconcile_compounded_equity(self):
        frame, args = self.fixture()
        trades, _, _ = self.backtest.simulate(frame, args)
        result = diagnosis.decompose(trades, frame, args, self.backtest)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result.gross_label_r, 1.5)
        np.testing.assert_allclose(result.net_label_r_after_slippage, 1.2)
        self.assertAlmostEqual(result.iloc[0].fee_pnl_initial_capital, .02 / 1.3 * .2)
        self.assertAlmostEqual(result.iloc[1].equity_before, trades[0]["equity"])
        self.assertAlmostEqual(result.net_pnl_initial_capital.sum(), trades[-1]["equity"] - 1)
        np.testing.assert_allclose(result.gross_pnl_initial_capital - result.fee_pnl_initial_capital -
                                   result.slippage_pnl_initial_capital, result.net_pnl_initial_capital)
        trades[0]["return_pct"] += .001
        with self.assertRaises(AssertionError):
            diagnosis.decompose(trades, frame, args, self.backtest)

    def test_reverse_recomputes_holding_and_is_not_negation(self):
        frame, args = self.fixture()
        trades, _, _ = self.backtest.simulate(frame, args)
        reverse, _, _ = self.backtest.simulate(diagnosis.flip_predictions(frame), args)
        self.assertEqual(len(trades), 2)
        self.assertEqual(len(reverse), 1)
        self.assertEqual(reverse[0]["holding_minutes"], 60)
        self.assertNotAlmostEqual(reverse[0]["return_pct"], -trades[0]["return_pct"])
        result = diagnosis.decompose(trades, frame, args, self.backtest)
        np.testing.assert_allclose(result.opposite_same_entry_net_label_r, -1.3)
        self.assertTrue(result.opposite_same_entry_holding_minutes.eq(60).all())
        self.assertTrue(result.opposite_same_entry_outcome.eq("stop_loss").all())

    def test_empty_groups_and_small_sample_marker(self):
        self.assertEqual(diagnosis.group_statistics(pd.DataFrame()), [])
        frame, args = self.fixture()
        trades, _, _ = self.backtest.simulate(frame, args)
        groups = diagnosis.group_statistics(diagnosis.decompose(trades, frame, args, self.backtest))
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(g["small_sample_under_20"] for g in groups))
        self.assertTrue(all(g["mean_losing_r"] is None for g in groups))

    def test_duplicate_input_timestamps_rejected(self):
        frame, args = self.fixture()
        frame["timestamp"] = 0
        with self.assertRaises(ValueError):
            diagnosis.decompose([], frame, args, self.backtest)


if __name__ == "__main__":
    unittest.main()
