import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from scripts import expected_return_research as research
from scripts import run_bounded_profitability_search as search


class ExpectedReturnResearchTests(unittest.TestCase):
    def test_fixed_six_configurations(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 6)
        self.assertEqual(len({c["id"] for c in configs}), 6)
        self.assertEqual({(c["include_trade"], c["take_profit_r"]) for c in configs},
                         {(f, t) for f in (False, True) for t in (1.0, 1.5, 3.0)})
        self.assertEqual(configs, search.configurations("expected_return"))

    def test_target_specific_columns(self):
        for target, tag in [(1, "1_0"), (1.5, "1_5"), (3, "3_0")]:
            columns = research.label_columns(target)
            self.assertEqual(len(columns), 6)
            self.assertTrue(all(f"_tp_{tag}r_" in c for c in columns))
            self.assertIn(f"short_tp_{tag}r_holding_minutes", columns)

    def test_cost_is_subtracted_once_from_selected_net_label(self):
        frame = pd.DataFrame({"stop_distance_pct": [.005, .01],
                              "long_tp_1_0r_return_r_net": [-1.2, -1.1],
                              "long_tp_1_5r_return_r_net": [1.3, 1.4],
                              "long_tp_3_0r_return_r_net": [2.8, 2.9]})
        np.testing.assert_allclose(research.target_returns(frame, "long", 1.5, 5), [1.2, 1.35])
        np.testing.assert_allclose(research.target_returns(frame, "long", 3, 5), [2.7, 2.85])
        frame.loc[0, "stop_distance_pct"] = 0
        with self.assertRaises(ValueError):
            research.target_returns(frame, "long", 3, 5)

    def test_signal_selection_without_ground_truth(self):
        frame = pd.DataFrame({"timestamp": range(5)})
        result = research.prediction_frame(frame, [.5, -.1, .3, .2, .4], [.2, -.2, .3, .7, .35], .1, .1)
        self.assertEqual(result.pred_action.tolist(), ["long", "no_trade", "no_trade", "short", "no_trade"])
        self.assertTrue(result.pred_confidence.eq(0).all())
        changed = frame.assign(long_tp_1_0r_return_r_net=999)
        self.assertEqual(research.prediction_frame(changed, [.5]*5, [.2]*5, .1, .1).pred_action.tolist(), ["long"]*5)

    def test_invalid_predictions_and_policy_rejected(self):
        frame = pd.DataFrame({"timestamp": [0]})
        for long, short, threshold, margin in [([np.nan], [0], .1, 0),
                                              ([], [0], .1, 0),
                                              ([[.5]], [[.2]], .1, 0),
                                              ([.5], [.2], .1, -1),
                                              ([.5], [.2], np.inf, 0)]:
            with self.assertRaises(ValueError):
                research.prediction_frame(frame, long, short, threshold, margin)

    def test_actual_simulator_uses_target_return_and_holding(self):
        action = search.load("16_action_profitability_walk_forward_model.py", "test_return_integration_action")
        backtest = search.load("8_model_signal_backtest.py", "test_return_integration_backtest")
        frame = pd.DataFrame({"timestamp": [0, 1200000], "datetime_utc": ["1970-01-01"]*2,
                              "entry_timestamp": [60000, 1260000], "entry_price": [100]*2,
                              "entry_rule": ["next_bar_open"]*2, "label_horizon_complete": [True]*2,
                              "pred_action": ["long"]*2, "pred_confidence": [0.0]*2,
                              "direction_target": [1]*2, "best_action": ["long"]*2,
                              "stop_distance_pct": [.005]*2, "round_trip_cost_bps": [10]*2})
        for tag, net, holding in [("1_0", .8, 1), ("3_0", 2.8, 60)]:
            for side in ("long", "short"):
                frame[f"{side}_tp_{tag}r_return_r_net"] = net
                frame[f"{side}_tp_{tag}r_holding_minutes"] = holding
        one = action.evaluate_candidate(backtest, frame, 5, 1.0)
        three = action.evaluate_candidate(backtest, frame, 5, 3.0)
        self.assertEqual(one["trades"], 2)
        self.assertEqual(three["trades"], 1)
        self.assertAlmostEqual(one["avg_return_r"], .7 / 1.3)
        self.assertAlmostEqual(three["avg_return_r"], 2.7 / 1.3)

    def test_backtest_receives_target_and_retains_old_default(self):
        action = search.load("16_action_profitability_walk_forward_model.py", "test_return_action")
        simulator = Mock()
        simulator.simulate.return_value = (None, None, None)
        simulator.summarize.return_value = dict(trades=1, win_rate_pct=100, avg_return_r=1,
                                                total_return_pct=2, max_drawdown_pct=0)
        frame = pd.DataFrame({"pred_action": ["long", "no_trade"]})
        for target in (1.5, 3.0):
            result = action.evaluate_candidate(simulator, frame, 10, target)
            signals, args = simulator.simulate.call_args.args
            self.assertEqual(args.take_profit_r, target)
            self.assertEqual(args.extra_slippage_bps, 10)
            self.assertEqual(args.min_confidence, 0)
            self.assertEqual(len(signals), 1)
            self.assertEqual(result["trades"], 1)
        action.evaluate_candidate(simulator, frame)
        self.assertEqual(simulator.simulate.call_args.args[1].take_profit_r, 1.0)


if __name__ == "__main__":
    unittest.main()
