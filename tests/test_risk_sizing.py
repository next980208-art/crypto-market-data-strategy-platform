import unittest

import pandas as pd

from scripts import search_risk_sizing as research


class RiskSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backtest = research.search.load("8_model_signal_backtest.py", "test_risk_backtest")
        cls.action = research.search.load("16_action_profitability_walk_forward_model.py", "test_risk_action")

    def frame(self, outcomes):
        times = pd.date_range("2024-01-01", periods=len(outcomes), freq="30min", tz="UTC")
        frame = pd.DataFrame({"timestamp": times.as_unit("ms").asi8, "datetime_utc": times.astype(str),
                              "entry_timestamp": times.as_unit("ms").asi8+60000, "entry_price": 100.,
                              "entry_rule": "next_bar_open", "label_horizon_complete": True,
                              "pred_action": "long", "pred_confidence": 0., "direction_target": "long",
                              "best_action": "long", "stop_distance_pct": .005, "round_trip_cost_bps": 10.})
        for side in ("long", "short"):
            frame[self.backtest.return_column(side, 1.5)] = outcomes
            frame[self.backtest.holding_column(side, 1.5)] = 1
        return frame

    def test_grid_keeps_all_baselines(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 48)
        self.assertEqual(len({c["id"] for c in configs}), 48)
        self.assertEqual(sum(c["risk_per_trade_pct"] == .02 for c in configs), 12)

    def test_sizing_and_invalid_inputs(self):
        self.assertAlmostEqual(research.notional_ratio(.02, .005, 15), 3.076923076923077)
        for risk in (0, -.01, .021, float("nan")):
            with self.assertRaises(ValueError):
                research.notional_ratio(risk, .005, 15)

    def test_stop_loss_and_r_units(self):
        frame = self.frame([-1.2])
        for risk in research.RISKS:
            metrics, summary, trades = research.evaluate(self.backtest, self.action, frame, risk, 5)
            self.assertAlmostEqual(metrics["total_return_pct"], -risk*100)
            self.assertAlmostEqual(metrics["avg_return_r"], -1.)
            self.assertAlmostEqual(trades.iloc[0].return_pct, -risk)
            self.assertEqual(summary["risk_per_trade_pct"], risk)

    def test_fixed_daily_guard_reruns_trade_selection(self):
        frame = self.frame([-1.2, .11, -1.2, -1.2, 1.3])
        large, _, _ = research.evaluate(self.backtest, self.action, frame, .02, 5)
        small, _, _ = research.evaluate(self.backtest, self.action, frame, .005, 5)
        self.assertEqual(large["trades"], 4)
        self.assertEqual(small["trades"], 5)

    def test_baseline_matches_existing_evaluator_and_empty_is_not_candidate(self):
        frame = self.frame([1.3, -1.2])
        for extra in (5, 10):
            metrics, _, _ = research.evaluate(self.backtest, self.action, frame, .02, extra)
            self.assertEqual(metrics, self.action.evaluate_candidate(self.backtest, frame, extra, 1.5))
        frame["pred_action"] = "no_trade"
        metrics, _, trades = research.evaluate(self.backtest, self.action, frame, .005, 5)
        self.assertTrue(trades.empty)
        folds = [{"fold": n, "normal": metrics, "stress": metrics} for n in (1, 2, 3)]
        self.assertFalse(research.prior.summarize({}, folds)["candidate"])

    def test_excess_risk_is_rejected_before_simulation(self):
        with self.assertRaises(ValueError):
            research.evaluate(self.backtest, self.action, self.frame([-1.2]), .1, 5)


if __name__ == "__main__":
    unittest.main()
