import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import search_flow_exhaustion as flow
from scripts import search_flow_rules as rules


class FlowResearchTests(unittest.TestCase):
    def source(self):
        count = 700
        close = 100 + np.sin(np.arange(count)/17)
        return pd.DataFrame({"close": close, "open": close-.1, "high": close+.2,
            "low": close-.3, "volume": np.arange(count)+1.,
            "agg_taker_volume_imbalance": np.sin(np.arange(count)/9),
            "agg_trade_volume_ratio_15m": np.ones(count)})

    def test_features_do_not_use_future_bars(self):
        raw = self.source()
        original, columns = flow.features(raw)
        raw.loc[501:, ["close", "volume", "agg_taker_volume_imbalance"]] *= 10
        changed, _ = flow.features(raw)
        pd.testing.assert_frame_equal(original.loc[:500, columns["flow"]], changed.loc[:500, columns["flow"]])
        prefix, _ = flow.features(raw.iloc[:501])
        pd.testing.assert_frame_equal(original.loc[:500, columns["flow"]], prefix[columns["flow"]])

    def test_outcome_columns_are_never_features(self):
        raw = self.source()
        raw["best_action"] = "long"
        raw["long_tp_3_0r_return_r_net"] = 1000.
        _, columns = flow.features(raw)
        self.assertNotIn("best_action", columns["flow"])
        self.assertNotIn("long_tp_3_0r_return_r_net", columns["flow"])
        self.assertTrue(set(columns["price"]).issubset(columns["flow"]))

    def test_full_horizon_purge_at_boundary(self):
        frame = pd.DataFrame({"entry_timestamp": [0, 10*flow.MINUTE, 11*flow.MINUTE],
                              "max_holding_minutes": [240, 240, 240]})
        result = flow.period(frame, 0, 250*flow.MINUTE)
        self.assertEqual(result.index.tolist(), [0])

    def test_selection_requires_trades_and_uses_severe_cost(self):
        def record(name, trades, severe):
            return {"id": name, "metrics": {cost: {"trades": trades,
                "total_return_pct": severe if cost == "severe" else 50.,
                "max_drawdown_pct": -2.} for cost in flow.COSTS}}
        records = [record("too_few", 1, 50), record("normal_only", 20, -1), record("better", 20, 1)]
        self.assertEqual(flow.select_validation(records)["id"], "better")
        self.assertIsNone(flow.select_validation(records[:1]))

    def test_negative_and_small_sample_are_recorded(self):
        metrics = {cost: {"trades": 2, "total_return_pct": -1., "max_drawdown_pct": -6.} for cost in flow.COSTS}
        self.assertEqual(len(flow.rejection_reasons(metrics)), 9)

    def test_funding_target_direction_and_no_double_cost(self):
        frame = pd.DataFrame({"entry_timestamp": [flow.MINUTE], "entry_price": [100.],
            "stop_distance_pct": [.005], "long_tp_1_0r_return_r_net": [.8],
            "short_tp_1_0r_return_r_net": [.8],
            "long_tp_1_0r_holding_minutes": [5], "short_tp_1_0r_holding_minutes": [5]})
        market = flow.funding.FundingMarket([{"fundingTime": 3*flow.MINUTE,
            "fundingRate": .001, "markPrice": 100}],
            pd.DataFrame({"event_ms": [0, flow.MINUTE], "close": [100., 100.]}))
        self.assertAlmostEqual(flow.funded_target(frame, "long", 1., market)[0], .5)
        self.assertAlmostEqual(flow.funded_target(frame, "short", 1., market)[0], .9)

    def test_refuses_to_overwrite_prior_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(flow, "digest", return_value=flow.EXPECTED_SHA):
                with self.assertRaises(FileExistsError):
                    flow.run(Path(tmp))

    def test_rule_symmetry_burst_and_no_future_label_use(self):
        frame = pd.DataFrame({"agg_taker_volume_imbalance_mean_15m": [.2, -.2, .2],
            "flow_fast_change": [-.1, .1, -.1], "price_return_15": [.01, -.01, .01],
            "agg_trade_volume_ratio_15m": [2., 2., 1.],
            "price_upper_wick": [.5, .5, .5], "price_lower_wick": [.5, .5, .5],
            "best_action": ["long", "short", "long"]})
        config = {"name": "exhaustion", "imbalance": .1, "volume_ratio": 1.5}
        self.assertEqual(rules.signals(frame, config).pred_action.tolist(), ["short", "long", "no_trade"])
        frame.best_action = "no_trade"
        self.assertEqual(rules.signals(frame, config).pred_action.tolist(), ["short", "long", "no_trade"])

    def test_rule_grid_is_unique_and_fixed(self):
        self.assertEqual(len(rules.configurations()), 36)
        self.assertEqual(len({r["id"] for r in rules.configurations()}), 36)


if __name__ == "__main__":
    unittest.main()
