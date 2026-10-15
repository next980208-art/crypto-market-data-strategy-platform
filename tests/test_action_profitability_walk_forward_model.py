import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "action_profitability_walk_forward_model",
    ROOT / "16_action_profitability_walk_forward_model.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeModel:
    def __init__(self, positive_probabilities):
        self.positive_probabilities = np.asarray(positive_probabilities, dtype=float)

    def predict_proba(self, features):
        return np.column_stack(
            [1.0 - self.positive_probabilities, self.positive_probabilities]
        )


def prediction_input(rows):
    return pd.DataFrame(
        {
            "feature": list(range(rows)),
            "direction_target": [1] * rows,
            "best_action": ["long"] * rows,
            "fixed_1r_direction": [1] * rows,
            "fixed_1r_best_action": ["long"] * rows,
        }
    )


class ActionProfitabilityWalkForwardTests(unittest.TestCase):
    def test_targets_include_extra_slippage(self):
        frame = pd.DataFrame(
            {
                "stop_distance_pct": [0.005, 0.005],
                "long_tp_1_0r_return_r_net": [0.11, 0.09],
                "short_tp_1_0r_return_r_net": [0.09, 0.11],
            }
        )

        result = MODULE.add_profitability_targets(frame, extra_slippage_bps=5.0)

        self.assertEqual([1, 0], result["long_profitability_target"].tolist())
        self.assertEqual([0, 1], result["short_profitability_target"].tolist())
        self.assertTrue(np.allclose(result["extra_slippage_r"], 0.1))

    def test_action_requires_threshold_and_probability_advantage(self):
        result = MODULE.prediction_frame(
            FakeModel([0.70, 0.57, 0.70, 0.60]),
            FakeModel([0.40, 0.62, 0.68, 0.60]),
            prediction_input(4),
            ["feature"],
            probability_threshold=0.55,
            probability_margin=0.05,
        )

        self.assertEqual(
            ["long", "short", "no_trade", "no_trade"],
            result["pred_action"].tolist(),
        )

    def test_force_abstain_never_emits_order(self):
        result = MODULE.prediction_frame(
            FakeModel([0.99, 0.01]),
            FakeModel([0.01, 0.99]),
            prediction_input(2),
            ["feature"],
            probability_threshold=0.52,
            probability_margin=0.0,
            force_abstain=True,
        )

        self.assertEqual(["no_trade", "no_trade"], result["pred_action"].tolist())
        self.assertEqual([-1, -1], result["pred_direction_target"].tolist())

    def test_probability_parser_rejects_unsafe_values(self):
        with self.assertRaises(ValueError):
            MODULE.parse_probability_thresholds("0.50,0.60")
        with self.assertRaises(ValueError):
            MODULE.parse_probability_margins("-0.01,0.05")


if __name__ == "__main__":
    unittest.main()
