import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "two_stage_walk_forward_model", ROOT / "15_two_stage_walk_forward_model.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TwoStageWalkForwardTests(unittest.TestCase):
    def test_gate_target_requires_return_and_fast_holding(self):
        frame = pd.DataFrame(
            {
                "fixed_1r_best_return": [0.8, 0.8, 0.1, -0.2],
                "two_stage_best_holding_minutes": [30, 90, 30, 20],
            }
        )

        result = MODULE.add_gate_target(frame, return_r=0.2, max_holding=60)

        self.assertEqual([1, 0, 0, 0], result["opportunity_target"].tolist())

    def test_force_abstain_never_emits_direction(self):
        class FakeModel:
            def predict_proba(self, features):
                return np.array([[0.01, 0.99], [0.99, 0.01]])

        frame = pd.DataFrame(
            {
                "feature": [1.0, 2.0],
                "direction_target": [1, 0],
                "best_action": ["long", "short"],
                "fixed_1r_direction": [1, 0],
                "fixed_1r_best_action": ["long", "short"],
            }
        )

        result = MODULE.prediction_frame(
            FakeModel(), FakeModel(), frame, ["feature"], 0.52, 0.52, force_abstain=True
        )

        self.assertEqual(["no_trade", "no_trade"], result["pred_action"].tolist())
        self.assertEqual([-1, -1], result["pred_direction_target"].tolist())


if __name__ == "__main__":
    unittest.main()
