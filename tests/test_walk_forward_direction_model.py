import importlib.util
import unittest
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "walk_forward_direction_model", ROOT / "14_walk_forward_direction_model.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fold_schedule_is_chronological_and_bounded():
    minimum = 0
    maximum = 2_000 * MODULE.DAY_MS - 60_000
    folds = MODULE.build_fold_specs(
        minimum, maximum, 365, 60, 60, 365, 4, "rolling"
    )

    assert len(folds) == 4
    for fold in folds:
        assert fold["train_start"] < fold["validation_start"]
        assert fold["validation_start"] < fold["test_start"]
        assert fold["test_start"] < fold["test_end"]
        assert fold["test_end"] <= maximum + 60_000


def test_expanding_window_keeps_original_train_start():
    folds = MODULE.build_fold_specs(
        0, 1_500 * MODULE.DAY_MS - 60_000, 365, 30, 30, 180, 5, "expanding"
    )

    assert len(folds) == 5
    assert {fold["train_start"] for fold in folds} == {0}
    assert folds[0]["validation_start"] < folds[-1]["validation_start"]


def test_split_purges_labels_crossing_boundaries():
    minute = 60_000
    frame = pd.DataFrame(
        {
            "timestamp": [index * minute for index in range(400)],
            "max_holding_minutes": [10] * 400,
        }
    )
    fold = {
        "fold": 1,
        "train_start": 0,
        "validation_start": 200 * minute,
        "test_start": 300 * minute,
        "test_end": 400 * minute,
    }

    split = MODULE.split_fold(frame, fold)

    assert split["train"]["timestamp"].max() + 10 * minute < fold["validation_start"]
    assert split["validation"]["timestamp"].max() + 10 * minute < fold["test_start"]
    assert split["train_purged_rows"] == 10
    assert split["validation_purged_rows"] == 10
    assert split["test"]["timestamp"].min() == fold["test_start"]


def test_approval_rejects_negative_aggregate_result():
    folds = [
        {
            "test_backtest": {"total_return_pct": 1.0, "trades": 30},
            "test_direction_metrics": {"balanced_accuracy": 0.51},
        },
        {
            "test_backtest": {"total_return_pct": -1.0, "trades": 30},
            "test_direction_metrics": {"balanced_accuracy": 0.51},
        },
    ]
    aggregate = {"total_return_pct": -2.0, "avg_return_r": -0.1, "max_drawdown_pct": -5.0}

    result = MODULE.approval_result(folds, aggregate, 20)

    assert result["approved_for_paper_connection"] is False
    assert result["decision"] == "research_only"


class ThresholdSafetyTests(unittest.TestCase):
    @staticmethod
    def load_training_module():
        training_spec = importlib.util.spec_from_file_location(
            "selective_direction_training_for_test",
            ROOT / "13_train_selective_direction_model.py",
        )
        training = importlib.util.module_from_spec(training_spec)
        training_spec.loader.exec_module(training)
        return training

    def test_prediction_threshold_one_forces_no_trade(self):
        class FakeModel:
            def predict_proba(self, features):
                return np.array([[0.0, 1.0], [1.0, 0.0]])

        training = self.load_training_module()
        frame = pd.DataFrame(
            {
                "feature": [1.0, 2.0],
                "direction_target": [1, 0],
                "best_action": ["long", "short"],
                "fixed_1r_direction": [1, 0],
                "fixed_1r_best_action": ["long", "short"],
            }
        )

        predictions = training.prediction_frame(FakeModel(), frame, ["feature"], 1.0)

        self.assertEqual(["no_trade", "no_trade"], predictions["pred_action"].tolist())
        self.assertEqual([-1, -1], predictions["pred_direction_target"].tolist())

    def test_negative_validation_candidates_force_abstention(self):
        training = self.load_training_module()
        candidates = [
            {
                "threshold": 0.60,
                "trades": 50,
                "avg_return_r": -0.01,
                "total_return_pct": -1.0,
                "max_drawdown_pct": -2.0,
                "selection_score": -2.0,
                "win_rate_pct": 45.0,
                "eligible": True,
            }
        ]

        selected = training.choose_threshold_candidate(candidates, 20)

        self.assertEqual(1.0, selected["threshold"])
        self.assertFalse(selected["deployable"])
        self.assertEqual("no_positive_expectancy_validation_candidate", selected["abstain_reason"])

    def test_positive_validation_candidate_remains_deployable(self):
        training = self.load_training_module()
        candidates = [
            {
                "threshold": 0.60,
                "trades": 50,
                "avg_return_r": 0.05,
                "total_return_pct": 2.0,
                "max_drawdown_pct": -2.0,
                "selection_score": 1.0,
                "win_rate_pct": 55.0,
                "eligible": True,
            }
        ]

        selected = training.choose_threshold_candidate(candidates, 20)

        self.assertEqual(0.60, selected["threshold"])
        self.assertTrue(selected["deployable"])
        self.assertIsNone(selected["abstain_reason"])
