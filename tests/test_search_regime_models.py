import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import search_regime_models as regime


class RegimeResearchTests(unittest.TestCase):
    def features(self):
        frame = pd.DataFrame(0., index=range(4), columns=regime.gate.GATE_COLUMNS)
        frame["volatility_15m"] = [1., 2., 3., 4.]
        frame["volume_ratio_60"] = [1., 2., 3., 4.]
        frame["is_long"] = [1, 0, 1, 0]
        frame["return_60m"] = [1., -1., -1., 0.]
        return frame

    def test_grid_and_model_limits(self):
        grid = regime.configurations()
        self.assertEqual(len(grid), 42)
        self.assertEqual(len({c["id"] for c in grid}), 42)
        tree = regime.make_model("tree")
        self.assertEqual(tree.max_depth, 2)
        self.assertEqual(tree.min_samples_leaf, 40)

    def test_cutoffs_training_only_and_partition(self):
        train = self.features()
        limits = regime.cutoffs(train)
        validation = train.copy()
        validation["volatility_15m"] = [2.5, 100, 1000, 10000]
        low = regime.regime_mask(validation, limits, "low_volatility")
        high = regime.regime_mask(validation, limits, "high_volatility")
        np.testing.assert_array_equal(low, [True, False, False, False])
        np.testing.assert_array_equal(low, ~high)
        self.assertEqual(limits["volatility_15m"], 2.5)

    def test_trend_uses_direction_and_excludes_flat(self):
        x = self.features()
        np.testing.assert_array_equal(regime.regime_mask(x, {}, "trend_aligned"), [True, True, False, False])
        np.testing.assert_array_equal(regime.regime_mask(x, {}, "trend_opposed"), [False, False, True, False])

    def test_filters_do_not_change_direction_or_input(self):
        x = self.features()
        frame = x.copy()
        frame["pred_action"] = ["long", "short", "long", "short"]
        frame["predicted_long_net_label_r"] = 1.
        frame["predicted_short_net_label_r"] = .5
        original = frame.copy()
        result = regime.apply_policy(frame, [.6, .6, .6, .6], regime.cutoffs(x), {"threshold": .5, "regime": "low_volatility"})
        self.assertEqual(result.pred_action.tolist(), ["long", "short", "no_trade", "no_trade"])
        pd.testing.assert_frame_equal(frame, original)

    def test_partial_profit_does_not_mean_candidate(self):
        self.assertEqual(regime.disposition({"candidate": False, "normal": {"positive_folds": 1}}), "partial_positive_rejected")
        self.assertEqual(regime.disposition({"candidate": True}), "research_candidate")

    def test_error_trial_is_preserved_and_next_trial_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("models", "trials", "trades"):
                (root/folder).mkdir()
            regime.search.write_json(root/"models.json", {})
            regime.search.write_json(root/"results.json", {"trials": []})
            configs = regime.configurations()[:2]
            plan = {"source_patterns": str(root), "configurations": configs}
            metric = {"trades": 0, "avg_return_r": 0, "total_return_pct": 0, "max_drawdown_pct": 0}
            result = regime.prior.summarize(configs[1], [{"normal": metric, "stress": metric} for _ in range(3)])
            result.update(status="completed", classification="rejected")
            with patch.object(regime, "verify_inputs", return_value=(plan, [])), \
                 patch.object(regime, "model_artifacts", return_value=({}, {})), \
                 patch.object(regime.search, "load"), \
                 patch.object(regime, "compare", side_effect=[ValueError("test failure"), result]):
                regime.run_worker(root, False)
            failed = regime.prior.read(root/"trials"/(configs[0]["id"]+".json"))
            saved = regime.prior.read(root/"results.json")
            self.assertEqual(failed["status"], "error")
            self.assertIn("test failure", failed["traceback"])
            self.assertEqual(saved["status"], "completed_with_errors")
            self.assertEqual(len(saved["trials"]), 2)
            self.assertEqual(saved["candidate_ids"], [])


if __name__ == "__main__":
    unittest.main()
