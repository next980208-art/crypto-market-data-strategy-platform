import unittest

import numpy as np
import pandas as pd

from scripts import run_cost_aware_gate as gate
from scripts import run_bounded_profitability_search as search


class CostAwareGateTests(unittest.TestCase):
    def fixture(self):
        frame = pd.DataFrame({"pred_action": ["long", "short"], "predicted_long_net_label_r": [.2, -.1],
                              "predicted_short_net_label_r": [.1, .3], "stop_distance_pct": [.005]*2,
                              "long_tp_1_5r_return_r_net": [1.3, -1.2], "short_tp_1_5r_return_r_net": [-1.2, .05]},
                             index=[10, 20])
        for column in gate.CAUSAL_COLUMNS:
            frame[column] = [.01, .02]
        return frame

    def test_feature_allowlist_excludes_future_labels(self):
        frame = self.fixture()
        expected = gate.gate_features(frame)
        self.assertEqual(expected.columns.tolist(), gate.GATE_COLUMNS)
        np.testing.assert_allclose(expected.selected_expected_r, [.2, .3])
        np.testing.assert_allclose(expected.direction_margin_r, [.1, .4])
        frame["long_tp_1_5r_return_r_net"] = 999
        frame["future_price"] = -999
        pd.testing.assert_frame_equal(gate.gate_features(frame), expected)
        frame.loc[10, "volatility_15m"] = np.nan
        with self.assertRaises(ValueError):
            gate.gate_features(frame)

    def test_target_uses_selected_side_and_cost_once(self):
        frame = self.fixture()
        np.testing.assert_allclose(gate.selected_net_returns(frame, 1.5, 5), [1.2, -.05])
        self.assertEqual(gate.selected_net_returns(frame, 1.5, 5).index.tolist(), [10, 20])
        frame.loc[10, "pred_action"] = "no_trade"
        with self.assertRaises(ValueError):
            gate.selected_net_returns(frame, 1.5, 5)

    def test_inner_split_purges_future_outcomes(self):
        boundary = 30 * search.DAY
        end = 45 * search.DAY
        frame = pd.DataFrame({"timestamp": [0, boundary-search.MINUTE, boundary, end-search.MINUTE],
                              "entry_timestamp": [search.MINUTE, boundary, boundary+search.MINUTE, end],
                              "max_holding_minutes": [240]*4})
        base, meta, bounds = gate.inner_parts(frame, {"train_start": 0, "validation_start": end})
        self.assertEqual(base.timestamp.tolist(), [0])
        self.assertEqual(meta.timestamp.tolist(), [boundary])
        self.assertEqual(bounds["validation_start"], boundary)
        with self.assertRaises(ValueError):
            gate.inner_parts(frame, {"train_start": 0, "validation_start": boundary})

    def test_gate_fixed_threshold_and_no_direction_changes(self):
        signals = self.fixture()
        output = gate.apply_gate(signals, [.499, .5])
        self.assertEqual(output.pred_action.tolist(), ["no_trade", "short"])
        self.assertEqual(signals.pred_action.tolist(), ["long", "short"])
        for values in ([.5], [np.nan, .5], [-.1, .5], [1.1, .5], [[.5], [.6]]):
            with self.assertRaises(ValueError):
                gate.apply_gate(signals, values)

    def test_insufficient_meta_history_blocks_fitting(self):
        x = gate.gate_features(self.fixture())
        self.assertIsNone(gate.fit_gate(x, pd.Series([0, 1], index=x.index)))
        with self.assertRaises(ValueError):
            gate.fit_gate(x, pd.Series([0, 1]))
        x = pd.DataFrame(np.zeros((200, 8)), columns=gate.GATE_COLUMNS)
        self.assertIsNone(gate.fit_gate(x, pd.Series([0]*190+[1]*10)))

    def test_scaler_fits_training_only_and_probabilities_reconstruct(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame(rng.normal(size=(240, 8)), columns=gate.GATE_COLUMNS)
        target = x.iloc[:, 0].gt(0).astype(int)
        model = gate.fit_gate(x, target)
        mean = model.steps[0][1].mean_.copy()
        future = x.iloc[:3].copy()+100
        probability = model.predict_proba(future)[:, 1]
        scaler, logistic = model.steps[0][1], model.steps[1][1]
        reconstructed = 1/(1+np.exp(-((future.to_numpy()-scaler.mean_)/scaler.scale_ @ logistic.coef_[0]+logistic.intercept_[0])))
        np.testing.assert_allclose(probability, reconstructed)
        np.testing.assert_allclose(mean, x.mean().to_numpy())
        np.testing.assert_array_equal(mean, scaler.mean_)

    def test_diagnostics_handles_no_accepted_or_single_class(self):
        report = gate.signal_diagnostics(pd.Series([-1.0, -.1]), np.array([.1, .2]), .3)
        self.assertEqual(report["accepted"], 0)
        self.assertIsNone(report["accepted_mean_net_label_r"])
        self.assertIsNone(report["roc_auc"])
        self.assertAlmostEqual(report["constant_train_prior_brier"], .09)
        report = gate.signal_diagnostics(pd.Series([1., -1.], index=[10, 20]), np.array([.6, .4]), .5)
        self.assertEqual(report["accepted_positive_rate"], 1)
        self.assertEqual(report["roc_auc"], 1)


if __name__ == "__main__":
    unittest.main()
