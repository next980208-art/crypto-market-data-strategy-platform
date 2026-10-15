import unittest

import numpy as np
import pandas as pd

from scripts import validate_return_bias as research


class ReturnBiasTests(unittest.TestCase):
    def samples(self):
        end = int(pd.Timestamp("2024-04-01", tz="UTC").timestamp()*1000)
        times = np.arange(end-90*research.search.DAY, end-241*research.search.MINUTE, 5*research.search.MINUTE)
        return pd.DataFrame({"timestamp": times, "entry_timestamp": times+research.search.MINUTE,
                             "max_holding_minutes": 240, "label": np.arange(len(times))}), end

    def test_fixed_grid(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({c["id"] for c in configs}), 12)
        self.assertTrue(all(c["risk_per_trade_pct"] == .005 for c in configs))

    def test_purged_chronological_split_and_nonoverlap(self):
        frame, end = self.samples()
        train, calibration = research.split_samples(frame, end)
        boundary = end-15*research.search.DAY
        self.assertTrue((train.entry_timestamp+240*research.search.MINUTE < boundary).all())
        self.assertTrue((calibration.timestamp >= boundary).all())
        self.assertTrue((calibration.entry_timestamp+240*research.search.MINUTE < end).all())
        self.assertTrue((calibration.entry_timestamp.diff().dropna() > 240*research.search.MINUTE).all())
        self.assertFalse(set(train.timestamp) & set(calibration.timestamp))
        frame["label"] = -frame["label"]
        train2, calibration2 = research.split_samples(frame, end)
        pd.testing.assert_series_equal(train.timestamp, train2.timestamp)
        pd.testing.assert_series_equal(calibration.timestamp, calibration2.timestamp)

    def test_overlap_and_insufficient_rows_fail(self):
        frame, end = self.samples()
        with self.assertRaises(ValueError):
            research.split_samples(frame.tail(10), end)
        with self.assertRaises(ValueError):
            research.split_samples(frame, end-240*research.search.MINUTE)

    def test_bias_sign_and_mean_correction(self):
        predicted = np.linspace(0, 1, 50)
        actual = predicted-.2
        bias = research.mean_bias(predicted, actual)
        self.assertAlmostEqual(bias, .2)
        self.assertAlmostEqual(float((predicted-bias).mean()), float(actual.mean()))
        self.assertAlmostEqual(research.mean_bias(actual, predicted), -.2)

    def test_invalid_bias_vectors_fail(self):
        for p, a in (([1]*49, [0]*49), ([1]*50, [0]*51), ([np.nan]*50, [0]*50)):
            with self.assertRaises(ValueError):
                research.mean_bias(p, a)

    def test_corrected_signals_do_not_read_evaluation_labels(self):
        frame = pd.DataFrame({"timestamp": [1, 2], "actual_future_r": [10, -10]})
        values = pd.DataFrame({"timestamp": [1, 2], "raw_long": [.4, .4], "raw_short": [0., 0.],
                               "corrected_long": [-.1, -.1], "corrected_short": [-.2, -.2]})
        self.assertEqual(research.predictions(frame, values, "raw", .05).pred_action.tolist(), ["long", "long"])
        self.assertEqual(research.predictions(frame, values, "bias_corrected", .05).pred_action.tolist(), ["no_trade", "no_trade"])
        frame["actual_future_r"] *= -1
        self.assertEqual(research.predictions(frame, values, "bias_corrected", .05).pred_action.tolist(), ["no_trade", "no_trade"])
        values["timestamp"] += 1
        with self.assertRaises(AssertionError):
            research.predictions(frame, values, "raw", .05)


if __name__ == "__main__":
    unittest.main()
