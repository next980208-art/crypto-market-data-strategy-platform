import unittest

import numpy as np
import pandas as pd

from scripts import audit_nonoverlapping_signals as audit


class NonoverlappingSignalTests(unittest.TestCase):
    def frame(self, minutes):
        entry = np.asarray(minutes, dtype=np.int64)*audit.MINUTE
        return pd.DataFrame({"timestamp": entry-audit.MINUTE, "entry_timestamp": entry,
                             "max_holding_minutes": [240]*len(entry), "actual_net_label_r": [1.0]*len(entry)})

    def test_closed_horizon_excludes_shared_boundary_bar(self):
        frame = self.frame([1, 2, 241, 242, 482, 483])
        self.assertEqual((audit.select_nonoverlapping(frame).entry_timestamp/audit.MINUTE).tolist(), [1, 242, 483])

    def test_sorted_selection_and_no_outcome_lookahead(self):
        frame = self.frame([500, 1, 242, 2])
        expected = audit.select_nonoverlapping(frame).entry_timestamp.tolist()
        frame.actual_net_label_r = [-999, 999, -100, 100]
        frame["actual_holding_minutes"] = [1, 240, 10, 100]
        frame["gate_probability"] = [0, 1, 0, 1]
        self.assertEqual(audit.select_nonoverlapping(frame).entry_timestamp.tolist(), expected)
        self.assertEqual(frame.entry_timestamp.tolist(), [500*audit.MINUTE, audit.MINUTE, 242*audit.MINUTE, 2*audit.MINUTE])

    def test_empty_input(self):
        frame = self.frame([])
        self.assertTrue(audit.select_nonoverlapping(frame).empty)
        self.assertIsNone(audit.describe(frame)["positive_rate"])
        self.assertFalse(audit.describe(frame)["meets_existing_meta_training_minimum"])

    def test_invalid_timestamps_or_horizons_rejected(self):
        for field, value in (("timestamp", np.nan), ("entry_timestamp", 1.5),
                             ("entry_timestamp", 60001), ("max_holding_minutes", 60)):
            frame = self.frame([1, 242])
            frame[field] = frame[field].astype(float)
            frame.loc[0, field] = value
            with self.assertRaises(ValueError):
                audit.select_nonoverlapping(frame)
        with self.assertRaises(ValueError):
            audit.select_nonoverlapping(self.frame([1, 1]))

    def test_class_and_sample_minimum_are_preserved(self):
        frame = self.frame(range(1, 201))
        frame.actual_net_label_r = [1.0]*20 + [0.0]*180
        self.assertTrue(audit.describe(frame)["meets_existing_meta_training_minimum"])
        self.assertFalse(audit.describe(frame.iloc[:-1])["meets_existing_meta_training_minimum"])
        frame.loc[0, "actual_net_label_r"] = -1
        self.assertFalse(audit.describe(frame)["meets_existing_meta_training_minimum"])

    def test_cross_fold_union_does_not_double_count_times(self):
        report = audit.count_union_time_windows([self.frame([1, 242]), self.frame([242, 483])])
        self.assertEqual(report, {"summed_rows": 4, "unique_entry_times": 3,
                                  "repeated_entry_times": 1, "nonoverlapping_time_windows": 3})
        self.assertEqual(audit.count_union_time_windows([])["nonoverlapping_time_windows"], 0)


if __name__ == "__main__":
    unittest.main()
