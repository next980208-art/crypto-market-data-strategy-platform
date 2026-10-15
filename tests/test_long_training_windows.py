import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scripts import search_long_training_windows as research


class LongWindowTests(unittest.TestCase):
    def test_fixed_schedule_and_grid(self):
        schedules = research.collection.schedules()
        self.assertEqual(len(schedules), 3)
        self.assertEqual(schedules[0]["validation_start"], 1671753600000)
        self.assertTrue(all(s["validation_end"]-s["validation_start"] == 30*research.search.DAY for s in schedules))
        configs = research.configurations()
        self.assertEqual(len(configs), 18)
        self.assertEqual(len({c["id"] for c in configs}), 18)

    def test_shared_validation_and_purged_training(self):
        day, minute = research.search.DAY, research.search.MINUTE
        schedule = {"validation_start": 370*day, "validation_end": 400*day}
        times = np.arange(0, 400*day, 60*minute)
        frame = pd.DataFrame({"timestamp": times, "entry_timestamp": times+minute, "max_holding_minutes": 240})
        previous = None
        sizes = []
        for days in (90, 180, 365):
            train, future, spec = research.select_window(frame, schedule, days)
            self.assertGreaterEqual(train.timestamp.min(), spec["train_start"])
            self.assertLess((train.entry_timestamp+240*minute).max(), schedule["validation_start"])
            if previous is not None:
                pd.testing.assert_frame_equal(previous, future)
            previous = future
            sizes.append(len(train))
        self.assertEqual(sizes, sorted(set(sizes)))
        with self.assertRaises(ValueError):
            research.select_window(frame[frame.timestamp >= 300*day], schedule, 365)

    def test_mark_continuity_required(self):
        frame = pd.DataFrame({"event_ms": [0, 60000], "open": [1., 1.], "high": [1., 1.], "low": [1., 1.], "close": [1., 1.]})
        research.collection.validate_marks(frame, 0, 120000)
        with self.assertRaises(AssertionError):
            research.collection.validate_marks(frame.iloc[:1], 0, 120000)

    def test_bounded_read_filters_and_detects_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.parquet"
            frame = pd.DataFrame({"timestamp": np.arange(20000)*60000, "close": np.arange(20000, dtype=float)})
            frame.to_parquet(path, index=False)
            result = research.bounded_read(path, 16000*60000, 18000*60000)
            pd.testing.assert_frame_equal(result, frame.iloc[16000:18000].reset_index(drop=True))
            frame.drop(index=17000).to_parquet(path, index=False)
            with self.assertRaises(AssertionError):
                research.bounded_read(path, 16000*60000, 18000*60000)

    def test_numeric_digest_detects_feature_change(self):
        self.assertEqual(research.numeric_digest([1., 2.]), research.numeric_digest(np.array([1., 2.], dtype=np.float32)))
        self.assertNotEqual(research.numeric_digest([1., 2.]), research.numeric_digest([1., 2.001]))


if __name__ == "__main__":
    unittest.main()
