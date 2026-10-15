import unittest

import numpy as np
import pandas as pd

from scripts import search_sampled_training as research


class SampledTrainingTests(unittest.TestCase):
    def frame(self, minutes):
        t = np.array(minutes)*research.search.MINUTE
        return pd.DataFrame({'timestamp': t, 'entry_timestamp': t+research.search.MINUTE, 'max_holding_minutes': 240,
                             'future_return': np.arange(len(t))})

    def test_five_minute_time_gap_not_row_count(self):
        selected = research.sample(self.frame([0, 1, 5, 7, 11, 12, 16]), 'gap5')
        self.assertEqual((selected.timestamp/research.search.MINUTE).tolist(), [0, 5, 11, 16])

    def test_closed_horizon_boundary(self):
        selected = research.sample(self.frame([0, 240, 241, 481, 482]), 'closed240')
        self.assertEqual((selected.timestamp/research.search.MINUTE).tolist(), [0, 241, 482])

    def test_no_label_dependence_or_input_mutation(self):
        frame = self.frame(range(600))
        before = frame.copy()
        first = research.sample(frame, 'closed240')
        pd.testing.assert_frame_equal(frame, before)
        frame.future_return = -frame.future_return
        second = research.sample(frame, 'closed240')
        np.testing.assert_array_equal(first.timestamp, second.timestamp)

    def test_duplicates_and_unknown_mode_fail(self):
        with self.assertRaises(ValueError):
            research.sample(self.frame([0, 0]), 'gap5')
        with self.assertRaises(ValueError):
            research.sample(self.frame([0, 1]), 'random')

    def test_fixed_twelve_policies(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({x['id'] for x in configs}), 12)


if __name__ == '__main__':
    unittest.main()
