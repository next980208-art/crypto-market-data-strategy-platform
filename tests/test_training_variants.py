import unittest

import numpy as np
import pandas as pd

from scripts import search_training_variants as research


class TrainingVariantsTests(unittest.TestCase):
    def test_fixed_grid(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({x['id'] for x in configs}), 12)
        self.assertTrue(all(x['target_r'] == 1.5 for x in configs))

    def test_quality_and_empty_constant(self):
        result = research.quality([.3, .4], [-.3, -.2])
        self.assertAlmostEqual(result['bias_r'], .6)
        self.assertAlmostEqual(result['rank_correlation'], 1.)
        self.assertIsNone(research.quality([1, 1], [0, 1])['rank_correlation'])
        self.assertEqual(research.quality([], [])['rows'], 0)
        with self.assertRaises(ValueError):
            research.quality([np.nan], [1])
        with self.assertRaises(ValueError):
            research.quality([1, 2], [1])

    def test_diagnostic_bins_ties_and_time_sampling(self):
        times = np.arange(7)*research.search.MINUTE
        frame = pd.DataFrame({'timestamp': times, 'entry_timestamp': times+research.search.MINUTE,
                              'max_holding_minutes': 240, 'stop_distance_pct': .005,
                              'long_tp_1_5r_return_r_net': .8, 'short_tp_1_5r_return_r_net': -1.2})
        predictions = pd.DataFrame({'timestamp': times, 'long': [-.1, 0, .05, .15, .3, .4, 0],
                                    'short': [-.2, -.1, 0, 0, 0, 0, 0]})
        report = research.diagnostic(frame, predictions)
        self.assertEqual(report['all_overlapping_signals']['overall']['rows'], 6)
        self.assertEqual([x['rows'] for x in report['all_overlapping_signals']['bins']], [1, 1, 1, 1, 2])
        self.assertEqual(report['closed_240m_sample']['overall']['rows'], 1)
        frame['long_tp_1_5r_return_r_net'] = -5.
        changed = research.diagnostic(frame, predictions)
        self.assertEqual([x['rows'] for x in changed['all_overlapping_signals']['bins']], [1, 1, 1, 1, 2])
        predictions['timestamp'] += 1
        with self.assertRaises(AssertionError):
            research.diagnostic(frame, predictions)

    def test_model_config_is_fixed(self):
        old = {'base_config': {'params': {'max_depth': 3, 'n_estimators': 200}}}
        ridge = research.make_model('ridge30', old, 42)
        self.assertEqual(ridge.steps[-1][1].alpha, 100.)
        shallow = research.make_model('shallow90', old, 42)
        self.assertEqual(shallow.max_depth, 2)
        self.assertEqual(shallow.min_child_weight, 100)
        self.assertEqual(old['base_config']['params']['max_depth'], 3)


if __name__ == '__main__':
    unittest.main()
