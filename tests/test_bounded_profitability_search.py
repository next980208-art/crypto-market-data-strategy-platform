import unittest

import pandas as pd

from scripts.run_bounded_profitability_search import (
    DAY, MINUTE, build_folds, choose_candidate, configurations, policy_summary, purged_parts,
)


def metrics(trades=20, mean=0.1, total=2.0, drawdown=-5.0):
    return {"trades": trades, "avg_return_r": mean, "total_return_pct": total, "max_drawdown_pct": drawdown}


class BoundedSearchTests(unittest.TestCase):
    def test_fixed_small_configuration_grid(self):
        grid = configurations()
        self.assertEqual(6, len(grid))
        self.assertEqual(6, len({row['id'] for row in grid}))
        self.assertEqual(3, sum(row['include_trade'] for row in grid))

    def test_validation_does_not_enter_holdout(self):
        folds, holdout = build_folds(0, 95 * DAY)
        self.assertEqual(75 * DAY, holdout)
        self.assertEqual(3, len(folds))
        self.assertTrue(all(fold['validation_end'] <= holdout for fold in folds))
        for left, right in zip(folds, folds[1:]):
            self.assertEqual(left['validation_end'], right['validation_start'])

    def test_insufficient_reserved_period_rejected(self):
        with self.assertRaisesRegex(ValueError, '14 days'):
            build_folds(0, 80 * DAY)

    def test_purge_accounts_for_next_bar_entry(self):
        frame = pd.DataFrame({'timestamp': [0, MINUTE, 4 * MINUTE, 5 * MINUTE, 6 * MINUTE],
                              'entry_timestamp': [MINUTE, 2 * MINUTE, 5 * MINUTE, 6 * MINUTE, 7 * MINUTE],
                              'max_holding_minutes': [1, 2, 1, 2, 2]})
        train, validation = purged_parts(frame, {'train_start': 0, 'validation_start': 4 * MINUTE,
                                                'validation_end': 8 * MINUTE})
        self.assertEqual([0], train.timestamp.tolist())
        self.assertEqual([4 * MINUTE], validation.timestamp.tolist())

    def test_zero_trades_are_not_profitable_candidate(self):
        result = policy_summary([metrics(trades=0, mean=0, total=0)] * 3)
        self.assertFalse(result['validation_eligible'])

    def test_one_losing_fold_blocks_candidate(self):
        result = policy_summary([metrics(), metrics(), metrics(mean=-0.1, total=-2)])
        self.assertEqual(2, result['positive_folds'])
        self.assertFalse(result['validation_eligible'])

    def test_too_few_trades_or_excess_drawdown_blocks_candidate(self):
        self.assertFalse(policy_summary([metrics(trades=19)] * 3)['validation_eligible'])
        self.assertFalse(policy_summary([metrics(drawdown=-21)] * 3)['validation_eligible'])

    def test_stress_failure_cannot_be_selected(self):
        candidate = {**policy_summary([metrics()] * 3), 'stress_survived': False}
        self.assertIsNone(choose_candidate([candidate]))
        candidate['stress_survived'] = True
        self.assertEqual(candidate, choose_candidate([candidate]))


if __name__ == '__main__':
    unittest.main()
