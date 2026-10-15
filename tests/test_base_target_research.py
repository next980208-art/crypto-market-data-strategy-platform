import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import search_base_targets as targets
from scripts import research_ledger


class BaseTargetTests(unittest.TestCase):
    def test_small_fixed_grid(self):
        configs = targets.configurations()
        self.assertEqual(len(configs), 6)
        self.assertEqual(len({x['id'] for x in configs}), 6)
        self.assertEqual({x['target_r'] for x in configs}, {1., 1.5})

    def test_training_window_matches_reference(self):
        fold = {'validation_start': 120*targets.search.DAY, 'validation_end': 150*targets.search.DAY}
        self.assertEqual(targets.training_spec(fold)['train_start'], 90*targets.search.DAY)
        self.assertEqual(targets.training_spec(fold)['validation_end'], fold['validation_end'])

    def test_ledger_keeps_errors_and_partial_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root/'results.json'
            targets.search.write_json(file, {'trials': [
                {'config': {'id': 'failed'}, 'status': 'error', 'error': 'test error'},
                {'config': {'id': 'partial'}, 'candidate': False,
                 'normal': {'positive_folds': 1, 'fold_returns_pct': [1, -1, -2]},
                 'stress': {'fold_returns_pct': [-1, -2, -3]},
                 'folds': [{'normal': {'trades': 21}}]*3, 'rejection_reasons': ['negative fold']}]})
            with patch.object(research_ledger.search, 'ROOT', root):
                text = research_ledger.render([file])
            self.assertIn('failed', text)
            self.assertIn('test error', text)
            self.assertIn('partial', text)
            self.assertIn('+1.000', text)
            self.assertIn('negative fold', text)


if __name__ == '__main__':
    unittest.main()
