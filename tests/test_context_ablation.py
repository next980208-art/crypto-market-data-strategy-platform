import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts import search_context_ablation as research


class AblationTests(unittest.TestCase):
    def test_fixed_grid_and_disjoint_groups(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 18)
        self.assertEqual(len({c["id"] for c in configs}), 18)
        features = [f for group in research.GROUPS.values() for f in group]
        self.assertEqual(features, research.context_run.FEATURES)
        self.assertEqual(len(features), len(set(features)))
        for config in configs:
            self.assertEqual(config["risk_per_trade_pct"], .005)

    def test_feature_selection_does_not_include_other_groups(self):
        self.assertEqual(research.selected_features(["close"], "funding"), ["close", "public_last_funding"])
        with self.assertRaises(ValueError):
            research.selected_features(["public_last_funding"], "funding")

    def test_paired_delta_requires_matching_folds(self):
        def fold(n, value):
            return {"fold": n, **{cost: {"total_return_pct": value, "trades": 5} for cost, _ in research.consensus.COSTS}}
        values = research.paired_deltas([fold(1, 2)], {"folds": [fold(1, -1)]})
        self.assertEqual(len(values), 3)
        self.assertTrue(all(row["return_delta_pp"] == 3 for row in values))
        with self.assertRaises(ValueError):
            research.paired_deltas([fold(1, 2)], {"folds": [fold(2, 2)]})

    def test_audit_preserves_missing_funding_mark_price(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with closing(sqlite3.connect(root / "context.sqlite3")) as db, db:
                db.execute("CREATE TABLE observations(family TEXT, event_ms INTEGER, payload TEXT)")
                for n in (1, 2, 3):
                    db.execute("INSERT INTO observations VALUES(?,?,?)", (f"funding_{n}", 1, json.dumps({"markPrice": ""})))
            report = research.funding_audit(root)
            self.assertEqual(sum(r["missing_settlement_mark_price"] for r in report["folds"]), 3)
            self.assertFalse(report["funding_cashflows_modeled"])


if __name__ == "__main__":
    unittest.main()
