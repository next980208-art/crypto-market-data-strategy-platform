import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts import validate_consensus as research


class ConsensusTests(unittest.TestCase):
    def test_grid_is_fixed(self):
        configs = research.configurations()
        self.assertEqual(len(configs), 6)
        self.assertEqual(len({c["id"] for c in configs}), 6)
        self.assertTrue(all(c["risk_per_trade_pct"] == .005 for c in configs))

    def test_agreement_requires_both_thresholds_and_direction(self):
        frame = pd.DataFrame({"timestamp": range(4), "future_label": [1, 0, -1, 1]})
        one = pd.DataFrame({"timestamp": range(4), "long": [.5, .5, .5, .5], "short": [0., 0., 0., 0.]})
        two = pd.DataFrame({"timestamp": range(4), "long": [.4, .04, 0., .5], "short": [0., 0., .4, .5]})
        config = {"mode": "agreement", "threshold_r": .05}
        actual = research.combine(frame, one, two, config)
        self.assertEqual(actual.pred_action.tolist(), ["long", "no_trade", "no_trade", "no_trade"])
        frame["future_label"] *= -100
        self.assertEqual(research.combine(frame, one, two, config).pred_action.tolist(), actual.pred_action.tolist())
        config["mode"] = "mean"
        self.assertEqual(research.combine(frame, one, two, config).pred_action.tolist(), ["long"]*4)

    def test_alignment_and_nonfinite_are_rejected(self):
        frame = pd.DataFrame({"timestamp": [1]})
        values = pd.DataFrame({"timestamp": [2], "long": [.5], "short": [0.]})
        with self.assertRaises(AssertionError):
            research.combine(frame, values, values, {"mode": "mean", "threshold_r": .05})
        values["timestamp"] = 1
        values["long"] = np.nan
        with self.assertRaises(ValueError):
            research.combine(frame, values, values, {"mode": "mean", "threshold_r": .05})

    def test_weekly_includes_empty_days_and_trailing_partial_week(self):
        start = pd.Timestamp("2024-01-01", tz="UTC").value//1000000
        trades = pd.DataFrame({"exit_time_utc": ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", "2024-01-30T01:00:00Z"],
                               "return_pct": [.01, -.02, .03]})
        result = research.weekly_results(trades, start, start+30*research.search.DAY)
        self.assertEqual([b["days"] for b in result["blocks"]], [7, 7, 7, 7, 2])
        self.assertEqual(result["positive_complete_weeks"], 0)
        self.assertFalse(result["criterion_passed"])
        self.assertAlmostEqual(result["total_return_pct"], (1.01*.98*1.03-1)*100)
        self.assertGreater(result["blocks"][-1]["return_pct"], 0)

    def test_empty_trades_never_pass_and_outside_dates_fail(self):
        start = pd.Timestamp("2024-01-01", tz="UTC").value//1000000
        result = research.weekly_results(pd.DataFrame(), start, start+30*research.search.DAY)
        self.assertFalse(result["criterion_passed"])
        self.assertEqual(result["total_return_pct"], 0)
        trades = pd.DataFrame({"exit_time_utc": ["2024-02-01T00:00:00Z"], "return_pct": [.01]})
        with self.assertRaises(ValueError):
            research.weekly_results(trades, start, start+30*research.search.DAY)

    def test_all_gates_are_required(self):
        metrics = {"trades": 20, "avg_return_r": .1, "total_return_pct": 1., "max_drawdown_pct": -1.}
        folds = [{"fold": n, **{k: metrics.copy() for k, _ in research.COSTS}} for n in (1, 2, 3)]
        diagnostics = [{"fold": n, "cost_case": k, "weekly": {"criterion_passed": True}} for n in (1, 2, 3) for k, _ in research.COSTS]
        self.assertTrue(research.summarize({}, folds, diagnostics)["candidate"])
        diagnostics[-1]["weekly"]["criterion_passed"] = False
        self.assertFalse(research.summarize({}, folds, diagnostics)["candidate"])
        diagnostics[-1]["weekly"]["criterion_passed"] = True
        folds[0]["severe"]["avg_return_r"] = -.1
        self.assertFalse(research.summarize({}, folds, diagnostics)["candidate"])

    def test_progress_keeps_errors_and_pending(self):
        with tempfile.TemporaryDirectory(dir=research.search.ROOT) as folder:
            root = Path(folder)
            (root/"trials").mkdir()
            (root/"trades").mkdir()
            rows = [{"config": {"id": "failed"}, "status": "error", "error": "test failure"},
                    {"config": {"id": "pending"}, "status": "pending"}]
            research.save_progress(root, rows, "running")
            self.assertEqual(research.prior.read(root/"results.json")["trials"], rows)
            self.assertIn("test failure", (root/"LEDGER.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
