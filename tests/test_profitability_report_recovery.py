import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import finalize_profitability_search as recovery
from scripts import run_bounded_profitability_search as search


class ProfitabilityReportRecoveryTests(unittest.TestCase):
    def test_atomic_write_retries_temporary_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            original = Path.replace
            attempts = []

            def replace(source, target):
                attempts.append(target)
                if len(attempts) == 1:
                    raise PermissionError("temporary sharing violation")
                return original(source, target)

            with patch.object(Path, "replace", replace), patch.object(search.time, "sleep"):
                search.write_json(path, {"status": "completed"})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(path.read_text())["status"], "completed")

    def test_permanent_denial_is_bounded_and_preserves_old_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            search.write_json(path, {"status": "old"})
            with patch.object(Path, "replace", side_effect=PermissionError("denied")) as replace:
                with patch.object(search.time, "sleep"), self.assertRaises(PermissionError):
                    search.write_json(path, {"status": "new"})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(json.loads(path.read_text())["status"], "old")

    def fixture(self, folder, completed=True):
        dataset = folder / "data.bin"
        dataset.write_bytes(b"fixed dataset")
        config = {"id": "example", "include_trade": False}
        protocol = {"dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    "code_sha256": {}, "configurations": [config], "folds": [{}, {}, {}],
                    "thresholds": [.1], "margins": [0]}
        metrics = [{"trades": 25, "avg_return_r": -.1, "total_return_pct": -5, "max_drawdown_pct": -5}]*3
        candidate = {"trial_id": "example", "threshold": .1, "margin": 0,
                     **search.policy_summary(metrics), "stress_survived": False}
        details = {"status": "completed" if completed else "failed", "config": config,
                   "folds": [{}, {}, {}], "candidates": [candidate], "test_evaluated": False,
                   "real_trading_enabled": False, "registry_modified": False,
                   "finished_utc": "2026-09-10T00:00:10+00:00"}
        report = {"status": "running", "started_utc": "2026-09-10T00:00:00+00:00"}
        for name, data in [("protocol", protocol), ("example", details), ("search", report)]:
            search.write_json(folder / f"{name}.json", data)

    def test_recovery_uses_finished_trials_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            report = recovery.finalize(folder)
            self.assertEqual(report["status"], "completed")
            self.assertIsNone(report["validation_candidate"])
            self.assertEqual(report["elapsed_seconds"], 10)
            self.assertFalse(report["training_rerun"])
            self.assertTrue((folder / "REPORT.md").exists())
            with self.assertRaises(ValueError):
                recovery.finalize(folder)

    def test_incomplete_trial_does_not_become_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder, completed=False)
            with self.assertRaises(ValueError):
                recovery.finalize(folder)
            self.assertEqual(json.loads((folder / "search.json").read_text())["status"], "running")


if __name__ == "__main__":
    unittest.main()
