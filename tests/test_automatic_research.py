import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import automatic_research as runner
from scripts import automatic_research_worker as worker


class AutomaticResearchTests(unittest.TestCase):
    def state(self):
        return {"status": "ready", "charged_seconds": 0, "tasks": {t: {"status": "pending", "attempts": []} for t in runner.TASKS},
                "candidate": None, "model_approved": False, "orders_sent": 0}

    def plan(self):
        return {"limits": {"total_seconds": 60, "task_seconds": 10, "max_attempts_per_task": 2,
                           "memory_mib": 2048, "output_mib": 64, "minimum_free_gib": 0}}

    def test_live_memory_monitor(self):
        self.assertGreater(runner.resident_bytes(os.getpid()), 0)

    def test_monitor_terminates_and_waits_for_timed_out_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = runner.monitor([sys.executable, "-c", "import time; time.sleep(10)"], root, root, .1, self.plan()["limits"])
            self.assertEqual(result["stop_reason"], "task_time_limit")
            self.assertNotEqual(result["exit_code"], 0)
            self.assertLess(result["elapsed_seconds"], 5)

    def test_resource_limits(self):
        limits = self.plan()["limits"]
        self.assertIsNone(runner.limit_reason(0, 10, 1, 1, 1, limits))
        self.assertEqual(runner.limit_reason(10, 10, 0, 0, 0, limits), "task_time_limit")
        self.assertEqual(runner.limit_reason(0, 10, 2048*1024**2+1, 0, 0, limits), "worker_memory_limit")
        self.assertEqual(runner.limit_reason(0, 10, 0, 64*1024**2+1, 0, limits), "output_size_limit")
        limits["minimum_free_gib"] = 5
        self.assertEqual(runner.limit_reason(0, 10, 0, 0, 1, limits), "free_disk_limit")

    def test_completed_tasks_are_skipped(self):
        state = self.state()
        self.assertEqual(runner.next_task(state), "audit")
        state["tasks"]["audit"]["status"] = "completed"
        self.assertEqual(runner.next_task(state), "prepare_1")

    def test_data_block_stops_before_any_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def monitor(command, root, folder, allowed, limits):
                runner.search.write_json(folder/"result.json", {"status": "blocked", "reason": "bad_data"})
                return {"exit_code": 0, "stop_reason": None, "elapsed_seconds": .1, "peak_worker_rss_bytes": 1}
            with patch.object(runner, "verify_resume"), patch.object(runner, "monitor", side_effect=monitor) as mocked:
                state = runner.execute(root, self.plan(), self.state())
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(state["status"], "blocked")
            self.assertEqual(state["tasks"]["prepare_1"]["status"], "pending")
            self.assertAlmostEqual(state["charged_seconds"], .1)

    def test_failure_resume_preserves_attempts_and_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            measured = {"exit_code": 1, "stop_reason": None, "elapsed_seconds": 2, "peak_worker_rss_bytes": 1}
            with patch.object(runner, "verify_resume"), patch.object(runner, "monitor", return_value=measured) as mocked:
                state = runner.execute(root, self.plan(), self.state())
                state = runner.execute(root, self.plan(), state)
                state = runner.execute(root, self.plan(), state)
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(state["charged_seconds"], 4)
            self.assertEqual(state["stop_reason"], "attempt_limit_audit")
            self.assertTrue((root/"audit/attempt_1").exists())
            self.assertTrue((root/"audit/attempt_2").exists())

    def test_crash_reservation_cannot_reset_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            state = self.state()
            state["charged_seconds"] = 60
            state["tasks"]["audit"] = {"status": "running", "attempts": [{"status": "running", "reserved_seconds": 60}]}
            with patch.object(runner, "verify_resume"), patch.object(runner, "monitor") as monitor:
                state = runner.execute(Path(temp), self.plan(), state)
            monitor.assert_not_called()
            self.assertEqual(state["charged_seconds"], 60)
            self.assertEqual(state["status"], "budget_exhausted")

    def test_resume_rejects_modified_completed_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("dataset", "reference", "protocol.json", "artifact"):
                (root/name).write_text("original", encoding="utf-8")
            plan = {"dataset": str(root/"dataset"), "dataset_sha256": runner.checksum(root/"dataset"),
                    "reference_trial": str(root/"reference"), "reference_sha256": runner.checksum(root/"reference"),
                    "code_sha256": {}, "packages": {}}
            state = {"protocol_sha256": runner.checksum(root/"protocol.json"),
                     "tasks": {"audit": {"status": "completed", "artifact_sha256": {"artifact": runner.checksum(root/"artifact")}}}}
            runner.verify_resume(root, plan, state)
            (root/"artifact").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.verify_resume(root, plan, state)

    def test_lock_released_after_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):
                with runner.exclusive_lock(Path(temp)):
                    raise RuntimeError("interrupted")
            with runner.exclusive_lock(Path(temp)):
                pass

    def test_concurrent_runner_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with runner.exclusive_lock(Path(temp)):
                with self.assertRaises(OSError):
                    with runner.exclusive_lock(Path(temp)):
                        pass

    def test_fixed_schedule_excludes_final_year(self):
        folds = worker.make_folds(0, 5*365*worker.search.DAY)
        self.assertEqual([f["start"]//worker.search.DAY for f in folds], [0, 365, 730])
        self.assertTrue(all(f["validation_end"]-f["validation_start"] == 30*worker.search.DAY for f in folds))
        with self.assertRaises(worker.Blocked):
            worker.make_folds(0, 100*worker.search.DAY)

    def test_rebuilt_label_parity_and_source_preservation(self):
        n = 750
        t = np.arange(n)*60000
        price = 100+np.sin(np.arange(n)/9)
        raw = pd.DataFrame({"timestamp": t, "datetime_utc": pd.to_datetime(t, unit="ms", utc=True).astype(str),
                            "symbol": "BTCUSDT", "market": "usdm", "timeframe": "1m",
                            "open": price, "high": price+.3, "low": price-.3, "close": price+.01, "volume": 10.})
        labeler = worker.search.load("4_triple_barrier_labeler.py", "auto_test_labeler")
        labels = labeler.add_labels(raw, [1.0], .005, 240, 10, 128, 4)
        columns = [c for c in labels if c.startswith(("long_", "short_"))]
        raw = raw.merge(labels[["timestamp"]+columns], on="timestamp", how="left")
        for column in ("long_tp_1_0r_outcome", "short_tp_1_0r_outcome"):
            raw[column] = raw[column].astype("category")
        copy = raw.copy(deep=True)
        result = worker.rebuild_labels(raw)
        pd.testing.assert_frame_equal(copy, raw)
        self.assertIn("long_tp_1_5r_return_r_net", result)
        self.assertEqual(len(result), n-240)
        raw.loc[0, "long_tp_1_0r_return_r_net"] += .2
        with self.assertRaises(worker.Blocked):
            worker.rebuild_labels(raw)


if __name__ == "__main__":
    unittest.main()
