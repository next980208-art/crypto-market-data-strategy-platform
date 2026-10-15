import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kafka.structs import TopicPartition

from realtime.idempotency import IdempotencyStore, offsets_after_records
from realtime.risk_paper_service import audit_event
from scripts.verify_realtime_idempotency import evaluate_counts


class RealtimeIdempotencyTests(unittest.TestCase):
    def test_store_survives_reopen_and_ignores_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "idempotency.sqlite3"
            with IdempotencyStore(path) as store:
                self.assertEqual(2, store.mark_many("feature", ["f1", "f1", "f2"]))
                self.assertTrue(store.contains("feature", "f1"))
                self.assertEqual(2, store.count("feature"))
            with IdempotencyStore(path) as reopened:
                self.assertTrue(reopened.contains("feature", "f1"))
                self.assertEqual(0, reopened.mark_many("feature", ["f1", "f2"]))
                self.assertEqual(2, reopened.count("feature"))

    def test_offsets_commit_after_the_last_record_in_each_partition(self):
        first = TopicPartition("features", 0)
        second = TopicPartition("features", 1)
        offsets = offsets_after_records(
            {
                first: [SimpleNamespace(offset=10), SimpleNamespace(offset=12)],
                second: [SimpleNamespace(offset=7)],
            }
        )
        self.assertEqual(13, offsets[first].offset)
        self.assertEqual(8, offsets[second].offset)

    def test_audit_id_is_deterministic_for_signal_replays(self):
        signal = {
            "signal_id": "feature-1:blocked",
            "symbol": "BTCUSDT",
            "market": "usdm",
            "timeframe": "1m",
            "event_time_ms": 1_780_000_000_000,
        }
        first = audit_event(signal, "rejected", "test")
        second = audit_event(signal, "rejected", "test")
        self.assertEqual("feature-1:blocked:risk_audit", first["audit_id"])
        self.assertEqual(first["audit_id"], second["audit_id"])

    def test_probe_requires_exactly_one_committed_output(self):
        counts = {
            "inference_signals": 1,
            "inference_risk_audits": 1,
            "direct_risk_audits": 1,
            "probe_paper_orders": 0,
        }
        inference = {"counters": {"duplicate_features_skipped": 2}}
        risk = {"counters": {"duplicate_signals_skipped": 2}}
        self.assertTrue(
            evaluate_counts(
                counts,
                inference,
                risk,
                minimum_inference_skips=2,
                minimum_risk_skips=2,
            )
        )
        counts["inference_signals"] = 2
        self.assertFalse(evaluate_counts(counts, inference, risk))


if __name__ == "__main__":
    unittest.main()
