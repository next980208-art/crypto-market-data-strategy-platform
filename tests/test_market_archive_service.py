import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from realtime.market_archive_service import MarketArchive, run


def event(offset, value=b'{"event_id":"x"}'):
    return SimpleNamespace(topic="book", partition=0, offset=offset,
                           timestamp=1000, key=b"BTCUSDT", value=value)


class MarketArchiveTests(unittest.TestCase):
    def test_storage_budget_stops_before_consuming_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(archive_file=Path(root) / "archive.sqlite3",
                                   report_json=str(Path(root) / "report.json"),
                                   bootstrap_servers="unused", consumer_group="test",
                                   duration_seconds=0, max_archive_bytes=1)
            consumer = MagicMock()
            consumer.partitions_for_topic.return_value = {0}
            consumer.beginning_offsets.side_effect = lambda partitions: {tp: 0 for tp in partitions}
            consumer.end_offsets.side_effect = lambda partitions: {tp: 0 for tp in partitions}
            with patch("realtime.market_archive_service.KafkaConsumer", return_value=consumer), \
                 patch("realtime.market_archive_service.signal.signal"), \
                 patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "storage budget"):
                    run(args)
            self.assertFalse(consumer.poll.called)
            report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
            self.assertEqual("failed", report["status"])
            self.assertEqual(0, report["inserted_this_run"])

    def test_replay_after_reopen_keeps_one_row(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "archive.sqlite3"
            store = MarketArchive(path)
            self.assertEqual(1, store.save([event(42)]))
            store.close()
            store = MarketArchive(path)
            self.assertEqual(0, store.save([event(42)]))
            self.assertEqual(43, store.position("book", 0))
            self.assertEqual(1, store.summary()["stored_rows"])
            store.close()

    def test_batch_rolls_back_on_gap(self):
        with tempfile.TemporaryDirectory() as root:
            store = MarketArchive(Path(root) / "archive.sqlite3")
            store.save([event(5)])
            with self.assertRaisesRegex(RuntimeError, "gap"):
                store.save([event(6), event(8)])
            self.assertEqual(6, store.position("book", 0))
            self.assertEqual(1, store.summary()["stored_rows"])
            store.close()

    def test_recreated_topic_conflict_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            store = MarketArchive(Path(root) / "archive.sqlite3")
            store.save([event(0)])
            with self.assertRaisesRegex(RuntimeError, "Conflicting"):
                store.save([event(0, b"different")])
            store.close()
