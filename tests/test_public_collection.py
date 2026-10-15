import asyncio
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.capture_public_raw_window import capture, encode_record
from scripts.run_public_collection_batch import commands


class PublicCollectionTests(unittest.TestCase):
    def test_raw_message_preserved_exactly(self):
        raw = '{ "data": {"e":"bookTicker", "b":"1.000"} }'
        kind, encoded = encode_record(raw, "2026-09-13T00:00:00Z")
        self.assertEqual(kind, "bookTicker")
        self.assertEqual(json.loads(encoded)["raw_message"], raw)

    def test_invalid_limits_do_not_start_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new"
            with self.assertRaises(ValueError):
                asyncio.run(capture(path, seconds=121))
            self.assertFalse(path.exists())

    def test_batch_does_not_start_services_and_rejects_long_range(self):
        tasks = commands(Path("out"), "2026-09-08", "2026-09-13")
        self.assertEqual(len(tasks), 4)
        self.assertIn("--no-kafka", tasks[0][1])
        with self.assertRaises(ValueError):
            commands(Path("out"), "2026-09-01", "2026-09-13")

    def test_capture_stops_on_message_limit_and_reads_back(self):
        class Socket:
            def __init__(self):
                self.events = iter(("aggTrade", "bookTicker", "depthUpdate"))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def recv(self):
                return json.dumps({"data": {"e": next(self.events)}})

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "window"
            with patch("scripts.capture_public_raw_window.websockets.connect", return_value=Socket()):
                report = asyncio.run(capture(path, seconds=1, max_messages=3))
            self.assertEqual(report["stop_reason"], "message_limit")
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["counts"], report["readback_counts"])
            self.assertEqual(sum(report["counts"].values()), 3)
            with gzip.open(path / "messages.jsonl.gz", "rt") as source:
                self.assertEqual(len(list(source)), 3)

    def test_byte_cap_stops_before_writing_oversized_message(self):
        class Socket:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def recv(self):
                return '{"data":{"e":"aggTrade"}}'

        with tempfile.TemporaryDirectory() as folder:
            with patch("scripts.capture_public_raw_window.websockets.connect", return_value=Socket()):
                report = asyncio.run(capture(Path(folder) / "window", seconds=1, max_bytes=1))
            self.assertEqual(report["stop_reason"], "byte_limit")
            self.assertEqual(report["uncompressed_bytes"], 0)
            self.assertEqual(report["status"], "partial")

    def test_plain_trade_fallback_preserves_trade_event_type(self):
        class Socket:
            def __init__(self):
                self.events = iter(("trade", "bookTicker", "depthUpdate"))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def recv(self):
                return json.dumps({"data": {"e": next(self.events)}})

        with tempfile.TemporaryDirectory() as folder:
            with patch("scripts.capture_public_raw_window.websockets.connect", return_value=Socket()):
                report = asyncio.run(capture(Path(folder) / "window", seconds=1, max_messages=3, trade_stream="trade"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["counts"]["trade"], 1)
            self.assertNotIn("aggTrade", report["counts"])


if __name__ == "__main__":
    unittest.main()
