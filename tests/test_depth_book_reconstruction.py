import unittest

from realtime.depth_book import DepthSequenceGap, LocalDepthBook
from realtime.kafka_feature_worker import MinuteState, update_depth
from scripts.verify_depth_kafka_messages import validate_event


def snapshot(last_update_id=100):
    return {
        "lastUpdateId": last_update_id,
        "bids": [["100.0", "2.0"], ["99.0", "3.0"], ["98.0", "4.0"]],
        "asks": [["101.0", "2.5"], ["102.0", "3.5"], ["103.0", "4.5"]],
    }


def event(first, final, previous, bids=None, asks=None):
    return {
        "U": first,
        "u": final,
        "pu": previous,
        "b": bids or [],
        "a": asks or [],
    }


class LocalDepthBookTests(unittest.TestCase):
    def test_snapshot_bridge_updates_and_removes_levels(self):
        book = LocalDepthBook("BTCUSDT", output_levels=2)
        book.load_snapshot(snapshot())

        payload = book.apply_event(
            event(99, 101, 98, bids=[["100.0", "0"], ["100.5", "7"]], asks=[["101.0", "5"]])
        )

        self.assertTrue(book.synced)
        self.assertEqual(101, book.last_update_id)
        self.assertEqual([["100.5", "7"], ["99.0", "3.0"]], payload["top_bids"])
        self.assertEqual([["101.0", "5"], ["102.0", "3.5"]], payload["top_asks"])
        self.assertEqual("synchronized", payload["sequence_status"])

    def test_events_older_than_snapshot_are_ignored(self):
        book = LocalDepthBook("BTCUSDT")
        book.load_snapshot(snapshot(200))

        self.assertIsNone(book.apply_event(event(190, 199, 189)))
        self.assertFalse(book.synced)
        self.assertEqual(1, book.stats["stale_before_bridge"])

    def test_initial_event_must_bridge_snapshot(self):
        book = LocalDepthBook("BTCUSDT")
        book.load_snapshot(snapshot(100))

        with self.assertRaises(DepthSequenceGap):
            book.apply_event(event(101, 102, 100))

    def test_subsequent_pu_must_match_previous_u(self):
        book = LocalDepthBook("BTCUSDT")
        book.load_snapshot(snapshot(100))
        book.apply_event(event(99, 101, 98))

        with self.assertRaises(DepthSequenceGap):
            book.apply_event(event(102, 103, 99))

    def test_reset_requires_new_snapshot(self):
        book = LocalDepthBook("BTCUSDT")
        book.load_snapshot(snapshot(100))
        book.apply_event(event(100, 101, 99))
        book.reset("test")

        self.assertFalse(book.synced)
        self.assertIsNone(book.snapshot_update_id)
        self.assertEqual(1, book.stats["reset_test"])

    def test_kafka_payload_contract_accepts_contiguous_events(self):
        book = LocalDepthBook("BTCUSDT")
        book.load_snapshot(snapshot(100))
        first = book.apply_event(event(100, 101, 99))
        second = book.apply_event(event(102, 103, 101, bids=[["99.0", "8"]]))

        first_u, first_errors = validate_event(
            {"event_type": "depth_delta", "symbol": "BTCUSDT", "payload": first},
            "BTCUSDT",
            None,
        )
        second_u, second_errors = validate_event(
            {"event_type": "depth_delta", "symbol": "BTCUSDT", "payload": second},
            "BTCUSDT",
            first_u,
        )

        self.assertEqual([], first_errors)
        self.assertEqual([], second_errors)
        self.assertEqual(103, second_u)

    def test_feature_worker_uses_only_synchronized_top_levels(self):
        state = MinuteState()
        update_depth(state, {"top_bids": [["100", "9"]], "top_asks": [["101", "1"]]})
        self.assertIsNone(state.book_imbalance_top5)

        update_depth(
            state,
            {
                "sequence_status": "synchronized",
                "top_bids": [["100", "9"]],
                "top_asks": [["101", "1"]],
            },
        )
        self.assertAlmostEqual(0.8, state.book_imbalance_top5)


if __name__ == "__main__":
    unittest.main()
