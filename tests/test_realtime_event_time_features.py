import unittest

from flink_jobs.realtime_kafka_feature_job import (
    apply_event,
    fresh_state,
    is_too_late,
    minute_start,
    window_deadline,
)


def trade(event_time_ms, price, quantity=1.0):
    return {
        "event_type": "agg_trade",
        "event_time_ms": event_time_ms,
        "payload": {
            "price": str(price),
            "quantity": str(quantity),
            "buyer_is_maker": False,
        },
    }


class RealtimeEventTimeFeatureTests(unittest.TestCase):
    def test_window_deadline_includes_allowed_lateness(self):
        minute = minute_start(125_123)
        self.assertEqual(190_000, window_deadline(minute, 10_000))
        self.assertFalse(is_too_late(minute, 189_999, 10_000))
        self.assertTrue(is_too_late(minute, 190_000, 10_000))

    def test_out_of_order_trades_keep_event_time_open_and_close(self):
        minute = 120_000
        state = fresh_state(minute)

        apply_event(state, trade(minute + 50_000, 105.0, 2.0))
        apply_event(state, trade(minute + 10_000, 100.0, 1.0))
        apply_event(state, trade(minute + 30_000, 110.0, 3.0))

        self.assertEqual(100.0, state["open"])
        self.assertEqual(105.0, state["close"])
        self.assertEqual(110.0, state["high"])
        self.assertEqual(100.0, state["low"])
        self.assertEqual(6.0, state["volume"])
        self.assertEqual(3, state["trade_count"])

    def test_older_book_event_does_not_overwrite_latest_value(self):
        state = fresh_state(120_000)
        newer = {
            "event_type": "book_ticker",
            "event_time_ms": 170_000,
            "payload": {"bid_price": "105", "ask_price": "106"},
        }
        older = {
            "event_type": "book_ticker",
            "event_time_ms": 130_000,
            "payload": {"bid_price": "99", "ask_price": "100"},
        }

        apply_event(state, newer)
        apply_event(state, older)

        self.assertEqual(105.0, state["bid_price"])
        self.assertEqual(106.0, state["ask_price"])


if __name__ == "__main__":
    unittest.main()
