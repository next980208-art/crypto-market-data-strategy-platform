import unittest

from scripts.verify_realtime_feature_continuity import analyze_features


def feature(minute, feature_id=None):
    return {
        "feature_id": feature_id or f"BTCUSDT:1m:{minute}",
        "event_time_ms": minute,
        "symbol": "BTCUSDT",
    }


class RealtimeFeatureContinuityTests(unittest.TestCase):
    def test_contiguous_unique_features_are_healthy(self):
        report = analyze_features([feature(60_000), feature(120_000), feature(180_000)], "BTCUSDT", 60_000)
        self.assertTrue(report["healthy"])
        self.assertEqual(3, report["unique_feature_ids"])

    def test_duplicate_feature_id_is_unhealthy(self):
        report = analyze_features([feature(60_000), feature(60_000)], "BTCUSDT", 0)
        self.assertFalse(report["healthy"])
        self.assertEqual(["BTCUSDT:1m:60000"], report["duplicate_feature_ids"])

    def test_minute_gap_is_unhealthy(self):
        report = analyze_features([feature(60_000), feature(180_000)], "BTCUSDT", 0)
        self.assertFalse(report["healthy"])
        self.assertEqual(1, report["minute_gap_count"])
        self.assertEqual(1, report["minute_gaps"][0]["gap_minutes"])

    def test_records_before_start_time_are_ignored(self):
        report = analyze_features([feature(60_000), feature(120_000), feature(180_000)], "BTCUSDT", 120_000)
        self.assertTrue(report["healthy"])
        self.assertEqual(2, report["records"])


if __name__ == "__main__":
    unittest.main()
