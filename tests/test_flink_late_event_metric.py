import unittest

from scripts.verify_flink_late_event_metric import evaluate_result


class FlinkLateEventMetricTests(unittest.TestCase):
    def test_one_drop_and_no_output_is_healthy(self):
        report = evaluate_result(
            baseline_dropped=3,
            final_dropped=4,
            baseline_accepted=100,
            final_accepted=102,
            output_features=0,
        )
        self.assertTrue(report["healthy"])

    def test_missing_drop_is_unhealthy(self):
        report = evaluate_result(
            baseline_dropped=3,
            final_dropped=3,
            baseline_accepted=100,
            final_accepted=102,
            output_features=0,
        )
        self.assertFalse(report["healthy"])

    def test_probe_feature_output_is_unhealthy(self):
        report = evaluate_result(
            baseline_dropped=3,
            final_dropped=4,
            baseline_accepted=100,
            final_accepted=100,
            output_features=1,
        )
        self.assertFalse(report["healthy"])


if __name__ == "__main__":
    unittest.main()
