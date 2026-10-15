import unittest

from scripts.audit_research_data_usage import declarations, ranges


class ResearchUsageTests(unittest.TestCase):
    def test_ranges_preserve_gaps_and_deduplicate(self):
        self.assertEqual(ranges([120000, 0, 0, 60000, 240000]), [
            {"start_ms": 0, "end_exclusive_ms": 180000, "rows": 3},
            {"start_ms": 240000, "end_exclusive_ms": 300000, "rows": 1}])
        self.assertEqual(ranges([]), [])

    def test_declarations_not_inferred_from_arbitrary_values(self):
        self.assertEqual(declarations({"rows": 1000, "test_start": False, "test_end": True}), [])
        result = declarations({"folds": [{"validation_start": 1600000000000, "validation_end": 1600000060000}]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "validation_declared")
        self.assertEqual(result[0]["json_pointer"], "$.folds[0]")


if __name__ == "__main__":
    unittest.main()
