import unittest

from scripts.collect_valuation_marks import normalize, MINUTE


class MarkCollectionTests(unittest.TestCase):
    def test_valid_and_gap_not_filled(self):
        rows = [[0, 100, 102, 99, 101, 0, MINUTE - 1],
                [2*MINUTE, 101, 102, 100, 101, 0, 3*MINUTE - 1]]
        self.assertEqual(len(normalize(rows, 0, 3*MINUTE)), 2)

    def test_invalid_and_duplicates_rejected(self):
        row = [0, 100, 102, 99, 101, 0, MINUTE - 1]
        for rows in ([], [row, row], [[0, 100, 98, 99, 101, 0, MINUTE - 1]],
                     [[0, 100, 102, 99, 101, 0, MINUTE - 2]]):
            with self.assertRaises(ValueError):
                normalize(rows, 0, MINUTE)


if __name__ == "__main__":
    unittest.main()
