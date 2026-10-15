from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from demo.run_end_to_end_demo import verify_committed_snapshot
from demo.serving.feature_store_api import (
    build_dataset_view,
    frame_records,
    load_feature_frame,
    read_run_report,
)


class DemoServingTest(unittest.TestCase):
    def make_parquet(self, folder: Path) -> Path:
        path = folder / "features.parquet"
        pd.DataFrame(
            {
                "timestamp": [2, 1],
                "datetime_utc": pd.to_datetime(
                    ["2026-01-01T00:01:00Z", "2026-01-01T00:00:00Z"], utc=True
                ),
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "market": ["usdm", "usdm"],
                "timeframe": ["1m", "1m"],
                "open": [101.0, 100.0],
                "high": [102.0, 101.0],
                "low": [100.0, 99.0],
                "close": [101.5, 100.5],
                "volume": [11.0, 10.0],
                "ma_5": [101.0, 100.5],
                "return_1m": [0.01, 0.0],
            }
        ).to_parquet(path, index=False)
        return path

    def test_build_dataset_view_reads_and_sorts_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_parquet(Path(temporary))
            view = build_dataset_view(path)

            self.assertEqual("healthy", view["status"])
            self.assertEqual(2, view["row_count"])
            self.assertEqual(0, view["duplicate_timestamps"])
            self.assertEqual(0, view["missing_required_values"])
            self.assertEqual(101.5, view["latest_close"])
            self.assertEqual(2, view["latest_rows"][-1]["timestamp"])

    def test_missing_required_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_parquet(Path(temporary))
            frame = pd.read_parquet(path).drop(columns=["close"])
            frame.to_parquet(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_feature_frame(path)

    def test_frame_records_limits_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = load_feature_frame(self.make_parquet(Path(temporary)))
            rows = frame_records(frame, 1)
            self.assertEqual(1, len(rows))
            self.assertEqual(2, rows[0]["timestamp"])

    def test_dataset_view_does_not_expose_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_parquet(Path(temporary))
            view = build_dataset_view(path)
            self.assertEqual("features.parquet", view["file"])

    def test_run_report_missing_and_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            self.assertEqual("not_available", read_run_report(path)["status"])
            path.write_text(json.dumps({"status": "success"}), encoding="utf-8")
            self.assertEqual("success", read_run_report(path)["status"])

    def test_committed_demo_snapshot_is_self_consistent(self) -> None:
        summary = verify_committed_snapshot()
        self.assertEqual("passed", summary["status"])
        self.assertEqual(1_000, summary["expected_count"])
        self.assertTrue(all(summary["checks"].values()))


if __name__ == "__main__":
    unittest.main()
