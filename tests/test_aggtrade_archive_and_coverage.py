import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load_module("aggtrade_collector_for_archive_test", "10_aggtrade_collector.py")
archive = load_module("aggtrade_archive_collector_for_test", "aggtrade_archive_collector.py")
coverage = load_module("aggtrade_coverage_for_test", "scripts/report_aggtrade_coverage.py")
daily = load_module("daily_runner_for_archive_test", "daily_collection_runner.py")
backfill = load_module("aggtrade_archive_backfill_for_test", "aggtrade_archive_backfill_runner.py")


def archive_rows(start_ms, count=3):
    return [
        {
            "agg_trade_id": str(100 + index),
            "price": str(100.0 + index),
            "quantity": "0.1",
            "first_trade_id": str(1000 + index),
            "last_trade_id": str(1000 + index),
            "transact_time": str(start_ms + index * 1000),
            "is_buyer_maker": "true" if index % 2 == 0 else "false",
        }
        for index in range(count)
    ]


class AggTradeArchiveTests(unittest.TestCase):
    def test_archive_rows_are_aggregated_with_trade_id_continuity(self):
        start_ms = 1_700_000_000_000 - (1_700_000_000_000 % collector.MINUTE_MS)
        aggregates, stats, total = archive.aggregate_rows(
            archive_rows(start_ms), start_ms, start_ms + collector.MINUTE_MS, collector
        )
        self.assertEqual(3, total)
        self.assertEqual(100, stats["first_trade_id"])
        self.assertEqual(102, stats["last_trade_id"])
        self.assertEqual(3, aggregates[start_ms]["trade_count"])
        self.assertAlmostEqual(0.2, aggregates[start_ms]["taker_sell_volume"])

    def test_archive_trade_id_gap_is_rejected(self):
        start_ms = 1_700_000_000_000 - (1_700_000_000_000 % collector.MINUTE_MS)
        rows = archive_rows(start_ms)
        rows[1]["agg_trade_id"] = "105"
        with self.assertRaisesRegex(RuntimeError, "연속적이지"):
            archive.aggregate_rows(rows, start_ms, start_ms + collector.MINUTE_MS, collector)

    def test_old_daily_run_uses_archive_route(self):
        args = SimpleNamespace(
            symbol="BTC/USDT",
            timeframe="1m",
            trade_context_folder="trade_context_store_v2",
            minimum_aggtrade_training_days=90,
            max_aggtrade_pages=1000,
            max_total_aggtrade_pages=10000,
            aggtrade_window_minutes=15,
            feature_folder="feature_store_v2",
            context_folder="futures_context_store_v2",
        )
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        command = daily.command_for("aggtrade_archive", args, start, start + timedelta(days=1))
        self.assertIn("aggtrade_archive_collector.py", command)
        self.assertIn("2026-08-01", command)

    def test_route_prefers_rest_immediately_and_archive_after_publication_delay(self):
        self.assertEqual("rest", daily.preferred_aggtrade_route(timedelta(hours=1)))
        self.assertEqual("archive", daily.preferred_aggtrade_route(timedelta(hours=6)))

    def test_stored_project_path_accepts_windows_separator(self):
        resolved = daily.stored_project_path(
            "trade_context_store_v2\\market=usdm\\symbol=BTCUSDT\\file.parquet"
        )
        self.assertEqual(
            daily.PROJECT_ROOT
            / "trade_context_store_v2"
            / "market=usdm"
            / "symbol=BTCUSDT"
            / "file.parquet",
            resolved,
        )


class AggTradeCoverageTests(unittest.TestCase):
    def write_day(self, root: Path, day: datetime):
        start_ms = int(day.timestamp() * 1000)
        end_ms = int((day + timedelta(days=1)).timestamp() * 1000)
        context = collector.build_context({}, "BTC/USDT", start_ms, end_ms)
        validation = collector.validate_context(context, start_ms, end_ms)
        args = SimpleNamespace(
            symbol="BTC/USDT",
            context_folder=str(root),
            window_minutes=1440,
            max_pages=0,
            max_total_pages=0,
        )
        collector.write_context(
            context,
            args,
            day,
            day + timedelta(days=1),
            [],
            0,
            0,
            validation,
            marker_overrides={"source_type": "test_archive"},
        )

    def test_report_detects_gap_and_uses_longest_streak(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = datetime(2026, 1, 1, tzinfo=timezone.utc)
            self.write_day(root, first)
            self.write_day(root, first + timedelta(days=2))
            report = coverage.build_report(root, "BTC/USDT", minimum_days=2)
            self.assertEqual("gaps_or_invalid_days_detected", report["status"])
            self.assertEqual(["2026-01-02"], report["missing_dates_inside_observed_range"])
            self.assertEqual(1, report["longest_continuous_days"])
            self.assertFalse(report["model_training_allowed"])

    def test_report_opens_training_gate_for_continuous_days(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = datetime(2026, 1, 1, tzinfo=timezone.utc)
            self.write_day(root, first)
            self.write_day(root, first + timedelta(days=1))
            report = coverage.build_report(root, "BTCUSDT", minimum_days=2)
            self.assertEqual("healthy_ready", report["status"])
            self.assertEqual(2, report["longest_continuous_days"])
            self.assertTrue(report["model_training_allowed"])


class AggTradeArchiveBackfillTests(unittest.TestCase):
    def test_available_end_waits_for_archive_publication_delay(self):
        now = datetime(2026, 9, 4, 5, 0, tzinfo=timezone.utc)
        self.assertEqual(
            datetime(2026, 9, 3).date(),
            backfill.available_target_end(datetime(2026, 9, 4).date(), now, 6),
        )

    def test_pending_days_are_selected_newest_first(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = datetime(2026, 1, 1, tzinfo=timezone.utc)
            helper = AggTradeCoverageTests()
            helper.write_day(root, first + timedelta(days=2))
            selected, missing = backfill.pending_days(
                first.date(),
                (first + timedelta(days=3)).date(),
                str(root),
                "BTCUSDT",
                1,
            )
            self.assertEqual(2, missing)
            self.assertEqual([(first + timedelta(days=1)).date()], selected)


if __name__ == "__main__":
    unittest.main()
