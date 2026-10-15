import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECTOR = load("9_futures_context_collector.py", "timeframe_collector")
CHECKER = load("scripts/verify_futures_context_store.py", "timeframe_checker")


class ContextTimeframeTests(unittest.TestCase):
    def test_writer_preserves_minute_unit(self):
        with tempfile.TemporaryDirectory() as root:
            frame = pd.DataFrame({"mark_price": [100.0], "mark_price_status": ["observed"],
                                  "funding_rate_status": ["available"], "open_interest_status": ["available"]})
            path, marker, _ = COLLECTOR.write_context(
                frame, root, "BTC/USDT", "1m", datetime(2026, 9, 1, tzinfo=timezone.utc),
                datetime(2026, 9, 2, tzinfo=timezone.utc),
            )
            self.assertEqual("timeframe=1m", path.parent.name)
            self.assertIn("CONTEXT_1m_", path.name)
            self.assertEqual("1m", json.loads(marker.read_text(encoding="utf-8"))["timeframe"])

    def test_checker_uses_explicit_timeframe_for_legacy_uppercase_run_id(self):
        with tempfile.TemporaryDirectory() as root:
            marker_dir = Path(root) / "_markers"
            marker_dir.mkdir()
            (marker_dir / "_SUCCESS_BTCUSDT_USDM_CONTEXT_1M_20260901_20260902.json").write_text(
                json.dumps({"run_id": "BTCUSDT_USDM_CONTEXT_1M_20260901_20260902", "timeframe": "1m",
                            "rows": 1440, "context_schema_version": COLLECTOR.FUTURES_CONTEXT_SCHEMA_VERSION}),
                encoding="utf-8",
            )
            ranges, errors = CHECKER.read_matching_markers(SimpleNamespace(
                context_folder=root, symbol="BTCUSDT", market="usdm", timeframe="1m",
                expected_schema=COLLECTOR.FUTURES_CONTEXT_SCHEMA_VERSION,
            ))
            self.assertEqual([], errors)
            self.assertEqual(1, len(ranges))


if __name__ == "__main__":
    unittest.main()
