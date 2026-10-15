import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts import fixed_observation_dashboard as dashboard


class FixedObservationDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=dashboard.PROJECT_ROOT)
        self.root = Path(self.temp.name)
        protocol = {
            "symbol": "BTCUSDT", "timeframe": "15m", "rule_id": "fixed", "rule": "test rule",
            "observation_start_utc": "2026-09-22T11:00:00+00:00",
            "review_after_utc": "2026-12-21T11:00:00+00:00",
            "minimum_observation_days": 90, "minimum_monthly_trades": 20,
            "weekly_gate": "three positive weeks", "storage_limit_bytes": 1024,
            "risk_per_trade_pct": .005, "stop_distance_pct": .01,
            "take_profit_r": 1.5, "max_holding_minutes": 1440,
        }
        state = {"status": "collecting", "last_run_utc": "2026-09-24T05:39:01+00:00",
                 "runs": [{"exchange_server_ms": 1, "observed_days": 1.5, "status": "collecting"}]}
        metrics = {"signals": 5, "long_signals": 5, "short_signals": 0, "trades": 2,
                   "win_rate_pct": 0, "avg_return_r": -1, "total_return_pct": -1,
                   "max_drawdown_pct": -1}
        report = {"status": "collecting", "decision": "no_trade", "orders_sent": 0,
                  "credentials_used": False, "observed_days": 1.5, "contract_bars": 200,
                  "mark_bars": 200, "funding_events": 5, "mature_signal_rows": 50,
                  "database_bytes": 256,
                  "results": {cost: {"metrics": metrics} for cost in dashboard.COST_CASES}}
        for name, value in (("protocol.json", protocol), ("state.json", state), ("latest_report.json", report)):
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        weekly = {
            "generated_utc": "2026-09-24T06:00:00+00:00",
            "quality": {"status": "healthy", "warning_count": 0, "critical_count": 0, "checks": []},
            "weekly": {"complete_weeks": 0, "positive_weeks": 0, "failed_weeks": 0,
                       "current_week": {"week": 1, "status": "collecting"}, "weeks": [],
                       "four_week_blocks": []},
            "monthly": {"periods": []},
            "review": {"status": "collecting", "decision": "no_trade"},
        }
        (self.root / "weekly_evaluation.json").write_text(json.dumps(weekly), encoding="utf-8")
        pd.DataFrame({"timestamp": [0, 900000], "close": [100.0, 101.0]}).to_parquet(
            self.root / "bars_15m.parquet", index=False)
        pd.DataFrame({"entry_time_utc": ["2026-09-22T14:00:00+00:00"], "equity": [.995]}).to_parquet(
            self.root / "current_normal_trades.parquet", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_summary_reads_current_artifacts(self):
        value = dashboard.build_summary(self.root)
        self.assertEqual(value["decision"], "no_trade")
        self.assertEqual(value["orders_sent"], 0)
        self.assertEqual(value["costs"]["normal"]["trades"], 2)
        self.assertEqual(value["market"]["latest_close"], 101.0)
        self.assertAlmostEqual(value["progress_pct"], 100 * 1.5 / 90)
        self.assertEqual(value["monitoring"]["quality_status"], "healthy")
        self.assertEqual(value["monitoring"]["current_week"]["week"], 1)

    def test_table_allowlist_and_limit(self):
        rows = dashboard.load_table(self.root, "normal", "trades", 1)
        self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError):
            dashboard.load_table(self.root, "../../outside", "trades")

    def test_root_must_stay_inside_project(self):
        with self.assertRaises(ValueError):
            dashboard.validate_root(Path(tempfile.gettempdir()))


if __name__ == "__main__":
    unittest.main()
