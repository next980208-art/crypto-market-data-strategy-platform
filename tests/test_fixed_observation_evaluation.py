import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts import evaluate_fixed_observation as evaluation


class FixedObservationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=evaluation.PROJECT_ROOT)
        self.root = Path(self.temp.name)
        self.start_ms = 1_800_000_000_000
        protocol = {
            "observation_start_ms": self.start_ms,
            "observation_start_utc": evaluation.utc_iso(self.start_ms),
            "review_after_utc": evaluation.utc_iso(self.start_ms + 90 * evaluation.DAY_MS),
            "rule_id": "fixed-test",
            "minimum_observation_days": 90,
            "minimum_monthly_trades": 20,
            "storage_limit_bytes": 1024 * 1024,
            "orders_allowed": False,
        }
        self._write_json("protocol.json", protocol)
        protocol_hash = hashlib.sha256((self.root / "protocol.json").read_bytes()).hexdigest()
        self._write_json("state.json", {"protocol_sha256": protocol_hash, "runs": []})
        end_ms = self.start_ms + 8 * evaluation.DAY_MS
        bars_start = self.start_ms - 48 * evaluation.STEP_MS
        timestamps = list(range(bars_start, end_ms, evaluation.STEP_MS))
        bars = pd.DataFrame({
            "timestamp": timestamps,
            "period_end_ms": [value + evaluation.STEP_MS for value in timestamps],
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
        })
        bars.to_parquet(self.root / "bars_15m.parquet", index=False)
        bars.to_parquet(self.root / "mark_15m.parquet", index=False)
        self._write_json("funding.json", {})
        (self.root / "funding.json").write_text("[]", encoding="utf-8")
        report = {
            "status": "collecting",
            "exchange_server_utc": evaluation.utc_iso(end_ms),
            "observed_days": 8.0,
            "contract_bars": len(bars),
            "mark_bars": len(bars),
            "funding_events": 0,
            "database_bytes": 1024,
            "credentials_used": False,
            "orders_sent": 0,
            "results": {
                cost: {"metrics": {"trades": 1, "total_return_pct": 1.0, "max_drawdown_pct": 0.0}}
                for cost in evaluation.COST_CASES
            },
        }
        self._write_json("latest_report.json", report)
        exit_time = pd.Timestamp(self.start_ms + evaluation.DAY_MS, unit="ms", tz="UTC").isoformat()
        for cost in evaluation.COST_CASES:
            pd.DataFrame({"exit_time_utc": [exit_time], "return_pct": [0.01]}).to_parquet(
                self.root / f"current_{cost}_trades.parquet", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def _write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def test_complete_week_passes_only_when_all_cost_cases_are_positive(self):
        value = evaluation.build_evaluation(self.root)
        self.assertEqual(value["quality"]["status"], "healthy")
        self.assertEqual(value["weekly"]["complete_weeks"], 1)
        self.assertEqual(value["weekly"]["positive_weeks"], 1)
        self.assertEqual(value["weekly"]["weeks"][0]["status"], "pass")
        self.assertEqual(value["weekly"]["weeks"][1]["status"], "collecting")
        self.assertEqual(value["review"]["status"], "collecting")

    def test_nonpositive_cost_case_fails_completed_week(self):
        pd.DataFrame({
            "exit_time_utc": [pd.Timestamp(self.start_ms + evaluation.DAY_MS, unit="ms", tz="UTC").isoformat()],
            "return_pct": [-0.01],
        }).to_parquet(self.root / "current_severe_trades.parquet", index=False)
        value = evaluation.build_evaluation(self.root)
        week = value["weekly"]["weeks"][0]
        self.assertEqual(week["status"], "fail")
        self.assertIn("nonpositive_cost_cases:severe", week["reasons"])

    def test_bar_gap_is_a_critical_quality_error(self):
        path = self.root / "bars_15m.parquet"
        frame = pd.read_parquet(path).drop(index=50).reset_index(drop=True)
        frame.to_parquet(path, index=False)
        report = evaluation.read_json(self.root / "latest_report.json")
        report["contract_bars"] = len(frame)
        self._write_json("latest_report.json", report)
        value = evaluation.build_evaluation(self.root)
        self.assertEqual(value["quality"]["status"], "critical")
        continuity = next(row for row in value["quality"]["checks"] if row["id"] == "bars_15m.parquet:continuity")
        self.assertEqual(continuity["details"]["missing_bars"], 1)

    def test_alert_history_does_not_duplicate_unchanged_state(self):
        value = evaluation.build_evaluation(self.root)
        evaluation.write_outputs(self.root, value)
        evaluation.write_outputs(self.root, value)
        lines = (self.root / "alert_history.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["status"], "clear")


if __name__ == "__main__":
    unittest.main()
