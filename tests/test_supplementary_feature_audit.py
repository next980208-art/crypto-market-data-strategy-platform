import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import audit_supplementary_features as audit

M = audit.MINUTE


def records(family, count, received=1000 * M):
    step = audit.FAMILIES[family]
    result = []
    for i in range(count):
        stamp = i * step
        if family == "open_interest_5m":
            payload = {"symbol": "BTCUSDT", "timestamp": stamp, "sumOpenInterest": str(100+i),
                       "sumOpenInterestValue": "10000"}
            end = stamp
        else:
            payload = [stamp, "100", "110", "90", str(100 + i / 100), "1", stamp + step - 1]
            end = stamp + step
        result.append((stamp, end, received, "fixture", json.dumps(payload)))
    return result


class SupplementaryFeatureAuditTests(unittest.TestCase):
    def test_normalization_checks_payload_timestamp_and_values(self):
        valid, quality = audit.normalize("spot_1m", records("spot_1m", 2))
        self.assertTrue(quality["passed"])
        self.assertEqual(len(valid), 2)
        bad = records("spot_1m", 1)
        payload = json.loads(bad[0][4])
        payload[0] = M
        bad[0] = (*bad[0][:4], json.dumps(payload))
        _, quality = audit.normalize("spot_1m", bad)
        self.assertEqual(quality["invalid_rows"], 1)
        self.assertFalse(quality["passed"])

    def test_missing_grid_is_not_compressed_or_filled(self):
        rows = records("open_interest_5m", 15)
        del rows[5]
        frame, quality = audit.normalize("open_interest_5m", rows)
        self.assertEqual(quality["missing_grid_rows"], 1)
        derived = audit.derive_family(frame, "open_interest_5m")
        self.assertTrue(np.isnan(derived.loc[6, audit.FEATURES[0]]))
        self.assertTrue(np.isnan(derived.loc[12, audit.FEATURES[1]]))
        self.assertAlmostEqual(derived.loc[7, audit.FEATURES[0]], 107/106-1)

    def test_feature_dependency_receipt_uses_latest_input_receipt(self):
        rows = records("open_interest_5m", 13)
        rows[0] = (*rows[0][:2], 2000 * M, *rows[0][3:])
        frame, _ = audit.normalize("open_interest_5m", rows)
        result = audit.derive_family(frame, "open_interest_5m")
        self.assertEqual(result.iloc[-1].dependency_received_ms, 2000*M)

    def test_received_and_historical_assumption_are_separate(self):
        frame, _ = audit.normalize("spot_1m", records("spot_1m", 20))
        source = audit.derive_family(frame, "spot_1m")
        decisions = pd.DataFrame({"decision_ms": [19*M]})
        strict = audit.join_family(decisions, source, "spot_1m", "received")
        replay = audit.join_family(decisions, source, "spot_1m", "historical_assumption")
        self.assertTrue(strict["spot_1m__value"].isna().all())
        self.assertEqual(replay["spot_1m__event_ms"].iloc[0], 17*M)
        self.assertLessEqual(replay["spot_1m__available_ms"].iloc[0], 19*M)

    def test_stale_context_not_carried_forever(self):
        frame, _ = audit.normalize("index_1m", records("index_1m", 2))
        source = audit.derive_family(frame, "index_1m")
        result = audit.join_family(pd.DataFrame({"decision_ms": [3*M, 4*M]}), source,
                                   "index_1m", "historical_assumption")
        self.assertFalse(pd.isna(result.iloc[0]["index_1m__value"]))
        self.assertTrue(pd.isna(result.iloc[1]["index_1m__value"]))

    def test_older_late_arrival_does_not_replace_newer_record(self):
        source = pd.DataFrame({"event_ms": [0, M], "period_end_ms": [M, 2*M],
                               "dependency_received_ms": [2*M+10, 2*M], "value": [99, 100]})
        result = audit.join_family(pd.DataFrame({"decision_ms": [2*M+20]}), source, "spot_1m", "received")
        self.assertEqual(result["spot_1m__value"].iloc[0], 100)

    def test_appending_future_source_does_not_change_past_features(self):
        frames = []
        for count in (20, 30):
            normalized, _ = audit.normalize("spot_1m", records("spot_1m", count))
            source = audit.derive_family(normalized, "spot_1m")
            frames.append(audit.join_family(pd.DataFrame({"decision_ms": [18*M, 19*M]}), source,
                                             "spot_1m", "historical_assumption"))
        pd.testing.assert_frame_equal(*frames)

    def test_empty_source_has_no_complete_features(self):
        sources = {f: audit.normalize(f, [])[0] for f in audit.FAMILIES}
        base = pd.DataFrame({"timestamp": [0], "entry_timestamp": [M], "close": [100], "max_holding_minutes": [240]})
        joined = audit.assemble(base, sources, "received")
        self.assertEqual(audit.coverage(joined)["complete_rows"], 0)

    def test_end_to_end_read_only_features_and_blocked_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "context.sqlite3"
            with closing(sqlite3.connect(database)) as db:
                db.execute("CREATE TABLE observations(family TEXT,event_ms INTEGER,period_end_ms INTEGER,first_received_ms INTEGER,source TEXT,payload TEXT)")
                for family, count in [("open_interest_5m", 20), ("spot_1m", 100), ("index_1m", 100)]:
                    db.executemany("INSERT INTO observations VALUES(?,?,?,?,?,?)", [(family, *r) for r in records(family, count)])
                db.commit()
            original = audit.sha256(database)
            base = pd.DataFrame({"timestamp": np.arange(100)*M, "entry_timestamp": (np.arange(100)+1)*M,
                                 "close": 101.0, "max_holding_minutes": 240})
            dataset = root / "base.parquet"
            base.to_parquet(dataset, index=False)
            digest = audit.sha256(dataset)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"dataset_file": str(dataset), "dataset_sha256": digest}), encoding="utf-8")
            protocol = root / "prior.json"
            protocol.write_text(json.dumps({"dataset_sha256": digest, "holdout_start_ms": 90*M,
                                            "folds": [{"fold": i+1,"train_start": 0,"validation_start": 30*M,"validation_end": 60*M} for i in range(3)]}), encoding="utf-8")
            with redirect_stdout(StringIO()):
                report = audit.run(database, manifest, protocol, root / "out")
            self.assertEqual(report["comparison"]["status"], "blocked")
            self.assertEqual(report["comparison"]["models_fitted"], 0)
            self.assertEqual(report["received"]["complete_rows"], 0)
            self.assertGreater(report["historical_assumption"]["complete_rows"], 0)
            self.assertEqual(audit.sha256(database), original)
            sidecar = pd.read_parquet(root / "out/historical_assumption_features_only.parquet")
            self.assertTrue(sidecar.model_training_allowed.eq(False).all())
            self.assertFalse(any("_tp_" in c for c in sidecar))
            self.assertTrue(sidecar.reserved_period.any())


if __name__ == "__main__":
    unittest.main()
