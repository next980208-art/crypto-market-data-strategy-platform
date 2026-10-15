import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load("16_action_profitability_walk_forward_model.py", "test_trade_model")
BUILDER = load("17_build_trade_enriched_dataset.py", "test_trade_builder")
CYCLE = load("scripts/run_trade_research_cycle.py", "test_trade_cycle")


class TradeResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "dataset.parquet"
        self.manifest_path = Path(self.temp.name) / "manifest.json"
        pd.DataFrame({"timestamp": np.arange(90 * 1440) * 60000}).to_parquet(self.path)
        _, columns = BUILDER.add_causal_trade_features(pd.DataFrame(columns=["timestamp"] + BUILDER.BASE_TRADE_COLUMNS))
        self.manifest = {
            "dataset_file": str(self.path), "dataset_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "model_training_allowed": True, "minimum_training_days": 90,
            "trade_feature_columns": columns + ["agg_trade_vwap_to_close"],
        }

    def verify(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return MODEL.verified_trade_columns(self.manifest_path, self.path)

    def test_accepts_only_verified_causal_feature_contract(self):
        self.assertEqual(22, len(self.verify()))

    def test_blocks_unapproved_dataset(self):
        self.manifest["model_training_allowed"] = False
        with self.assertRaisesRegex(RuntimeError, "training gate"):
            self.verify()

    def test_blocks_changed_file(self):
        self.manifest["dataset_sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.verify()

    def test_blocks_label_as_feature(self):
        self.manifest["trade_feature_columns"].append("long_tp_1_0r_return_r_net")
        with self.assertRaisesRegex(RuntimeError, "causal feature contract"):
            self.verify()

    def test_blocks_interior_time_gap(self):
        timestamps = np.arange(90 * 1440) * 60000
        timestamps[100:] += 60000
        pd.DataFrame({"timestamp": timestamps}).to_parquet(self.path)
        self.manifest["dataset_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "continuous"):
            self.verify()

    def test_excludes_partial_context_without_removing_rows(self):
        frame = pd.DataFrame({"open_interest": [1.0, np.nan], "open_interest_value": [3.0, 4.0], "close": [10, 11]})
        result, excluded = CYCLE.drop_incomplete_optional_context(frame)
        self.assertEqual(2, len(result))
        self.assertNotIn("open_interest", result)
        self.assertIn("open_interest_value", result)
        self.assertEqual(1, excluded["open_interest"]["unavailable_rows"])


if __name__ == "__main__":
    unittest.main()
