import importlib.util
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "trade_enriched_dataset", ROOT / "17_build_trade_enriched_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trade_frame(rows, *, count_offset=0):
    count = np.arange(rows, dtype=float) + 10 + count_offset
    volume = count * 0.1
    price = 100.0 + np.arange(rows) * 0.01
    return pd.DataFrame(
        {
            "timestamp": np.arange(rows, dtype=np.int64) * MODULE.MINUTE_MS,
            "trade_count": count,
            "trade_volume": volume,
            "trade_notional": volume * price,
            "trade_vwap": price,
            "trade_first_price": price - 0.01,
            "trade_last_price": price + 0.01,
            "taker_buy_volume": volume * 0.55,
            "taker_sell_volume": volume * 0.45,
            "taker_buy_notional": volume * price * 0.55,
            "taker_sell_notional": volume * price * 0.45,
            "taker_volume_imbalance": np.full(rows, 0.1),
        }
    )


class TradeEnrichedDatasetTests(unittest.TestCase):
    def test_output_training_gate_rejects_time_gaps_and_volume_mismatch(self):
        for gap, volume_error, expected in ((False, 0.0, True), (True, 0.0, False), (False, 0.02, False)):
            with self.subTest(gap=gap, volume_error=volume_error), tempfile.TemporaryDirectory() as root:
                timestamps = np.arange(1440) * MODULE.MINUTE_MS
                if gap:
                    timestamps[100:] += MODULE.MINUTE_MS
                dataset = pd.DataFrame({"timestamp": timestamps})
                args = SimpleNamespace(output_folder=root, market="usdm", symbol="BTCUSDT", timeframe="1m",
                                       minimum_training_days=1, report_json=str(Path(root) / "report.json"),
                                       report_md=str(Path(root) / "report.md"))
                with patch.object(MODULE, "write_markdown"):
                    _, _, _, marker = MODULE.write_outputs(
                        args, dataset, [], "labels", {},
                        {"aggtrade_vs_kline_volume": {"daily_relative_error_max": volume_error}},
                    )
                self.assertEqual(expected, marker["model_training_allowed"])

    def test_overlap_resolution_prefers_complete_source(self):
        complete = trade_frame(120)
        snippet = trade_frame(3, count_offset=1000)

        resolved, report = MODULE.resolve_trade_overlaps([snippet, complete])

        self.assertEqual(120, len(resolved))
        self.assertEqual(3, report["overlap_timestamps"])
        self.assertEqual(3, report["conflicting_overlap_timestamps"])
        self.assertEqual(10.0, resolved.iloc[0]["trade_count"])

    def test_causal_features_match_prefix_calculation(self):
        raw = trade_frame(180)
        full, columns = MODULE.add_causal_trade_features(raw)

        MODULE.verify_causal_trade_features(raw, full, columns)

        self.assertIn("agg_trade_count_ratio_60m", columns)
        self.assertIn("agg_taker_volume_imbalance_mean_15m", columns)
        self.assertTrue(pd.isna(full.iloc[58]["agg_trade_count_ratio_60m"]))
        self.assertTrue(np.isfinite(full.iloc[59]["agg_trade_count_ratio_60m"]))

    def test_notional_and_volume_features_are_finite_after_warmup(self):
        raw = trade_frame(120)
        features, columns = MODULE.add_causal_trade_features(raw)
        ready = features.dropna(subset=columns)

        self.assertEqual(61, len(ready))
        self.assertTrue(np.isfinite(ready[columns].to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
