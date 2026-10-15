import unittest

import pandas as pd

from scripts.verify_batch_streaming_feature_parity import compare_frames, realtime_features_from_ohlcv


def raw_frame():
    closes = [100.0, 101.0, 99.0, 102.0, 104.0, 103.0]
    return pd.DataFrame(
        {
            "timestamp": [60_000 * (index + 1) for index in range(len(closes))],
            "open": [value - 0.5 for value in closes],
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [10.0 + index for index in range(len(closes))],
            "run_id": "parity-test",
            "symbol": "BTC/USDT",
            "market": "usdm",
            "timeframe": "1m",
        }
    )


def matching_batch(raw):
    batch = raw[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    batch["ma_5"] = batch["close"].rolling(5, min_periods=1).mean()
    batch["return_1m"] = batch["close"].pct_change().fillna(0.0)
    return batch


class BatchStreamingFeatureParityTests(unittest.TestCase):
    def test_ready_rows_match_batch_features(self):
        raw = raw_frame()
        realtime = realtime_features_from_ohlcv(raw, "BTCUSDT")
        report = compare_frames(matching_batch(raw), realtime, rtol=1e-9, atol=1e-9)

        self.assertTrue(report["healthy"])
        self.assertEqual(4, report["warmup_rows"])
        self.assertEqual(2, report["compared_rows"])

    def test_changed_batch_value_is_reported(self):
        raw = raw_frame()
        batch = matching_batch(raw)
        batch.loc[5, "ma_5"] += 1.0
        realtime = realtime_features_from_ohlcv(raw, "BTCUSDT")
        report = compare_frames(batch, realtime, rtol=1e-9, atol=1e-9)

        self.assertFalse(report["healthy"])
        self.assertEqual(1, report["mismatch_count"])
        self.assertEqual("ma_5", report["mismatches"][0]["column"])


if __name__ == "__main__":
    unittest.main()
