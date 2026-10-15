from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from scripts import multi_timeframe_funding_backtest as engine
from scripts import search_low_frequency_targets as research


BACKTEST = research.search.load("8_model_signal_backtest.py", "lowfreq_test_backtest")


def args():
    return SimpleNamespace(risk_per_trade_pct=.005, extra_slippage_bps=5, take_profit_r=1.5,
        max_new_entries_per_day=5, max_consecutive_losses=3, min_minutes_between_entries=15,
        daily_stop_pct=.04, hard_daily_stop_pct=.06, min_confidence=0)


def signal(step=5, holding=720, payoff=1):
    stamp = step*60000
    return pd.DataFrame([{"timestamp": 0, "entry_timestamp": stamp, "entry_price": 100.,
        "stop_distance_pct": .01, "round_trip_cost_bps": 10., "entry_rule": "next_bar_open",
        "label_horizon_complete": True, "max_holding_minutes": 720, "pred_action": "long",
        "pred_confidence": 0., "long_tp_1_5r_holding_minutes": holding,
        "short_tp_1_5r_holding_minutes": holding, "long_tp_1_5r_return_r_net": payoff,
        "short_tp_1_5r_return_r_net": payoff}])


class LowFrequencyTests(unittest.TestCase):
    def test_grid_frozen(self):
        self.assertEqual(len(research.target_specs()), 8)
        self.assertEqual(len(research.configurations()), 48)
        self.assertEqual(len({c["id"] for c in research.configurations()}), 48)

    def test_aggregate_requires_complete_aligned_bars(self):
        raw = pd.DataFrame({"timestamp": np.arange(10)*60000, "open": np.arange(10)+1.,
            "high": np.arange(10)+2., "low": np.arange(10)+.5, "close": np.arange(10)+1.5,
            "volume": np.ones(10)})
        bars = research.aggregate(raw, 5)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars.iloc[0].open, 1)
        self.assertEqual(bars.iloc[0].close, 5.5)
        self.assertEqual(bars.iloc[0].volume, 5)
        with self.assertRaises(ValueError):
            research.aggregate(raw.drop(index=4), 5)

    def test_features_do_not_change_past_when_future_changes(self):
        n = 120
        bars = pd.DataFrame({"timestamp": np.arange(n)*300000, "open": np.linspace(100, 120, n),
            "high": np.linspace(101, 121, n), "low": np.linspace(99, 119, n),
            "close": np.linspace(100.5, 120.5, n), "volume": np.linspace(10, 20, n)})
        bars["datetime_utc"] = pd.to_datetime(bars.timestamp, unit="ms", utc=True)
        bars["symbol"], bars["market"], bars["timeframe"], bars["feature_schema_version"] = "BTCUSDT", "usdm", "5m", "test"
        first, names = research.add_features(bars)
        bars.loc[n-1, ["open", "high", "low", "close", "volume"]] *= 2
        second, _ = research.add_features(bars)
        pd.testing.assert_frame_equal(first.iloc[:-1][names].reset_index(drop=True), second.iloc[:-1][names].reset_index(drop=True))

    def test_multi_minute_engine_and_horizon(self):
        trades, _, _, events, _ = engine.simulate(signal(), args(), BACKTEST)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["holding_minutes"], 720)
        self.assertEqual(events[-1]["timestamp"], 725*60000)
        broken = signal()
        broken.loc[0, "entry_timestamp"] = 6*60000
        trades, _, _, _, _ = engine.simulate(broken, args(), BACKTEST)
        self.assertEqual(len(trades), 1)
        inconsistent = pd.concat([signal(), signal()], ignore_index=True)
        inconsistent.loc[1, ["timestamp", "entry_timestamp"]] = [10*60000, 16*60000]
        with self.assertRaises(ValueError):
            engine.simulate(inconsistent, args(), BACKTEST)

    def test_holding_must_match_bar_step(self):
        with self.assertRaises(ValueError):
            engine.simulate(signal(holding=721), args(), BACKTEST)


if __name__ == "__main__":
    unittest.main()
