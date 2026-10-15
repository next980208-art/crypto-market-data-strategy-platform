import unittest

import pandas as pd

from scripts import backtest_rsi_wick_dca_scenario as scenario


class FakeMarket:
    def __init__(self, events=None):
        self.events = events or []

    def settlements(self, entry, exit_ms):
        return [row for row in self.events if entry < row["timestamp"] < exit_ms]


def bars(rows):
    frame = pd.DataFrame(rows)
    frame["volume"] = 1.0
    frame["signal"] = frame.get("signal", "")
    return frame


class RsiWickDcaScenarioTests(unittest.TestCase):
    def test_signal_is_symmetric_and_uses_closed_bar(self):
        rows = []
        for index in range(20):
            close = 100.0 - index
            rows.append({"timestamp": index * scenario.STEP, "open": close + 0.2,
                         "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1.0})
        rows[-1].update(open=82.0, high=82.2, low=79.0, close=81.5)
        result = scenario.add_signal_columns(pd.DataFrame(rows))
        self.assertEqual(result.iloc[-1].signal, "long")

    def test_quantity_respects_risk_and_leverage_caps(self):
        quantity = scenario.initial_unit_quantity(1.0, 100.0, 99.0, 7.5)
        full_quantity = 3.0 * quantity
        stop_loss_with_conservative_cost = full_quantity * 1.0 + 6.0 * quantity * 100.0 * 0.00075
        self.assertLessEqual(stop_loss_with_conservative_cost, 0.02 + 1e-12)
        self.assertLessEqual(full_quantity * 100.0, 10.0 + 1e-12)

    def test_same_bar_collision_uses_stop(self):
        row = pd.Series({"open": 100.0, "high": 104.0, "low": 96.0})
        self.assertEqual(scenario.bar_exit("long", row, 98.0, 102.0), (98.0, "stop"))
        self.assertEqual(scenario.bar_exit("short", row, 102.0, 98.0), (102.0, "stop"))

    def test_second_fill_is_two_units_on_following_bar_only(self):
        step = scenario.STEP
        frame = bars([
            {"timestamp": 0, "open": 100.0, "high": 101.0, "low": 98.0, "close": 100.0, "signal": "long"},
            {"timestamp": step, "open": 100.0, "high": 100.5, "low": 99.0, "close": 99.5, "signal": ""},
            {"timestamp": 2*step, "open": 99.5, "high": 100.0, "low": 98.8, "close": 99.0, "signal": ""},
            {"timestamp": 3*step, "open": 99.0, "high": 103.0, "low": 98.9, "close": 102.5, "signal": ""},
        ])
        trades, _ = scenario.simulate_fold(frame, 1, FakeMarket(), 5.0)
        self.assertEqual(len(trades), 1)
        self.assertTrue(bool(trades.iloc[0].dca_filled))
        self.assertAlmostEqual(trades.iloc[0].dca_quantity, 2.0 * trades.iloc[0].first_quantity)
        self.assertEqual(int(trades.iloc[0].dca_timestamp), 3 * step)

    def test_positive_funding_costs_long_and_benefits_short(self):
        market = FakeMarket([{"timestamp": 100, "price": 100.0, "rate": 0.001}])
        long_value, _ = scenario.funding_cashflow(market, "long", 0, 200, 1.0, 0.0, None)
        short_value, _ = scenario.funding_cashflow(market, "short", 0, 200, 1.0, 0.0, None)
        self.assertAlmostEqual(long_value, -0.1)
        self.assertAlmostEqual(short_value, 0.1)


if __name__ == "__main__":
    unittest.main()
