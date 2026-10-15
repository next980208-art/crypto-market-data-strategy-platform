from types import SimpleNamespace
import unittest

import pandas as pd

from scripts import funding_event_backtest as engine
from scripts import run_bounded_profitability_search as search
from scripts import validate_funding_event_execution as replay


BACKTEST = search.load("8_model_signal_backtest.py", "funding_test_base")


def args(**overrides):
    return SimpleNamespace(**{**dict(risk_per_trade_pct=.005, extra_slippage_bps=0, take_profit_r=1.5,
        max_new_entries_per_day=5, max_consecutive_losses=3, min_minutes_between_entries=0,
        daily_stop_pct=.04, hard_daily_stop_pct=.06, min_confidence=0), **overrides})


def signal(entry=60000, holding=5, side="long", payoff=0):
    row = dict(timestamp=entry-60000, entry_timestamp=entry, entry_price=100., stop_distance_pct=.005,
               round_trip_cost_bps=10., entry_rule="next_bar_open", label_horizon_complete=True,
               pred_action=side, pred_confidence=0.)
    for s in ("long", "short"):
        row[s+"_tp_1_5r_holding_minutes"] = holding
        row[s+"_tp_1_5r_return_r_net"] = payoff
    return row


def market(stamp=180000, rate=.001, price="100"):
    return engine.FundingMarket([dict(fundingTime=stamp, fundingRate=rate, markPrice=price)],
                                pd.DataFrame(dict(event_ms=[0, 60000, 120000, 180000], close=[100., 100., 100., 10000.])))


class FundingEngineTests(unittest.TestCase):
    def test_funding_updates_cash_and_blocks_next_entry(self):
        frame = pd.DataFrame([signal(), signal(entry=420000)])
        trades, _, skipped, events, _ = engine.simulate(frame, args(daily_stop_pct=.0005), BACKTEST, market())
        self.assertEqual(len(trades), 1)
        self.assertEqual(skipped["daily_stop"], 1)
        self.assertEqual([e["kind"] for e in events], ["funding", "trade_exit"])
        self.assertLess(events[0]["cash_after"], 1)
        self.assertAlmostEqual(trades[0]["funding_pnl"], -.005/.006*.001)

    def test_short_receives_positive_rate(self):
        trades, _, _, _, _ = engine.simulate(pd.DataFrame([signal(side="short")]), args(), BACKTEST, market())
        self.assertGreater(trades[0]["funding_pnl"], 0)

    def test_no_payment_at_entry_or_exit(self):
        for stamp in (60000, 360000):
            self.assertEqual(market(stamp).settlements(60000, 360000), [])

    def test_proxy_never_reads_future_candle(self):
        item = market(price="").settlements(60000, 360000)[0]
        self.assertEqual(item["price"], 100)
        self.assertLessEqual(item["price_available_ms"], item["timestamp"])
        with self.assertRaises(ValueError):
            market(stamp=500000, price="").settlements(400000, 600000)

    def test_no_exit_profit_booked_before_exit(self):
        trades, _, skipped, events, _ = engine.simulate(pd.DataFrame([signal(payoff=1), signal(entry=120000)]), args(), BACKTEST)
        self.assertEqual(len(trades), 1)
        self.assertEqual(skipped["position_already_open"], 1)
        self.assertEqual(events[0]["timestamp"], 360000)
        self.assertEqual(events[0]["cash_before"], 1)

    def test_new_day_funding_and_exit_separate_days(self):
        entry = engine.DAY-2*engine.MINUTE
        m = market(stamp=engine.DAY-engine.MINUTE)
        trades, _, _, events, daily = engine.simulate(pd.DataFrame([signal(entry=entry, payoff=-1)]), args(), BACKTEST, m)
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(daily[1]["cash_at_first_event"], events[0]["cash_after"])
        self.assertLess(daily[0]["realized_pnl"], 0)
        self.assertLess(daily[1]["realized_pnl"], 0)

    def test_position_size_frozen_across_funding(self):
        m = engine.FundingMarket([dict(fundingTime=t, fundingRate=.001, markPrice="100") for t in (120000, 240000)],
                                 pd.DataFrame(dict(event_ms=[0], close=[100.])))
        _, _, _, events, _ = engine.simulate(pd.DataFrame([signal()]), args(), BACKTEST, m)
        self.assertAlmostEqual(events[0]["amount"], events[1]["amount"])

    def test_invalid_and_zero_trade(self):
        row = signal(side="no_trade")
        trades, curve, _, events, _ = engine.simulate(pd.DataFrame([row]), args(), BACKTEST)
        self.assertEqual(trades, [])
        self.assertEqual(events, [])
        self.assertEqual(curve, [1.])
        with self.assertRaises(ValueError):
            engine.simulate(pd.DataFrame([signal(), signal()]), args(), BACKTEST)

    def test_fractional_holding_rejected(self):
        with self.assertRaises(ValueError):
            engine.simulate(pd.DataFrame([signal(holding=1.5)]), args(), BACKTEST)

    def test_weekly_cashflow_uses_funding_day_not_exit_day(self):
        entry = 7*engine.DAY - 2*engine.MINUTE
        _, _, _, events, _ = engine.simulate(pd.DataFrame([signal(entry=entry, payoff=1)]), args(), BACKTEST,
                                            market(stamp=7*engine.DAY-engine.MINUTE))
        weeks = replay.consensus.weekly_results(pd.DataFrame(events), 0, 30*engine.DAY)
        self.assertLess(weeks["blocks"][0]["return_pct"], 0)
        self.assertGreater(weeks["blocks"][1]["return_pct"], 0)
        self.assertAlmostEqual(weeks["total_return_pct"], (events[-1]["cash_after"]-1)*100)

    def test_runner_empty_and_control_metrics(self):
        action = search.load("16_action_profitability_walk_forward_model.py", "funding_test_action")
        metrics, _, tables = replay.evaluate(pd.DataFrame([signal(side="no_trade")]), action, BACKTEST, 5, market())
        self.assertEqual(metrics["trades"], 0)
        self.assertEqual(metrics["total_return_pct"], 0)
        self.assertTrue(tables["events"].empty)
        metrics, _, _ = replay.evaluate(pd.DataFrame([signal()]), action, BACKTEST, 5, market())
        control, _, _ = replay.evaluate(pd.DataFrame([signal()]), action, BACKTEST, 5, None)
        self.assertLess(metrics["total_return_pct"], control["total_return_pct"])

    def test_weekly_mixed_millisecond_timestamps(self):
        _, _, _, events, _ = engine.simulate(pd.DataFrame([signal(payoff=1)]), args(), BACKTEST, market(stamp=180013))
        result = replay.cash_weekly(pd.DataFrame(events), 0, 30*engine.DAY)
        self.assertAlmostEqual(result["total_return_pct"], (events[-1]["cash_after"]-1)*100)


if __name__ == "__main__":
    unittest.main()
