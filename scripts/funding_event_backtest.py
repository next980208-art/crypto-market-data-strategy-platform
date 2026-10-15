"""Chronological, single-position funding cash ledger for offline research.

Reuses the project's label payoff and position-sizing contract. Trading costs
are included at exit; daily guards use realized cash including funding, not MTM.
Missing settlement marks use only a previously completed one-minute mark bar.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

MINUTE = 60000
DAY = 86400000


class FundingMarket:
    def __init__(self, events, marks):
        self.events = sorted(events, key=lambda row: int(row["fundingTime"]))
        self.times = np.array([int(row["fundingTime"]) for row in self.events], dtype=np.int64)
        if len(self.times) and (np.diff(self.times) <= 0).any():
            raise ValueError("Duplicate funding events")
        marks = marks.sort_values("event_ms")
        self.available = marks.event_ms.to_numpy(dtype=np.int64) + MINUTE
        self.prices = marks.close.to_numpy(dtype=float)
        if (np.diff(self.available) <= 0).any() or not np.isfinite(self.prices).all() or (self.prices <= 0).any():
            raise ValueError("Invalid mark series")

    def settlements(self, entry, exit_ms):
        # At equal timestamps: close old position, settle funding, open new position.
        left = self.times.searchsorted(entry, side="right")
        right = self.times.searchsorted(exit_ms, side="left")
        output = []
        for source in self.events[left:right]:
            stamp, rate = int(source["fundingTime"]), float(source["fundingRate"])
            if not np.isfinite(rate):
                raise ValueError("Nonfinite funding rate")
            proxy = source.get("markPrice") in (None, "")
            if proxy:
                index = self.available.searchsorted(stamp, side="right") - 1
                if index < 0 or stamp - self.available[index] >= MINUTE:
                    raise ValueError("Missing causal mark proxy")
                price = float(self.prices[index])
                available_ms = int(self.available[index])
            else:
                price, available_ms = float(source["markPrice"]), stamp
            if not np.isfinite(price) or price <= 0:
                raise ValueError("Invalid funding mark")
            output.append({"timestamp": stamp, "rate": rate, "price": price, "proxy": proxy,
                           "price_available_ms": available_ms,
                           "boundary_uncertain": min(stamp-entry, exit_ms-stamp) <= MINUTE})
        return output


def simulate(predictions, args, backtest, market=None):
    backtest.validate_args(args)
    if args.max_new_entries_per_day < 1 or args.max_consecutive_losses < 1 or args.min_minutes_between_entries < 0:
        raise ValueError("Invalid entry limits")
    if not 0 < args.daily_stop_pct <= args.hard_daily_stop_pct < 1:
        raise ValueError("Invalid daily stops")
    predictions = predictions.sort_values("entry_timestamp").reset_index(drop=True)
    if predictions.entry_timestamp.duplicated().any():
        raise ValueError("Duplicate entry timestamps")
    if not predictions.entry_rule.eq("next_bar_open").all() or not predictions.label_horizon_complete.eq(True).all():
        raise ValueError("Invalid entry/label contract")
    cash, trades, events, curve = 1., [], [], [1.]
    day_starts, day_pnl = {}, defaultdict(float)
    day_entries, losses = defaultdict(int), defaultdict(int)
    skipped = {name: 0 for name in ("low_confidence", "no_trade_prediction", "daily_entry_limit", "daily_stop",
               "hard_daily_stop", "min_interval", "consecutive_loss_stop", "position_already_open")}
    active, last_entry = None, None

    def touch(stamp):
        day = int(stamp) // DAY
        day_starts.setdefault(day, cash)
        return day

    def book(stamp, amount, kind, extra=None):
        nonlocal cash
        day, before = touch(stamp), cash
        cash += amount
        if not np.isfinite(cash) or cash <= 0:
            raise ValueError("Insolvent or nonfinite cash balance")
        day_pnl[day] += amount
        curve.append(cash)
        events.append({"timestamp": int(stamp), "exit_time_utc": pd.Timestamp(stamp, unit="ms", tz="UTC").isoformat(),
                       "kind": kind, "cash_before": before, "cash_after": cash, "amount": amount,
                       "return_pct": amount / before, **(extra or {})})

    def advance(until):
        nonlocal active
        if active is None:
            return
        while active["funding_index"] < len(active["funding"]):
            event = active["funding"][active["funding_index"]]
            if event["timestamp"] > until:
                break
            sign = 1 if active["side"] == "long" else -1
            amount = -sign * active["quantity"] * event["price"] * event["rate"]
            book(event["timestamp"], amount, "funding", event)
            active["funding_pnl"] += amount
            active["funding_index"] += 1
        if active["exit_ms"] <= until:
            book(active["exit_ms"], active["entry_cash"] * active["base_return_pct"], "trade_exit")
            net = active["base_return_pct"] + active["funding_pnl"] / active["entry_cash"]
            exit_day = active["exit_ms"] // DAY
            losses[exit_day] = losses[exit_day] + 1 if net < 0 else 0
            row = active["row"]
            trades.append({"signal_timestamp": int(row["timestamp"]), "entry_timestamp": active["entry_ms"],
                           "entry_time_utc": pd.Timestamp(active["entry_ms"], unit="ms", tz="UTC").isoformat(),
                           "exit_time_utc": pd.Timestamp(active["exit_ms"], unit="ms", tz="UTC").isoformat(),
                           "action": active["side"], "entry_price": float(row["entry_price"]),
                           "quantity": active["quantity"], "entry_cash": active["entry_cash"],
                           "return_pct": net, "return_r": net / args.risk_per_trade_pct,
                           "base_return_pct": active["base_return_pct"], "funding_pnl": active["funding_pnl"],
                           "funding_events": len(active["funding"]), "funding_proxy_events": sum(e["proxy"] for e in active["funding"]),
                           "boundary_uncertain_events": sum(e["boundary_uncertain"] for e in active["funding"]),
                           "holding_minutes": active["holding"], "equity": cash})
            active = None

    for row in predictions.to_dict("records"):
        stamp = int(row["entry_timestamp"])
        if stamp != row["entry_timestamp"] or stamp % MINUTE or stamp - int(row["timestamp"]) != MINUTE:
            raise ValueError("Unexpected entry clock")
        advance(stamp)
        day = touch(stamp)
        side = row["pred_action"]
        checks = ((active is not None, "position_already_open"), (side == "no_trade", "no_trade_prediction"),
                  (row["pred_confidence"] < args.min_confidence, "low_confidence"),
                  (day_entries[day] >= args.max_new_entries_per_day, "daily_entry_limit"),
                  (day_pnl[day] <= -day_starts[day] * args.hard_daily_stop_pct, "hard_daily_stop"),
                  (day_pnl[day] <= -day_starts[day] * args.daily_stop_pct, "daily_stop"),
                  (losses[day] >= args.max_consecutive_losses, "consecutive_loss_stop"),
                  (last_entry is not None and stamp-last_entry < args.min_minutes_between_entries*MINUTE, "min_interval"))
        reason = next((name for condition, name in checks if condition), None)
        if reason:
            skipped[reason] += 1
            continue
        if side not in ("long", "short"):
            raise ValueError("Unknown action")
        price, stop, cost = float(row["entry_price"]), float(row["stop_distance_pct"]), float(row["round_trip_cost_bps"])
        holding_value = float(row[backtest.holding_column(side, args.take_profit_r)])
        if not np.isfinite([price, stop, cost, holding_value]).all() or min(price, stop) <= 0 or cost < 0 or not 0 <= holding_value <= 240 or not holding_value.is_integer():
            raise ValueError("Invalid position inputs")
        holding = int(holding_value)
        label = float(row[backtest.return_column(side, args.take_profit_r)])
        _, base_return = backtest.account_return_from_label(row, label, args)
        if not np.isfinite(base_return) or base_return <= -1:
            raise ValueError("Invalid exit payoff")
        exit_ms = stamp + holding*MINUTE
        quantity = cash * args.risk_per_trade_pct / (stop + (cost + args.extra_slippage_bps)/10000) / price
        active = {"row": row, "entry_ms": stamp, "exit_ms": exit_ms, "entry_cash": cash,
                  "side": side, "quantity": quantity, "holding": holding, "base_return_pct": base_return,
                  "funding_pnl": 0., "funding_index": 0,
                  "funding": market.settlements(stamp, exit_ms) if market else []}
        day_entries[day] += 1
        last_entry = stamp
    if active is not None:
        advance(active["exit_ms"])
    np.testing.assert_allclose(np.prod([1+t["return_pct"] for t in trades]), cash, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.prod([1+e["return_pct"] for e in events]), cash, atol=1e-12, rtol=1e-12)
    daily = [{"day_start_ms": day*DAY, "cash_at_first_event": day_starts[day], "realized_pnl": day_pnl[day],
              "entries": day_entries[day], "consecutive_losses_at_end": losses[day]} for day in sorted(day_starts)]
    return trades, curve, skipped, events, daily
