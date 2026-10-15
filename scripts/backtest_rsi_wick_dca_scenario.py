"""Backtest a causal 15-minute RSI/wick reversal scenario with one DCA fill.

This is offline research only. It reads frozen historical OHLCV and funding
artifacts, never loads credentials, and never sends an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.funding_event_backtest import FundingMarket
except ModuleNotFoundError:  # Support direct execution: python scripts/<name>.py
    from funding_event_backtest import FundingMarket


ROOT = Path(__file__).resolve().parents[1]
MINUTE = 60_000
STEP = 15 * MINUTE
DAY = 24 * 60 * MINUTE
INITIAL_EQUITY = 1.0
RISK_PER_TRADE = 0.02
MAX_LEVERAGE = 10.0
TAKE_PROFIT_R = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
STOP_BUFFER_BPS = 5.0
BASE_EXECUTION_BPS = 5.0
EXTRA_ROUND_TRIP_BPS = {"normal": 5.0, "stress": 10.0, "severe": 20.0}

DATASET = (
    ROOT
    / "ml_dataset_5y_20260828"
    / "market=usdm"
    / "symbol=BTCUSDT"
    / "timeframe=1m"
    / "ml_dataset_BTCUSDT_USDM_1m.parquet"
)
VALUATION = ROOT / "runtime_reports" / "long_window_marks_20260920_v2"
SOURCE_PROTOCOL = ROOT / "runtime_reports" / "low_frequency_targets_20260922" / "protocol.json"
DEFAULT_OUTPUT = ROOT / "runtime_reports" / "rsi_wick_dca_20261001"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate_15m(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    if raw.empty or int(raw.timestamp.iloc[0]) % STEP:
        raise ValueError("Source data is empty or not aligned to a 15-minute boundary")
    if not raw.timestamp.diff().dropna().eq(MINUTE).all():
        raise ValueError("Source one-minute bars are not continuous")
    frame = raw.copy()
    frame["bucket"] = frame.timestamp // STEP * STEP
    grouped = frame.groupby("bucket", sort=True, observed=True)
    if not grouped.size().eq(15).all():
        raise ValueError("An aggregate bucket does not contain exactly 15 one-minute bars")
    bars = (
        grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"bucket": "timestamp"})
    )
    values = bars[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (bars.low <= 0).any():
        raise ValueError("Invalid aggregate OHLCV values")
    return bars


def add_signal_columns(bars: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    close = result.close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    loss = -delta.clip(upper=0).rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result["rsi_14"] = 100.0 - 100.0 / (1.0 + relative_strength)

    span = result.high - result.low
    body = (result.close - result.open).abs()
    lower_wick = np.minimum(result.open, result.close) - result.low
    upper_wick = result.high - np.maximum(result.open, result.close)
    close_location = (result.close - result.low) / span.replace(0, np.nan)
    wick_floor = np.maximum(body, 0.20 * span)

    long_signal = (
        result.rsi_14.le(RSI_OVERSOLD)
        & lower_wick.ge(wick_floor)
        & close_location.ge(0.50)
    )
    short_signal = (
        result.rsi_14.ge(RSI_OVERBOUGHT)
        & upper_wick.ge(wick_floor)
        & close_location.le(0.50)
    )
    if (long_signal & short_signal).any():
        raise ValueError("A bar emitted both a long and short signal")
    result["signal"] = np.select([long_signal, short_signal], ["long", "short"], default="")
    result["lower_wick"] = lower_wick
    result["upper_wick"] = upper_wick
    result["close_location"] = close_location
    return result


def execution_bps(extra_round_trip_bps: float) -> float:
    """Per-fill charge matching 10bps base round trip plus project stress costs."""
    return BASE_EXECUTION_BPS + extra_round_trip_bps / 2.0


def initial_unit_quantity(
    equity: float,
    entry_price: float,
    stop_price: float,
    per_fill_bps: float,
) -> float:
    """Size one unit so a possible 1+2-unit position stays within 2% price risk."""
    distance = abs(entry_price - stop_price)
    if not np.isfinite([equity, entry_price, stop_price, per_fill_bps]).all():
        raise ValueError("Nonfinite sizing input")
    if equity <= 0 or entry_price <= 0 or stop_price <= 0 or distance <= 0 or per_fill_bps < 0:
        raise ValueError("Invalid sizing input")
    cost_rate = per_fill_bps / 10_000.0
    # Three units of stop distance plus six conservative execution notionals:
    # 1 unit first entry, 2 units second entry, and 3 units at exit.
    loss_per_unit = 3.0 * distance + 6.0 * max(entry_price, stop_price) * cost_rate
    risk_quantity = equity * RISK_PER_TRADE / loss_per_unit
    leverage_quantity = equity * MAX_LEVERAGE / (3.0 * max(entry_price, stop_price))
    return min(risk_quantity, leverage_quantity)


def target_price(side: str, average_entry: float, stop_price: float) -> float:
    distance = abs(average_entry - stop_price)
    return average_entry + TAKE_PROFIT_R * distance if side == "long" else average_entry - TAKE_PROFIT_R * distance


def bar_exit(side: str, row: Any, stop_price: float, take_profit: float) -> tuple[float, str] | None:
    """Resolve a bar conservatively; a same-bar stop/target collision is a stop."""
    open_price = float(row.open)
    if side == "long":
        if open_price <= stop_price:
            return open_price, "stop_gap"
        if open_price >= take_profit:
            return open_price, "take_profit_gap"
        stop_hit, target_hit = float(row.low) <= stop_price, float(row.high) >= take_profit
    else:
        if open_price >= stop_price:
            return open_price, "stop_gap"
        if open_price <= take_profit:
            return open_price, "take_profit_gap"
        stop_hit, target_hit = float(row.high) >= stop_price, float(row.low) <= take_profit
    if stop_hit:
        return stop_price, "stop"
    if target_hit:
        return take_profit, "take_profit"
    return None


def funding_cashflow(
    market: FundingMarket,
    side: str,
    entry_ms: int,
    exit_ms: int,
    first_quantity: float,
    dca_quantity: float,
    dca_timestamp: int | None,
) -> tuple[float, int]:
    direction = 1.0 if side == "long" else -1.0
    amount = 0.0
    events = market.settlements(entry_ms, exit_ms)
    for event in events:
        quantity = first_quantity
        if dca_timestamp is not None and int(event["timestamp"]) > dca_timestamp:
            quantity += dca_quantity
        amount += -direction * quantity * float(event["price"]) * float(event["rate"])
    return amount, len(events)


def close_trade(
    active: dict[str, Any],
    exit_price: float,
    exit_ms: int,
    reason: str,
    equity: float,
    market: FundingMarket,
    per_fill_bps: float,
) -> tuple[dict[str, Any], float]:
    sign = 1.0 if active["side"] == "long" else -1.0
    fills = active["fills"]
    total_quantity = sum(fill["quantity"] for fill in fills)
    gross = sum(sign * fill["quantity"] * (exit_price - fill["price"]) for fill in fills)
    cost_rate = per_fill_bps / 10_000.0
    entry_cost = sum(fill["quantity"] * fill["price"] * cost_rate for fill in fills)
    exit_cost = total_quantity * exit_price * cost_rate
    dca_fill = fills[1] if len(fills) == 2 else None
    funding, funding_events = funding_cashflow(
        market,
        active["side"],
        active["entry_ms"],
        exit_ms,
        fills[0]["quantity"],
        dca_fill["quantity"] if dca_fill else 0.0,
        dca_fill["timestamp"] if dca_fill else None,
    )
    net = gross - entry_cost - exit_cost + funding
    equity_after = equity + net
    if not np.isfinite(equity_after) or equity_after <= 0:
        raise ValueError("Backtest equity became nonpositive or nonfinite")
    average_entry = sum(fill["quantity"] * fill["price"] for fill in fills) / total_quantity
    maximum_notional = max(active["maximum_notional"], total_quantity * average_entry)
    row = {
        "fold": active["fold"],
        "side": active["side"],
        "signal_timestamp": active["signal_ms"],
        "entry_timestamp": active["entry_ms"],
        "dca_timestamp": dca_fill["timestamp"] if dca_fill else np.nan,
        "exit_timestamp": int(exit_ms),
        "entry_time_utc": pd.Timestamp(active["entry_ms"], unit="ms", tz="UTC").isoformat(),
        "exit_time_utc": pd.Timestamp(exit_ms, unit="ms", tz="UTC").isoformat(),
        "first_entry_price": fills[0]["price"],
        "dca_entry_price": dca_fill["price"] if dca_fill else np.nan,
        "average_entry_price": average_entry,
        "stop_price": active["stop_price"],
        "take_profit_price": active["take_profit"],
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "first_quantity": fills[0]["quantity"],
        "dca_quantity": dca_fill["quantity"] if dca_fill else 0.0,
        "total_quantity": total_quantity,
        "dca_filled": bool(dca_fill),
        "holding_minutes": (int(exit_ms) - active["entry_ms"]) / MINUTE,
        "gross_pnl": gross,
        "execution_cost": entry_cost + exit_cost,
        "funding_pnl": funding,
        "funding_events": funding_events,
        "net_pnl": net,
        "return_pct_of_entry_equity": net / active["entry_equity"] * 100.0,
        "risk_budget_pct": RISK_PER_TRADE * 100.0,
        "maximum_leverage_used": maximum_notional / active["entry_equity"],
        "equity_before": equity,
        "equity_after": equity_after,
    }
    return row, equity_after


def simulate_fold(
    frame: pd.DataFrame,
    fold: int,
    market: FundingMarket,
    extra_round_trip_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_fill_bps = execution_bps(extra_round_trip_bps)
    equity = INITIAL_EQUITY
    trades: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    skipped_active = 0
    invalid_stop = 0

    for index in range(1, len(frame)):
        row = frame.iloc[index]
        previous = frame.iloc[index - 1]

        if active is not None:
            outcome = bar_exit(active["side"], row, active["stop_price"], active["take_profit"])
            if outcome is not None:
                trade, equity = close_trade(
                    active,
                    outcome[0],
                    int(row.timestamp),
                    outcome[1],
                    equity,
                    market,
                    per_fill_bps,
                )
                trades.append(trade)
                active = None
                continue

            if index == active["dca_evaluation_index"]:
                first_price = active["fills"][0]["price"]
                adverse = float(row.close) < first_price if active["side"] == "long" else float(row.close) > first_price
                if adverse:
                    quantity = 2.0 * active["fills"][0]["quantity"]
                    active["fills"].append(
                        {"timestamp": int(row.timestamp) + STEP, "price": float(row.close), "quantity": quantity}
                    )
                    total_quantity = sum(fill["quantity"] for fill in active["fills"])
                    average_entry = sum(fill["quantity"] * fill["price"] for fill in active["fills"]) / total_quantity
                    active["take_profit"] = target_price(active["side"], average_entry, active["stop_price"])
                    active["maximum_notional"] = max(
                        active["maximum_notional"], total_quantity * average_entry
                    )
            if previous.signal:
                skipped_active += 1
            continue

        if not previous.signal or index >= len(frame) - 1:
            continue

        side = str(previous.signal)
        entry_price = float(row.open)
        buffer_rate = STOP_BUFFER_BPS / 10_000.0
        stop_price = (
            float(previous.low) * (1.0 - buffer_rate)
            if side == "long"
            else float(previous.high) * (1.0 + buffer_rate)
        )
        if (side == "long" and entry_price <= stop_price) or (side == "short" and entry_price >= stop_price):
            invalid_stop += 1
            continue
        quantity = initial_unit_quantity(equity, entry_price, stop_price, per_fill_bps)
        active = {
            "fold": fold,
            "side": side,
            "signal_ms": int(previous.timestamp),
            "entry_ms": int(row.timestamp),
            "entry_equity": equity,
            "stop_price": stop_price,
            "take_profit": target_price(side, entry_price, stop_price),
            "dca_evaluation_index": index + 1,
            "fills": [{"timestamp": int(row.timestamp), "price": entry_price, "quantity": quantity}],
            "maximum_notional": quantity * entry_price,
        }
        outcome = bar_exit(side, row, stop_price, active["take_profit"])
        if outcome is not None:
            trade, equity = close_trade(
                active,
                outcome[0],
                int(row.timestamp),
                outcome[1],
                equity,
                market,
                per_fill_bps,
            )
            trades.append(trade)
            active = None

    if active is not None:
        last = frame.iloc[-1]
        trade, equity = close_trade(
            active,
            float(last.close),
            int(last.timestamp) + STEP,
            "fold_end",
            equity,
            market,
            per_fill_bps,
        )
        trades.append(trade)

    table = pd.DataFrame(trades)
    signal_counts = frame.signal.value_counts()
    if table.empty:
        metrics = {
            "signals": int(frame.signal.ne("").sum()),
            "long_signals": int(signal_counts.get("long", 0)),
            "short_signals": int(signal_counts.get("short", 0)),
            "trades": 0,
            "total_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "max_trade_loss_pct": 0.0,
            "dca_trades": 0,
            "funding_pnl_pct_of_initial": 0.0,
            "forced_exits": 0,
            "maximum_leverage_used": 0.0,
            "average_holding_hours": 0.0,
            "skipped_while_position_open": skipped_active,
            "invalid_stop_entries": invalid_stop,
        }
        return table, metrics

    curve = np.r_[INITIAL_EQUITY, table.equity_after.to_numpy(dtype=float)]
    peaks = np.maximum.accumulate(curve)
    drawdown = curve / peaks - 1.0
    metrics = {
        "signals": int(frame.signal.ne("").sum()),
        "long_signals": int(signal_counts.get("long", 0)),
        "short_signals": int(signal_counts.get("short", 0)),
        "trades": int(len(table)),
        "total_return_pct": float((equity / INITIAL_EQUITY - 1.0) * 100.0),
        "win_rate_pct": float(table.net_pnl.gt(0).mean() * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "max_trade_loss_pct": float(table.return_pct_of_entry_equity.min()),
        "dca_trades": int(table.dca_filled.sum()),
        "funding_pnl_pct_of_initial": float(table.funding_pnl.sum() * 100.0),
        "forced_exits": int(table.exit_reason.eq("fold_end").sum()),
        "maximum_leverage_used": float(table.maximum_leverage_used.max()),
        "average_holding_hours": float(table.holding_minutes.mean() / 60.0),
        "skipped_while_position_open": skipped_active,
        "invalid_stop_entries": invalid_stop,
    }
    return table, metrics


def load_fold(schedule: dict[str, Any]) -> pd.DataFrame:
    start = int(schedule["validation_start"])
    end = int(schedule["validation_end"])
    warmup_start = start - 3 * DAY
    raw = pd.read_parquet(
        DATASET,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[("timestamp", ">=", warmup_start), ("timestamp", "<", end)],
    )
    expected = np.arange(warmup_start, end, MINUTE, dtype=np.int64)
    actual = raw.sort_values("timestamp").timestamp.to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(actual, expected)
    bars = add_signal_columns(aggregate_15m(raw))
    result = bars[(bars.timestamp >= start) & (bars.timestamp < end)].reset_index(drop=True)
    if len(result) != 30 * 24 * 4 or not result.timestamp.diff().dropna().eq(STEP).all():
        raise ValueError("Fold does not contain exactly 30 continuous days of 15-minute bars")
    return result


def summarize_cost(folds: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["total_return_pct"]) for row in folds]
    return {
        "fold_returns_pct": returns,
        "fold_trades": [int(row["trades"]) for row in folds],
        "fold_win_rates_pct": [float(row["win_rate_pct"]) for row in folds],
        "positive_folds": int(sum(value > 0 for value in returns)),
        "average_fold_return_pct": float(np.mean(returns)),
        "worst_fold_return_pct": float(min(returns)),
        "total_trades": int(sum(row["trades"] for row in folds)),
        "worst_drawdown_pct": float(min(row["max_drawdown_pct"] for row in folds)),
        "worst_trade_loss_pct": float(min(row["max_trade_loss_pct"] for row in folds)),
        "dca_trades": int(sum(row["dca_trades"] for row in folds)),
        "funding_pnl_pct_of_initial": float(sum(row["funding_pnl_pct_of_initial"] for row in folds)),
        "maximum_leverage_used": float(max(row["maximum_leverage_used"] for row in folds)),
    }


def build_protocol(schedules: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = [DATASET, SOURCE_PROTOCOL, VALUATION / "collection.json"]
    for fold in (1, 2, 3):
        inputs.extend([VALUATION / f"funding_{fold}.json", VALUATION / f"mark1m_{fold}.parquet"])
    return {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": "Offline scenario evaluation only; no model approval and no orders",
        "symbol": "BTCUSDT",
        "market": "Binance USD-M historical artifacts",
        "timeframe": "15m",
        "schedules": schedules,
        "signal": {
            "rsi_period": RSI_PERIOD,
            "long": "RSI <= 30, lower wick >= max(body, 20% of range), close in upper half",
            "short": "RSI >= 70, upper wick >= max(body, 20% of range), close in lower half",
        },
        "execution": {
            "first_entry": "next 15-minute bar open after the completed signal bar",
            "second_entry": "two units at the close of the following bar only when adverse; at most one fill per candle",
            "same_bar_collision": "stop first",
            "stop": "signal low/high plus 5bps outward buffer",
            "take_profit": "2R from current weighted average entry to the fixed structural stop",
            "maximum_holding": "none; any open trade is marked out at the 30-day fold boundary",
        },
        "risk": {
            "risk_per_trade_pct": RISK_PER_TRADE * 100.0,
            "maximum_leverage": MAX_LEVERAGE,
            "dca_unit_ratio": [1, 2],
            "scope": "2% sizing budget covers structural-stop price loss and execution costs; funding is reported separately",
        },
        "costs": {
            name: {
                "base_round_trip_bps": 2 * BASE_EXECUTION_BPS,
                "extra_round_trip_bps": extra,
                "per_fill_bps": execution_bps(extra),
            }
            for name, extra in EXTRA_ROUND_TRIP_BPS.items()
        },
        "funding": "Actual historical rates with causal one-minute mark-price proxies",
        "limitations": [
            "Three reused 30-day development folds are not a new independent holdout.",
            "OHLC bars cannot reveal the exact intrabar path; stop wins a stop/target collision.",
            "Execution costs are fixed assumptions, not account-specific fills.",
            "Funding can make realized trade loss differ from the 2% structural-stop sizing budget.",
            "No liquidation, latency, order-book impact, or exchange rejection simulation.",
        ],
        "inputs": {str(path.relative_to(ROOT)): file_sha256(path) for path in inputs},
        "credentials_used": False,
        "orders_allowed": False,
    }


def render_summary(results: dict[str, Any]) -> str:
    lines = [
        "# RSI 과매수·과매도 꼬리 반전 + 1회 추가 진입 결과",
        "",
        "이 파일은 고정된 과거 데이터로 수행한 오프라인 연구 결과다. 실제 주문은 0건이다.",
        "",
        "| 비용 조건 | 구간 1 | 구간 2 | 구간 3 | 양수 구간 | 총 거래 | 최악 낙폭 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("normal", "stress", "severe"):
        item = results["cost_summary"][name]
        values = item["fold_returns_pct"]
        lines.append(
            f"| {name} | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% | "
            f"{item['positive_folds']}/3 | {item['total_trades']} | {item['worst_drawdown_pct']:.3f}% |"
        )
    lines += [
        "",
        f"판정: **{results['decision']}**",
        "",
        "승인 기준은 기본·스트레스 비용 모두 세 구간 양수, 구간당 10건 이상, 최악 낙폭 -20% 이내다.",
        "이 결과는 개발 구간 재평가이므로 기준을 통과하더라도 독립 미래 검증 전에는 거래에 사용할 수 없다.",
        "",
    ]
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not DATASET.exists() or not VALUATION.exists() or not SOURCE_PROTOCOL.exists():
        raise FileNotFoundError("Required frozen research inputs are missing")
    output.mkdir(parents=True)
    ledger_dir = output / "trades"
    ledger_dir.mkdir()

    source = read_json(SOURCE_PROTOCOL)
    schedules = source["schedules"]
    protocol = build_protocol(schedules)
    write_json(output / "protocol.json", protocol)

    fold_frames = {int(row["fold"]): load_fold(row) for row in schedules}
    markets = {
        fold: FundingMarket(
            read_json(VALUATION / f"funding_{fold}.json"),
            pd.read_parquet(VALUATION / f"mark1m_{fold}.parquet"),
        )
        for fold in (1, 2, 3)
    }

    by_cost: dict[str, list[dict[str, Any]]] = {}
    artifact_hashes: dict[str, str] = {}
    for cost_name, extra_bps in EXTRA_ROUND_TRIP_BPS.items():
        fold_metrics = []
        for fold in (1, 2, 3):
            trades, metrics = simulate_fold(fold_frames[fold], fold, markets[fold], extra_bps)
            path = ledger_dir / f"fold{fold}_{cost_name}.parquet"
            trades.to_parquet(path, index=False)
            artifact_hashes[str(path.relative_to(output))] = file_sha256(path)
            fold_metrics.append({"fold": fold, **metrics})
        by_cost[cost_name] = fold_metrics

    cost_summary = {name: summarize_cost(rows) for name, rows in by_cost.items()}
    normal, stress = cost_summary["normal"], cost_summary["stress"]
    rejection_reasons = []
    if normal["positive_folds"] != 3:
        rejection_reasons.append("normal costs were not positive in all three folds")
    if stress["positive_folds"] != 3:
        rejection_reasons.append("stress costs were not positive in all three folds")
    if min(normal["fold_trades"]) < 10:
        rejection_reasons.append("at least one fold had fewer than 10 trades")
    if normal["worst_drawdown_pct"] < -20.0:
        rejection_reasons.append("normal-cost drawdown exceeded 20%")
    if normal["maximum_leverage_used"] > MAX_LEVERAGE + 1e-9:
        rejection_reasons.append("leverage cap was breached")

    results = {
        "status": "completed",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "folds": by_cost,
        "cost_summary": cost_summary,
        "decision": "rejected" if rejection_reasons else "development_survivor_only",
        "rejection_reasons": rejection_reasons,
        "independent_holdout_evaluated": False,
        "model_approved": False,
        "orders_sent": 0,
        "artifact_sha256": artifact_hashes,
    }
    write_json(output / "results.json", results)
    (output / "SUMMARY.md").write_text(render_summary(results), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"],
        "normal": result["cost_summary"]["normal"],
        "orders_sent": result["orders_sent"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
