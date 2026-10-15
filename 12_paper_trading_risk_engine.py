import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


REPLAY_WARNING = (
    "오프라인 재생 전용: 이 스크립트는 거래소 주문을 보내지 않습니다. "
    "미래 라벨 결과를 사용하므로 실시간 페이퍼 트레이딩 성과로 해석하면 안 됩니다."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create risk-limited paper trade plans from model prediction replay data."
    )
    parser.add_argument("--prediction-folder", default="predictions_v2")
    parser.add_argument("--prediction-file")
    parser.add_argument("--output-folder", default="paper_trading_v2")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="usdm")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--account-balance", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--max-margin-pct", type=float, default=0.35)
    parser.add_argument("--take-profit-r", type=float, default=1.5)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-minutes-between-entries", type=int, default=15)
    parser.add_argument("--max-new-entries-per-day", type=int, default=3)
    parser.add_argument("--daily-stop-pct", type=float, default=0.04)
    parser.add_argument("--weekly-stop-pct", type=float, default=0.08)
    parser.add_argument("--max-consecutive-losses", type=int, default=3)
    parser.add_argument("--extra-slippage-bps", type=float, default=5.0)
    parser.add_argument(
        "--liquidation-distance-pct",
        type=float,
        required=True,
        help=(
            "Conservative entry-to-liquidation distance from the exchange's actual isolated-margin "
            "calculation, expressed as decimal. Example: 0.10 for 10%%. This is not inferred from leverage."
        ),
    )
    parser.add_argument(
        "--liquidation-buffer-pct",
        type=float,
        default=0.01,
        help="Additional price-distance buffer required between stop and liquidation price. 0.01 = 1%%.",
    )
    parser.add_argument("--report-path", default="docs/paper_trading_risk_replay_report_v2.md")
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def format_r(value):
    return str(value).replace(".", "_").replace("-", "m")


def market_root(folder, market, symbol, timeframe):
    root = Path(folder)
    if market != "spot":
        root = root / f"market={safe_value(market)}"
    return root / f"symbol={safe_value(symbol)}" / f"timeframe={safe_value(timeframe)}"


def prediction_path(args):
    if args.prediction_file:
        path = Path(args.prediction_file)
    else:
        suffix = "" if args.market == "spot" else f"_{safe_value(args.market).upper()}"
        path = market_root(args.prediction_folder, args.market, args.symbol, args.timeframe) / (
            f"direction_predictions_{safe_value(args.symbol)}{suffix}_{safe_value(args.timeframe)}.parquet"
        )
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return path


def validate_args(args):
    positive_values = {
        "account balance": args.account_balance,
        "risk per trade": args.risk_per_trade_pct,
        "leverage": args.leverage,
        "liquidation distance": args.liquidation_distance_pct,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    if not 0 < args.max_margin_pct <= 1:
        raise ValueError("--max-margin-pct must be greater than 0 and at most 1.")
    if args.risk_per_trade_pct > 0.02:
        raise ValueError("Risk policy violation: --risk-per-trade-pct cannot exceed 0.02.")
    if args.daily_stop_pct < args.risk_per_trade_pct:
        raise ValueError("--daily-stop-pct must be at least --risk-per-trade-pct.")
    if args.weekly_stop_pct < args.daily_stop_pct:
        raise ValueError("--weekly-stop-pct must be at least --daily-stop-pct.")


def required_columns(args):
    suffix = format_r(args.take_profit_r)
    return [
        "timestamp",
        "datetime_utc",
        "entry_timestamp",
        "entry_price",
        "entry_rule",
        "label_horizon_complete",
        "stop_distance_pct",
        "max_holding_minutes",
        "round_trip_cost_bps",
        "pred_action",
        "pred_confidence",
        f"long_tp_{suffix}r_outcome",
        f"short_tp_{suffix}r_outcome",
        f"long_tp_{suffix}r_return_r_gross",
        f"short_tp_{suffix}r_return_r_gross",
        f"long_tp_{suffix}r_holding_minutes",
        f"short_tp_{suffix}r_holding_minutes",
    ]


def load_predictions(args):
    path = prediction_path(args)
    predictions = pd.read_parquet(path).sort_values("entry_timestamp").reset_index(drop=True)
    missing = [column for column in required_columns(args) if column not in predictions.columns]
    if missing:
        raise RuntimeError(
            "Prediction replay data is missing required columns. "
            f"Use labels generated with --take-profit-r including {args.take_profit_r}: {missing}"
        )
    if not predictions["label_horizon_complete"].eq(True).all():
        raise RuntimeError("Prediction replay data contains incomplete label horizons.")
    if not predictions["entry_rule"].eq("next_bar_open").all():
        raise RuntimeError("Prediction replay data must use the next_bar_open entry rule.")
    predictions["datetime_utc"] = pd.to_datetime(predictions["datetime_utc"], utc=True)
    predictions["entry_datetime_utc"] = pd.to_datetime(
        predictions["entry_timestamp"], unit="ms", utc=True
    )
    return predictions, path


def week_key(value):
    iso = value.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def side_prices(side, entry_price, stop_distance_pct, take_profit_r, liquidation_distance_pct):
    if side == "long":
        return (
            entry_price * (1 - stop_distance_pct),
            entry_price * (1 + stop_distance_pct * take_profit_r),
            entry_price * (1 - liquidation_distance_pct),
        )
    return (
        entry_price * (1 + stop_distance_pct),
        entry_price * (1 - stop_distance_pct * take_profit_r),
        entry_price * (1 + liquidation_distance_pct),
    )


def make_trade_plan(row, args, equity):
    side = str(row["pred_action"])
    entry_price = float(row["entry_price"])
    stop_distance_pct = float(row["stop_distance_pct"])
    base_cost_pct = float(row["round_trip_cost_bps"]) / 10_000
    extra_slippage_pct = args.extra_slippage_bps / 10_000
    effective_stop_pct = stop_distance_pct + base_cost_pct + extra_slippage_pct
    risk_budget = equity * args.risk_per_trade_pct
    notional = risk_budget / effective_stop_pct
    required_margin = notional / args.leverage
    quantity = notional / entry_price
    stop_price, take_price, estimated_liquidation_price = side_prices(
        side, entry_price, stop_distance_pct, args.take_profit_r, args.liquidation_distance_pct
    )
    liquidation_distance_pct = args.liquidation_distance_pct
    liquidation_buffer_ok = liquidation_distance_pct >= (
        effective_stop_pct + args.liquidation_buffer_pct
    )
    gross_return_r = float(row[f"{side}_tp_{format_r(args.take_profit_r)}r_return_r_gross"])
    realized_return_pct = gross_return_r * stop_distance_pct - base_cost_pct - extra_slippage_pct
    return {
        "entry_price": entry_price,
        "stop_distance_pct": stop_distance_pct,
        "base_cost_pct": base_cost_pct,
        "extra_slippage_pct": extra_slippage_pct,
        "effective_stop_pct": effective_stop_pct,
        "risk_budget": risk_budget,
        "notional": notional,
        "required_margin": required_margin,
        "required_margin_pct": required_margin / equity,
        "quantity": quantity,
        "stop_price": stop_price,
        "take_price": take_price,
        "estimated_liquidation_price": estimated_liquidation_price,
        "liquidation_buffer_ok": liquidation_buffer_ok,
        "gross_return_r": gross_return_r,
        "realized_return_pct": realized_return_pct,
        "realized_pnl": notional * realized_return_pct,
        "outcome": str(row[f"{side}_tp_{format_r(args.take_profit_r)}r_outcome"]),
        "holding_minutes": int(row[f"{side}_tp_{format_r(args.take_profit_r)}r_holding_minutes"]),
    }


def replay(predictions, args):
    equity = args.account_balance
    peak_equity = equity
    accepted = []
    rejected = []
    rejection_counts = Counter()
    active_until = None
    last_entry_time = None
    day_entries = defaultdict(int)
    day_pnl = defaultdict(float)
    week_pnl = defaultdict(float)
    day_start_equity = {}
    week_start_equity = {}
    consecutive_losses = 0

    for _, row in predictions.iterrows():
        timestamp = row["entry_datetime_utc"]
        day = timestamp.date()
        week = week_key(timestamp)
        day_start_equity.setdefault(day, equity)
        week_start_equity.setdefault(week, equity)

        reason = None
        side = str(row["pred_action"])
        confidence = float(row["pred_confidence"])
        if side not in {"long", "short"}:
            reason = "no_trade_prediction"
        elif confidence < args.min_confidence:
            reason = "low_confidence"
        elif active_until is not None and timestamp < active_until:
            reason = "position_already_open"
        elif day_entries[day] >= args.max_new_entries_per_day:
            reason = "daily_entry_limit"
        elif day_pnl[day] <= -(day_start_equity[day] * args.daily_stop_pct):
            reason = "daily_stop"
        elif week_pnl[week] <= -(week_start_equity[week] * args.weekly_stop_pct):
            reason = "weekly_stop"
        elif consecutive_losses >= args.max_consecutive_losses:
            reason = "consecutive_loss_stop"
        elif last_entry_time is not None and (timestamp - last_entry_time).total_seconds() < args.min_minutes_between_entries * 60:
            reason = "minimum_entry_interval"

        plan = None
        if reason is None:
            plan = make_trade_plan(row, args, equity)
            if plan["required_margin_pct"] > args.max_margin_pct:
                reason = "max_margin_pct_exceeded"
            elif not plan["liquidation_buffer_ok"]:
                reason = "liquidation_buffer_failed"

        if reason is not None:
            rejection_counts[reason] += 1
            rejected.append(
                {
                    "timestamp": timestamp,
                    "side": side,
                    "confidence": confidence,
                    "reason": reason,
                }
            )
            continue

        pnl = plan["realized_pnl"]
        equity_before = equity
        equity += pnl
        peak_equity = max(peak_equity, equity)
        day_entries[day] += 1
        last_entry_time = timestamp
        active_until = timestamp + timedelta(minutes=plan["holding_minutes"])
        realized_day = active_until.date()
        realized_week = week_key(active_until)
        day_start_equity.setdefault(realized_day, equity_before)
        week_start_equity.setdefault(realized_week, equity_before)
        day_pnl[realized_day] += pnl
        week_pnl[realized_week] += pnl
        consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
        accepted.append(
            {
                "signal_time_utc": row["datetime_utc"],
                "entry_time_utc": timestamp,
                "exit_time_utc": active_until,
                "side": side,
                "confidence": confidence,
                "outcome": plan["outcome"],
                "entry_price": plan["entry_price"],
                "stop_price": plan["stop_price"],
                "take_price": plan["take_price"],
                "estimated_liquidation_price": plan["estimated_liquidation_price"],
                "quantity_btc": plan["quantity"],
                "notional_usdt": plan["notional"],
                "required_margin_usdt": plan["required_margin"],
                "required_margin_pct": plan["required_margin_pct"],
                "risk_budget_usdt": plan["risk_budget"],
                "effective_stop_pct": plan["effective_stop_pct"],
                "gross_return_r": plan["gross_return_r"],
                "realized_return_pct": plan["realized_return_pct"],
                "realized_pnl_usdt": pnl,
                "equity_before_usdt": equity_before,
                "equity_after_usdt": equity,
                "drawdown_pct": (equity / peak_equity) - 1,
            }
        )

    return accepted, rejected, rejection_counts


def build_summary(accepted, rejected, rejection_counts, args, prediction_file):
    trades = pd.DataFrame(accepted)
    final_equity = float(trades.iloc[-1]["equity_after_usdt"]) if not trades.empty else args.account_balance
    total_pnl = final_equity - args.account_balance
    return {
        "warning": REPLAY_WARNING,
        "prediction_file": str(prediction_file),
        "account_balance_usdt": args.account_balance,
        "final_equity_usdt": final_equity,
        "total_pnl_usdt": total_pnl,
        "total_return_pct": total_pnl / args.account_balance * 100,
        "accepted_trades": len(accepted),
        "rejected_signals": len(rejected),
        "win_rate_pct": float((trades["realized_pnl_usdt"] > 0).mean() * 100) if not trades.empty else 0.0,
        "max_drawdown_pct": float(trades["drawdown_pct"].min() * 100) if not trades.empty else 0.0,
        "max_realized_loss_pct": float((-trades["realized_pnl_usdt"] / trades["equity_before_usdt"]).max() * 100)
        if not trades.empty
        else 0.0,
        "rejection_counts": dict(rejection_counts),
        "settings": {
            "entry_rule": "next_bar_open",
            "risk_per_trade_pct": args.risk_per_trade_pct,
            "leverage": args.leverage,
            "max_margin_pct": args.max_margin_pct,
            "take_profit_r": args.take_profit_r,
            "min_confidence": args.min_confidence,
            "daily_stop_pct": args.daily_stop_pct,
            "weekly_stop_pct": args.weekly_stop_pct,
            "max_consecutive_losses": args.max_consecutive_losses,
            "extra_slippage_bps": args.extra_slippage_bps,
            "liquidation_distance_pct_input": args.liquidation_distance_pct,
            "liquidation_buffer_pct": args.liquidation_buffer_pct,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_markdown(summary, trades_path, rejected_path, json_path):
    lines = [
        "# 페이퍼 트레이딩 리스크 재생 보고서",
        "",
        f"작성 시각: {summary['created_at_utc']}",
        "",
        f"> {summary['warning']}",
        "",
        "## 결과",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 시작 자산 | {summary['account_balance_usdt']:.2f} USDT |",
        f"| 종료 자산 | {summary['final_equity_usdt']:.2f} USDT |",
        f"| 총수익률 | {summary['total_return_pct']:.2f}% |",
        f"| 허용 거래 | {summary['accepted_trades']} |",
        f"| 거절 신호 | {summary['rejected_signals']} |",
        f"| 승률 | {summary['win_rate_pct']:.2f}% |",
        f"| 최대 낙폭 | {summary['max_drawdown_pct']:.2f}% |",
        f"| 최대 실현 손실 | {summary['max_realized_loss_pct']:.2f}% |",
        "",
        "## 리스크 설정",
        "",
        "```text",
    ]
    for key, value in summary["settings"].items():
        lines.append(f"{key} = {value}")
    lines.extend(
        [
            "```",
            "",
            "## 신호 거절 사유",
            "",
            "| 사유 | 횟수 |",
            "| --- | ---: |",
        ]
    )
    for reason, count in sorted(summary["rejection_counts"].items()):
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## 결과 파일",
            "",
            f"- 허용 거래: `{trades_path}`",
            f"- 거절 신호: `{rejected_path}`",
            f"- JSON 요약: `{json_path}`",
            "",
            "## 해석",
            "",
            "위 결과는 이후 캔들로 만든 Triple Barrier 라벨을 사용했습니다. "
            "오프라인 연구 재생에만 유효합니다. 실제 페이퍼 트레이딩에서는 미래 결과 대신 "
            "실시간 시장 가격과 가상 주문 상태 이벤트를 사용해야 합니다.",
            "",
            "`estimated_liquidation_price`는 명령행 입력값을 사용한 거리 기반 보호값일 뿐입니다. "
            "거래소의 실제 청산가 계산이 아니며 실거래 주문을 허용하는 근거가 될 수 없습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_outputs(args, accepted, rejected, summary):
    output_root = market_root(args.output_folder, args.market, args.symbol, args.timeframe)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trades_path = output_root / f"risk_replay_trades_{run_id}.parquet"
    rejected_path = output_root / f"risk_replay_rejected_{run_id}.parquet"
    json_path = output_root / f"risk_replay_summary_{run_id}.json"
    pd.DataFrame(accepted).to_parquet(trades_path, index=False)
    pd.DataFrame(rejected).to_parquet(rejected_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown(summary, trades_path, rejected_path, json_path), encoding="utf-8")
    return trades_path, rejected_path, json_path, report_path


def main():
    args = parse_args()
    validate_args(args)
    predictions, source_path = load_predictions(args)
    accepted, rejected, rejection_counts = replay(predictions, args)
    summary = build_summary(accepted, rejected, rejection_counts, args, source_path)
    trades_path, rejected_path, json_path, report_path = save_outputs(args, accepted, rejected, summary)

    print(REPLAY_WARNING)
    print(f"Prediction source: {source_path}")
    print(f"Accepted trades: {summary['accepted_trades']}")
    print(f"Rejected signals: {summary['rejected_signals']}")
    print(f"Final equity: {summary['final_equity_usdt']:.2f} USDT")
    print(f"Largest realized loss: {summary['max_realized_loss_pct']:.2f}%")
    print(f"Trades: {trades_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
