import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest model-predicted direction signals.")
    parser.add_argument("--prediction-folder", default="predictions_v2")
    parser.add_argument("--prediction-file")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="spot")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--take-profit-r", type=float, default=1.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--min-minutes-between-entries", type=int, default=15)
    parser.add_argument("--max-new-entries-per-day", type=int, default=5)
    parser.add_argument("--daily-stop-pct", type=float, default=0.04)
    parser.add_argument("--hard-daily-stop-pct", type=float, default=0.06)
    parser.add_argument("--max-consecutive-losses", type=int, default=3)
    parser.add_argument("--extra-slippage-bps", type=float, default=5.0)
    parser.add_argument("--report-path", default="docs/model_signal_backtest_report_v2.md")
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Write JSON and trade Parquet only when a higher-level report owns the Markdown summary.",
    )
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def format_r(value):
    return str(value).replace(".", "_").replace("-", "m")


def find_prediction_file(args):
    if args.prediction_file:
        path = Path(args.prediction_file)
        if not path.exists():
            raise FileNotFoundError(f"예측 파일이 없습니다: {path}")
        return path

    root = Path(args.prediction_folder)
    if args.market != "spot":
        root = root / f"market={safe_value(args.market)}"
    market_suffix = "" if args.market == "spot" else f"_{safe_value(args.market).upper()}"
    path = root / f"symbol={safe_value(args.symbol)}" / f"timeframe={safe_value(args.timeframe)}" / f"direction_predictions_{safe_value(args.symbol)}{market_suffix}_{safe_value(args.timeframe)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"예측 파일이 없습니다: {path}")
    return path


def return_column(action, take_profit_r):
    return f"{action}_tp_{format_r(take_profit_r)}r_return_r_net"


def outcome_column(action, take_profit_r):
    return f"{action}_tp_{format_r(take_profit_r)}r_outcome"


def holding_column(action, take_profit_r):
    return f"{action}_tp_{format_r(take_profit_r)}r_holding_minutes"


def validate_args(args):
    if not 0 < args.risk_per_trade_pct <= 0.02:
        raise ValueError("--risk-per-trade-pct must be greater than 0 and at most 0.02.")
    if args.daily_stop_pct <= 0 or args.hard_daily_stop_pct < args.daily_stop_pct:
        raise ValueError("Daily stop values must be positive and hard stop must not be below daily stop.")
    if args.extra_slippage_bps < 0:
        raise ValueError("--extra-slippage-bps cannot be negative.")
    if args.max_new_entries_per_day <= 0 or args.max_consecutive_losses <= 0:
        raise ValueError("Entry and consecutive-loss limits must be positive.")


def account_return_from_label(row, return_r, args):
    stop_distance_pct = float(row["stop_distance_pct"])
    if stop_distance_pct <= 0:
        raise RuntimeError("stop_distance_pct must be positive.")
    label_cost_r = (float(row["round_trip_cost_bps"]) / 10_000) / stop_distance_pct
    extra_slippage_r = (args.extra_slippage_bps / 10_000) / stop_distance_pct
    adjusted_return_r = float(return_r) - extra_slippage_r
    stop_loss_r_with_costs = 1.0 + label_cost_r + extra_slippage_r
    risk_normalized_r = adjusted_return_r / stop_loss_r_with_costs
    return risk_normalized_r, risk_normalized_r * args.risk_per_trade_pct


def max_drawdown(equity_values):
    peak = 1.0
    mdd = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        drawdown = (equity / peak) - 1
        mdd = min(mdd, drawdown)
    return mdd


def max_consecutive_losses(trades):
    current = 0
    best = 0
    for trade in trades:
        if trade["return_pct"] < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def simulate(predictions, args):
    required = [
        "timestamp",
        "datetime_utc",
        "entry_timestamp",
        "entry_price",
        "entry_rule",
        "label_horizon_complete",
        "pred_action",
        "pred_confidence",
        "direction_target",
        "best_action",
        "stop_distance_pct",
        "round_trip_cost_bps",
        return_column("long", args.take_profit_r),
        return_column("short", args.take_profit_r),
        holding_column("long", args.take_profit_r),
        holding_column("short", args.take_profit_r),
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise RuntimeError(f"백테스트에 필요한 컬럼이 없습니다: {missing}")
    if not predictions["label_horizon_complete"].eq(True).all():
        raise RuntimeError("불완전한 미래 구간 라벨이 예측 파일에 포함되어 있습니다.")
    if not predictions["entry_rule"].eq("next_bar_open").all():
        raise RuntimeError("지원하지 않는 진입 규칙입니다. next_bar_open 예측 파일을 사용하세요.")

    predictions = predictions.sort_values("entry_timestamp").reset_index(drop=True)
    predictions["entry_datetime_utc"] = pd.to_datetime(
        predictions["entry_timestamp"], unit="ms", utc=True
    )
    predictions["date_utc"] = predictions["entry_datetime_utc"].dt.date

    equity = 1.0
    equity_curve = [equity]
    trades = []
    skipped = {
        "low_confidence": 0,
        "no_trade_prediction": 0,
        "daily_entry_limit": 0,
        "daily_stop": 0,
        "hard_daily_stop": 0,
        "min_interval": 0,
        "consecutive_loss_stop": 0,
        "position_already_open": 0,
    }

    day_entries = defaultdict(int)
    day_realized_pnl = defaultdict(float)
    day_start_equity = {}
    last_entry_time = None
    consecutive_losses_by_day = defaultdict(int)
    active_until = None

    for _, row in predictions.iterrows():
        row_time = row["entry_datetime_utc"]
        row_day = row["date_utc"]
        day_start_equity.setdefault(row_day, equity)

        action = str(row["pred_action"])
        confidence = float(row["pred_confidence"])

        if active_until is not None and row_time < active_until:
            skipped["position_already_open"] += 1
            continue
        if action == "no_trade":
            skipped["no_trade_prediction"] += 1
            continue
        if confidence < args.min_confidence:
            skipped["low_confidence"] += 1
            continue
        if day_entries[row_day] >= args.max_new_entries_per_day:
            skipped["daily_entry_limit"] += 1
            continue
        if day_realized_pnl[row_day] <= -(day_start_equity[row_day] * args.hard_daily_stop_pct):
            skipped["hard_daily_stop"] += 1
            continue
        if day_realized_pnl[row_day] <= -(day_start_equity[row_day] * args.daily_stop_pct):
            skipped["daily_stop"] += 1
            continue
        if consecutive_losses_by_day[row_day] >= args.max_consecutive_losses:
            skipped["consecutive_loss_stop"] += 1
            continue
        if last_entry_time is not None:
            minutes_since_last = (row_time - last_entry_time).total_seconds() / 60
            if minutes_since_last < args.min_minutes_between_entries:
                skipped["min_interval"] += 1
                continue

        ret_col = return_column(action, args.take_profit_r)
        out_col = outcome_column(action, args.take_profit_r)
        label_return_r = float(row[ret_col])
        return_r, return_pct = account_return_from_label(row, label_return_r, args)
        holding_minutes = int(row[holding_column(action, args.take_profit_r)])
        active_until = row_time + timedelta(minutes=max(0, holding_minutes))
        realized_day = active_until.date()
        equity_before = equity
        equity *= 1 + return_pct
        equity_curve.append(equity)

        day_entries[row_day] += 1
        day_start_equity.setdefault(realized_day, equity_before)
        day_realized_pnl[realized_day] += equity - equity_before
        last_entry_time = row_time
        if return_pct < 0:
            consecutive_losses_by_day[realized_day] += 1
        else:
            consecutive_losses_by_day[realized_day] = 0

        trades.append(
            {
                "signal_timestamp": int(row["timestamp"]),
                "signal_datetime_utc": str(row["datetime_utc"]),
                "entry_timestamp": int(row["entry_timestamp"]),
                "entry_time_utc": row_time.isoformat(),
                "entry_price": float(row["entry_price"]),
                "exit_time_utc": active_until.isoformat(),
                "date_utc": str(row_day),
                "action": action,
                "confidence": confidence,
                "take_profit_r": args.take_profit_r,
                "outcome": str(row[out_col]) if out_col in row else "unknown",
                "return_r": return_r,
                "label_return_r_net": label_return_r,
                "return_pct": return_pct,
                "holding_minutes": holding_minutes,
                "equity": equity,
                "actual_best_action": str(row["best_action"]) if "best_action" in row else "",
            }
        )

    return trades, equity_curve, skipped


def summarize(trades, equity_curve, skipped, predictions, args):
    total_return_pct = (equity_curve[-1] - 1) * 100 if equity_curve else 0.0
    mdd_pct = max_drawdown(equity_curve) * 100 if equity_curve else 0.0
    wins = [trade for trade in trades if trade["return_pct"] > 0]
    losses = [trade for trade in trades if trade["return_pct"] < 0]

    return {
        "symbol": args.symbol,
        "market": args.market,
        "timeframe": args.timeframe,
        "take_profit_r": args.take_profit_r,
        "risk_per_trade_pct": args.risk_per_trade_pct,
        "min_confidence": args.min_confidence,
        "min_minutes_between_entries": args.min_minutes_between_entries,
        "max_new_entries_per_day": args.max_new_entries_per_day,
        "daily_stop_pct": args.daily_stop_pct,
        "hard_daily_stop_pct": args.hard_daily_stop_pct,
        "extra_slippage_bps": args.extra_slippage_bps,
        "test_rows": len(predictions),
        "trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0.0,
        "loss_rate_pct": len(losses) / len(trades) * 100 if trades else 0.0,
        "avg_return_r": sum(trade["return_r"] for trade in trades) / len(trades) if trades else 0.0,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": mdd_pct,
        "max_consecutive_losses": max_consecutive_losses(trades),
        "largest_realized_loss_pct": min((trade["return_pct"] for trade in trades), default=0.0) * 100,
        "skipped": skipped,
    }


def build_markdown(summary, prediction_file):
    skipped = summary["skipped"]
    lines = [
        "# 모델 신호 기반 백테스트 리포트",
        "",
        f"작성일: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "이 리포트는 1차 direction 모델이 예측한 long/short/no_trade 신호만 사용한 기초 백테스트다.",
        "",
        "## 실행 조건",
        "",
        "```text",
        f"예측 파일: {prediction_file}",
        f"심볼: {summary['symbol']}",
        f"시장: {summary['market']}",
        f"타임프레임: {summary['timeframe']}",
        f"익절 기준: {summary['take_profit_r']}R",
        f"1회 손실폭: 계좌의 {summary['risk_per_trade_pct'] * 100:.2f}%",
        f"최소 신뢰도: {summary['min_confidence']}",
        f"최소 진입 간격: {summary['min_minutes_between_entries']}분",
        f"하루 최대 신규 진입: {summary['max_new_entries_per_day']}회",
        f"하루 손실 중단: -{summary['daily_stop_pct'] * 100:.2f}%",
        f"하루 하드 중단: -{summary['hard_daily_stop_pct'] * 100:.2f}%",
        f"추가 슬리피지 가정: {summary['extra_slippage_bps']:.2f} bps",
        "진입 시점: 신호 확정 다음 봉 시가",
        "동시 포지션: 1개만 허용",
        "비용 포함 스탑 손실: 계좌 위험 한도에 맞게 정규화",
        "청산: 허용하지 않는 전제",
        "스탑: 정상 체결 전제",
        "```",
        "",
        "## 결과 요약",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 테스트 rows | {summary['test_rows']} |",
        f"| 실제 거래 수 | {summary['trades']} |",
        f"| 승률 | {summary['win_rate_pct']:.2f}% |",
        f"| 손실률 | {summary['loss_rate_pct']:.2f}% |",
        f"| 평균 R | {summary['avg_return_r']:.4f} |",
        f"| 총 복리 수익률 | {summary['total_return_pct']:.2f}% |",
        f"| 최대 낙폭 | {summary['max_drawdown_pct']:.2f}% |",
        f"| 최장 연속 손실 | {summary['max_consecutive_losses']} |",
        f"| 거래 1회의 최대 실현 손실 | {summary['largest_realized_loss_pct']:.2f}% |",
        "",
        "## Skip 사유",
        "",
        "| 사유 | 횟수 |",
        "| --- | ---: |",
    ]

    for key, value in skipped.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 예측 파일의 시간순 테스트 구간만 사용한 오프라인 연구 결과다.",
            "- 한 포지션이 종료되기 전의 신규 신호는 제외한다.",
            "- 연속 손실 한도에 도달하면 해당 UTC 날짜의 신규 진입을 중단하고 다음 날짜에 재개한다.",
            "- 라벨 비용과 추가 슬리피지를 포함한 스탑 손실이 계좌 위험 한도를 넘지 않도록 손익을 정규화한다.",
            "- 미래 캔들로 만든 Triple Barrier 결과를 사용하므로 실시간 페이퍼 트레이딩 성과가 아니다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary, trades, args, prediction_file):
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_markdown:
        report_path.write_text(build_markdown(summary, prediction_file), encoding="utf-8")

    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    trades_path = report_path.with_name(report_path.stem + "_trades.parquet")
    pd.DataFrame(trades).to_parquet(trades_path, index=False)
    return report_path, json_path, trades_path


def main():
    args = parse_args()
    validate_args(args)
    prediction_file = find_prediction_file(args)
    predictions = pd.read_parquet(prediction_file)

    trades, equity_curve, skipped = simulate(predictions, args)
    summary = summarize(trades, equity_curve, skipped, predictions, args)
    report_path, json_path, trades_path = write_outputs(summary, trades, args, prediction_file)

    if args.skip_markdown:
        print("모델 신호 백테스트 Markdown: 상위 통합 보고서에서 관리")
    else:
        print(f"모델 신호 백테스트 리포트 저장: {report_path}")
    print(f"모델 신호 백테스트 JSON 저장: {json_path}")
    print(f"거래 로그 저장: {trades_path}")
    print(
        f"trades={summary['trades']}, win={summary['win_rate_pct']:.2f}%, "
        f"avgR={summary['avg_return_r']:.4f}, return={summary['total_return_pct']:.2f}%, "
        f"MDD={summary['max_drawdown_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
