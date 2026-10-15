import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTCOME_PATTERN = re.compile(r"^(long|short)_tp_(.+)r_outcome$")


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest Triple Barrier R labels.")
    parser.add_argument("--label-folder", default="label_store_v2")
    parser.add_argument("--label-file")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="spot")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    parser.add_argument("--entry-interval-minutes", type=int, default=15)
    parser.add_argument("--extra-slippage-bps", type=float, default=5.0)
    parser.add_argument("--report-path", default="docs/triple_barrier_backtest_report_v2.md")
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def find_label_file(args):
    if args.label_file:
        path = Path(args.label_file)
        if not path.exists():
            raise FileNotFoundError(f"라벨 파일이 없습니다: {path}")
        return path

    root = Path(args.label_folder)
    if args.market != "spot":
        root = root / f"market={safe_value(args.market)}"
    root = root / f"symbol={safe_value(args.symbol)}" / f"timeframe={safe_value(args.timeframe)}"
    paths = sorted(root.glob("**/labels_*.parquet"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"라벨 파일이 없습니다: {root}")
    return paths[0]


def infer_step_minutes(df):
    timestamps = df["timestamp"].sort_values().drop_duplicates()
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return 1
    return max(1, round(float(diffs.median()) / 60_000))


def display_r(r_key):
    return r_key.replace("_", ".")


def max_consecutive_losses(returns):
    current = 0
    best = 0
    for value in returns:
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def equity_metrics(return_r_values, risk_per_trade_pct):
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for return_r in return_r_values:
        trade_return = float(return_r) * risk_per_trade_pct
        equity *= 1 + trade_return
        peak = max(peak, equity)
        drawdown = (equity / peak) - 1
        max_drawdown = min(max_drawdown, drawdown)

    return {
        "compound_return_pct": (equity - 1) * 100,
        "max_drawdown_pct": max_drawdown * 100,
    }


def select_non_overlapping(df, holding_col):
    selected_indexes = []
    active_until = None
    for index, row in df.sort_values("entry_timestamp").iterrows():
        timestamp = int(row["entry_timestamp"])
        if active_until is not None and timestamp < active_until:
            continue
        selected_indexes.append(index)
        active_until = timestamp + int(row[holding_col]) * 60_000
    return df.loc[selected_indexes].copy(), len(df) - len(selected_indexes)


def risk_normalized_returns(df, return_col, extra_slippage_bps):
    stop_distance = df["stop_distance_pct"].astype(float)
    if (stop_distance <= 0).any():
        raise RuntimeError("stop_distance_pct must be positive.")
    label_cost_r = (df["round_trip_cost_bps"].astype(float) / 10_000) / stop_distance
    extra_slippage_r = (extra_slippage_bps / 10_000) / stop_distance
    return (df[return_col].astype(float) - extra_slippage_r) / (
        1.0 + label_cost_r + extra_slippage_r
    )


def summarize_case(df, side, r_key, args):
    outcome_col = f"{side}_tp_{r_key}r_outcome"
    return_col = f"{side}_tp_{r_key}r_return_r_net"
    holding_col = f"{side}_tp_{r_key}r_holding_minutes"

    df, overlapping_entries_removed = select_non_overlapping(df, holding_col)
    outcomes = df[outcome_col]
    returns = risk_normalized_returns(df, return_col, args.extra_slippage_bps)

    take_profit_count = int((outcomes == "take_profit").sum())
    stop_loss_count = int((outcomes == "stop_loss").sum())
    time_expired_count = int((outcomes == "time_expired").sum())
    total = len(df)

    simple_return_pct = float(returns.sum() * args.risk_per_trade_pct * 100)
    equity = equity_metrics(returns, args.risk_per_trade_pct)

    return {
        "side": side,
        "take_profit_r": display_r(r_key),
        "trades": total,
        "take_profit_rate_pct": take_profit_count / total * 100 if total else 0,
        "stop_loss_rate_pct": stop_loss_count / total * 100 if total else 0,
        "time_expired_rate_pct": time_expired_count / total * 100 if total else 0,
        "avg_return_r_net": float(returns.mean()) if total else 0,
        "simple_return_pct": simple_return_pct,
        "compound_return_pct": equity["compound_return_pct"],
        "max_drawdown_pct": equity["max_drawdown_pct"],
        "max_consecutive_losses": max_consecutive_losses(returns),
        "avg_holding_minutes": float(df[holding_col].mean()) if total and holding_col in df.columns else 0,
        "overlapping_entries_removed": overlapping_entries_removed,
    }


def find_cases(df):
    cases = []
    for column in df.columns:
        match = OUTCOME_PATTERN.match(column)
        if match:
            side, r_key = match.groups()
            return_col = f"{side}_tp_{r_key}r_return_r_net"
            if return_col in df.columns:
                cases.append((side, r_key))
    return sorted(cases)


def build_markdown(report, label_file, args):
    lines = [
        "# Triple Barrier 라벨 백테스트 리포트",
        "",
        f"작성일: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "이 리포트는 머신러닝 전략 성과가 아니라, 생성된 Triple Barrier 라벨을 기준으로 R 단위 결과 분포를 확인하는 기초 백테스트다.",
        "",
        "## 실행 조건",
        "",
        "```text",
        f"라벨 파일: {label_file}",
        f"심볼: {args.symbol}",
        f"시장: {args.market}",
        f"타임프레임: {args.timeframe}",
        f"1회 손실폭: 계좌의 {args.risk_per_trade_pct * 100:.2f}%",
        f"가상 진입 간격: {args.entry_interval_minutes}분",
        f"추가 슬리피지: {args.extra_slippage_bps:.2f} bps",
        "진입 시점: 신호 확정 다음 봉 시가",
        "동시 포지션: 1개",
        "청산: 허용하지 않는 전제",
        "스탑: 정상 체결 전제",
        "```",
        "",
        "## 결과 요약",
        "",
        "| 방향 | 익절 R | 거래 수 | 중복제외 | TP% | SL% | 시간만료% | 평균 R | 단순수익률% | 복리수익률% | MDD% | 최장연속손실 | 평균보유분 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in report["cases"]:
        lines.append(
            "| {side} | {take_profit_r} | {trades} | {overlapping_entries_removed} | {take_profit_rate_pct:.2f} | "
            "{stop_loss_rate_pct:.2f} | {time_expired_rate_pct:.2f} | {avg_return_r_net:.4f} | "
            "{simple_return_pct:.2f} | {compound_return_pct:.2f} | {max_drawdown_pct:.2f} | "
            "{max_consecutive_losses} | {avg_holding_minutes:.1f} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 이 결과는 모든 신호를 머신러닝이 고른 것이 아니라, 일정 간격으로 가상 진입했을 때의 라벨 분포를 보는 기초 검증이다.",
            "- 실제 모델은 이 라벨을 학습해서 기대값이 낮은 구간을 거래하지 않는 방향으로 개선해야 한다.",
            "- 단순수익률과 복리수익률이 좋아 보여도, 최대 낙폭과 연속 손실이 크면 실전 적용 전 필터가 필요하다.",
            "- 다음 단계는 이 라벨을 학습 데이터에 붙여 ML 모델이 롱/숏/관망과 익절 R 후보를 선택하게 만드는 것이다.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if not 0 < args.risk_per_trade_pct <= 0.02:
        raise ValueError("--risk-per-trade-pct must be greater than 0 and at most 0.02.")
    if args.extra_slippage_bps < 0:
        raise ValueError("--extra-slippage-bps cannot be negative.")
    if args.entry_interval_minutes <= 0:
        raise ValueError("--entry-interval-minutes must be positive.")
    label_file = find_label_file(args)
    labels = pd.read_parquet(label_file)
    required_metadata = {"label_horizon_complete", "entry_timestamp", "entry_price", "entry_rule"}
    missing_metadata = sorted(required_metadata - set(labels.columns))
    if missing_metadata:
        raise RuntimeError(
            "오래된 라벨 파일입니다. 4_triple_barrier_labeler.py로 다시 생성하세요. "
            f"누락 컬럼={missing_metadata}"
        )
    labels = labels[labels["label_horizon_complete"].eq(True)].reset_index(drop=True)
    if not labels["entry_rule"].eq("next_bar_open").all():
        raise RuntimeError("지원하지 않는 진입 규칙입니다. next_bar_open 라벨을 사용하세요.")
    if "market" not in labels.columns:
        labels["market"] = "spot"
    labels = labels[labels["market"].astype(str) == args.market].reset_index(drop=True)
    if labels.empty:
        raise RuntimeError(f"시장 조건에 맞는 라벨이 없습니다: market={args.market}")

    step_minutes = infer_step_minutes(labels)
    step_rows = max(1, round(args.entry_interval_minutes / step_minutes))
    sampled = labels.iloc[::step_rows].reset_index(drop=True)

    cases = []
    for side, r_key in find_cases(sampled):
        cases.append(summarize_case(sampled, side, r_key, args))

    report = {
        "label_file": str(label_file),
        "symbol": args.symbol,
        "market": args.market,
        "timeframe": args.timeframe,
        "risk_per_trade_pct": args.risk_per_trade_pct,
        "entry_interval_minutes": args.entry_interval_minutes,
        "extra_slippage_bps": args.extra_slippage_bps,
        "sampled_rows": len(sampled),
        "cases": cases,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown(report, label_file, args), encoding="utf-8")

    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"백테스트 리포트 저장: {report_path}")
    print(f"백테스트 JSON 저장: {json_path}")
    for item in cases:
        print(
            f"{item['side']} {item['take_profit_r']}R: "
            f"trades={item['trades']}, TP={item['take_profit_rate_pct']:.2f}%, "
            f"avgR={item['avg_return_r_net']:.4f}, compound={item['compound_return_pct']:.2f}%, "
            f"MDD={item['max_drawdown_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
