import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "ohlcv_basic_v2_boundary4"
REQUIRED_COLUMNS = [
    "timestamp",
    "datetime_utc",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "feature_schema_version",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create Triple Barrier labels for BTCUSDT features.")
    parser.add_argument("--feature-folder", default="feature_store_v2")
    parser.add_argument("--label-folder", default="label_store_v2")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=["spot", "usdm"], default="spot")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--stop-distance-pct", type=float, default=0.005, help="Price stop distance. 0.005 = 0.5%%.")
    parser.add_argument("--take-profit-r", default="1,1.5,3", help="Comma-separated R targets.")
    parser.add_argument("--max-holding-minutes", type=int, default=240)
    parser.add_argument("--round-trip-cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Vectorized label calculation batch size. Lower this when memory is limited.",
    )
    parser.add_argument(
        "--verify-sample-size",
        type=int,
        default=100,
        help="Rows checked at both the start and end against the legacy per-bar implementation. Set 0 to skip.",
    )
    return parser.parse_args()


def safe_value(value):
    return str(value).replace("/", "").replace(":", "").replace(" ", "")


def parse_take_profit_r(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("--take-profit-r는 하나 이상의 숫자가 필요합니다.")
    return values


def market_root(folder, market, symbol, timeframe):
    root = Path(folder)
    if market != "spot":
        root = root / f"market={safe_value(market)}"
    return root / f"symbol={safe_value(symbol)}" / f"timeframe={safe_value(timeframe)}"


def load_features(feature_folder, symbol, market, timeframe):
    partition_root = market_root(feature_folder, market, symbol, timeframe)
    if partition_root.exists():
        paths = sorted(partition_root.glob("**/*.parquet"))
    else:
        paths = sorted(Path(feature_folder).glob("**/*.parquet"))

    if not paths:
        raise RuntimeError(f"Feature parquet 파일이 없습니다: {feature_folder}")

    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    if "market" not in df.columns:
        df["market"] = "spot"

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"라벨링에 필요한 Feature 컬럼이 없습니다: {missing}")

    df = df[df["symbol"].astype(str).str.replace("/", "", regex=False) == safe_value(symbol)]
    df = df[df["market"].astype(str) == market]
    df = df[df["timeframe"].astype(str) == timeframe]
    df = df.drop_duplicates(subset=["symbol", "market", "timeframe", "timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"라벨링할 데이터가 없습니다: symbol={symbol}, timeframe={timeframe}")
    if set(df["feature_schema_version"].astype(str)) != {FEATURE_SCHEMA_VERSION}:
        raise RuntimeError("지원하지 않는 Feature Store 스키마 버전입니다.")
    return df


def infer_step_minutes(df):
    timestamps = df["timestamp"].sort_values().drop_duplicates()
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return 1
    median_ms = float(diffs.median())
    return max(1, round(median_ms / 60_000))


def format_r(value):
    return str(value).replace(".", "_").replace("-", "m")


def make_event_result(side, entry, exit_price, stop_distance_pct, round_trip_cost_pct, gross_return_r, exit_index, outcome):
    if side == "long":
        gross_return_pct = (exit_price - entry) / entry
    else:
        gross_return_pct = (entry - exit_price) / entry

    if outcome in ["take_profit", "stop_loss"]:
        gross_r = gross_return_r
    else:
        gross_r = gross_return_pct / stop_distance_pct

    net_r = (gross_return_pct - round_trip_cost_pct) / stop_distance_pct
    if outcome == "take_profit":
        net_r = gross_r - (round_trip_cost_pct / stop_distance_pct)
    elif outcome == "stop_loss":
        net_r = gross_r - (round_trip_cost_pct / stop_distance_pct)

    return outcome, gross_r, net_r, exit_index


def evaluate_barrier(df, index, side, take_profit_r, stop_distance_pct, max_bars, round_trip_cost_pct):
    entry_index = index + 1
    if entry_index >= len(df):
        return "invalid", 0.0, 0.0, index
    entry = float(df.at[entry_index, "open"])
    if entry <= 0:
        return "invalid", 0.0, 0.0, index

    if side == "long":
        stop_price = entry * (1 - stop_distance_pct)
        take_price = entry * (1 + stop_distance_pct * take_profit_r)
    else:
        stop_price = entry * (1 + stop_distance_pct)
        take_price = entry * (1 - stop_distance_pct * take_profit_r)

    end_index = min(index + max_bars, len(df) - 1)
    if end_index <= index:
        return "time_expired", 0.0, 0.0, index

    for cursor in range(entry_index, end_index + 1):
        high = float(df.at[cursor, "high"])
        low = float(df.at[cursor, "low"])

        if side == "long":
            stop_hit = low <= stop_price
            take_hit = high >= take_price
            if stop_hit:
                return make_event_result(side, entry, stop_price, stop_distance_pct, round_trip_cost_pct, -1.0, cursor, "stop_loss")
            if take_hit:
                return make_event_result(side, entry, take_price, stop_distance_pct, round_trip_cost_pct, take_profit_r, cursor, "take_profit")
        else:
            stop_hit = high >= stop_price
            take_hit = low <= take_price
            if stop_hit:
                return make_event_result(side, entry, stop_price, stop_distance_pct, round_trip_cost_pct, -1.0, cursor, "stop_loss")
            if take_hit:
                return make_event_result(side, entry, take_price, stop_distance_pct, round_trip_cost_pct, take_profit_r, cursor, "take_profit")

    exit_price = float(df.at[end_index, "close"])
    return make_event_result(side, entry, exit_price, stop_distance_pct, round_trip_cost_pct, 0.0, end_index, "time_expired")


def evaluate_barrier_vectorized(
    entries,
    close,
    high,
    low,
    side,
    take_profit_r,
    stop_distance_pct,
    max_bars,
    round_trip_cost_pct,
    batch_size,
):
    """Evaluate one side/target in batches while preserving stop-first intrabar behavior."""
    row_count = len(close)
    outcomes = np.full(row_count, "time_expired", dtype=object)
    gross_returns = np.zeros(row_count, dtype=float)
    net_returns = np.zeros(row_count, dtype=float)
    holding_bars = np.zeros(row_count, dtype=int)

    padded_high = np.concatenate([high[1:], np.full(max_bars, np.nan)])
    padded_low = np.concatenate([low[1:], np.full(max_bars, np.nan)])
    high_windows = np.lib.stride_tricks.sliding_window_view(padded_high, max_bars)
    low_windows = np.lib.stride_tricks.sliding_window_view(padded_low, max_bars)
    no_hit = max_bars
    cost_r = round_trip_cost_pct / stop_distance_pct

    for start in range(0, row_count, batch_size):
        end = min(start + batch_size, row_count)
        batch_entries = entries[start:end]
        valid_entries = batch_entries > 0
        horizon = np.minimum(max_bars, row_count - 1 - np.arange(start, end))

        batch_high = high_windows[start:end]
        batch_low = low_windows[start:end]
        if side == "long":
            stop_hit = batch_low <= (batch_entries[:, None] * (1 - stop_distance_pct))
            take_hit = batch_high >= (batch_entries[:, None] * (1 + stop_distance_pct * take_profit_r))
        else:
            stop_hit = batch_high >= (batch_entries[:, None] * (1 + stop_distance_pct))
            take_hit = batch_low <= (batch_entries[:, None] * (1 - stop_distance_pct * take_profit_r))

        stop_any = stop_hit.any(axis=1)
        take_any = take_hit.any(axis=1)
        first_stop = np.where(stop_any, stop_hit.argmax(axis=1), no_hit)
        first_take = np.where(take_any, take_hit.argmax(axis=1), no_hit)
        stop_selected = valid_entries & (first_stop <= first_take) & (first_stop < no_hit)
        take_selected = valid_entries & ~stop_selected & (first_take < no_hit)
        time_selected = valid_entries & ~stop_selected & ~take_selected

        exit_offsets = horizon.copy()
        exit_offsets[stop_selected] = first_stop[stop_selected] + 1
        exit_offsets[take_selected] = first_take[take_selected] + 1
        exit_indexes = np.arange(start, end) + exit_offsets

        outcomes[start:end][stop_selected] = "stop_loss"
        outcomes[start:end][take_selected] = "take_profit"
        outcomes[start:end][~valid_entries] = "invalid"
        gross_returns[start:end][stop_selected] = -1.0
        gross_returns[start:end][take_selected] = take_profit_r

        if time_selected.any():
            time_entries = batch_entries[time_selected]
            time_exits = close[exit_indexes[time_selected]]
            if side == "long":
                time_return_pct = (time_exits - time_entries) / time_entries
            else:
                time_return_pct = (time_entries - time_exits) / time_entries
            gross_returns[start:end][time_selected] = time_return_pct / stop_distance_pct

        net_returns[start:end] = gross_returns[start:end] - cost_r
        net_returns[start:end][~valid_entries] = 0.0
        # The legacy implementation returns 0R when no future bar exists at all.
        net_returns[start:end][valid_entries & (horizon == 0)] = 0.0
        holding_bars[start:end] = exit_offsets
        holding_bars[start:end][~valid_entries] = 0

    return outcomes, gross_returns, net_returns, holding_bars


def verify_vectorized_sample(
    df,
    labels,
    take_profit_values,
    stop_distance_pct,
    max_bars,
    round_trip_cost_pct,
    step_minutes,
    sample_size,
):
    sample_size = min(sample_size, len(df))
    if sample_size <= 0:
        return

    sample_indexes = list(range(sample_size))
    final_start = max(sample_size, len(df) - sample_size)
    sample_indexes.extend(range(final_start, len(df)))

    for take_profit_r in take_profit_values:
        r_key = format_r(take_profit_r)
        for side in ["long", "short"]:
            prefix = f"{side}_tp_{r_key}r"
            for index in sample_indexes:
                expected = evaluate_barrier(
                    df,
                    index,
                    side,
                    take_profit_r,
                    stop_distance_pct,
                    max_bars,
                    round_trip_cost_pct,
                )
                actual = (
                    labels.at[index, f"{prefix}_outcome"],
                    labels.at[index, f"{prefix}_return_r_gross"],
                    labels.at[index, f"{prefix}_return_r_net"],
                    index + round(labels.at[index, f"{prefix}_holding_minutes"] / step_minutes),
                )
                if expected[0] != actual[0] or not np.isclose(expected[1], actual[1]) or not np.isclose(expected[2], actual[2]) or expected[3] != actual[3]:
                    raise RuntimeError(
                        "Vectorized label verification failed: "
                        f"index={index}, side={side}, tp={take_profit_r}, expected={expected}, actual={actual}"
                    )


def add_labels(df, take_profit_values, stop_distance_pct, max_holding_minutes, round_trip_cost_bps, batch_size, verify_sample_size):
    step_minutes = infer_step_minutes(df)
    max_bars = max(1, round(max_holding_minutes / step_minutes))
    round_trip_cost_pct = round_trip_cost_bps / 10_000
    if batch_size <= 0:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    labels = df[["timestamp", "datetime_utc", "symbol", "market", "timeframe", "close"]].copy()
    labels["stop_distance_pct"] = stop_distance_pct
    labels["max_holding_minutes"] = max_holding_minutes
    labels["round_trip_cost_bps"] = round_trip_cost_bps

    close = df["close"].to_numpy(dtype=float)
    open_prices = df["open"].to_numpy(dtype=float)
    entries = np.concatenate([open_prices[1:], np.array([np.nan])])
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    entry_timestamps = np.concatenate([timestamps[1:], np.array([-1], dtype=np.int64)])
    labels["entry_timestamp"] = entry_timestamps
    labels["entry_price"] = entries
    labels["entry_rule"] = "next_bar_open"

    for take_profit_r in take_profit_values:
        r_key = format_r(take_profit_r)
        for side in ["long", "short"]:
            outcomes, gross_returns, net_returns, holding_bars = evaluate_barrier_vectorized(
                entries,
                close,
                high,
                low,
                side,
                take_profit_r,
                stop_distance_pct,
                max_bars,
                round_trip_cost_pct,
                batch_size,
            )

            prefix = f"{side}_tp_{r_key}r"
            labels[f"{prefix}_outcome"] = outcomes
            labels[f"{prefix}_return_r_gross"] = gross_returns
            labels[f"{prefix}_return_r_net"] = net_returns
            labels[f"{prefix}_holding_minutes"] = holding_bars * step_minutes

    verify_vectorized_sample(
        df,
        labels,
        take_profit_values,
        stop_distance_pct,
        max_bars,
        round_trip_cost_pct,
        step_minutes,
        verify_sample_size,
    )

    row_indexes = np.arange(len(df))
    horizon_indexes = row_indexes + max_bars
    complete_horizon = horizon_indexes < len(df)
    valid_indexes = row_indexes[complete_horizon]
    complete_horizon[valid_indexes] = (
        timestamps[horizon_indexes[valid_indexes]] - timestamps[valid_indexes]
        == max_bars * step_minutes * 60_000
    )
    labels["label_horizon_complete"] = complete_horizon
    removed = int((~complete_horizon).sum())
    labels = labels[labels["label_horizon_complete"]].reset_index(drop=True)
    labels["entry_datetime_utc"] = pd.to_datetime(labels["entry_timestamp"], unit="ms", utc=True)
    labels.attrs["incomplete_horizon_rows_removed"] = removed
    return labels


def write_labels(labels, label_folder, symbol, market, timeframe, stop_distance_pct, max_holding_minutes):
    output_root = Path(label_folder)
    if market != "spot":
        output_root = output_root / f"market={safe_value(market)}"
    output_dir = (
        output_root / f"symbol={safe_value(symbol)}"
        / f"timeframe={safe_value(timeframe)}"
        / f"stop={stop_distance_pct:.4f}"
        / f"hold={max_holding_minutes}m"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    market_suffix = "" if market == "spot" else f"_{safe_value(market).upper()}"
    output_path = output_dir / f"labels_{safe_value(symbol)}{market_suffix}_{safe_value(timeframe)}_stop{stop_distance_pct:.4f}_hold{max_holding_minutes}m.parquet"
    labels.to_parquet(output_path, index=False)

    verified = pd.read_parquet(output_path)
    if len(verified) != len(labels):
        raise RuntimeError(f"라벨 저장 검증 실패: expected={len(labels)}, actual={len(verified)}")

    marker_path = output_dir / "_LABEL_SUCCESS.json"
    marker = {
        "symbol": safe_value(symbol),
        "market": market,
        "timeframe": timeframe,
        "stop_distance_pct": stop_distance_pct,
        "max_holding_minutes": max_holding_minutes,
        "entry_rule": "next_bar_open",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "rows": len(labels),
        "incomplete_horizon_rows_removed": int(
            labels.attrs.get("incomplete_horizon_rows_removed", 0)
        ),
        "label_file": str(output_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, marker_path


def main():
    args = parse_args()
    take_profit_values = parse_take_profit_r(args.take_profit_r)
    if not 0 < args.stop_distance_pct < 1:
        raise ValueError("--stop-distance-pct는 0보다 크고 1보다 작아야 합니다.")
    if any(value <= 0 for value in take_profit_values):
        raise ValueError("--take-profit-r 값은 모두 0보다 커야 합니다.")
    if args.max_holding_minutes <= 0:
        raise ValueError("--max-holding-minutes는 1 이상이어야 합니다.")
    if args.round_trip_cost_bps < 0:
        raise ValueError("--round-trip-cost-bps는 음수일 수 없습니다.")

    print(f"Feature 로드: market={args.market}, symbol={args.symbol}, timeframe={args.timeframe}")
    features = load_features(args.feature_folder, args.symbol, args.market, args.timeframe)
    print(f"Feature rows: {len(features):,}")

    print(
        "Triple Barrier 라벨링 시작: "
        f"stop={args.stop_distance_pct:.4f}, tp={take_profit_values}, "
        f"hold={args.max_holding_minutes}m, cost={args.round_trip_cost_bps}bps"
    )
    labels = add_labels(
        features,
        take_profit_values,
        args.stop_distance_pct,
        args.max_holding_minutes,
        args.round_trip_cost_bps,
        args.batch_size,
        args.verify_sample_size,
    )
    print(
        "불완전하거나 시간 누락을 지나는 라벨 제외: "
        f"{labels.attrs.get('incomplete_horizon_rows_removed', 0):,} rows"
    )

    output_path, marker_path = write_labels(
        labels,
        args.label_folder,
        args.symbol,
        args.market,
        args.timeframe,
        args.stop_distance_pct,
        args.max_holding_minutes,
    )
    print(f"라벨 저장 완료: {output_path} ({len(labels):,} rows)")
    print(f"라벨 성공 마커: {marker_path}")


if __name__ == "__main__":
    main()
