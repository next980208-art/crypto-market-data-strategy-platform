"""Bounded 5m/15m target research with chronological costs and funding; offline only."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import joblib
import numpy as np
import pandas as pd

from scripts import collect_long_window_marks as collection
from scripts import multi_timeframe_funding_backtest as engine
from scripts import search_long_training_windows as previous

replay, windows = previous.replay, previous.windows
search, prior, runner = previous.search, previous.prior, previous.runner
consensus, sizing = previous.consensus, previous.sizing


def target_specs():
    values = []
    for timeframe, holds in (("5m", (240, 720)), ("15m", (720, 1440))):
        for hold in holds:
            for stop in (.005, .01):
                values.append({"id": f"{timeframe}_hold{hold}_stop{stop*100:g}pct", "timeframe": timeframe,
                               "step_minutes": int(timeframe[:-1]), "max_holding_minutes": hold,
                               "stop_distance_pct": stop})
    return values


def configurations():
    return [{"id": f"{spec['id']}_{family}_entry{threshold}", "target_id": spec["id"],
             "family": family, "threshold_r": threshold, "target_r": 1.5, "risk_per_trade_pct": .005}
            for spec in target_specs() for family in ("shallow", "ridge") for threshold in (.05, .15, .3)]


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def aggregate(raw, step_minutes):
    step = step_minutes*search.MINUTE
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    if raw.empty or raw.timestamp.iloc[0] % step or not raw.timestamp.diff().dropna().eq(search.MINUTE).all():
        raise ValueError("Source must be continuous and aligned")
    raw = raw.copy()
    raw["bucket"] = raw.timestamp//step*step
    grouped = raw.groupby("bucket", sort=True, observed=True)
    counts = grouped.size()
    if not counts.eq(step_minutes).all():
        raise ValueError("Incomplete aggregate bar")
    frame = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                        close=("close", "last"), volume=("volume", "sum")).reset_index().rename(columns={"bucket": "timestamp"})
    if not np.isfinite(frame[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("Nonfinite aggregate")
    frame["datetime_utc"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    frame["symbol"], frame["market"], frame["timeframe"] = "BTCUSDT", "usdm", f"{step_minutes}m"
    frame["feature_schema_version"] = "low_frequency_ohlcv_v1"
    return frame


def add_features(frame):
    result = frame.copy()
    close, high, low, open_, volume = (pd.to_numeric(result[c], errors="raise") for c in ("close", "high", "low", "open", "volume"))
    names = []
    for bars in (1, 2, 3, 6, 12, 24):
        name = f"return_{bars}bar"
        result[name] = close.pct_change(bars, fill_method=None)
        names.append(name)
    means = {}
    returns = close.pct_change(fill_method=None)
    for bars in (3, 6, 12, 48):
        means[bars] = close.rolling(bars, min_periods=bars).mean()
        result[f"close_to_ma_{bars}bar"] = close/means[bars]-1
        result[f"volatility_{bars}bar"] = returns.rolling(bars, min_periods=bars).std(ddof=0)
        result[f"volume_ratio_{bars}bar"] = volume/volume.rolling(bars, min_periods=bars).mean()-1
        names += [f"close_to_ma_{bars}bar", f"volatility_{bars}bar", f"volume_ratio_{bars}bar"]
    result["trend_ma_12_48"] = means[12]/means[48]-1
    previous_close = close.shift(1)
    tr = pd.concat([high-low, (high-previous_close).abs(), (low-previous_close).abs()], axis=1).max(axis=1)
    result["atr_14bar_pct"] = tr.rolling(14, min_periods=14).mean()/close
    result["candle_body_pct"] = (close-open_)/open_
    result["candle_range_pct"] = (high-low)/open_
    span = (high-low).replace(0, np.nan)
    result["close_location"] = (close-low)/span
    result["upper_wick_pct"] = (high-np.maximum(open_, close))/open_
    result["lower_wick_pct"] = (np.minimum(open_, close)-low)/open_
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    result["rsi_14bar"] = (100-100/(1+gain/loss.replace(0, np.nan)))/100
    rolling_low, rolling_high = low.rolling(48, min_periods=48).min(), high.rolling(48, min_periods=48).max()
    result["range_position_48bar"] = (close-rolling_low)/(rolling_high-rolling_low).replace(0, np.nan)
    dt = pd.to_datetime(result.timestamp, unit="ms", utc=True)
    minute = dt.dt.hour*60+dt.dt.minute
    result["minute_sin"], result["minute_cos"] = np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440)
    result["weekday_sin"], result["weekday_cos"] = np.sin(2*np.pi*dt.dt.dayofweek/7), np.cos(2*np.pi*dt.dt.dayofweek/7)
    names += ["trend_ma_12_48", "atr_14bar_pct", "candle_body_pct", "candle_range_pct", "close_location",
              "upper_wick_pct", "lower_wick_pct", "rsi_14bar", "range_position_48bar",
              "minute_sin", "minute_cos", "weekday_sin", "weekday_cos"]
    result[names] = result[names].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=names).reset_index(drop=True)
    if not np.isfinite(result[names].to_numpy()).all():
        raise ValueError("Nonfinite feature")
    return result, names


def build_frame(dataset, schedule, spec, labeler):
    warmup = 3*search.DAY
    start = schedule["validation_start"]-365*search.DAY-warmup
    raw = pd.read_parquet(dataset, columns=["timestamp", "open", "high", "low", "close", "volume"],
                          filters=[("timestamp", ">=", start), ("timestamp", "<", schedule["validation_end"])])
    np.testing.assert_array_equal(raw.sort_values("timestamp").timestamp, np.arange(start, schedule["validation_end"], search.MINUTE))
    bars = aggregate(raw, spec["step_minutes"])
    del raw
    features, names = add_features(bars)
    labels = labeler.add_labels(bars, [1.5], spec["stop_distance_pct"], spec["max_holding_minutes"], 10, 8192, 10)
    label_columns = [c for c in labels if c not in ("datetime_utc", "symbol", "market", "timeframe", "close")]
    frame = features.merge(labels[label_columns], on="timestamp", how="inner", validate="one_to_one")
    frame = frame[frame.timestamp >= schedule["validation_start"]-365*search.DAY].reset_index(drop=True)
    return frame, names


def partitions(frame, schedule, spec):
    outcome = frame.entry_timestamp+frame.max_holding_minutes*search.MINUTE
    start = schedule["validation_start"]-365*search.DAY
    train = frame[(frame.timestamp >= start) & (outcome < schedule["validation_start"])].copy()
    future = frame[(frame.timestamp >= schedule["validation_start"]) & (frame.timestamp < schedule["validation_end"])
                   & (outcome < schedule["validation_end"])].copy()
    if train.empty or future.empty or train.timestamp.min() > start+spec["step_minutes"]*search.MINUTE:
        raise ValueError("Incomplete train/evaluation partition")
    return train.reset_index(drop=True), future.reset_index(drop=True)


def evaluate(prediction, action, backtest, extra, market):
    args = action.backtest_args(extra)
    args.risk_per_trade_pct, args.take_profit_r = .005, 1.5
    signals = prediction[prediction.pred_action.ne("no_trade")].copy()
    trades, curve, skipped, events, daily = engine.simulate(signals, args, backtest, market)
    summary = backtest.summarize(trades, curve, skipped, signals, args)
    counts = [int(signals.pred_action.eq(side).sum()) for side in ("long", "short")]
    metrics = {"signals": len(signals), "long_signals": counts[0], "short_signals": counts[1],
               "maximum_direction_share": max(counts)/len(signals) if len(signals) else 1.}
    metrics.update({key: summary[key] for key in ("trades", "win_rate_pct", "avg_return_r", "total_return_pct", "max_drawdown_pct")})
    tables = {"trades": pd.DataFrame(trades, columns=list(trades[0]) if trades else ["signal_timestamp", "equity"]),
              "events": pd.DataFrame(events, columns=list(dict.fromkeys(k for e in events for k in e)) if events else ["timestamp", "exit_time_utc", "return_pct"]),
              "daily": pd.DataFrame(daily, columns=list(daily[0]) if daily else ["day_start_ms", "realized_pnl"])}
    return metrics, summary, tables


def save(root, rows, status):
    consensus.save_progress(root, rows, status)
    report = prior.read(root/"results.json")
    artifacts = list((root/"models").glob("*.joblib"))+list(root.glob("validation_*.parquet"))+list(root.glob("predictions_*.parquet"))
    if (root/"training.json").exists():
        artifacts.append(root/"training.json")
        report["models_fitted"] = len(prior.read(root/"training.json"))
    report["artifact_sha256"].update({str(p.relative_to(root)): runner.checksum(p) for p in artifacts})
    search.write_json(root/"results.json", report)


def prepare(root, plan, verify):
    labeler = search.load("4_triple_barrier_labeler.py", "lowfreq_labels")
    metadata, frames = [], {}
    specs = {row["id"]: row for row in plan["target_specs"]}
    for schedule in plan["schedules"]:
        n = schedule["fold"]
        for spec in specs.values():
            frame, features = build_frame(plan["dataset"], schedule, spec, labeler)
            train, future = partitions(frame, schedule, spec)
            key = f"{spec['id']}_{n}"
            validation_path = root/f"validation_{key}.parquet"
            if verify:
                pd.testing.assert_frame_equal(future, pd.read_parquet(validation_path), check_exact=True)
            else:
                future.to_parquet(validation_path, index=False)
            frames[n, spec["id"]] = future
            for family in ("shallow", "ridge"):
                values = {"timestamp": future.timestamp.to_numpy()}
                for offset, side in enumerate(("long", "short")):
                    model_id = f"{key}_{family}_{side}"
                    model_path = root/"models"/(model_id+".joblib")
                    target = sizing.returns.target_returns(train, side, 1.5, 5)
                    if verify:
                        model = joblib.load(model_path)
                    else:
                        model = windows.make_model(family+"90", plan, 1200+n*100+offset)
                        model.fit(train[features], target)
                        joblib.dump(model, model_path)
                    values[side] = model.predict(future[features])
                    metadata.append({"id": model_id, "fold": n, "target_id": spec["id"], "family": family, "side": side,
                        "features": features, "training_rows": len(train), "validation_rows": len(future),
                        "first_train_ms": int(train.timestamp.min()),
                        "last_train_outcome_ms": int((train.entry_timestamp+train.max_holding_minutes*search.MINUTE).max()),
                        "validation_start_ms": schedule["validation_start"], "model_sha256": runner.checksum(model_path),
                        "feature_sha256": digest(train[features]), "target_sha256": digest(target),
                        "quality": windows.quality(values[side], sizing.returns.target_returns(future, side, 1.5, 5))})
                    if not verify:
                        search.write_json(root/"training.json", metadata)
                    del model
                predictions = pd.DataFrame(values)
                path = root/f"predictions_{key}_{family}.parquet"
                if verify:
                    pd.testing.assert_frame_equal(predictions, pd.read_parquet(path), check_exact=True)
                else:
                    predictions.to_parquet(path, index=False)
                print(f"fold{n} {spec['id']} {family}: train={len(train)} validation={len(future)}", flush=True)
            del frame, train
            gc.collect()
    if verify and metadata != prior.read(root/"training.json"):
        raise ValueError("Training replay mismatch")
    return frames


def run_worker(root, verify):
    plan, state = prior.read(root/"protocol.json"), prior.read(root/"state.json")
    replay.check_hashes(plan, state, root)
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Package changed: "+name)
    if verify:
        saved = prior.read(root/"results.json")
        if saved["status"] != "completed":
            raise ValueError("Incomplete experiment")
        for name, value in saved["artifact_sha256"].items():
            if runner.checksum(root/name) != value:
                raise ValueError("Artifact changed: "+name)
    rows = [{"status": "pending", "config": c} for c in plan["configurations"]]
    try:
        frames = prepare(root, plan, verify)
    except Exception as exc:
        if verify:
            raise
        error = {"error": str(exc), "traceback": traceback.format_exc(), "stage": "training"}
        search.write_json(root/"training_error.json", error)
        rows = [{"status": "error", "config": c, **error} for c in plan["configurations"]]
        for row in rows:
            search.write_json(root/"trials"/(row["config"]["id"]+".json"), row)
        save(root, rows, "completed_with_errors")
        return
    action = search.load("16_action_profitability_walk_forward_model.py", "lowfreq_action")
    backtest = search.load("8_model_signal_backtest.py", "lowfreq_backtest")
    valuation = Path(plan["valuation"])
    markets = {n: replay.engine.FundingMarket(prior.read(valuation/f"funding_{n}.json"), pd.read_parquet(valuation/f"mark1m_{n}.parquet")) for n in (1, 2, 3)}
    for index, config in enumerate(plan["configurations"]):
        path = root/"trials"/(config["id"]+".json")
        if not verify:
            rows[index] = {"status": "running", "config": config}
            search.write_json(path, rows[index])
            save(root, rows, "running")
        try:
            folds, diagnostics = [], []
            for schedule in plan["schedules"]:
                n, frame = schedule["fold"], frames[schedule["fold"], config["target_id"]]
                values = pd.read_parquet(root/f"predictions_{config['target_id']}_{n}_{config['family']}.parquet")
                np.testing.assert_array_equal(values.timestamp, frame.timestamp)
                prediction = sizing.returns.prediction_frame(frame, values.long, values.short, config["threshold_r"], 0)
                fold = {"fold": n}
                for cost, extra in consensus.COSTS:
                    metrics, summary, tables = evaluate(prediction, action, backtest, extra, markets[n])
                    weekly = replay.cash_weekly(tables["events"], schedule["validation_start"], schedule["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold[cost] = metrics
                    diagnostics.append({"fold": n, "cost_case": cost, "summary": summary, "weekly": weekly})
                    for name, table in tables.items():
                        dest = root/"trades"/f"{config['id']}_{n}_{cost}_{name}.parquet"
                        if verify:
                            pd.testing.assert_frame_equal(table, pd.read_parquet(dest), check_exact=True)
                        else:
                            table.to_parquet(dest, index=False)
                folds.append(fold)
            result = consensus.summarize(config, folds, diagnostics)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(path):
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(path, result)
        rows[index] = result
        if not verify:
            save(root, rows, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if saved["trials"] != rows:
            raise ValueError("Results mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "models_reloaded": 96,
            "target_frames_rebuilt": 24, "policies_replayed": 48, "backtests_replayed": 432,
            "trade_ledger_files": 1296, "weekly_reconciliations": 432,
            "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
    else:
        save(root, rows, "completed" if all(r["status"] == "completed" for r in rows) else "completed_with_errors")


def initialize(root, valuation, limits):
    if shutil.disk_usage(search.ROOT).free < limits["minimum_free_gib"]*1024**3:
        raise ValueError("Insufficient disk space")
    old = replay.context_run.inputs(search.ROOT/"runtime_reports/market_context_validation_20260918")[3]
    collection_report = prior.read(valuation/"collection.json")
    if collection_report["status"] != "completed":
        raise ValueError("Valuation source incomplete")
    files = [Path(__file__).resolve(), Path(engine.__file__).resolve(), Path(old["dataset"]), valuation/"collection.json",
             valuation/"protocol.json", search.ROOT/"runtime_reports/data_usage_audit_20260920/audit.json"]
    files += list(valuation.glob("mark1m_*.parquet"))+list(valuation.glob("funding_*.json"))
    files += [Path(m.__file__).resolve() for m in (previous, replay, windows, search, prior, runner, consensus, sizing, sizing.returns)]
    files += [search.ROOT/name for name in ("4_triple_barrier_labeler.py", "8_model_signal_backtest.py",
                                            "16_action_profitability_walk_forward_model.py")]
    root.mkdir(parents=True, exist_ok=False)
    for name in ("models", "trades", "trials"):
        (root/name).mkdir()
    plan = {"created_utc": search.utc_now(), "dataset": old["dataset"], "base_config": old["base_config"],
            "schedules": collection.schedules(), "target_specs": target_specs(), "configurations": configurations(),
            "valuation": str(valuation), "hashes": {str(p): runner.checksum(p) for p in files},
            "packages": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "scikit-learn", "xgboost", "pyarrow", "joblib")},
            "limits": {**limits, "seconds": 600}, "training": "365 days, all completed 5m/15m bars, causal OHLCV features",
            "targets": "1.5R, stops 0.5%/1%, holds 240/720m at 5m and 720/1440m at 15m; next-bar-open; stop-first intrabar",
            "costs": "10bps in labels plus 5/10/20bps, and chronological funding cashflows",
            "admission": "All 3 folds and all costs: >=20 trades, positive mean R/return, cash MDD>=-20%, >=3/4 positive full weeks",
            "limitations": "Reused development data; OHLC intrabar ambiguity uses stop-first; funding proxies; no fills/margin/liquidation/MTM; no independent holdout",
            "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False}
    search.write_json(root/"protocol.json", plan)
    search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json")})
    shutil.copyfile(__file__, root/"code_snapshot.py")
    shutil.copyfile(engine.__file__, root/"engine_snapshot.py")
    rows = [{"status": "pending", "config": c} for c in configurations()]
    for row in rows:
        search.write_json(root/"trials"/(row["config"]["id"]+".json"), row)
    save(root, rows, "ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valuation", type=Path, default=Path("runtime_reports/long_window_marks_20260920_v2"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use project subdirectory")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 384, "minimum_free_gib": 5}
    if not args.verify:
        initialize(root, args.valuation.resolve(), limits)
    with runner.exclusive_lock(root):
        state = prior.read(root/"state.json")
        folder = root/("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.search_low_frequency_targets", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root/"state.json", state)
        measured = runner.monitor(command, root, folder, 600, limits)
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root/"results.json")["status"] if success else "stopped",
                         measurement=measured, finished_utc=search.utc_now())
            search.write_json(root/"state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
