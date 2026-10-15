"""Paired baseline/context research on historical public data; no orders or auto-deployment."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import joblib
import numpy as np
import pandas as pd

from scripts import validate_return_bias as previous
from scripts import collect_research_context as collection

consensus, sizing, sampled = previous.consensus, previous.sizing, previous.sampled
search, prior, runner = previous.search, previous.prior, previous.runner
FEATURES = ["public_spot_return15", "public_spot_return60", "public_spot_taker_ratio",
            "public_index_return15", "public_mark_index_basis", "public_futures_spot_lag_basis", "public_last_funding"]


def configurations():
    return [{"id": f"{family}_{group}_entry{threshold}", "family": family, "group": group,
             "threshold_r": threshold, "risk_per_trade_pct": .005, "target_r": 1.5}
            for family in ("shallow90", "ridge90") for group in ("baseline", "context") for threshold in (.05, .15, .3)]


def enrich(frame, sources):
    result = frame.sort_values("entry_timestamp").reset_index(drop=True).copy()
    for family in ("spot15", "index15", "mark15", "funding"):
        source = sources[family].sort_values("event_ms").copy()
        if source.event_ms.duplicated().any():
            raise ValueError("Duplicate context timestamps")
        if family != "funding":
            grid = np.arange(int(source.event_ms.min()), int(source.event_ms.max())+collection.STEP, collection.STEP)
            source = source.set_index("event_ms").reindex(grid).rename_axis("event_ms").reset_index()
            source.loc[~source.complete_candle.eq(True), "value"] = np.nan
            source["available_ms"] = source.event_ms+collection.STEP+search.MINUTE
            if family == "spot15":
                source["public_spot_return15"] = source.value.pct_change(fill_method=None)
                source["public_spot_return60"] = source.value.pct_change(4, fill_method=None).where(source.value.rolling(5).count().eq(5))
                source["public_spot_taker_ratio"] = source.taker_ratio.where(source.value.notna())
            if family == "index15":
                source["public_index_return15"] = source.value.pct_change(fill_method=None)
            tolerance = collection.STEP-1
        else:
            source["available_ms"] = source.event_ms+search.MINUTE
            source["public_last_funding"] = source.value
            tolerance = 24*60*search.MINUTE
        keep = ["available_ms", "value"]+[f for f in FEATURES if f in source]
        source = source[keep].rename(columns={"available_ms": family+"_available_ms", "value": family+"_value"})
        result = pd.merge_asof(result, source, left_on="entry_timestamp", right_on=family+"_available_ms", direction="backward", tolerance=tolerance)
        if (result[family+"_available_ms"].dropna() > result.loc[result[family+"_available_ms"].notna(), "entry_timestamp"]).any():
            raise ValueError("Future context leak")
    result["public_mark_index_basis"] = result.mark15_value/result.index15_value-1
    result["public_futures_spot_lag_basis"] = result.close/result.spot15_value-1
    complete = np.isfinite(result[FEATURES].to_numpy()).all(axis=1)
    return result.loc[complete].reset_index(drop=True), {"input_rows": len(frame), "common_rows": int(complete.sum()), "excluded_rows": int((~complete).sum())}


def inputs(root):
    plan = prior.read(root/"protocol.json")
    state = prior.read(root/"state.json")
    if runner.checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for file, digest in plan["hashes"].items():
        if runner.checksum(Path(file)) != digest:
            raise ValueError("Frozen input changed: "+file)
    reference = Path(plan["reference"])
    source, _ = sizing.check_reference(reference)
    _, _, _, old = sampled.verify_inputs(reference)
    context = Path(plan["context"])
    report = prior.read(context/"collection.json")
    if report["status"] != "completed":
        raise ValueError("Collection incomplete")
    for task in report["tasks"]:
        if task["status"] == "completed" and runner.checksum(context/task["path"]) != task["sha256"]:
            raise ValueError("Context artifact changed")
    return plan, reference, source, old, context


def fit_or_verify(root, reference, source, old, context, verify):
    metadata, coverage, frames = [], [], {}
    for n in (1, 2, 3):
        sources = {family: pd.read_parquet(context/f"{family}_{n}.parquet") for family in collection.SPECS}
        sample = pd.read_parquet(reference/f"samples/gap5_{n}.parquet")
        prices = pd.read_parquet(old["dataset"], columns=["timestamp", "close"],
                                 filters=[("timestamp", ">=", int(sample.timestamp.min())), ("timestamp", "<=", int(sample.timestamp.max()))])
        sample = sample.merge(prices, on="timestamp", how="left", validate="one_to_one")
        if not np.isfinite(sample.close).all():
            raise ValueError("Missing original close prices for training samples")
        train, train_coverage = enrich(sample, sources)
        future, future_coverage = enrich(pd.read_parquet(source/f"validation_{n}.parquet"), sources)
        if any(row["common_rows"] < .95*row["input_rows"] for row in (train_coverage, future_coverage)):
            raise ValueError(f"Common coverage below 95% in fold {n}")
        if (train.entry_timestamp+240*search.MINUTE).max() >= future.timestamp.min():
            raise ValueError("Training outcomes overlap evaluation")
        frames[n] = future
        coverage.append({"fold": n, "train": train_coverage, "validation": future_coverage})
        for family in ("shallow90", "ridge90"):
            for group in ("baseline", "context"):
                features = old["features"]+(FEATURES if group == "context" else [])
                values = {"timestamp": future.timestamp.to_numpy()}
                for offset, side in enumerate(("long", "short")):
                    key = f"{family}_{group}_{n}_{side}"
                    path = root/"models"/(key+".joblib")
                    if verify:
                        model = joblib.load(path)
                    else:
                        model = sampled.windows.make_model(family, old, 902+n*10+offset)
                        model.fit(train[features], sizing.returns.target_returns(train, side, 1.5, 5))
                        joblib.dump(model, path)
                    values[side] = model.predict(future[features])
                    metadata.append({"id": key, "fold": n, "family": family, "group": group, "side": side,
                                     "train_rows": len(train), "validation_rows": len(future), "features": features,
                                     "last_train_outcome_ms": int((train.entry_timestamp+240*search.MINUTE).max()), "model_sha256": runner.checksum(path)})
                path = root/f"predictions_{family}_{group}_{n}.parquet"
                values = pd.DataFrame(values)
                if verify:
                    pd.testing.assert_frame_equal(values, pd.read_parquet(path), check_exact=True)
                else:
                    values.to_parquet(path, index=False)
                    search.write_json(root/"training.json", metadata)
                print(f"{family} {group} fold{n}: {len(features)} features, train={len(train)}, validation={len(future)}", flush=True)
        sidecar = future[["timestamp", "entry_timestamp"]+FEATURES+[f+"_available_ms" for f in collection.SPECS]]
        path = root/f"context_features_{n}.parquet"
        if verify:
            pd.testing.assert_frame_equal(sidecar, pd.read_parquet(path), check_exact=True)
        else:
            sidecar.to_parquet(path, index=False)
            search.write_json(root/"coverage.json", coverage)
    if verify and (metadata != prior.read(root/"training.json") or coverage != prior.read(root/"coverage.json")):
        raise ValueError("Model/coverage replay mismatch")
    return frames


def save_progress(root, results, status):
    previous.save_progress(root, results, status)
    report = prior.read(root/"results.json")
    files = list(root.glob("context_features_*.parquet"))
    if (root/"coverage.json").exists():
        files.append(root/"coverage.json")
    report["artifact_sha256"].update({str(p.relative_to(root)): runner.checksum(p) for p in files})
    search.write_json(root/"results.json", report)
    (root/"LEDGER.md").write_text(consensus.ledger.render([root/"results.json"]), encoding="utf-8")


def run_worker(root, verify):
    plan, reference, source, old, context = inputs(root)
    if verify:
        saved = prior.read(root/"results.json")
        if saved["status"] != "completed":
            raise ValueError("Cannot verify incomplete research")
        for path, digest in saved["artifact_sha256"].items():
            if runner.checksum(root/path) != digest:
                raise ValueError("Artifact changed: "+path)
    frames = fit_or_verify(root, reference, source, old, context, verify)
    schedules = {r["fold"]: r["schedule"] for r in prior.read(source/"training.json")}
    action = search.load("16_action_profitability_walk_forward_model.py", "context_action")
    backtest = search.load("8_model_signal_backtest.py", "context_backtest")
    results = [{"status": "pending", "config": c} for c in plan["configurations"]]
    for i, config in enumerate(plan["configurations"]):
        path = root/"trials"/(config["id"]+".json")
        if not verify:
            results[i] = {"status": "running", "config": config}
            search.write_json(path, results[i])
            save_progress(root, results, "running")
        try:
            folds, diagnostics = [], []
            for n, frame in frames.items():
                values = pd.read_parquet(root/f"predictions_{config['family']}_{config['group']}_{n}.parquet")
                np.testing.assert_array_equal(frame.timestamp, values.timestamp)
                prediction = sizing.returns.prediction_frame(frame, values.long, values.short, config["threshold_r"], 0)
                fold = {"fold": n}
                for label, extra in consensus.COSTS:
                    metrics, summary, trades = sizing.evaluate(backtest, action, prediction, .005, extra)
                    weekly = consensus.weekly_results(trades, schedules[n]["validation_start"], schedules[n]["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold[label] = metrics
                    diagnostics.append({"fold": n, "cost_case": label, "summary": summary, "weekly": weekly})
                    trade_path = root/"trades"/f"{config['id']}_{n}_{label}.parquet"
                    if verify:
                        pd.testing.assert_frame_equal(trades, pd.read_parquet(trade_path))
                    else:
                        trades.to_parquet(trade_path, index=False)
                folds.append(fold)
            result = consensus.summarize(config, folds, diagnostics)
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if prior.read(path) != result:
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(path, result)
        results[i] = result
        if not verify:
            save_progress(root, results, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if results != saved["trials"]:
            raise ValueError("Results replay mismatch")
        search.write_json(root/"verification.json", {"status": "passed", "models_reloaded": 24, "context_joins_replayed": 6,
                          "policies_replayed": 12, "metric_backtests": 108, "trade_files": 108, "weekly_reconciliations": 108,
                          "results_sha256": runner.checksum(root/"results.json"), "orders_sent": 0})
    else:
        save_progress(root, results, "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use project subdirectory")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 64, "minimum_free_gib": 5}
    if not args.verify:
        context = args.context.resolve()
        reference = (search.ROOT/"runtime_reports/literature_sampling_search_20260917").resolve()
        sizing.check_reference(reference)
        if prior.read(context/"collection.json")["status"] != "completed" or shutil.disk_usage(context).free < 5*1024**3:
            raise ValueError("Incomplete collection or insufficient disk")
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("trials", "models", "trades"):
            (root/folder).mkdir()
        files = [context/n for n in ("protocol.json", "collection.json")]+[reference/n for n in ("protocol.json", "state.json", "results.json")]
        files += [Path(m.__file__).resolve() for m in (previous, consensus, sizing, sampled, collection, consensus.ledger)]+[Path(__file__).resolve()]
        plan = {"created_utc": search.utc_now(), "reference": str(reference), "context": str(context), "configurations": configurations(),
                "hashes": {str(p): runner.checksum(p) for p in files}, "features": FEATURES, "limits": {**limits, "seconds": 600},
                "join": "Historical replay assumption: candle close plus 1 minute; past funding plus 1 minute, max age 24h; missing/incomplete candles excluded, no interpolation",
                "pairing": "Identical complete rows and seeds for 45-feature baseline vs 52-feature context; minimum 95% common coverage",
                "admission": "All 3 folds and costs15/20/30bps: >=20 trades, positive mean R/return, MDD>=-20%; >=3/4 positive full weeks",
                "limitations": "Not point-in-time verified; reused outer validation; funding as feature only, no cashflows/liquidation/mark-to-market; not bot performance data",
                "orders_allowed": False, "deployment_allowed": False}
        search.write_json(root/"protocol.json", plan)
        (root/"code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        search.write_json(root/"state.json", {"status": "ready", "protocol_sha256": runner.checksum(root/"protocol.json")})
        rows = [{"status": "pending", "config": c} for c in configurations()]
        for row in rows:
            search.write_json(root/"trials"/(row["config"]["id"]+".json"), row)
        save_progress(root, rows, "ready")
    with runner.exclusive_lock(root):
        state = prior.read(root/"state.json")
        folder = root/("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.validate_market_context", "--context", str(args.context.resolve()), "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root/"state.json", state)
        try:
            measured = runner.monitor(command, root, folder, 600, limits)
        except BaseException as exc:
            if not args.verify:
                state.update(status="interrupted", reason=type(exc).__name__)
                search.write_json(root/"state.json", state)
            raise
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root/("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root/"results.json")["status"] if success else "stopped", measurement=measured, finished_utc=search.utc_now())
            search.write_json(root/"state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
