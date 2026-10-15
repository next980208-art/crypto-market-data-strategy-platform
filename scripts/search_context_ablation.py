"""Finite feature-group ablation on frozen historical folds. No orders or deployment."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import traceback

import joblib
import numpy as np
import pandas as pd

from scripts import validate_market_context as context_run

search, prior, runner = context_run.search, context_run.prior, context_run.runner
consensus, sizing, sampled = context_run.consensus, context_run.sizing, context_run.sampled
GROUPS = {
    "spot": context_run.FEATURES[:3],
    "reference_prices": context_run.FEATURES[3:6],
    "funding": context_run.FEATURES[6:],
}


def configurations():
    return [{"id": f"{family}_{group}_entry{threshold}", "family": family, "group": group,
             "threshold_r": threshold, "risk_per_trade_pct": .005, "target_r": 1.5}
            for family in ("shallow90", "ridge90") for group in GROUPS for threshold in (.05, .15, .3)]


def selected_features(baseline, group):
    features = list(baseline) + GROUPS[group]
    if len(features) != len(set(features)):
        raise ValueError("Duplicate model features")
    return features


def verified_reference(root):
    verification = prior.read(root / "verification.json")
    if verification["status"] != "passed" or verification["results_sha256"] != runner.checksum(root / "results.json"):
        raise ValueError("Reference requires matching passed verification")
    report = prior.read(root / "results.json")
    if report["status"] != "completed":
        raise ValueError("Reference incomplete")
    for name, digest in report["artifact_sha256"].items():
        if runner.checksum(root / name) != digest:
            raise ValueError("Reference artifact changed: " + name)
    return report


def funding_audit(context):
    rows = []
    with closing(sqlite3.connect((context / "context.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        for n in (1, 2, 3):
            values = [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM observations WHERE family=? ORDER BY event_ms", (f"funding_{n}",))]
            missing = sum(v.get("markPrice") in (None, "") for v in values)
            rows.append({"fold": n, "funding_records": len(values), "missing_settlement_mark_price": missing})
    return {"folds": rows, "funding_cashflows_modeled": False,
            "reason": "Funding features are historical inputs, not realized payments. Missing markPrice is not imputed.",
            "collection_started": False, "orders_sent": 0}


def paired_deltas(folds, baseline):
    controls = {row["fold"]: row for row in baseline["folds"]}
    if {row["fold"] for row in folds} != set(controls):
        raise ValueError("Control fold mismatch")
    return [{"fold": row["fold"], "cost_case": cost,
             "return_delta_pp": row[cost]["total_return_pct"] - controls[row["fold"]][cost]["total_return_pct"],
             "trade_count_delta": row[cost]["trades"] - controls[row["fold"]][cost]["trades"]}
            for row in folds for cost, _ in consensus.COSTS]


def save(root, rows, status):
    context_run.save_progress(root, rows, status)
    report = prior.read(root / "results.json")
    path = root / "funding_source_audit.json"
    if path.exists():
        report["artifact_sha256"][path.name] = runner.checksum(path)
    report["feature_groups"] = GROUPS
    search.write_json(root / "results.json", report)


def prepare(root, reference, verify):
    controls = verified_reference(reference)
    _, sampled_root, source, old, context = context_run.inputs(reference)
    audit = funding_audit(context)
    audit_path = root / "funding_source_audit.json"
    if verify:
        if audit != prior.read(audit_path):
            raise ValueError("Funding source audit mismatch")
    else:
        search.write_json(audit_path, audit)
    frames, metadata, coverage = {}, [], []
    for n in (1, 2, 3):
        sources = {family: pd.read_parquet(context / f"{family}_{n}.parquet") for family in context_run.collection.SPECS}
        sample = pd.read_parquet(sampled_root / f"samples/gap5_{n}.parquet")
        prices = pd.read_parquet(old["dataset"], columns=["timestamp", "close"],
                                 filters=[("timestamp", ">=", int(sample.timestamp.min())), ("timestamp", "<=", int(sample.timestamp.max()))])
        sample = sample.merge(prices, on="timestamp", how="left", validate="one_to_one")
        if not np.isfinite(sample.close).all():
            raise ValueError("Missing original training prices")
        train, tc = context_run.enrich(sample, sources)
        future, vc = context_run.enrich(pd.read_parquet(source / f"validation_{n}.parquet"), sources)
        coverage.append({"fold": n, "train": tc, "validation": vc})
        if any(c["common_rows"] < .95 * c["input_rows"] for c in (tc, vc)):
            raise ValueError("Common coverage below 95%")
        if (train.entry_timestamp + 240 * search.MINUTE).max() >= future.timestamp.min():
            raise ValueError("Training labels overlap evaluation")
        expected = pd.read_parquet(reference / f"context_features_{n}.parquet")
        pd.testing.assert_frame_equal(future[expected.columns], expected, check_exact=True)
        frames[n] = future
        for family in ("shallow90", "ridge90"):
            for group in GROUPS:
                features = selected_features(old["features"], group)
                if not np.isfinite(train[features].to_numpy()).all():
                    raise ValueError("Nonfinite training input")
                values = {"timestamp": future.timestamp.to_numpy()}
                for offset, side in enumerate(("long", "short")):
                    key = f"{family}_{group}_{n}_{side}"
                    path = root / "models" / (key + ".joblib")
                    if verify:
                        model = joblib.load(path)
                    else:
                        model = sampled.windows.make_model(family, old, 902 + n * 10 + offset)
                        model.fit(train[features], sizing.returns.target_returns(train, side, 1.5, 5))
                        joblib.dump(model, path)
                    values[side] = model.predict(future[features])
                    metadata.append({"id": key, "features": features, "train_rows": len(train), "validation_rows": len(future),
                                     "last_train_outcome_ms": int((train.entry_timestamp + 240 * search.MINUTE).max()),
                                     "model_sha256": runner.checksum(path)})
                    if not verify:
                        search.write_json(root / "training.json", metadata)
                path = root / f"predictions_{family}_{group}_{n}.parquet"
                values = pd.DataFrame(values)
                if verify:
                    pd.testing.assert_frame_equal(values, pd.read_parquet(path), check_exact=True)
                else:
                    values.to_parquet(path, index=False)
                print(f"{family} {group} fold{n}: {len(features)} features", flush=True)
    if coverage != prior.read(reference / "coverage.json"):
        raise ValueError("Rows changed versus controls")
    if verify:
        if metadata != prior.read(root / "training.json") or coverage != prior.read(root / "coverage.json"):
            raise ValueError("Training/coverage replay mismatch")
    else:
        search.write_json(root / "coverage.json", coverage)
    schedules = {row["fold"]: row["schedule"] for row in prior.read(source / "training.json")}
    return frames, schedules, {row["config"]["id"]: row for row in controls["trials"]}


def run_worker(root, verify):
    plan, state = prior.read(root / "protocol.json"), prior.read(root / "state.json")
    if runner.checksum(root / "protocol.json") != state["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for name, digest in plan["hashes"].items():
        if runner.checksum(Path(name)) != digest:
            raise ValueError("Frozen dependency changed: " + name)
    if verify:
        saved = prior.read(root / "results.json")
        if saved["status"] != "completed":
            raise ValueError("Incomplete experiment")
        for name, digest in saved["artifact_sha256"].items():
            if runner.checksum(root / name) != digest:
                raise ValueError("Artifact changed: " + name)
    frames, schedules, controls = prepare(root, Path(plan["reference"]), verify)
    action = search.load("16_action_profitability_walk_forward_model.py", "ablation_action")
    backtest = search.load("8_model_signal_backtest.py", "ablation_backtest")
    results = [{"status": "pending", "config": c} for c in plan["configurations"]]
    for i, config in enumerate(plan["configurations"]):
        path = root / "trials" / (config["id"] + ".json")
        if not verify:
            results[i] = {"status": "running", "config": config}
            search.write_json(path, results[i])
            save(root, results, "running")
        try:
            folds, diagnostics = [], []
            for n, frame in frames.items():
                values = pd.read_parquet(root / f"predictions_{config['family']}_{config['group']}_{n}.parquet")
                np.testing.assert_array_equal(frame.timestamp, values.timestamp)
                prediction = sizing.returns.prediction_frame(frame, values.long, values.short, config["threshold_r"], 0)
                fold = {"fold": n}
                for cost, extra in consensus.COSTS:
                    metrics, summary, trades = sizing.evaluate(backtest, action, prediction, .005, extra)
                    weekly = consensus.weekly_results(trades, schedules[n]["validation_start"], schedules[n]["validation_end"])
                    np.testing.assert_allclose(weekly["total_return_pct"], metrics["total_return_pct"], atol=1e-10, rtol=0)
                    fold[cost] = metrics
                    diagnostics.append({"fold": n, "cost_case": cost, "summary": summary, "weekly": weekly})
                    trade_path = root / "trades" / f"{config['id']}_{n}_{cost}.parquet"
                    if verify:
                        pd.testing.assert_frame_equal(trades, pd.read_parquet(trade_path), check_exact=True)
                    else:
                        trades.to_parquet(trade_path, index=False)
                folds.append(fold)
            result = consensus.summarize(config, folds, diagnostics)
            for control in ("baseline", "context"):
                key = f"{config['family']}_{control}_entry{config['threshold_r']}"
                result["versus_" + control] = paired_deltas(folds, controls[key])
        except Exception as exc:
            if verify:
                raise
            result = {"status": "error", "config": config, "error": str(exc), "traceback": traceback.format_exc()}
        if verify:
            if result != prior.read(path):
                raise ValueError("Trial replay mismatch")
        else:
            search.write_json(path, result)
        results[i] = result
        if not verify:
            save(root, results, "running")
        print(f"{config['id']}: {result['status']} candidate={result.get('candidate')}", flush=True)
    if verify:
        if saved["trials"] != results:
            raise ValueError("Results replay mismatch")
        search.write_json(root / "verification.json", {"status": "passed", "models_reloaded": 36,
                          "context_joins_replayed": 6, "policies_replayed": 18, "metric_backtests": 162,
                          "trade_files": 162, "weekly_reconciliations": 162,
                          "results_sha256": runner.checksum(root / "results.json"), "orders_sent": 0})
    else:
        save(root, results, "completed" if all(r["status"] == "completed" for r in results) else "completed_with_errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runtime_reports/market_context_validation_20260918"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if root == search.ROOT or not root.is_relative_to(search.ROOT):
        raise ValueError("Use a project subdirectory")
    if args.worker:
        run_worker(root, args.verify)
        return 0
    limits = {"memory_mib": 2048, "output_mib": 96, "minimum_free_gib": 5}
    if not args.verify:
        reference = args.reference.resolve()
        verified_reference(reference)
        _, _, _, _, context = context_run.inputs(reference)
        if shutil.disk_usage(reference).free < 5 * 1024**3:
            raise ValueError("Insufficient disk space")
        files = [reference / n for n in ("protocol.json", "state.json", "results.json", "verification.json")]
        files += [context / "context.sqlite3", Path(__file__).resolve(), Path(context_run.__file__).resolve()]
        files += [Path(m.__file__).resolve() for m in (search, prior, runner, consensus, sizing, sampled, context_run.previous)]
        root.mkdir(parents=True, exist_ok=False)
        for directory in ("models", "trials", "trades"):
            (root / directory).mkdir()
        plan = {"created_utc": search.utc_now(), "reference": str(reference), "configurations": configurations(),
                "feature_groups": GROUPS, "hashes": {str(p): runner.checksum(p) for p in files},
                "limits": {**limits, "seconds": 600}, "controls": "Frozen 45-feature baseline and 52-feature context",
                "pairing": "Identical common rows, seed 902+fold*10+side, 90d gap5 training, same three historical 30d folds",
                "admission": "All folds and costs15/20/30bps: >=20 trades, mean R/return >0, MDD>=-20%, >=3/4 positive full weeks",
                "limitations": "Reused evaluation; no independent test; funding as feature only; no intratrade drawdown, margin, liquidation or live execution",
                "orders_allowed": False, "deployment_allowed": False, "collection_allowed": False}
        search.write_json(root / "protocol.json", plan)
        (root / "code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        search.write_json(root / "state.json", {"status": "ready", "protocol_sha256": runner.checksum(root / "protocol.json")})
        pending = [{"status": "pending", "config": c} for c in configurations()]
        for row in pending:
            search.write_json(root / "trials" / (row["config"]["id"] + ".json"), row)
        save(root, pending, "ready")
    with runner.exclusive_lock(root):
        state = prior.read(root / "state.json")
        folder = root / ("verify_log" if args.verify else "run_log")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.search_context_ablation", "--output", str(root), "--worker"]
        if args.verify:
            command.append("--verify")
        else:
            state.update(status="running", started_utc=search.utc_now())
            search.write_json(root / "state.json", state)
        try:
            measured = runner.monitor(command, root, folder, 600, limits)
        except BaseException as exc:
            if not args.verify:
                state.update(status="interrupted", reason=type(exc).__name__)
                search.write_json(root / "state.json", state)
            raise
        success = measured["exit_code"] == 0 and not measured["stop_reason"]
        search.write_json(root / ("verify_execution.json" if args.verify else "execution.json"), measured)
        if not args.verify:
            state.update(status=prior.read(root / "results.json")["status"] if success else "stopped", measurement=measured,
                         finished_utc=search.utc_now())
            search.write_json(root / "state.json", state)
    print(json.dumps(measured))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
