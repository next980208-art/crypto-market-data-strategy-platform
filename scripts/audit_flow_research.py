"""Verify saved flow experiments and generate the complete trial ledger."""

import json

import joblib
import numpy as np
import pandas as pd

from scripts import search_flow_exhaustion as flow
from scripts import search_flow_rules as rules


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    roots = [flow.search.ROOT / "runtime_reports" / name for name in
             ("flow_exhaustion_20261001", "flow_rules_20261001")]
    audit = {"checked_utc": flow.search.utc_now(), "dataset_sha256": flow.digest(flow.DATASET), "runs": []}
    assert audit["dataset_sha256"] == flow.EXPECTED_SHA
    lines = ["# Flow Research Trial Ledger (2026-10-01)", "",
             "Historical development evidence only. No independent holdout or real-trading approval.", "",
             "Normal/stress/severe round-trip costs: 15/20/30 bps plus historical funding.",
             "R = fixed 0.5% price stop; account risk = 0.5% per trade. Returns below are percent.",
             "Validation IDs repeat across periods: these are not independent successful discoveries.", "",
             "| Experiment | Fold | Policy | Trades (normal) | Normal % | Stress % | Severe % | Rejection reasons |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    raw = pd.read_parquet(flow.DATASET)
    frame, columns = flow.features(raw)
    finite = np.isfinite(frame[columns["flow"]].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite & frame.entry_timestamp.mod(15*flow.MINUTE).eq(0)]
    market, _ = flow.funding_market(raw)
    backtest = flow.search.load("8_model_signal_backtest.py", "flow_audit_backtest")
    action = flow.search.load("16_action_profitability_walk_forward_model.py", "flow_audit_action")
    for root in roots:
        protocol, result = read(root / "protocol.json"), read(root / "result.json")
        assert result["status"] == "completed"
        assert all(flow.digest(flow.search.ROOT/name) == value for name, value in protocol["source_hashes"].items())
        counts, positive, passed = 0, 0, 0
        for number in (1, 2, 3):
            folder = root / f"fold_{number}"
            trials = read(folder / "validation_trials.json")
            assert len(trials) == len({r["id"] for r in trials}) == 36
            choice = read(folder / "selection_before_test.json")
            if root == roots[0]:
                assert choice == {name: flow.select_validation([r for r in trials if r["feature_set"] == name])
                                  for name in ("price", "flow")}
            else:
                assert choice == flow.select_validation(trials)
            for row in trials:
                metrics = row["metrics"]
                assert row["rejection_reasons"] == flow.rejection_reasons(metrics)
                counts += 1
                positive += metrics["normal"]["total_return_pct"] > 0
                passed += not row["rejection_reasons"]
                values = " | ".join(f"{metrics[c]['total_return_pct']:.6f}" for c in flow.COSTS)
                lines.append(f"| {root.name} | {number} | {row['id']} | {metrics['normal']['trades']} | {values} | {'; '.join(row['rejection_reasons']) or 'none'} |")
            if root == roots[1]:
                fold = result["folds"][number-1]
                if choice is None:
                    continue
                test = flow.period(frame, int(pd.Timestamp(fold["test_start_utc"]).timestamp()*1000),
                                   int(pd.Timestamp(fold["test_end_exclusive_utc"]).timestamp()*1000))
                predictions = rules.signals(test, choice["config"])
                pd.testing.assert_frame_equal(predictions.reset_index(drop=True),
                                              pd.read_parquet(folder / "test_predictions.parquet").reset_index(drop=True))
                metrics, detail = flow.evaluate(predictions, choice["config"]["target"], market, backtest, action)
                assert metrics == fold["metrics"]
                assert detail == read(folder / "test_ledger.json")
                for item in detail.values():
                    np.testing.assert_allclose(np.prod([1+t["return_pct"] for t in item["trades"]]), item["cash_curve"][-1])
                    stamps = [e["timestamp"] for e in item["events"]]
                    assert stamps == sorted(stamps)
        audit["runs"].append({"path": str(root.relative_to(flow.search.ROOT)), "validation_trials": counts,
            "positive_normal": int(positive), "passed_validation_gate": int(passed), "source_hashes_match": True,
            "bytes_before_audit": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())})
    # Reload the best positive ML validation artifact, without fitting again.
    model_file = roots[0] / "fold_2/price_xgb_3.0.joblib"
    saved = joblib.load(model_file)
    validation = flow.period(frame, int(pd.Timestamp("2026-07-26", tz="UTC").timestamp()*1000),
                            int(pd.Timestamp("2026-08-05", tz="UTC").timestamp()*1000))
    models = saved["models"]
    predictions = flow.returns.prediction_frame(validation, models["long"].predict(validation[saved["features"]]),
                                                models["short"].predict(validation[saved["features"]]), .05, 0.)
    metrics, _ = flow.evaluate(predictions, 3., market, backtest, action)
    source = next(r for r in read(roots[0] / "fold_2/validation_trials.json") if r["id"] == "price_xgb_3.0_threshold_0.05")
    assert metrics == source["metrics"]
    audit.update(rule_test_replay_matched=True, saved_ml_model_replay_matched=True,
                 funding_cash_reconciled=True, all_validation_selections_verified=True)
    flow.search.write_json(roots[1] / "integrity_check.json", audit)
    ledger = flow.search.ROOT / "docs/flow_research_trial_ledger_2026-10-01.md"
    ledger.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
