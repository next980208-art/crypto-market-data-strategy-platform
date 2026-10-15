"""Read back gate artifacts and reconstruct probabilities and constrained trades without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_cost_aware_gate as gate


def verify(output):
    search, diagnosis = gate.search, gate.diagnosis
    plan = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    reference = json.loads((diagnosis.REFERENCE / "protocol.json").read_text(encoding="utf-8"))
    assert result["status"] == execution["status"] == "completed"
    assert not result["model_approved"] and not result["independent_test_evaluated"] and result["orders_sent"] == 0
    assert plan["reference_protocol_sha256"] == diagnosis.digest(diagnosis.REFERENCE / "protocol.json")
    assert plan["reference_trial_sha256"] == diagnosis.digest(diagnosis.REFERENCE / f"{diagnosis.TRIAL}.json")
    diagnosis.verify_reference(reference)
    for name, checksum in plan["current_code_sha256"].items():
        assert diagnosis.digest(output / "code_snapshot" / name) == diagnosis.digest(search.ROOT / name) == checksum
    walk = search.load("14_walk_forward_direction_model.py", "verify_gate_walk")
    training = search.load("13_train_selective_direction_model.py", "verify_gate_training")
    extended = search.load("15_two_stage_walk_forward_model.py", "verify_gate_extended")
    action = search.load("16_action_profitability_walk_forward_model.py", "verify_gate_action")
    backtest = search.load("8_model_signal_backtest.py", "verify_gate_backtest")
    rows = []
    assert len(plan["folds"]) == len(result["folds"]) == 3
    for bounds, row in zip(plan["folds"], result["folds"]):
        number = row["fold"]
        assert number == bounds["fold"]
        raw = walk.load_fold_source(Path(plan["dataset"]), {**bounds, "test_end": bounds["validation_end"]},
                                    reference["start_ms"], gate.returns.label_columns(plan["config"]["take_profit_r"]))
        source, _ = extended.add_two_stage_features(raw, training)
        source = source.set_index("timestamp", verify_integrity=True)
        meta = pd.read_parquet(output / f"gate_training_signals_fold_{number}.parquet")
        signals = pd.read_parquet(output / f"validation_signals_fold_{number}.parquet")
        assert len(meta) == row["meta_signals"] and len(signals) == row["baseline"]["signals"]
        assert row["inner_base_last_outcome_limit_ms"] < row["inner_bounds"]["validation_start"]
        assert row["meta_last_outcome_limit_ms"] < bounds["validation_start"]
        assert row["validation_last_outcome_limit_ms"] < bounds["validation_end"] <= plan["holdout_start_ms"]
        for audit, start, end in ((meta, row["inner_bounds"]["validation_start"], bounds["validation_start"]),
                                  (signals, bounds["validation_start"], bounds["validation_end"])):
            assert audit.timestamp.is_unique
            assert audit.timestamp.ge(start).all() and (audit.entry_timestamp + audit.max_holding_minutes*search.MINUTE).lt(end).all()
            original = source.loc[audit.timestamp].reset_index()
            np.testing.assert_allclose(audit[gate.CAUSAL_COLUMNS], original[gate.CAUSAL_COLUMNS], atol=1e-12)
            np.testing.assert_array_equal(audit.entry_timestamp, original.entry_timestamp)
            original["pred_action"] = audit.pred_action.to_numpy()
            np.testing.assert_allclose(audit.actual_net_label_r, gate.selected_net_returns(original, plan["config"]["take_profit_r"], plan["costs_bps"]["extra"]))
        params = row["gate_parameters"]
        if row["gate_fitted"]:
            assert params["features"] == gate.GATE_COLUMNS
            np.testing.assert_allclose(meta[gate.GATE_COLUMNS].mean(), params["mean"], atol=1e-12)
            scales = meta[gate.GATE_COLUMNS].std(ddof=0).to_numpy()
            scales[scales == 0] = 1
            np.testing.assert_allclose(scales, params["scale"], atol=1e-12)
            x = (signals[gate.GATE_COLUMNS].to_numpy()-params["mean"])/params["scale"]
            logits = x @ np.asarray(params["coefficients"]) + params["intercept"]
            probability = 1/(1+np.exp(-np.clip(logits, -700, 700)))
        else:
            probability = np.zeros(len(signals))
        np.testing.assert_allclose(probability, signals.gate_probability, atol=1e-12)
        np.testing.assert_array_equal(probability >= gate.GATE_THRESHOLD, signals.gate_accepted)
        original = source.loc[signals.timestamp].reset_index()
        original["pred_action"] = signals.pred_action.to_numpy()
        original["pred_confidence"] = 0.0
        filtered = gate.apply_gate(original, probability)
        for key, frame, cost in (("baseline", original, plan["costs_bps"]["extra"]),
                                 ("gated", filtered, plan["costs_bps"]["extra"]),
                                 ("gated_stress", filtered, plan["costs_bps"]["stress_extra"])):
            actual = action.evaluate_candidate(backtest, frame, cost, plan["config"]["take_profit_r"])
            for field in ("signals", "trades", "win_rate_pct", "avg_return_r", "total_return_pct", "max_drawdown_pct"):
                np.testing.assert_allclose(actual[field], row[key][field], atol=1e-10)
        args = action.backtest_args(plan["costs_bps"]["extra"])
        args.take_profit_r = plan["config"]["take_profit_r"]
        trades, _, skipped = backtest.simulate(filtered[filtered.pred_action.ne("no_trade")], args)
        saved = pd.read_parquet(output / f"gated_trades_fold_{number}.parquet")
        assert len(saved) == len(trades)
        if trades:
            pd.testing.assert_frame_equal(saved, pd.DataFrame(trades))
        rows.append({"fold": number, "gate_fitted": row["gate_fitted"], "training_signals": len(meta),
                     "validation_signals": len(signals), "accepted_signals": int(signals.gate_accepted.sum()),
                     "reproduced_trades": len(trades), "skipped": dict(skipped)})
    return {"status": "passed", "verified_utc": search.utc_now(), "folds": rows,
            "verifier_sha256": diagnosis.digest(Path(__file__)),
            "checks": ["reference and source hashes", "causal features and net labels from original dataset",
                       "chronological purge boundaries", "training-only scaling", "saved coefficient probabilities",
                       "fixed threshold selection", "baseline/gated/stress replay", "saved executed trades"],
            "models_retrained": 0, "orders_sent": 0, "independent_test_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output)
    gate.search.write_json(args.output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
