"""Verify a completed automatic run from saved inputs, gate coefficients and backtest replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import automatic_research as runner
from scripts import run_cost_aware_gate as gate


def verify(root):
    plan = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    state = json.loads((root/"state.json").read_text(encoding="utf-8"))
    runner.verify_resume(root, plan, state)
    assert state["status"] == "completed" and state["orders_sent"] == 0 and not state["model_approved"]
    assert all(t["status"] == "completed" for t in state["tasks"].values())
    def result(task):
        return json.loads((root/state["tasks"][task]["result_path"]).read_text(encoding="utf-8"))
    audit = result("audit")
    action = gate.search.load("16_action_profitability_walk_forward_model.py", "verify_auto_action")
    backtest = gate.search.load("8_model_signal_backtest.py", "verify_auto_backtest")
    verified = []
    for number in (1, 2, 3):
        prepared = result(f"prepare_{number}")
        meta = pd.read_parquet(root/prepared["meta_path"])
        validation = pd.read_parquet(root/prepared["validation_path"])
        assert prepared["stored_1R_parity"] and prepared["feature_count"] == 45
        assert meta.timestamp.is_unique and meta.entry_timestamp.diff().dropna().gt(240*60000).all()
        assert len(meta) == prepared["meta_signals_after"]
        assert int(meta.actual_net_label_r.gt(0).sum()) == prepared["meta_positive"]
        assert prepared["schedule"]["validation_end"] <= audit["end_exclusive_ms"]-365*gate.search.DAY
        for chunk in prepared["chunks"]:
            assert chunk["last_train_outcome_ms"] < chunk["validation_start"]
            assert chunk["last_prediction_outcome_ms"] < chunk["validation_end"]
        assert (meta.entry_timestamp+240*60000).lt(prepared["schedule"]["validation_start"]).all()
        assert validation.timestamp.ge(prepared["schedule"]["validation_start"]).all()
        assert (validation.entry_timestamp+240*60000).lt(prepared["schedule"]["validation_end"]).all()
        baseline = action.evaluate_candidate(backtest, validation, 5, 1.5)
        for field in ("signals", "trades", "avg_return_r", "total_return_pct", "max_drawdown_pct"):
            np.testing.assert_allclose(baseline[field], prepared["baseline"][field], atol=1e-10)
        for task in ("trial_c1", "trial_c01"):
            trial = result(task)
            fold = trial["folds"][number-1]
            folder = (root/state["tasks"][task]["result_path"]).parent
            decisions = pd.read_parquet(folder/f"decisions_{number}.parquet")
            np.testing.assert_array_equal(decisions.timestamp, validation.timestamp)
            np.testing.assert_array_equal(decisions.original_action, validation.pred_action)
            params = fold["parameters"]
            if fold["gate_fitted"]:
                assert len(meta) >= 200 and min(prepared["meta_positive"], prepared["meta_nonpositive"]) >= 20
                assert params["columns"] == gate.GATE_COLUMNS
                np.testing.assert_allclose(meta[gate.GATE_COLUMNS].mean(), params["mean"], atol=1e-12)
                scales = meta[gate.GATE_COLUMNS].std(ddof=0).to_numpy()
                scales[scales == 0] = 1
                np.testing.assert_allclose(scales, params["scale"], atol=1e-12)
                inputs = (gate.gate_features(validation).to_numpy()-params["mean"])/params["scale"]
                probability = 1/(1+np.exp(-np.clip(inputs @ params["coef"]+params["intercept"], -700, 700)))
            else:
                probability = np.zeros(len(validation))
            np.testing.assert_allclose(probability, decisions.probability, atol=1e-12)
            np.testing.assert_array_equal(probability >= .5, decisions.accepted)
            filtered = gate.apply_gate(validation, probability)
            for key, cost in (("gated", 5), ("stress", 10)):
                actual = action.evaluate_candidate(backtest, filtered, cost, 1.5)
                for field in ("signals", "trades", "win_rate_pct", "avg_return_r", "total_return_pct", "max_drawdown_pct"):
                    np.testing.assert_allclose(actual[field], fold[key][field], atol=1e-10)
            args = action.backtest_args(5)
            args.take_profit_r = 1.5
            trades, _, _ = backtest.simulate(filtered[filtered.pred_action.ne("no_trade")], args)
            saved = pd.read_parquet(folder/f"trades_{number}.parquet")
            assert len(saved) == len(trades)
            if trades:
                pd.testing.assert_frame_equal(saved, pd.DataFrame(trades))
            verified.append({"trial": task, "fold": number, "meta_samples": len(meta), "trades": len(saved)})
    candidates = []
    for task in ("trial_c1", "trial_c01"):
        trial = result(task)
        expected = all(f["gate_fitted"] for f in trial["folds"]) and gate.search.policy_summary([f["gated"] for f in trial["folds"]])["validation_eligible"] and gate.search.policy_summary([f["stress"] for f in trial["folds"]])["validation_eligible"]
        assert trial["research_candidate"] == expected
        if expected:
            candidates.append((trial["summary"]["worst_fold_avg_r"], task))
    assert state["candidate"] == (max(candidates)[1] if candidates else None)
    return {"status": "passed", "verified_utc": gate.search.utc_now(), "fold_trials": verified,
            "checks": ["frozen hashes/packages", "chronological purge and excluded final year",
                       "nonoverlapping meta samples", "training-only scaling", "gate probability reconstruction",
                       "baseline/gated/stress replay", "saved trades", "candidate admission"],
            "verifier_sha256": runner.checksum(Path(__file__)), "models_trained": 0, "orders_sent": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output.resolve())
    gate.search.write_json(args.output/"verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
