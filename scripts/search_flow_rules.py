"""Fixed trade-flow hypotheses; validation selects before diagnostic test.

This is a separate, historical development experiment, not a retuning of the
ML run or permission to change the existing prospective observer.
"""

import argparse
import json
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import search_flow_exhaustion as flow


def configurations():
    return [{"id": f"{name}_imb_{imb}_volume_{volume}_tp_{target}",
             "name": name, "imbalance": imb, "volume_ratio": volume, "target": target}
            for name, imb, volume, target in product(
                ("continuation", "exhaustion", "absorption"), (.1, .2), (1.5, 2.), (1., 1.5, 3.))]


def signals(frame, config):
    slow = frame.agg_taker_volume_imbalance_mean_15m
    change = frame.flow_fast_change
    move = frame.price_return_15
    burst = frame.agg_trade_volume_ratio_15m >= config["volume_ratio"]
    buy, sell = slow >= config["imbalance"], slow <= -config["imbalance"]
    if config["name"] == "continuation":
        long = buy & (change > .05) & (move > 0)
        short = sell & (change < -.05) & (move < 0)
    elif config["name"] == "exhaustion":
        short = buy & (change < -.05) & (move > 0) & (frame.price_upper_wick >= .25)
        long = sell & (change > .05) & (move < 0) & (frame.price_lower_wick >= .25)
    elif config["name"] == "absorption":
        short = buy & (move <= 0)
        long = sell & (move >= 0)
    else:
        raise ValueError("Unknown hypothesis")
    output = frame.copy()
    output["pred_action"] = np.where(burst & long, "long", np.where(burst & short, "short", "no_trade"))
    output["pred_confidence"] = 0.
    return output


def run(output):
    started = time.monotonic()
    if flow.digest(flow.DATASET) != flow.EXPECTED_SHA:
        raise ValueError("Dataset changed")
    output.mkdir(parents=True, exist_ok=False)
    source_protocol = flow.search.ROOT / "runtime_reports/flow_exhaustion_20261001/protocol.json"
    protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
    for inherited in ("feature_sets", "models", "targets_R", "thresholds_R", "model_parameters"):
        protocol.pop(inherited)
    protocol.update(created_utc=flow.search.utc_now(), experiment="direct_trade_flow_rules",
                    configurations=configurations(), prior_ml_validation_seen=True,
                    source_ml_protocol_sha256=flow.digest(source_protocol),
                    source_sha256=flow.digest(Path(__file__)),
                    selection="one winner per fold by validation severe return, >=10 trades all costs; same research gate as ML",
                    training="no fitted model: 45 day training range unused; same validation/test dates as ML",
                    additional_fixed_thresholds={"fast_imbalance_change": .05, "wick_fraction": .25})
    protocol["source_hashes"]["scripts/search_flow_rules.py"] = flow.digest(Path(__file__))
    flow.search.write_json(output / "protocol.json", protocol)
    report = {"status": "running", "folds": [], "real_trading_enabled": False,
              "independent_holdout": False, "frozen_observer_modified": False}
    try:
        raw = pd.read_parquet(flow.DATASET)
        market, _ = flow.funding_market(raw)
        frame, columns = flow.features(raw)
        finite = np.isfinite(frame[columns["flow"]].to_numpy(dtype=float)).all(axis=1)
        frame = frame.loc[finite & frame.entry_timestamp.mod(15*flow.MINUTE).eq(0)]
        backtest = flow.search.load("8_model_signal_backtest.py", "flow_rule_backtest")
        action = flow.search.load("16_action_profitability_walk_forward_model.py", "flow_rule_action")
        for number, definition in enumerate(protocol["folds"], 1):
            start = int(pd.Timestamp(definition["train_start"]).timestamp()*1000)
            val_start, test_start, test_end = start+45*flow.DAY, start+55*flow.DAY, start+65*flow.DAY
            validation, test = flow.period(frame, val_start, test_start), flow.period(frame, test_start, test_end)
            folder = output / f"fold_{number}"
            folder.mkdir()
            records = []
            for config in protocol["configurations"]:
                if time.monotonic()-started > 1800:
                    raise TimeoutError("Predeclared runtime limit")
                metrics, _ = flow.evaluate(signals(validation, config), config["target"], market, backtest, action)
                records.append({"id": config["id"], "config": config, "metrics": metrics,
                                "rejection_reasons": flow.rejection_reasons(metrics)})
                flow.search.write_json(folder / "validation_trials.json", records)
            selected = flow.select_validation(records)
            flow.search.write_json(folder / "selection_before_test.json", selected)
            fold = {"fold": number, "validation_trials": len(records),
                    "test_start_utc": pd.Timestamp(test_start, unit="ms", tz="UTC").isoformat(),
                    "test_end_exclusive_utc": pd.Timestamp(test_end, unit="ms", tz="UTC").isoformat()}
            if selected is None:
                fold["status"] = "no_validation_candidate"
            else:
                config = selected["config"]
                predicted = signals(test, config)
                predicted.to_parquet(folder / "test_predictions.parquet", index=False)
                metrics, detail = flow.evaluate(predicted, config["target"], market, backtest, action)
                flow.search.write_json(folder / "test_ledger.json", detail)
                fold.update(status="diagnostic_only", selected_policy=config["id"], metrics=metrics,
                            validation_rejection_reasons=selected["rejection_reasons"],
                            test_rejection_reasons=flow.rejection_reasons(metrics))
            report["folds"].append(fold)
            flow.search.write_json(output / "result.json", report)
            print(f"fold={number} status={fold['status']}", flush=True)
        report["research_gate_passed"] = all(f["status"] == "diagnostic_only"
            and not f["validation_rejection_reasons"] and not f["test_rejection_reasons"] for f in report["folds"])
        report.update(status="completed", elapsed_seconds=round(time.monotonic()-started, 3), completed_utc=flow.search.utc_now())
        flow.search.write_json(output / "result.json", report)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        flow.search.write_json(output / "result.json", report)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=flow.search.ROOT / "runtime_reports/flow_rules_20261001")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "research_gate_passed": result["research_gate_passed"]}))
