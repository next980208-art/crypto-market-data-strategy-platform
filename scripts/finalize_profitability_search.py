"""Recover a summary from complete trial results without training again."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

try:
    from scripts import run_bounded_profitability_search as search
except ModuleNotFoundError:
    import run_bounded_profitability_search as search


def finalize(folder):
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((folder / "search.json").read_text(encoding="utf-8"))
    if report["status"] == "completed":
        raise ValueError("Summary already completed; refusing to replace it")
    if hashlib.sha256(Path(protocol["dataset"]).read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Dataset no longer matches the experiment")
    for name, digest in protocol["code_sha256"].items():
        if hashlib.sha256((folder / "code_snapshot" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source snapshot mismatch: {name}")
    trials, candidates, finishes = [], [], []
    for config in protocol["configurations"]:
        details = json.loads((folder / f"{config['id']}.json").read_text(encoding="utf-8"))
        expected_policies = len(protocol["thresholds"]) * len(protocol["margins"])
        if (details["status"] != "completed" or details["config"] != config
                or len(details["folds"]) != len(protocol["folds"])
                or len(details["candidates"]) != expected_policies
                or details["test_evaluated"] or details["real_trading_enabled"] or details["registry_modified"]):
            raise ValueError(f"Incomplete or incompatible trial: {config['id']}")
        candidates.extend(details["candidates"])
        qualified = [row for row in details["candidates"] if row["folds_with_enough_trades"] == 3]
        best = max(qualified or details["candidates"], key=lambda row: (row["worst_fold_avg_r"], row["weighted_avg_r"]))
        trials.append({"id": config["id"], "status": "completed", "verified_from_trial_json": True,
                       "best_diagnostic": best})
        finishes.append(datetime.fromisoformat(details["finished_utc"]))
    # The failed supervisor's monotonic stopwatch cannot be recovered.
    finished = max(finishes)
    report.update(status="completed", trials=trials, validation_candidate=search.choose_candidate(candidates),
                  model_policy_combinations_evaluated=len(candidates),
                  finished_utc=finished.isoformat(),
                  elapsed_seconds=round((finished - datetime.fromisoformat(report["started_utc"])).total_seconds(), 3),
                  elapsed_measurement="wall_clock_start_to_last_trial_finish_not_original_monotonic_timer",
                  summary_recovered_utc=search.utc_now(), training_rerun=False,
                  supervisor_error="PermissionError WinError 5 replacing search.tmp with search.json",
                  output_bytes_before_final_report=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()))
    search.write_json(folder / "search.json", report)
    search.write_markdown(report, folder / "REPORT.md")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = finalize(args.folder)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "candidate": result["validation_candidate"], "training_rerun": False}))
