"""Compare frozen baseline/context runs without training or placing orders."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COSTS = ("normal", "stress", "severe")


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trade_effect(baseline, context):
    def indexed(frame):
        if frame.empty:
            frame = pd.DataFrame(columns=["entry_timestamp", "action", "return_pct"])
        if frame.duplicated(["entry_timestamp", "action"]).any():
            raise ValueError("Duplicate trade keys")
        values = frame.set_index(["entry_timestamp", "action"])["return_pct"].astype(float)
        if not np.isfinite(values).all() or (values <= -1).any():
            raise ValueError("Invalid account return")
        return np.log1p(values)

    b, c = indexed(baseline), indexed(context)
    common = b.index.intersection(c.index)
    removed = b.index.difference(c.index)
    added = c.index.difference(b.index)
    # Log returns are additive; this is accounting, not causal attribution.
    parts = {
        "common_trades": len(common), "removed_trades": len(removed), "added_trades": len(added),
        "common_log_return_delta": float(c.loc[common].sum() - b.loc[common].sum()),
        "removed_log_return_effect": float(-b.loc[removed].sum()),
        "added_log_return_effect": float(c.loc[added].sum()),
        "total_log_return_delta": float(c.sum() - b.sum()),
    }
    np.testing.assert_allclose(
        sum(parts[k] for k in ("common_log_return_delta", "removed_log_return_effect", "added_log_return_effect")),
        parts["total_log_return_delta"], atol=1e-12, rtol=0,
    )
    return parts


def paired_trials(trials):
    groups = {}
    for row in trials:
        if row["status"] != "completed":
            raise ValueError("Incomplete trial")
        config = row["config"]
        key = (config["family"], config["threshold_r"])
        pair = groups.setdefault(key, {})
        if config["group"] in pair:
            raise ValueError("Duplicate trial group")
        pair[config["group"]] = row
    for key, pair in sorted(groups.items()):
        if set(pair) != {"baseline", "context"}:
            raise ValueError("Unpaired feature comparison")
        yield key, pair["baseline"], pair["context"]


def analyze(root):
    results_path = root / "results.json"
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    verification = read(root / "verification.json")
    if verification["status"] != "passed" or verification["results_sha256"] != checksum(results_path):
        raise ValueError("Matching replay verification required")
    results = read(results_path)
    for name, digest in results["artifact_sha256"].items():
        path = (root / name.replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or checksum(path) != digest:
            raise ValueError(f"Artifact mismatch: {name}")
    rows = []
    for (family, threshold), baseline, context in paired_trials(results["trials"]):
        bfolds = {f["fold"]: f for f in baseline["folds"]}
        cfolds = {f["fold"]: f for f in context["folds"]}
        if set(bfolds) != set(cfolds):
            raise ValueError("Unpaired folds")
        for fold in sorted(bfolds):
            for cost in COSTS:
                b, c = bfolds[fold][cost], cfolds[fold][cost]
                frames = [pd.read_parquet(root / "trades" / f"{t['config']['id']}_{fold}_{cost}.parquet")
                          for t in (baseline, context)]
                for frame, metrics in zip(frames, (b, c)):
                    if len(frame) != metrics["trades"]:
                        raise ValueError("Trade count mismatch")
                    actual_return = np.expm1(np.log1p(frame.return_pct).sum()) * 100 if len(frame) else 0.0
                    np.testing.assert_allclose(actual_return, metrics["total_return_pct"], atol=1e-10, rtol=0)
                    if len(frame):
                        np.testing.assert_allclose(frame.return_r.mean(), metrics["avg_return_r"], atol=1e-12, rtol=0)
                rows.append({
                    "family": family, "threshold_r": threshold, "fold": fold, "cost": cost,
                    "baseline": b, "context": c,
                    "return_delta_pp": c["total_return_pct"] - b["total_return_pct"],
                    "mean_r_delta": c["avg_return_r"] - b["avg_return_r"] if b["trades"] and c["trades"] else None,
                    "trade_count_delta": c["trades"] - b["trades"],
                    "trade_set_effect": trade_effect(*frames),
                })
    summary = {}
    for cost in COSTS:
        selected = [r for r in rows if r["cost"] == cost]
        summary[cost] = {
            "pairs": len(selected),
            "improved_return": sum(r["return_delta_pp"] > 1e-12 for r in selected),
            "worse_return": sum(r["return_delta_pp"] < -1e-12 for r in selected),
            "unchanged_return": sum(abs(r["return_delta_pp"]) <= 1e-12 for r in selected),
            "improved_mean_r_both_traded": sum(r["mean_r_delta"] is not None and r["mean_r_delta"] > 1e-12 for r in selected),
        }
    return {"results_sha256": checksum(results_path), "summary": summary, "pairs": rows,
            "candidate_ids": results["candidate_ids"], "orders_sent": 0,
            "interpretation": "Paired historical comparisons, not independent tests or causal effects. Zero trades are not positive evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root == ROOT or not root.is_relative_to(ROOT):
        raise ValueError("Use a project subdirectory")
    report = analyze(root)
    report["analysis_code_sha256"] = checksum(Path(__file__))
    path = root / "context_effect_analysis.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(path)


if __name__ == "__main__":
    main()
