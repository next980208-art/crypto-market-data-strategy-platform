"""Offline context audit and label-free feature sidecar; never starts collection or training."""

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_bounded_profitability_search as research

MINUTE = research.MINUTE
DAY = research.DAY
FAMILIES = {"open_interest_5m": 5 * MINUTE, "spot_1m": MINUTE, "index_1m": MINUTE}
FEATURES = ["context_oi_change_5m", "context_oi_change_60m", "context_spot_return_1m",
            "context_spot_return_15m", "context_index_return_1m",
            "context_futures_spot_basis", "context_futures_index_basis", "context_spot_index_basis"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def database_hashes(path):
    return {suffix or "main": sha256(Path(str(path) + suffix)) for suffix in ("", "-wal")
            if Path(str(path) + suffix).exists()}


def utc(value):
    return None if value is None else pd.Timestamp(int(value), unit="ms", tz="UTC").isoformat()


def normalize(family, records):
    step = FAMILIES[family]
    parsed, invalid = [], 0
    digest = hashlib.sha256()
    for event, end, received, source, payload in records:
        digest.update(json.dumps([event, end, received, source, payload], ensure_ascii=False).encode("utf-8"))
        try:
            row = json.loads(payload)
            if event % step or received < end:
                raise ValueError("Off-grid timestamp or receipt before close")
            if family == "open_interest_5m":
                value = float(row["sumOpenInterest"])
                notional = float(row["sumOpenInterestValue"])
                valid = (row["symbol"] == "BTCUSDT" and int(row["timestamp"]) == event
                         and end == event and np.isfinite(notional) and notional > 0)
            else:
                o, high, low, value = map(float, row[1:5])
                valid = (int(row[0]) == event and int(row[6]) + 1 == end == event + step
                         and np.isfinite([o, high, low, value]).all()
                         and 0 < low <= min(o, value) <= max(o, value) <= high)
                if family == "spot_1m":
                    valid = valid and np.isfinite(float(row[5])) and float(row[5]) >= 0
            if not valid or not np.isfinite(value) or value <= 0:
                raise ValueError("Invalid payload")
            parsed.append({"event_ms": event, "period_end_ms": end,
                           "first_received_ms": received, "value": value})
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            invalid += 1
    frame = pd.DataFrame(parsed, columns=["event_ms", "period_end_ms", "first_received_ms", "value"])
    stamps = sorted({r[0] for r in records})
    expected = (stamps[-1] - stamps[0]) // step + 1 if stamps else 0
    duplicate = len(records) - len(stamps)
    quality = {"rows": len(records), "valid_rows": len(frame), "invalid_rows": invalid,
               "duplicate_events": duplicate, "missing_grid_rows": max(0, expected - len(stamps)),
               "first_event_utc": utc(stamps[0]) if stamps else None,
               "last_event_utc": utc(stamps[-1]) if stamps else None,
               "first_receipt_utc": utc(min((r[2] for r in records), default=None)),
               "last_receipt_utc": utc(max((r[2] for r in records), default=None)),
               "canonical_records_sha256": digest.hexdigest(),
               "passed": bool(records) and invalid == 0 and duplicate == 0 and expected == len(stamps)}
    if duplicate:
        raise ValueError(f"Duplicate source timestamps: {family}")
    return frame, quality


def derive_family(frame, family):
    step = FAMILIES[family]
    if frame.empty:
        return frame.copy()
    indexed = frame.set_index("event_ms").sort_index()
    # Explicit grid prevents a missing five-minute observation becoming a five-minute return.
    index = pd.Index(range(int(indexed.index.min()), int(indexed.index.max()) + step, step), name="event_ms")
    expanded = indexed.reindex(index)
    horizons = {"open_interest_5m": [(1, FEATURES[0]), (12, FEATURES[1])],
                "spot_1m": [(1, FEATURES[2]), (15, FEATURES[3])],
                "index_1m": [(1, FEATURES[4])]}[family]
    for periods, name in horizons:
        values = expanded.value.pct_change(periods, fill_method=None)
        complete = expanded.value.rolling(periods + 1).count().eq(periods + 1)
        expanded[name] = values.where(complete)
    window = max(h[0] for h in horizons) + 1
    expanded["dependency_received_ms"] = expanded.first_received_ms.rolling(window, min_periods=1).max()
    return expanded.reset_index()


def join_family(decisions, family_frame, family, mode):
    if mode not in ("received", "historical_assumption"):
        raise ValueError("Unknown availability mode")
    if family_frame.empty:
        return pd.DataFrame(index=decisions.index)
    right = family_frame.dropna(subset=["period_end_ms", "dependency_received_ms"]).copy()
    if right.empty:
        return pd.DataFrame(index=decisions.index)
    # Replay lags are explicit hypotheses, not measured exchange publication delays.
    right["available_ms"] = (right.period_end_ms + FAMILIES[family] if mode == "historical_assumption"
                             else right[["period_end_ms", "dependency_received_ms"]].max(axis=1))
    right = right.sort_values(["available_ms", "event_ms"])
    # A late-arriving older observation must not replace a newer available observation.
    right = right[right.event_ms.eq(right.event_ms.cummax())]
    right["available_ms"] = right.available_ms.astype("int64")
    joined = pd.merge_asof(decisions[["decision_ms"]].sort_values("decision_ms"), right,
                           left_on="decision_ms", right_on="available_ms", direction="backward")
    age = joined.decision_ms - joined.period_end_ms
    fresh = age.ge(0) & age.lt(2 * FAMILIES[family])
    joined.loc[~fresh, right.columns] = np.nan
    joined = joined.drop(columns="decision_ms").add_prefix(family + "__")
    joined.index = decisions.sort_values("decision_ms").index
    return joined.reindex(decisions.index)


def assemble(base, sources, mode):
    result = base[["timestamp", "entry_timestamp", "close", "max_holding_minutes"]].copy()
    result["decision_ms"] = result.entry_timestamp
    for family in FAMILIES:
        result = result.join(join_family(result, sources[family], family, mode))
    for feature in FEATURES[:5]:
        matched = [c for c in result if c.endswith("__" + feature)]
        result[feature] = result[matched[0]] if matched else np.nan
    spot = result.get("spot_1m__value", pd.Series(np.nan, index=result.index))
    index = result.get("index_1m__value", pd.Series(np.nan, index=result.index))
    result[FEATURES[5]] = result.close / spot - 1
    result[FEATURES[6]] = result.close / index - 1
    result[FEATURES[7]] = spot / index - 1
    result["features_complete"] = np.isfinite(result[FEATURES].to_numpy(dtype=float)).all(axis=1)
    for family in FAMILIES:
        column = family + "__available_ms"
        if column in result and (result[column].dropna() > result.loc[result[column].notna(), "decision_ms"]).any():
            raise AssertionError("Future context joined")
    return result


def coverage(frame):
    stamps = frame.loc[frame.features_complete, "timestamp"].sort_values()
    if stamps.empty:
        return {"complete_rows": 0, "covered_days": 0, "longest_continuous_minutes": 0,
                "first_utc": None, "last_utc": None}
    groups = stamps.diff().ne(MINUTE).cumsum()
    return {"complete_rows": len(stamps), "covered_days": len(stamps) / 1440,
            "longest_continuous_minutes": int(stamps.groupby(groups).size().max()),
            "first_utc": utc(stamps.min()), "last_utc": utc(stamps.max())}


def compare_gate(strict, replay, protocol, quality_passed):
    counts = []
    for fold in protocol["folds"]:
        row = {"fold": fold["fold"]}
        for name, frame in (("received", strict), ("historical_assumption", replay)):
            end = frame.entry_timestamp + frame.max_holding_minutes * MINUTE
            common = frame.features_complete
            row[name] = {"train_rows": int((common & frame.timestamp.ge(fold["train_start"])
                                            & end.lt(fold["validation_start"])).sum()),
                         "validation_rows": int((common & frame.timestamp.ge(fold["validation_start"])
                                                  & end.lt(fold["validation_end"])).sum())}
        counts.append(row)
    reasons = []
    if not quality_passed:
        reasons.append("source_quality_failed")
    if coverage(strict)["longest_continuous_minutes"] < 90 * 1440:
        reasons.append("received_time_features_below_existing_90_day_requirement")
    if any(c["received"]["train_rows"] < 44 * 1440 or c["received"]["validation_rows"] < 9 * 1440 for c in counts):
        reasons.append("received_time_features_insufficient_for_fixed_45_day_train_10_day_validation")
    if any(c["historical_assumption"]["train_rows"] == 0 for c in counts):
        reasons.append("even_historical_replay_has_no_joint_training_rows_in_existing_folds")
    return {"status": "blocked" if reasons else "ready_not_run", "reasons": reasons, "fold_common_rows": counts,
            "models_fitted": 0, "profitability_comparison_executed": False,
            "holdout_outcomes_read": False, "minimum_days_not_relaxed": 90,
            "baseline_metrics": None, "enriched_metrics": None}


def run(database, manifest_path, prior_protocol_path, output):
    output.mkdir(parents=True, exist_ok=False)
    before = database_hashes(database)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_protocol_path.read_text(encoding="utf-8"))
    dataset = Path(manifest["dataset_file"])
    if sha256(dataset) != manifest["dataset_sha256"] or prior["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Dataset/protocol checksum mismatch")
    # Only prices/times, never outcome labels, including in the reserved date range.
    base = pd.read_parquet(dataset, columns=["timestamp", "entry_timestamp", "close", "max_holding_minutes"])
    if (not base.timestamp.diff().dropna().eq(MINUTE).all() or base.timestamp.duplicated().any()
            or not base.entry_timestamp.eq(base.timestamp + MINUTE).all()
            or not base.max_holding_minutes.eq(240).all()
            or not np.isfinite(base.close).all() or not base.close.gt(0).all()):
        raise ValueError("Invalid base timeline or prices")
    plan = {"started_utc": research.utc_now(), "dataset_sha256": manifest["dataset_sha256"],
            "database_hashes_before": before, "previous_protocol": str(prior_protocol_path.resolve()),
            "previous_protocol_sha256": sha256(prior_protocol_path), "folds": prior["folds"],
            "holdout_start_ms": prior["holdout_start_ms"], "feature_columns": FEATURES,
            "historical_assumption_lags_ms": FAMILIES, "max_age_exclusive_ms": {f: 2*s for f,s in FAMILIES.items()},
            "strict_availability": "max(period_end_ms, max first_received_ms of feature dependencies)",
            "historical_assumption": "period_end_ms plus one source interval; not proven point-in-time availability",
            "missing_policy": "no interpolation/backfill; bounded backward asof with complete rolling windows",
            "code_sha256": sha256(Path(__file__)), "collection_started": False,
            "real_trading_enabled": False, "deployment_enabled": False}
    research.write_json(output / "protocol.json", plan)
    sources, qualities = {}, {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        inventory = [{"family": f, "rows": n, "first_event_utc": utc(a), "last_event_utc": utc(b)}
                     for f,n,a,b in connection.execute("SELECT family,COUNT(*),MIN(event_ms),MAX(event_ms) FROM observations GROUP BY family")]
        for family in FAMILIES:
            records = connection.execute("SELECT event_ms,period_end_ms,first_received_ms,source,payload "
                                         "FROM observations WHERE family=? ORDER BY event_ms", (family,)).fetchall()
            normalized, qualities[family] = normalize(family, records)
            sources[family] = derive_family(normalized, family)
    strict = assemble(base, sources, "received")
    replay = assemble(base, sources, "historical_assumption")
    after = database_hashes(database)
    if before != after or sha256(dataset) != manifest["dataset_sha256"]:
        raise RuntimeError("Source changed during audit; refusing to publish a stable-source report")
    quality_passed = check == "ok" and all(q["passed"] for q in qualities.values())
    comparison = compare_gate(strict, replay, prior, quality_passed)
    saved = []
    for name, frame in (("received", strict), ("historical_assumption", replay)):
        complete = frame[frame.features_complete].copy()
        columns = ["timestamp", "decision_ms"] + FEATURES
        columns += [c for c in complete if c.endswith(("__event_ms", "__period_end_ms", "__available_ms", "__dependency_received_ms"))]
        complete = complete[columns]
        complete["availability_mode"] = name
        complete["model_training_allowed"] = False
        complete["reserved_period"] = complete.timestamp.ge(prior["holdout_start_ms"])
        path = output / f"{name}_features_only.parquet"
        complete.to_parquet(path, index=False, compression="zstd")
        saved.append({"path": str(path.resolve()), "rows": len(complete), "sha256": sha256(path), "bytes": path.stat().st_size})
    preview = replay.loc[replay.features_complete, ["timestamp", "decision_ms"] + FEATURES].head(5)
    report = {"status": "audit_completed_comparison_" + comparison["status"], "completed_utc": research.utc_now(),
              "sqlite_check": check, "inventory": inventory, "source_quality": qualities,
              "quality_passed": quality_passed, "source_files_unchanged": True,
              "base_rows": len(base), "base_start_utc": utc(base.timestamp.min()), "base_end_utc": utc(base.timestamp.max()),
              "received": coverage(strict), "historical_assumption": coverage(replay),
              "historical_before_reserved": coverage(replay[replay.timestamp.lt(prior["holdout_start_ms"])]),
              "comparison": comparison, "artifacts": saved, "historical_feature_preview": preview.to_dict("records"),
              "model_training_allowed": False, "collectors_started": False, "orders_sent": False}
    research.write_json(output / "audit.json", report)
    (output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: report[k] for k in ("status", "received", "historical_assumption", "historical_before_reserved", "comparison")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("supplementary_market_store/market_context.sqlite3"))
    parser.add_argument("--manifest", type=Path, default=research.DEFAULT_MANIFEST)
    parser.add_argument("--prior-protocol", type=Path, default=Path("runtime_reports/expected_return_search_20260910/protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.database, args.manifest, args.prior_protocol, args.output)


if __name__ == "__main__":
    main()
