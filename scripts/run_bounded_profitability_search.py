"""Bounded validation-only experiments on an existing, checksum-verified dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DAY = 86_400_000
MINUTE = 60_000
DEFAULT_MANIFEST = ROOT / "runtime_reports/trade_research_20260908/dataset.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def configurations(experiment="classification"):
    if experiment == "expected_return":
        from scripts.expected_return_research import configurations as return_configurations
        return return_configurations()
    variants = [
        {"name": "balanced_reference", "balanced": True,
         "params": {"n_estimators": 100}},
        {"name": "natural_class_frequency", "balanced": False,
         "params": {"n_estimators": 100}},
        {"name": "natural_shallow_regularized", "balanced": False,
         "params": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03,
                    "min_child_weight": 20, "reg_lambda": 10.0, "reg_alpha": 0.5}},
    ]
    return [{"id": f"{features}_{variant['name']}", "include_trade": features == "trade",
             **variant} for features in ("base", "trade") for variant in variants]


def build_folds(start, end):
    holdout_start = start + 75 * DAY
    if end - holdout_start < 14 * DAY:
        raise ValueError("At least 14 days must remain excluded after validation")
    return [{"fold": index + 1, "train_start": start + index * 10 * DAY,
             "validation_start": start + (45 + index * 10) * DAY,
             "validation_end": start + (55 + index * 10) * DAY}
            for index in range(3)], holdout_start


def purged_parts(frame, fold):
    horizon = pd.to_numeric(frame["max_holding_minutes"], errors="raise") * MINUTE
    # Include next-bar entry, not just the signal timestamp, in the purge horizon.
    outcome_limit = pd.to_numeric(frame["entry_timestamp"], errors="raise") + horizon
    train = frame[(frame.timestamp >= fold["train_start"])
                  & (outcome_limit < fold["validation_start"])].copy()
    validation = frame[(frame.timestamp >= fold["validation_start"])
                       & (frame.timestamp < fold["validation_end"])
                       & (outcome_limit < fold["validation_end"])].copy()
    if train.empty or validation.empty:
        raise ValueError("Purge produced an empty partition")
    return train, validation


class CachedProbabilities:
    def __init__(self, model, frame, columns):
        self.values = model.predict_proba(frame[columns])

    def predict_proba(self, frame):
        if len(frame) != len(self.values):
            raise ValueError("Cached probabilities belong to a different partition")
        return self.values


def policy_summary(metrics, minimum_trades=20):
    total_trades = sum(row["trades"] for row in metrics)
    enough = len(metrics) == 3 and all(row["trades"] >= minimum_trades for row in metrics)
    positive = sum(row["avg_return_r"] > 0 and row["total_return_pct"] > 0 for row in metrics)
    worst_r = min((row["avg_return_r"] for row in metrics), default=0.0)
    return {"validation_eligible": enough and positive == 3
            and all(row["max_drawdown_pct"] >= -20 for row in metrics),
            "positive_folds": positive, "folds_with_enough_trades": sum(row["trades"] >= minimum_trades for row in metrics),
            "trades": total_trades, "worst_fold_avg_r": worst_r,
            "weighted_avg_r": sum(row["avg_return_r"] * row["trades"] for row in metrics) / total_trades if total_trades else 0.0,
            "fold_returns_pct": [row["total_return_pct"] for row in metrics],
            "worst_drawdown_pct": min((row["max_drawdown_pct"] for row in metrics), default=0.0)}


def choose_candidate(candidates):
    eligible = [row for row in candidates if row["validation_eligible"] and row["stress_survived"]]
    return max(eligible, key=lambda row: (row["worst_fold_avg_r"], row["weighted_avg_r"])) if eligible else None


def run_trial(protocol_path, trial_id):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment") == "expected_return":
        from scripts.expected_return_research import run_trial as return_trial
        return return_trial(protocol_path, trial_id)
    config = next(row for row in protocol["configurations"] if row["id"] == trial_id)
    output = protocol_path.parent
    report_path = output / f"{trial_id}.json"
    action = load("16_action_profitability_walk_forward_model.py", "search_action")
    walk = load("14_walk_forward_direction_model.py", "search_walk")
    training = load("13_train_selective_direction_model.py", "search_training")
    two_stage = load("15_two_stage_walk_forward_model.py", "search_features")
    backtest = load("8_model_signal_backtest.py", "search_backtest")
    dataset_path = Path(protocol["dataset"])
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed after protocol was frozen")
    trade_columns = action.verified_trade_columns(protocol["manifest"], dataset_path)
    report = {"trial_id": trial_id, "status": "running", "started_utc": utc_now(),
              "config": config, "folds": [], "candidates": [], "test_evaluated": False,
              "real_trading_enabled": False, "registry_modified": False}
    policies = [(threshold, margin) for threshold in protocol["thresholds"] for margin in protocol["margins"]]
    results = {pair: [] for pair in policies}
    stress_results = {pair: [] for pair in policies}
    try:
        for fold in protocol["folds"]:
            read_fold = {**fold, "test_end": fold["validation_end"]}
            raw = walk.load_fold_source(dataset_path, read_fold, protocol["start_ms"], trade_columns)
            frame, features = two_stage.add_two_stage_features(raw, training)
            if config["include_trade"]:
                features += trade_columns
            if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
                raise ValueError("Nonfinite model input")
            if not frame.round_trip_cost_bps.eq(protocol["round_trip_cost_bps"]).all():
                raise ValueError("Label cost differs from the fixed protocol")
            if not frame.label_horizon_complete.eq(True).all():
                raise ValueError("Incomplete label horizon")
            frame = action.add_profitability_targets(frame, protocol["extra_slippage_bps"])
            train, validation = purged_parts(frame, fold)
            models, weights_used = [], {}
            for offset, side in enumerate(("long", "short")):
                target = train[f"{side}_profitability_target"].astype(int)
                if target.nunique() != 2:
                    raise ValueError("Both classes required")
                model = training.make_model(protocol["seed"] + fold["fold"] * 10 + offset)
                model.set_params(**config["params"], n_jobs=protocol["n_jobs"])
                weights = training.sample_weights(target) if config["balanced"] else None
                model.fit(train[features], target, sample_weight=weights)
                models.append(CachedProbabilities(model, validation, features))
                weights_used[side] = float(target.mean())
            fold_result = {"fold": fold["fold"], "train_rows": len(train), "validation_rows": len(validation),
                           "features": len(features), "feature_columns": features,
                           "last_train_outcome_limit_ms": int((train.entry_timestamp + train.max_holding_minutes * MINUTE).max()),
                           "last_validation_outcome_limit_ms": int((validation.entry_timestamp + validation.max_holding_minutes * MINUTE).max()),
                           "positive_training_share": weights_used, "policies": []}
            for threshold, margin in policies:
                predictions = action.prediction_frame(*models, validation, features, threshold, margin)
                metrics = action.evaluate_candidate(backtest, predictions, protocol["extra_slippage_bps"])
                stress = action.evaluate_candidate(backtest, predictions, protocol["stress_slippage_bps"])
                results[threshold, margin].append(metrics)
                stress_results[threshold, margin].append(stress)
                fold_result["policies"].append({"threshold": threshold, "margin": margin, "base": metrics, "stress": stress})
            report["folds"].append(fold_result)
            write_json(report_path, report)
            print(f"{trial_id}: fold {fold['fold']}/3 completed", flush=True)
        for pair, metrics in results.items():
            summary = policy_summary(metrics, protocol["minimum_validation_trades"])
            stress = policy_summary(stress_results[pair], protocol["minimum_validation_trades"])
            report["candidates"].append({"trial_id": trial_id, "threshold": pair[0], "margin": pair[1],
                                         **summary, "stress": stress, "stress_survived": stress["validation_eligible"]})
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_utc"] = utc_now()
        write_json(report_path, report)


def write_markdown(report, path):
    protocol = json.loads((path.parent / "protocol.json").read_text(encoding="utf-8"))
    policy_count = len(protocol['thresholds']) * len(protocol['margins'])
    regression = protocol.get("experiment") == "expected_return"
    lines = ["# 기존 데이터 기반 제한 탐색 결과", "", f"실행 시작: {report['started_utc']}",
             f"실행 상태: {report['status']}", "", "## 실행 범위", "",
             "새 수집·실거래·모델 자동 배포 없이 기존 95일 고정 데이터를 사용했다.",
             f"{len(protocol['configurations'])}개 XGBoost 설정과 설정별 {policy_count}개 진입 정책을 시간순 검증 3구간에서 비교했다.",
             ("예상 순손익 회귀: 정책의 기준값과 차이는 비용 차감 후 라벨 R 단위이며 확률이 아니다. 목표는 1R/1.5R/3R, 최대 보유 시간은 모두 240분이다."
              if regression else "이익 발생 확률 분류: 정책의 기준값과 차이는 확률 단위다."),
             "실험 설정과 데이터 해시는 실행 전 protocol.json에 고정했다.",
             "기존 라벨 왕복 비용 10bps + 추가 슬리피지 5bps를 유지하고, 추가 10bps 조건도 검사했다.",
             "실제 수수료·펀딩 현금흐름·부분 체결·지연을 완전히 재현한 것은 아니다.", "",
             "## 검증 결과", "",
             "각 행은 거래 수 기준을 충족한 정책을 우선해 최저 구간 평균 R로 선택한 진단이다. 기준을 충족한 정책이 없으면 미충족 정책을 표시한다.",
             "이는 검증 데이터에서 골라낸 진단 수치이지 독립 테스트 성과가 아니다.", "",
             "| 설정 | 상태 | 정책 | 충분한 거래 구간 | 양수 구간 | 거래 수 | 평균 R | 최저 구간 평균 R |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for trial in report["trials"]:
        best = trial.get("best_diagnostic")
        if not best:
            lines.append(f"| {trial['id']} | {trial['status']} | - | - | - | - | - | - |")
            continue
        lines.append(f"| {trial['id']} | {trial['status']} | {best['threshold']}/{best['margin']} | "
                     f"{best['folds_with_enough_trades']}/3 | {best['positive_folds']}/3 | {best['trades']} | "
                     f"{best['weighted_avg_r']:.4f} | {best['worst_fold_avg_r']:.4f} |")
    selected = report.get("validation_candidate")
    lines += ["", "## 결론", "",
              (f"검증 후보: {selected['trial_id']}. 독립 평가와 추가 검증 전이며 배포하지 않았다." if selected
               else "이번 탐색에서 모든 검증 구간과 비용 스트레스 조건을 통과한 후보는 없었다. no_trade 유지."),
              "", "## 해석의 한계", "",
              "- 평가용으로 남긴 마지막 약 19일은 이번 탐색에서 예측·정책 선택·백테스트에 사용하지 않았다.",
              "- 이 시장 기간은 과거 프로젝트 실험에서 사용됐을 수 있어 완전히 새로운 미관측 데이터라고 주장하지 않는다.",
              f"- 이번에 {report['model_policy_combinations_evaluated']}개 모델·정책 조합을 비교했다. 이전 탐색도 있어 우연히 좋은 결과를 고를 위험이 있으며 통계적 유의성을 확정하지 않았다.",
              "- 거래 수가 구간별 20건 미만이면 후보에서 제외했다. 0거래·0수익을 안정적인 성과로 평가하지 않았다.",
              "- 수익성 승인·실거래·장기 페이퍼 승인은 모두 변경하지 않았다.",
              "- 5년 데이터 전체를 이번에 재학습한 것이 아니며, 새 OI·현물·지수·호가 원본도 아직 결합하지 않았다.",
              "- 출력은 프로토콜·실험 로그·요약 JSON이다. 대용량 예측 파일이나 모델 묶음은 생성하지 않았다.",
              "", "근거: [실행 JSON](search.json), [고정 실험 조건](protocol.json).", "",
              "반복 탐색의 과적합 위험: [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_search(args):
    output = args.output_folder.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = Path(manifest["dataset_file"])
    action = load("16_action_profitability_walk_forward_model.py", "bounded_action_verify")
    action.verified_trade_columns(manifest_path, dataset)
    start = int(datetime.fromisoformat(manifest["start_utc"]).timestamp() * 1000)
    end = int(datetime.fromisoformat(manifest["end_utc"]).timestamp() * 1000) + MINUTE
    folds, holdout = build_folds(start, end)
    protocol = {"created_utc": utc_now(), "dataset": str(dataset), "manifest": str(manifest_path),
                "dataset_sha256": manifest["dataset_sha256"], "start_ms": start, "end_exclusive_ms": end,
                "folds": folds, "holdout_start_ms": holdout, "experiment": args.experiment,
                "configurations": configurations(args.experiment),
                "thresholds": [0.05, 0.10, 0.20] if args.experiment == "expected_return" else [0.52, 0.55, 0.58, 0.60, 0.65, 0.70],
                "margins": [0.0, 0.10] if args.experiment == "expected_return" else [0.0, 0.03, 0.05, 0.10],
                "round_trip_cost_bps": 10.0, "extra_slippage_bps": 5.0, "stress_slippage_bps": 10.0,
                "minimum_validation_trades": 20, "seed": 242, "n_jobs": 2,
                "max_total_seconds": args.max_seconds, "max_trial_seconds": args.trial_seconds,
                "selection": "all 3 validation folds positive with >=20 trades and drawdown >=-20%; same under cost stress",
                "test_evaluated": False, "registry_modified": False, "real_trading_enabled": False,
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                ["scripts/run_bounded_profitability_search.py", "8_model_signal_backtest.py",
                                 "13_train_selective_direction_model.py", "14_walk_forward_direction_model.py",
                                 "15_two_stage_walk_forward_model.py", "16_action_profitability_walk_forward_model.py"]}}
    if args.experiment == "expected_return":
        name = "scripts/expected_return_research.py"
        protocol["code_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        protocol["max_holding_minutes"] = 240
        protocol["policy_unit"] = "net_label_R_after_extra_slippage_not_probability"
    protocol["packages"] = {name: importlib.metadata.version(name) for name in
                            ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost")}
    for name in protocol["code_sha256"]:
        snapshot = output / "code_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
    write_json(output / "protocol.json", protocol)
    report = {"started_utc": utc_now(), "status": "running", "trials": [], "validation_candidate": None,
              "test_evaluated": False, "real_trading_enabled": False, "registry_modified": False,
              "approved_for_paper_connection": False, "live_policy": "unchanged_no_trade"}
    started = time.monotonic()
    candidates = []
    for config in protocol["configurations"]:
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            report["status"] = "budget_exhausted"
            break
        print(f"Starting {config['id']}", flush=True)
        trial = {"id": config["id"], "status": "running"}
        report["trials"].append(trial)
        write_json(output / "search.json", report)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-id", config["id"],
                   "--protocol", str(output / "protocol.json")]
        with (output / f"{config['id']}.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=min(args.trial_seconds, remaining))
                trial["status"] = "completed" if result.returncode == 0 else "failed"
                trial["exit_code"] = result.returncode
            except subprocess.TimeoutExpired:
                trial["status"] = "timeout"
        if trial["status"] == "completed":
            details = json.loads((output / f"{config['id']}.json").read_text(encoding="utf-8"))
            candidates += details["candidates"]
            qualified = [row for row in details["candidates"] if row["folds_with_enough_trades"] == 3]
            trial["best_diagnostic"] = max(qualified or details["candidates"],
                                          key=lambda row: (row["worst_fold_avg_r"], row["weighted_avg_r"]))
        write_json(output / "search.json", report)
        print(f"Finished {config['id']}: {trial['status']}", flush=True)
    else:
        report["status"] = "completed" if all(t["status"] == "completed" for t in report["trials"]) else "partial"
    report["validation_candidate"] = choose_candidate(candidates) if report["status"] == "completed" else None
    report["model_policy_combinations_evaluated"] = len(candidates)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["finished_utc"] = utc_now()
    report["output_bytes_before_final_report"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    write_json(output / "search.json", report)
    write_markdown(report, output / "REPORT.md")
    print(json.dumps({"status": report["status"], "candidate": report["validation_candidate"],
                      "elapsed_seconds": report["elapsed_seconds"], "report": str(output / "REPORT.md")}))
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-folder", type=Path, default=ROOT / "runtime_reports" / datetime.now(timezone.utc).strftime("profitability_search_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--trial-seconds", type=int, default=300)
    parser.add_argument("--experiment", choices=["classification", "expected_return"], default="classification")
    parser.add_argument("--worker-id")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.worker_id:
        if args.protocol is None:
            parser.error("worker requires protocol")
        run_trial(args.protocol, args.worker_id)
        return 0
    if not 1 <= args.max_seconds <= 1800 or not 1 <= args.trial_seconds <= 300:
        parser.error("max-seconds=1..1800 and trial-seconds=1..300 required")
    return run_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
