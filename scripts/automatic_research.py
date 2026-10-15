"""Finite, resumable offline research queue with resource limits and immutable inputs."""

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts import run_bounded_profitability_search as search

TASKS = ["audit", "prepare_1", "prepare_2", "prepare_3", "trial_c1", "trial_c01"]
SOURCE_FILES = ["scripts/automatic_research.py", "scripts/automatic_research_worker.py",
                "scripts/run_bounded_profitability_search.py", "scripts/run_cost_aware_gate.py",
                "scripts/expected_return_research.py", "scripts/analyze_loss_causes.py",
                "scripts/audit_nonoverlapping_signals.py", "scripts/verify_cost_aware_gate.py",
                "4_triple_barrier_labeler.py", "8_model_signal_backtest.py", "13_train_selective_direction_model.py",
                "14_walk_forward_direction_model.py", "15_two_stage_walk_forward_model.py",
                "16_action_profitability_walk_forward_model.py"]
DEFAULT_DATASET = search.ROOT / "ml_dataset_5y_20260828/market=usdm/symbol=BTCUSDT/timeframe=1m/ml_dataset_BTCUSDT_USDM_1m.parquet"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive_lock(root):
    # OS lock ownership is released on exit/crash; a leftover file is not a stale lock.
    with (root/".runner.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def resident_bytes(pid):
    if os.name != "nt":
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
                    ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x1000 | 0x10, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Cannot monitor worker memory")
    try:
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counter), counter.cb):
            raise OSError(ctypes.get_last_error(), "Cannot read worker memory")
        return counter.working
    finally:
        kernel.CloseHandle(handle)


def output_size(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def limit_reason(elapsed, allowed, rss, size, free, limits):
    if elapsed >= allowed:
        return "task_time_limit"
    if rss > limits["memory_mib"]*1024**2:
        return "worker_memory_limit"
    if size > limits["output_mib"]*1024**2:
        return "output_size_limit"
    if free < limits["minimum_free_gib"]*1024**3:
        return "free_disk_limit"
    return None


def monitor(command, root, folder, allowed, limits):
    started = time.monotonic()
    peak = 0
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONIOENCODING": "utf-8"}
    with (folder/"execution.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=search.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        reason = None
        try:
            while process.poll() is None:
                try:
                    rss = resident_bytes(process.pid)
                except OSError:
                    if process.poll() is not None:
                        break
                    raise
                peak = max(peak, rss)
                reason = limit_reason(time.monotonic()-started, allowed, rss, output_size(root), shutil.disk_usage(root).free, limits)
                if reason:
                    process.kill()
                    break
                time.sleep(.5)
            code = process.wait()
            reason = reason or limit_reason(time.monotonic()-started, allowed, peak, output_size(root), shutil.disk_usage(root).free, limits)
            return {"exit_code": code, "stop_reason": reason, "elapsed_seconds": time.monotonic()-started, "peak_worker_rss_bytes": peak}
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def initialize(root, max_seconds):
    reference = search.ROOT/"runtime_reports/expected_return_search_20260910/base_return_tp_1_5.json"
    old = json.loads(reference.read_text(encoding="utf-8"))
    plan = {"created_utc": search.utc_now(), "dataset": str(DEFAULT_DATASET), "dataset_sha256": checksum(DEFAULT_DATASET),
            "reference_trial": str(reference), "reference_sha256": checksum(reference),
            "features": old["folds"][0]["feature_columns"], "base_config": old["config"],
            "tasks": TASKS, "trials": [{"id": "trial_c1", "C": 1}, {"id": "trial_c01", "C": .1}],
            "limits": {"total_seconds": max_seconds, "task_seconds": 180, "memory_mib": 2048,
                       "output_mib": 64, "minimum_free_gib": 5, "max_attempts_per_task": 2},
            "schedule": {"start_offsets_days": [0, 365, 730], "base_train_days": 30, "meta_days": 90,
                         "validation_days": 30, "meta_prediction_chunk_days": 30, "excluded_final_days": 365},
            "sampling": "closed nonoverlapping 240-minute meta label windows, earliest entry first",
            "threshold": .5, "minimum_meta_signals": 200, "minimum_each_meta_class": 20,
            "minimum_validation_trades_per_fold": 20, "slippage_bps": [5, 10],
            "admission": "all 3 folds positive mean R/return and MDD >= -20%, at least 20 trades each, also stress",
            "outer_validation_previously_used_possible": True, "independent_test_evaluated": False,
            "collection_allowed": False, "orders_allowed": False, "auto_deployment_allowed": False,
            "packages": {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow", "scikit-learn", "xgboost")},
            "code_sha256": {name: checksum(search.ROOT/name) for name in SOURCE_FILES}}
    search.write_json(root/"protocol.json", plan)
    for name in SOURCE_FILES:
        target = root/"code_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((search.ROOT/name).read_bytes())
    state = {"status": "ready", "protocol_sha256": checksum(root/"protocol.json"), "charged_seconds": 0,
             "tasks": {task: {"status": "pending", "attempts": []} for task in TASKS}, "orders_sent": 0,
             "model_approved": False, "collection_started": False, "candidate": None}
    search.write_json(root/"state.json", state)
    return plan, state


def verify_resume(root, plan, state):
    if checksum(root/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    if checksum(Path(plan["dataset"])) != plan["dataset_sha256"] or checksum(Path(plan["reference_trial"])) != plan["reference_sha256"]:
        raise ValueError("Frozen dataset/reference changed; start a new reviewed run")
    for name, expected in plan["code_sha256"].items():
        if checksum(search.ROOT/name) != expected or checksum(root/"code_snapshot"/name) != expected:
            raise ValueError("Frozen source code changed: "+name)
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Package version changed: "+name)
    for task in state["tasks"].values():
        if task["status"] == "completed":
            for name, expected in task["artifact_sha256"].items():
                if checksum(root/name) != expected:
                    raise ValueError("Completed artifact changed: "+name)


def next_task(state):
    return next((name for name in TASKS if state["tasks"][name]["status"] != "completed"), None)


def report(root, plan, state):
    lines = ["# Automatic Offline Research", "", f"Status: **{state['status']}**", "",
             "No collection, orders, automatic deployment or independent final-test evaluation.",
             "This finite queue does not guarantee a profitable strategy.", "",
             f"Charged time: {state['charged_seconds']:.3f} / {plan['limits']['total_seconds']} seconds.",
             f"Stop reason: {state.get('stop_reason', 'none')}", "", "| Stage | Status | Attempts |", "| --- | --- | ---: |"]
    for task in TASKS:
        row = state["tasks"][task]
        lines.append(f"| {task} | {row['status']} | {len(row['attempts'])} |")
    for task in ("trial_c1", "trial_c01"):
        row = state["tasks"][task]
        if row["status"] != "completed":
            continue
        result = json.loads((root/row["result_path"]).read_text(encoding="utf-8"))
        lines += ["", f"## {task}", "", "| Fold | Baseline % | Gated % | Trades | Meta samples |", "| --- | ---: | ---: | ---: | ---: |"]
        for f in result["folds"]:
            lines.append(f"| {f['fold']} | {f['baseline']['total_return_pct']:.3f} | {f['gated']['total_return_pct']:.3f} | {f['gated']['trades']} | {f['training_signals']} |")
    lines += ["", f"Research candidate: {state.get('candidate') or 'none'}. Model approval remains false.", "",
              "Commands: use --status to inspect; --resume verifies hashes and skips completed stages.",
              "A resource stop or blocked data contract is not a successful profitability result."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def execute(root, plan, state):
    verify_resume(root, plan, state)
    state["status"] = "running"
    state.pop("stop_reason", None)
    while (name := next_task(state)) is not None:
        task = state["tasks"][name]
        if task["status"] == "blocked":
            state.update(status="blocked", stop_reason=task.get("reason", "data_contract_blocked"))
            break
        if task["status"] == "running":
            # Keep the reserved time charged after a crash; never silently reset the cumulative budget.
            task["status"] = "interrupted"
            task["attempts"][-1]["status"] = "interrupted"
        remaining = plan["limits"]["total_seconds"]-state["charged_seconds"]
        if remaining < 1:
            state.update(status="budget_exhausted", stop_reason="cumulative_time_budget")
            break
        if len(task["attempts"]) >= plan["limits"]["max_attempts_per_task"]:
            state.update(status="stopped", stop_reason="attempt_limit_"+name)
            break
        reason = limit_reason(0, remaining, 0, output_size(root), shutil.disk_usage(root).free, plan["limits"])
        if reason:
            state.update(status="stopped", stop_reason=reason)
            break
        allowed = min(remaining, plan["limits"]["task_seconds"])
        folder = root/name/f"attempt_{len(task['attempts'])+1}"
        folder.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", "-m", "scripts.automatic_research_worker", "--root", str(root), "--task", name, "--output", str(folder)]
        attempt = {"status": "running", "started_utc": search.utc_now(), "reserved_seconds": allowed,
                   "command": command, "folder": str(folder.relative_to(root))}
        task["attempts"].append(attempt)
        task["status"] = "running"
        state["charged_seconds"] += allowed
        search.write_json(root/"state.json", state)
        print(f"Starting {name}, allowance={allowed:.1f}s", flush=True)
        try:
            measured = monitor(command, root, folder, allowed, plan["limits"])
        except BaseException:
            task["status"] = attempt["status"] = "interrupted"
            state.update(status="stopped", stop_reason="worker_or_monitor_interrupted")
            search.write_json(root/"state.json", state)
            report(root, plan, state)
            raise
        state["charged_seconds"] += min(measured["elapsed_seconds"], allowed)-allowed
        attempt.update(measured, finished_utc=search.utc_now())
        path = folder/"result.json"
        if measured["stop_reason"] or measured["exit_code"] != 0 or not path.exists():
            task["status"] = attempt["status"] = "failed"
            state.update(status="stopped", stop_reason=measured["stop_reason"] or "worker_failed_"+name)
            search.write_json(root/"state.json", state)
            break
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") not in ("completed", "blocked"):
            raise ValueError("Invalid worker result status")
        task["status"] = attempt["status"] = result["status"]
        task["result_path"] = str(path.relative_to(root))
        task["reason"] = result.get("reason")
        task["artifact_sha256"] = {str(p.relative_to(root)): checksum(p) for p in folder.rglob("*") if p.is_file()}
        search.write_json(root/"state.json", state)
        report(root, plan, state)
        print(f"Finished {name}: {result['status']}", flush=True)
    else:
        verify_resume(root, plan, state)
        candidates = []
        for task_name in ("trial_c1", "trial_c01"):
            result = json.loads((root/state["tasks"][task_name]["result_path"]).read_text(encoding="utf-8"))
            if result["research_candidate"]:
                candidates.append((result["summary"]["worst_fold_avg_r"], task_name))
        state.update(status="completed", candidate=max(candidates)[1] if candidates else None)
    state["updated_utc"] = search.utc_now()
    search.write_json(root/"state.json", state)
    report(root, plan, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to(search.ROOT.resolve()) or root == search.ROOT.resolve():
        raise ValueError("Output must be a subdirectory inside the project")
    if args.status:
        print((root/"state.json").read_text(encoding="utf-8"))
        return 0
    if not 60 <= args.max_seconds <= 3600:
        raise ValueError("Time budget must be 60..3600 seconds")
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    with exclusive_lock(root):
        if args.resume:
            plan = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
            state = json.loads((root/"state.json").read_text(encoding="utf-8"))
        else:
            plan, state = initialize(root, args.max_seconds)
        state = execute(root, plan, state)
    print(json.dumps({k:state.get(k) for k in ("status", "stop_reason", "charged_seconds", "candidate", "orders_sent")}), flush=True)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
