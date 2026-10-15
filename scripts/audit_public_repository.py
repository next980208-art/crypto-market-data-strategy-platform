"""Fail CI when the public portfolio repository contains unsafe or broken artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024

REQUIRED_PATHS = {
    "README.md",
    "SECURITY.md",
    ".github/workflows/tests.yml",
    "demo/input/binance_usdm_btcusdt_1000.jsonl",
    "demo/output/latest_features.parquet",
    "demo/results/latest_end_to_end_run.json",
    "docs/PROJECT_EVIDENCE.md",
    "docs/TECHNICAL_DECISIONS.md",
}

FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}

FORBIDDEN_SUFFIXES = {
    ".env",
    ".key",
    ".pem",
    ".pkl",
    ".joblib",
    ".sqlite",
    ".sqlite3",
    ".wal",
    ".log",
}

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def check_required_paths(errors: list[str]) -> None:
    for relative_path in sorted(REQUIRED_PATHS):
        if not (ROOT / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")


def check_tracked_files(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT)
        if FORBIDDEN_PARTS.intersection(relative.parts):
            errors.append(f"generated directory is tracked: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"runtime or secret-like file is tracked: {relative.as_posix()}")
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"tracked file exceeds 10 MiB: {relative.as_posix()}")


def check_readme_links(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK.findall(readme):
        target = raw_target.strip().split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        decoded = target.replace("%20", " ")
        if not (ROOT / decoded).exists():
            errors.append(f"README link target does not exist: {raw_target}")


def main() -> int:
    errors: list[str] = []
    files = tracked_files()
    check_required_paths(errors)
    check_tracked_files(files, errors)
    check_readme_links(errors)

    if errors:
        print("Public repository audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Public repository audit passed: {len(files)} tracked files checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
