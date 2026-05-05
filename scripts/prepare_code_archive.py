#!/usr/bin/env python3
"""Validate that held-out test artifacts are pairwise consistent, optionally build a code zip.

Pairs: for every runs/<run>/test_eval/metrics_test.json, expects runs/<run>/test_eval/predictions.jsonl
unless --allow-missing-predictions.

Also checks results/attention_variants_test_summary.csv: every listed run must have
runs/<run>/test_eval/metrics_test.json.

Zip (--output-zip): includes files under the repo root that are not ignored by git check-ignore,
excluding .git/ and explicit extra patterns (checkpoints already ignored via .gitignore).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from pathlib import Path


def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def git_check_ignore(repo: Path, rel_posix: str) -> bool:
    """True if git considers the path ignored (exit 0 from check-ignore)."""
    cp = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-q", "--", rel_posix],
        capture_output=True,
        text=True,
    )
    return cp.returncode == 0


def iter_repo_files_for_zip(repo: Path):
    """Yield paths relative to repo (POSIX) to include in archive."""
    skip_prefixes = {".git"}
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(repo)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in skip_prefixes:
            continue
        rel_posix = rel.as_posix()
        if git_check_ignore(repo, rel_posix):
            continue
        yield rel_posix


def find_metrics_without_predictions(repo: Path) -> list[str]:
    missing: list[str] = []
    runs = repo / "runs"
    if not runs.is_dir():
        return missing
    for m in sorted(runs.glob("*/test_eval/metrics_test.json")):
        pred = m.parent / "predictions.jsonl"
        if not pred.is_file():
            missing.append(str(pred.relative_to(repo)))
    return missing


def runs_from_attention_variants_csv(repo: Path) -> list[str]:
    csv_path = repo / "results" / "attention_variants_test_summary.csv"
    if not csv_path.is_file():
        print(
            "Warning: results/attention_variants_test_summary.csv not found; CSV run check skipped.",
            file=sys.stderr,
        )
        return []
    runs: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "run" not in reader.fieldnames:
            return []
        for row in reader:
            r = (row.get("run") or "").strip()
            if r:
                runs.append(r)
    return runs


def validate_csv_runs_have_metrics(repo: Path) -> list[str]:
    bad: list[str] = []
    for run in runs_from_attention_variants_csv(repo):
        m = repo / "runs" / run / "test_eval" / "metrics_test.json"
        if not m.is_file():
            bad.append(str(m.relative_to(repo)))
    return bad


def validate(
    repo: Path,
    *,
    allow_missing_predictions: bool,
) -> int:
    mp = find_metrics_without_predictions(repo)
    if mp and not allow_missing_predictions:
        print("Missing predictions.jsonl for runs with metrics_test.json:", file=sys.stderr)
        for x in mp:
            print(f"  - {x}", file=sys.stderr)
        return 1
    if mp and allow_missing_predictions:
        print("Warning (--allow-missing-predictions): missing predictions.jsonl:", file=sys.stderr)
        for x in mp:
            print(f"  - {x}", file=sys.stderr)

    csv_bad = validate_csv_runs_have_metrics(repo)
    if csv_bad:
        print(
            "Runs listed in results/attention_variants_test_summary.csv "
            "lack test_eval/metrics_test.json:",
            file=sys.stderr,
        )
        for x in csv_bad:
            print(f"  - {x}", file=sys.stderr)
        return 1

    print("prepare_code_archive: validation OK.")
    csv_path = repo / "results" / "attention_variants_test_summary.csv"
    if csv_path.is_file() and not runs_from_attention_variants_csv(repo):
        print(
            "Note: results/attention_variants_test_summary.csv has no data rows in 'run' column.",
            file=sys.stderr,
        )
    return 0


def write_zip(repo: Path, zip_path: Path) -> None:
    zip_path = zip_path.expanduser().resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(iter_repo_files_for_zip(repo))
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for rel in files:
            abs_path = repo / rel
            zf.write(abs_path, arcname=rel)
    print(f"Wrote archive ({len(files)} files): {zip_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Do not exit 1 when predictions.jsonl is missing for a metrics_test.json",
    )
    p.add_argument(
        "--output-zip",
        type=str,
        default=None,
        help="If set, write a zip after successful validation",
    )
    args = p.parse_args()
    repo = infer_repo_root()

    code = validate(repo, allow_missing_predictions=args.allow_missing_predictions)
    if code != 0:
        return code
    if args.output_zip:
        write_zip(repo, Path(args.output_zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
