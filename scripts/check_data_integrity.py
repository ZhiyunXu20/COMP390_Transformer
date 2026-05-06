#!/usr/bin/env python3
"""Verify split TSV SHA-256 records match on-disk files (manifest + split_metadata)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


TSV_NAMES = ("train.tsv", "val.tsv", "test.tsv")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_integrity_check(
    repo: Path,
    *,
    split_subdir: str,
    report_path: Path | None,
    write_report: bool,
) -> int:
    split_dir = (repo / split_subdir).resolve()
    rows: list[dict[str, str]] = []
    ok = True

    actual: dict[str, str] = {}
    for name in TSV_NAMES:
        fp = split_dir / name
        if not fp.is_file():
            rows.append(
                {
                    "file": name,
                    "source": "(missing file)",
                    "recorded_hash": "—",
                    "actual_hash": "—",
                    "status": "FAIL (file missing)",
                }
            )
            ok = False
            continue
        actual[name] = sha256_file(fp)

    manifest = load_json(split_dir / "manifest.json") or {}
    manifest_sha = manifest.get("sha256") or {}
    if not isinstance(manifest_sha, dict):
        manifest_sha = {}

    meta = load_json(split_dir / "split_metadata.json") or {}
    meta_keys = {
        "train.tsv": "train_file_sha256_for_tokenizers",
        "val.tsv": "val_file_sha256",
        "test.tsv": "test_file_sha256",
    }

    for name in TSV_NAMES:
        act = actual.get(name)
        if act is None:
            continue

        rec_man = manifest_sha.get(name)
        if rec_man is None:
            rows.append(
                {
                    "file": name,
                    "source": "manifest.json",
                    "recorded_hash": "—",
                    "actual_hash": act,
                    "status": "FAIL (no manifest entry)",
                }
            )
            ok = False
        else:
            st = "PASS" if rec_man == act else "FAIL"
            if rec_man != act:
                ok = False
            rows.append(
                {
                    "file": name,
                    "source": "manifest.json",
                    "recorded_hash": rec_man,
                    "actual_hash": act,
                    "status": st,
                }
            )

        mk = meta_keys[name]
        rec_meta = meta.get(mk)
        if rec_meta is None:
            rows.append(
                {
                    "file": name,
                    "source": f"split_metadata.json ({mk})",
                    "recorded_hash": "—",
                    "actual_hash": act,
                    "status": "FAIL (missing metadata field)",
                }
            )
            ok = False
        else:
            st = "PASS" if rec_meta == act else "FAIL"
            if rec_meta != act:
                ok = False
            rows.append(
                {
                    "file": name,
                    "source": f"split_metadata.json ({mk})",
                    "recorded_hash": str(rec_meta),
                    "actual_hash": act,
                    "status": st,
                }
            )

    if write_report and report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Data integrity report",
            "",
            f"- **split directory**: `{split_dir}`",
            f"- **overall**: **{'PASS' if ok else 'FAIL'}**",
            "",
            "| file | source | recorded_hash | actual_hash | status |",
            "|------|--------|---------------|-------------|--------|",
        ]
        for r in rows:
            lines.append(
                "| {file} | {source} | `{recorded_hash}` | `{actual_hash}` | {status} |".format(**r)
            )
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=str,
        default=str(infer_repo_root()),
        help="Repository root (default: parent of scripts/)",
    )
    ap.add_argument(
        "--split-subdir",
        type=str,
        default="data/splits/en_fr_50k_seed42",
        help="Path relative to repo root for manifest + TSV files",
    )
    ap.add_argument(
        "--report",
        type=str,
        default="results/data_integrity_report.md",
        help="Write markdown report relative to repo root (default: results/data_integrity_report.md)",
    )
    ap.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write the markdown report",
    )
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    report = repo / args.report if not Path(args.report).is_absolute() else Path(args.report)
    code = run_integrity_check(
        repo,
        split_subdir=args.split_subdir,
        report_path=report,
        write_report=not args.no_report,
    )
    if not args.no_report:
        print(f"Wrote {report}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
