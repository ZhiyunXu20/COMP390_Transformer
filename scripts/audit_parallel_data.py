#!/usr/bin/env python3
"""Audit train/val/test TSV splits: overlap matrices and 1-to-many noise.

Parser matches TabParallelDataset: ``line.rstrip("\\n").split("\\t")``;
rows with ``len(parts) != 2`` are skipped (bad). Overlap stats use
``.strip()`` on both columns, matching loaded training examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _iter_pairs(path: Path) -> tuple[list[tuple[str, str]], int, int, int]:
    """Return (pairs, bad_lines, empty_after_strip, src_eq_tgt)."""
    pairs: list[tuple[str, str]] = []
    bad = 0
    empty = 0
    seq = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                bad += 1
                continue
            s, t = parts[0].strip(), parts[1].strip()
            if not s or not t:
                empty += 1
            if s == t:
                seq += 1
            pairs.append((s, t))
    return pairs, bad, empty, seq


def count_lines(path: str | Path) -> int:
    """Count lines that yield exactly two tab-separated fields (loader semantics)."""
    pairs, _, _, _ = _iter_pairs(Path(path))
    return len(pairs)


def _pair_set(pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    return set(pairs)


def _src_set(pairs: list[tuple[str, str]]) -> set[str]:
    return {s for s, _ in pairs}


def _tgt_set(pairs: list[tuple[str, str]]) -> set[str]:
    return {t for _, t in pairs}


def compute_overlap(train_path: str | Path, val_path: str | Path, test_path: str | Path) -> dict[str, Any]:
    train_p = Path(train_path)
    val_p = Path(val_path)
    test_p = Path(test_path)

    train, bad_tr, e_tr, seq_tr = _iter_pairs(train_p)
    val, bad_va, e_va, seq_va = _iter_pairs(val_p)
    test, bad_te, e_te, seq_te = _iter_pairs(test_p)

    Pt, Pv, Ps = _pair_set(train), _pair_set(val), _pair_set(test)
    St, Sv, Ss = _src_set(train), _src_set(val), _src_set(test)
    Tt, Tv, Ts = _tgt_set(train), _tgt_set(val), _tgt_set(test)

    all_pairs = train + val + test
    src_to_tgt: dict[str, set[str]] = defaultdict(set)
    tgt_to_src: dict[str, set[str]] = defaultdict(set)
    for s, t in all_pairs:
        src_to_tgt[s].add(t)
        tgt_to_src[t].add(s)

    n_multi_src = sum(1 for s, ts in src_to_tgt.items() if len(ts) > 1)
    n_multi_tgt = sum(1 for t, ss in tgt_to_src.items() if len(ss) > 1)

    return {
        "row_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "bad_rows_skipped": {"train": bad_tr, "val": bad_va, "test": bad_te},
        "empty_src_or_tgt_rows": {"train": e_tr, "val": e_va, "test": e_te},
        "src_eq_tgt_rows": {"train": seq_tr, "val": seq_va, "test": seq_te},
        "exact_pair_overlap": {
            "train-val": len(Pt & Pv),
            "train-test": len(Pt & Ps),
            "val-test": len(Pv & Ps),
        },
        "source_overlap": {
            "train-val": len(St & Sv),
            "train-test": len(St & Ss),
            "val-test": len(Sv & Ss),
        },
        "target_overlap": {
            "train-val": len(Tt & Tv),
            "train-test": len(Tt & Ts),
            "val-test": len(Tv & Ts),
        },
        "source_to_multiple_targets": n_multi_src,
        "target_to_multiple_sources": n_multi_tgt,
    }


def _write_md(path: Path, data: dict[str, Any], splits_dir: Path) -> None:
    split_pf = f"{splits_dir.as_posix()}/{{train,val,test}}.tsv"
    lines = [
        "# Data split overlap audit",
        "",
        r"Parser: line.rstrip('\n').split('\t')",
        f"Source files: {split_pf}",
        "Note: default csv.reader is unsafe here because DCEP text contains un-escaped natural-language double quotes.",
        "",
        "## Summary",
        "",
        f"- **row_counts**: {data['row_counts']}",
        f"- **bad_rows_skipped** (len(parts) != 2): {data['bad_rows_skipped']}",
        f"- **empty_src_or_tgt_rows** (after strip): {data['empty_src_or_tgt_rows']}",
        f"- **src_eq_tgt_rows**: {data['src_eq_tgt_rows']}",
        "",
        "## Exact (source, target) pair overlap",
        "",
        "| pair | count |",
        "|---|---:|",
    ]
    for k, v in data["exact_pair_overlap"].items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Source-string overlap",
            "",
            "| pair | count |",
            "|---|---:|",
        ]
    )
    for k, v in data["source_overlap"].items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Target-string overlap",
            "",
            "| pair | count |",
            "|---|---:|",
        ]
    )
    for k, v in data["target_overlap"].items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## 1-to-many (union of all splits)",
            "",
            f"- **source strings with >1 distinct target**: {data['source_to_multiple_targets']}",
            f"- **target strings with >1 distinct source**: {data['target_to_multiple_sources']}",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits/en_fr_50k_seed42"),
        help="Directory containing train.tsv, val.tsv, test.tsv",
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/data_split_overlap_audit.md"),
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/data_split_overlap_audit.json"),
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    split_dir = args.splits_dir.resolve() if args.splits_dir.is_absolute() else (repo / args.splits_dir).resolve()

    train_f = split_dir / "train.tsv"
    val_f = split_dir / "val.tsv"
    test_f = split_dir / "test.tsv"
    for p in (train_f, val_f, test_f):
        if not p.is_file():
            print(f"Missing split file: {p}", file=sys.stderr)
            return 1

    data = compute_overlap(train_f, val_f, test_f)

    out_md = args.output_md if args.output_md.is_absolute() else repo / args.output_md
    out_json = args.output_json if args.output_json.is_absolute() else repo / args.output_json
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        split_rel = split_dir.relative_to(repo)
    except ValueError:
        split_rel = split_dir
    _write_md(out_md, data, split_rel)

    out_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}", file=sys.stderr)
    print(f"Wrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
