#!/usr/bin/env python3
"""
只读各 run 的 runs/<run>/test_eval/metrics_test.json，汇总 held-out test 指标。
禁止使用 runs/<run>/metrics.json（训练期 / val）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RUNS: tuple[str, ...] = (
    "fast_dot",
    "fast_add",
    "head_1h_dot",
    "var_bilinear",
    "var_gated_dot_additive",
    "var_sparsemax",
    "var_entmax15",
    "var_local_window",
    "var_global_local",
)

FIELDNAMES: tuple[str, ...] = (
    "run",
    "attention_type",
    "family",
    "bleu",
    "chrf",
    "comet",
    "bertscore_f1",
    "num_examples",
    "checkpoint",
    "notes",
    "status",
)


def mechanism_family(attention_type: str | None) -> str:
    if not attention_type:
        return ""
    return {
        "dot_product": "scaled dot-product / softmax / dense full",
        "additive": "additive scoring / softmax / dense full",
        "bilinear": "bilinear score / softmax / dense full",
        "gated_dot_additive": "gated dot+additive mix / softmax / dense full",
        "sparsemax": "dot-product logits / sparsemax / dense full",
        "entmax15": "dot-product logits / entmax α=1.5 / dense full",
        "local_window": "dot-product / softmax / dense + structural local mask",
        "global_local": "dot-product / softmax / dense + structural global-local mask",
    }.get(attention_type, "")


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def row_from_metrics(run: str, data: dict[str, Any]) -> dict[str, Any]:
    att = data.get("attention_type")
    if att is not None:
        att = str(att)
    notes_parts: list[str] = []
    pkg = data.get("pkg")
    if pkg:
        notes_parts.append(f"pkg={pkg}")
    nh = data.get("n_heads")
    if nh is not None:
        notes_parts.append(f"n_heads={nh}")
    notes = "; ".join(notes_parts)

    return {
        "run": run,
        "attention_type": att or "",
        "family": mechanism_family(att),
        "bleu": data.get("BLEU"),
        "chrf": data.get("chrF"),
        "comet": data.get("COMET"),
        "bertscore_f1": data.get("BERTScore"),
        "num_examples": data.get("number_of_test_examples"),
        "checkpoint": data.get("checkpoint") or "",
        "notes": notes,
        "status": "ok",
    }


def missing_row(run: str) -> dict[str, Any]:
    return {
        "run": run,
        "attention_type": "",
        "family": "",
        "bleu": "",
        "chrf": "",
        "comet": "",
        "bertscore_f1": "",
        "num_examples": "",
        "checkpoint": "",
        "notes": f"missing runs/{run}/test_eval/metrics_test.json",
        "status": "missing_test_eval",
    }


def fmt_cell(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_md(
    path: Path,
    rows: list[dict[str, Any]],
    repo_root: Path,
    csv_path: Path,
) -> None:
    disclaimer = (
        "**All reported scores are from independent held-out test evaluation, "
        "not training-time validation metrics.**\n\n"
        "Sources are limited to `runs/<run>/test_eval/metrics_test.json` only "
        "(never `runs/<run>/metrics.json`).\n\n"
        "---\n\n"
    )
    headers = list(FIELDNAMES)
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    head = "| " + " | ".join(headers) + " |"
    lines = [
        "# Attention variants — held-out test summary",
        "",
        disclaimer,
        head,
        sep,
    ]
    for r in rows:
        cells = [fmt_cell(r.get(h)) for h in headers]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    try:
        rel_csv = csv_path.relative_to(repo_root)
    except ValueError:
        rel_csv = csv_path
    lines.extend(
        [
            "",
            f"_Companion CSV: `{rel_csv}`._",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize test_eval metrics only (held-out test).")
    p.add_argument(
        "--repo-root",
        type=str,
        default=str(REPO_ROOT),
        help="仓库根目录（默认为本脚本的上两级）",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        metavar="RUN",
        help=f"run 目录名（默认 {' '.join(DEFAULT_RUNS)}）",
    )
    p.add_argument(
        "--csv-out",
        type=str,
        default="results/attention_variants_test_summary.csv",
    )
    p.add_argument(
        "--md-out",
        type=str,
        default="results/attention_variants_test_summary.md",
    )
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    rows: list[dict[str, Any]] = []
    for run in args.runs:
        met_path = repo_root / "runs" / run / "test_eval" / "metrics_test.json"
        data = load_metrics(met_path)
        if data is None:
            rows.append(missing_row(run))
        else:
            rows.append(row_from_metrics(run, data))

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = repo_root / args.csv_out if not Path(args.csv_out).is_absolute() else Path(args.csv_out)
    md_path = repo_root / args.md_out if not Path(args.md_out).is_absolute() else Path(args.md_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDNAMES))
        w.writeheader()
        for row in rows:
            out_row = {k: row.get(k, "") for k in FIELDNAMES}
            w.writerow(out_row)

    write_md(md_path, rows, repo_root, csv_path)

    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
