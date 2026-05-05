#!/usr/bin/env python3
"""
只读各 run 的 runs/<run>/test_eval/metrics_test.json，汇总 held-out test 指标。
禁止使用 runs/<run>/metrics.json（训练期 / val）。

Status 来自 results/runs_sanity_report.json；训练侧/aux 字段来自 training_meta.json。
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
    "swap_fr_dot",
    "var_bilinear",
    "var_gated_dot_additive",
    "var_sparsemax",
    "var_entmax15",
    "var_local_window",
    "var_global_local",
)

FIELDNAMES: tuple[str, ...] = (
    "run",
    "pkg",
    "attention_type",
    "mechanism_family",
    "BLEU",
    "chrF",
    "chrF++",
    "COMET",
    "BERTScore",
    "number_of_test_examples",
    "parameter_count",
    "train_time_seconds",
    "peak_gpu_memory_mib",
    "status",
    "notes",
)


def _ensure_experiments_path() -> None:
    exp = REPO_ROOT / "experiments"
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))


def run_to_variant_family() -> dict[str, str]:
    _ensure_experiments_path()
    from variant_registry import VARIANT_REGISTRY  # noqa: E402

    return {m.run_name: m.family for m in VARIANT_REGISTRY.values()}


VARIANT_RUN_FAMILY: dict[str, str] = run_to_variant_family()


def infer_pkg(run: str) -> str:
    if run.startswith("head") or run == "head_1h_dot":
        return "small_head"
    if run.startswith("swap") or run == "swap_fr_dot":
        return "small_swap"
    return "small_try"


def mechanism_family(run: str, _attention_type: str | None) -> str:
    if run in VARIANT_RUN_FAMILY:
        return VARIANT_RUN_FAMILY[run]
    if run in ("fast_dot", "head_1h_dot", "swap_fr_dot", "fast_add"):
        return "scoring"
    return ""


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_sanity_status_by_run(repo_root: Path) -> dict[str, str]:
    path = repo_root / "results" / "runs_sanity_report.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, list):
        return {}
    out: dict[str, str] = {}
    for item in data:
        if isinstance(item, dict) and "run" in item and "status" in item:
            out[str(item["run"])] = str(item["status"])
    return out


def training_meta_columns(repo_root: Path, run: str) -> tuple[Any, Any, Any]:
    """parameter_count, train_time_seconds, peak_gpu_memory_mib — all None if repaired."""
    p = repo_root / "runs" / run / "training_meta.json"
    meta = load_json(p)
    if not meta:
        return None, None, None
    if meta.get("metadata_repaired") is True:
        return None, None, None
    return (
        meta.get("num_parameters"),
        meta.get("wall_time_seconds"),
        meta.get("peak_gpu_memory_mib"),
    )


def build_row(
    run: str,
    data: dict[str, Any] | None,
    sanity_map: dict[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    pkg = infer_pkg(run)
    att = None
    if data:
        a = data.get("attention_type")
        att = str(a) if a is not None else None

    mech = mechanism_family(run, att)

    param_c, train_t, peak_mib = training_meta_columns(repo_root, run)

    status = sanity_map.get(run, "")
    if not status and data is None:
        status = "missing_artifacts"

    notes_parts: list[str] = []
    if data:
        nh = data.get("n_heads")
        if nh is not None:
            notes_parts.append(f"n_heads={nh}")

    if run == "var_local_window":
        notes_parts.append(
            "val_loss=NaN; outputs empty; numerical-stability failure, not mechanism comparison."
        )

    if run.startswith("var_"):
        notes_parts.append(
            "single seed; ranking below |Δ| = 1 BLEU is unreliable due to seed variance"
        )

    notes = "; ".join(notes_parts)

    if data is None:
        return {
            "run": run,
            "pkg": pkg,
            "attention_type": att or "",
            "mechanism_family": mech,
            "BLEU": "",
            "chrF": "",
            "chrF++": "",
            "COMET": "",
            "BERTScore": "",
            "number_of_test_examples": "",
            "parameter_count": param_c if param_c is not None else "",
            "train_time_seconds": train_t if train_t is not None else "",
            "peak_gpu_memory_mib": peak_mib if peak_mib is not None else "",
            "status": status or "missing_artifacts",
            "notes": f"missing runs/{run}/test_eval/metrics_test.json; {notes}".strip("; "),
        }

    return {
        "run": run,
        "pkg": pkg,
        "attention_type": att or "",
        "mechanism_family": mech,
        "BLEU": data.get("BLEU"),
        "chrF": data.get("chrF"),
        "chrF++": data.get("chrF++"),
        "COMET": data.get("COMET"),
        "BERTScore": data.get("BERTScore"),
        "number_of_test_examples": data.get("number_of_test_examples"),
        "parameter_count": param_c if param_c is not None else "",
        "train_time_seconds": train_t if train_t is not None else "",
        "peak_gpu_memory_mib": peak_mib if peak_mib is not None else "",
        "status": status,
        "notes": notes,
    }


# Not our runs — appended after every MD regen; do not add as table rows.
MD_EXTERNAL_CONTEXT = (
    "### Context (external baselines — not our runs)\n\n"
    "For **scale context only** (do **not** rank this table against these numbers): "
    "Vaswani et al. 2017 report **38.1 / 41.8 BLEU** (EN–FR base/big, ~4.5M pairs, "
    "beam 4, length penalty 0.6); a typical IWSLT'17-style EN–FR tutorial is often "
    "**30+ BLEU** on ~225k pairs with beam. This project uses **~50k pairs**, **greedy** "
    "decoding, short training (**3000 steps**), and `max_seq_len=96`, so **~15.8 BLEU** "
    "(multi-seed mean on `fast_dot`) is in a plausible range versus those references—not "
    "evidence of a broken stack. **Do not** claim competitive parity with Vaswani 2017 on "
    "BLEU alone.\n\n"
    "---\n"
)


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
        "**status** comes from `results/runs_sanity_report.json` (re-run `python scripts/sanity_check_runs.py` "
        "to refresh).\n\n"
        "Training-time fields (`parameter_count`, `train_time_seconds`, `peak_gpu_memory_mib`) are read from "
        "`training_meta.json` and are **blank when `metadata_repaired=true`**.\n\n"
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
        cells = [fmt_cell(_) for _ in (r.get(h) for h in headers)]
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
            *MD_EXTERNAL_CONTEXT.strip().splitlines(),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    sanity_map = load_sanity_status_by_run(repo_root)

    rows: list[dict[str, Any]] = []
    for run in args.runs:
        met_path = repo_root / "runs" / run / "test_eval" / "metrics_test.json"
        data = load_json(met_path)
        rows.append(build_row(run, data, sanity_map, repo_root))

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
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})

    write_md(md_path, rows, repo_root, csv_path)

    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
