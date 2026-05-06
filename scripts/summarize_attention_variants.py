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
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RUNS: tuple[str, ...] = (
    "fast_dot",
    "fast_add",
    "fast_add_lr3e3",
    "head_1h_dot",
    "swap_fr_dot",
    "var_bilinear",
    "var_gated_dot_additive",
    "var_sparsemax",
    "var_entmax15",
    "var_local_window",
    "var_global_local",
)

# CSV columns (machine-oriented; BLEU/chrF/chrF++/COMET duplicate means where applicable).
CSV_FIELDNAMES: tuple[str, ...] = (
    "run",
    "pkg",
    "attention_type",
    "mechanism_family",
    "BLEU",
    "chrF",
    "chrF++",
    "COMET",
    "n_seeds",
    "Welch_p_vs_fast_dot_BLEU",
    "BERTScore",
    "number_of_test_examples",
    "parameter_count",
    "train_time_seconds",
    "peak_gpu_memory_mib",
    "status",
    "notes",
    "BLEU_mean",
    "BLEU_std",
    "BLEU_n_seeds",
    "chrF_mean",
    "chrF_std",
    "chrFpp_mean",
    "chrFpp_std",
    "COMET_mean",
    "COMET_std",
)

# Markdown table (Welch column title matches thesis-facing wording).
MD_COL_WELCH = "Welch p vs fast_dot (BLEU)"
MD_FIELDNAMES: tuple[str, ...] = (
    "run",
    "pkg",
    "attention_type",
    "mechanism_family",
    "BLEU",
    "chrF",
    "chrF++",
    "COMET",
    "n_seeds",
    MD_COL_WELCH,
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

# experiment column in results/ablation_per_seed.csv → aggregate 3-seed held-out metrics
RUN_TO_EXPERIMENT: dict[str, str] = {
    "fast_dot": "fast_dot",
    "fast_add": "fast_add",
    "fast_add_lr3e3": "fast_add_lr3e3",
    "var_bilinear": "var_bilinear",
    "var_gated_dot_additive": "var_gated_dot_additive",
    "var_entmax15": "var_entmax15",
}

# Display row has no `runs/<run>/test_eval/` — read representative seed for base fields / legacy blurb.
MULTISEED_ROW_ARTIFACT_RUN: dict[str, str] = {
    "fast_add_lr3e3": "fast_add_lr3e3_s1",
}


def load_multiseed_from_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    by_exp: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = (row.get("experiment") or "").strip()
            if not exp:
                continue
            if (row.get("checkpoint_kind") or "best").strip() != "best":
                continue
            by_exp.setdefault(exp, []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for exp, rows in by_exp.items():
        if len(rows) < 3:
            continue

        def col_f(key: str) -> list[float]:
            return [float(r[key]) for r in rows]

        bleu = col_f("BLEU")
        chrf = col_f("chrF")
        chrfpp = col_f("chrF++")
        comet = col_f("COMET")
        n = len(rows)
        out[exp] = {
            "n": n,
            "BLEU": bleu,
            "chrF": chrf,
            "chrF++": chrfpp,
            "COMET": comet,
            "bleu_mean": statistics.mean(bleu),
            "bleu_std": statistics.stdev(bleu) if n > 1 else 0.0,
            "chrf_mean": statistics.mean(chrf),
            "chrf_std": statistics.stdev(chrf) if n > 1 else 0.0,
            "chrfpp_mean": statistics.mean(chrfpp),
            "chrfpp_std": statistics.stdev(chrfpp) if n > 1 else 0.0,
            "comet_mean": statistics.mean(comet),
            "comet_std": statistics.stdev(comet) if n > 1 else 0.0,
        }
    return out


def welch_p_value_two_sample(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy.stats import ttest_ind

        return float(ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return None


def fmt_ms(mean: float, std: float, n: int) -> str:
    return f"{mean:.2f} ± {std:.2f} (n={n})"


def fmt_ms_comet(mean: float, std: float, n: int) -> str:
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def fmt_welch_md(p: float | None) -> str:
    if p is None:
        return ""
    s = f"{p:.4f}".rstrip("0").rstrip(".")
    return f"p={s}"


def legacy_single_seed_blurb(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    try:
        b = float(data["BLEU"])
        cp = float(data["chrF++"])
        co = float(data["COMET"])
        return f"BLEU={b:.2f}, chrF++={cp:.2f}, COMET={co:.4f}"
    except (TypeError, ValueError, KeyError):
        b = data.get("BLEU")
        cp = data.get("chrF++")
        co = data.get("COMET")
        return f"BLEU={b}, chrF++={cp}, COMET={co}"


def annotate_row_for_export(
    row: dict[str, Any],
    run: str,
    raw_metrics: dict[str, Any] | None,
    ms: dict[str, dict[str, Any]],
) -> None:
    """Set MD strings, n_seeds, Welch column, CSV *_mean/*_std, and notes."""
    exp = RUN_TO_EXPERIMENT.get(run)
    base_notes = (row.get("notes") or "").strip()

    if exp and exp in ms and ms[exp]["n"] >= 3:
        s = ms[exp]
        n = int(s["n"])
        leg = legacy_single_seed_blurb(raw_metrics)
        row["BLEU"] = fmt_ms(s["bleu_mean"], s["bleu_std"], n)
        row["chrF"] = fmt_ms(s["chrf_mean"], s["chrf_std"], n)
        row["chrF++"] = fmt_ms(s["chrfpp_mean"], s["chrfpp_std"], n)
        row["COMET"] = fmt_ms_comet(s["comet_mean"], s["comet_std"], n)
        row["n_seeds"] = n
        ref = ms.get("fast_dot")
        if exp == "fast_dot":
            row[MD_COL_WELCH] = "—"
            row["Welch_p_vs_fast_dot_BLEU"] = ""
        elif ref and len(ref["BLEU"]) >= 3:
            p = welch_p_value_two_sample(ref["BLEU"], s["BLEU"])
            row[MD_COL_WELCH] = fmt_welch_md(p)
            row["Welch_p_vs_fast_dot_BLEU"] = p if p is not None else ""
        else:
            row[MD_COL_WELCH] = ""
            row["Welch_p_vs_fast_dot_BLEU"] = ""

        row["BLEU_mean"] = s["bleu_mean"]
        row["BLEU_std"] = s["bleu_std"]
        row["BLEU_n_seeds"] = n
        row["chrF_mean"] = s["chrf_mean"]
        row["chrF_std"] = s["chrf_std"]
        row["chrFpp_mean"] = s["chrfpp_mean"]
        row["chrFpp_std"] = s["chrfpp_std"]
        row["COMET_mean"] = s["comet_mean"]
        row["COMET_std"] = s["comet_std"]

        leg_path = MULTISEED_ROW_ARTIFACT_RUN.get(run, run)
        if leg_path != run:
            extra = (
                f"3 seeds from `results/ablation_per_seed.csv`; "
                f"representative `runs/{leg_path}/test_eval` metrics (notes blurbs): {leg}."
            )
        else:
            extra = f"3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/{run}`: {leg}."
        row["notes"] = f"{extra} {base_notes}".strip()
        return

    st = str(row.get("status") or "")
    is_failed = st == "failed_nan" or run == "var_local_window"

    snap: dict[str, Any] = {k: row.get(k) for k in ("BLEU", "chrF", "chrF++", "COMET")}

    if is_failed:
        row["n_seeds"] = "failed_nan"
        row[MD_COL_WELCH] = ""
        row["Welch_p_vs_fast_dot_BLEU"] = ""
        for key in ("BLEU", "chrF", "chrF++", "COMET"):
            v = snap[key]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[key] = str(v)
            else:
                row[key] = fmt_cell(v)
        for k_src, k_m in (
            ("BLEU", "BLEU_mean"),
            ("chrF", "chrF_mean"),
            ("chrF++", "chrFpp_mean"),
            ("COMET", "COMET_mean"),
        ):
            v = snap[k_src]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                row[k_m] = float(v)
            else:
                row[k_m] = ""
        for k in ("BLEU_std", "chrF_std", "chrFpp_std", "COMET_std"):
            row[k] = ""
        row["BLEU_n_seeds"] = 0
    else:
        row["n_seeds"] = 1
        row[MD_COL_WELCH] = ""
        row["Welch_p_vs_fast_dot_BLEU"] = ""
        for key in ("BLEU", "chrF", "chrF++", "COMET"):
            v = snap[key]
            if v is None or v == "":
                row[key] = ""
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[key] = str(v)
                continue
            if isinstance(v, (int, float)):
                if key == "COMET":
                    row[key] = f"{float(v):.4f} (n=1)"
                else:
                    row[key] = f"{float(v):.2f} (n=1)"
        for k_src, k_m, k_sd in (
            ("BLEU", "BLEU_mean", "BLEU_std"),
            ("chrF", "chrF_mean", "chrF_std"),
            ("chrF++", "chrFpp_mean", "chrFpp_std"),
            ("COMET", "COMET_mean", "COMET_std"),
        ):
            v = snap[k_src]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                row[k_m] = float(v)
                row[k_sd] = ""
            else:
                row[k_m] = ""
                row[k_sd] = ""
        row["BLEU_n_seeds"] = 1

    caveat = "(single seed; not statistically tested vs fast_dot)"
    single_variants = {
        "var_sparsemax",
        "var_local_window",
        "var_global_local",
        "head_1h_dot",
        "swap_fr_dot",
    }
    if run in single_variants:
        row["notes"] = f"{caveat} {base_notes}".strip()

    if run.startswith("var_") and run not in RUN_TO_EXPERIMENT and run not in (
        "var_sparsemax",
        "var_local_window",
        "var_global_local",
    ):
        row["notes"] = f"{caveat} {base_notes}".strip()


def infer_pkg(run: str) -> str:
    if run.startswith("head") or run == "head_1h_dot":
        return "small_head"
    if run.startswith("swap") or run == "swap_fr_dot":
        return "small_swap"
    return "small_try"


def mechanism_family(run: str, _attention_type: str | None) -> str:
    if run in VARIANT_RUN_FAMILY:
        return VARIANT_RUN_FAMILY[run]
    if run in ("fast_dot", "head_1h_dot", "swap_fr_dot", "fast_add", "fast_add_lr3e3"):
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
    *,
    artifact_run: str | None = None,
) -> dict[str, Any]:
    ar = artifact_run if artifact_run is not None else run
    pkg = infer_pkg(run)
    att = None
    if data:
        a = data.get("attention_type")
        att = str(a) if a is not None else None

    mech = mechanism_family(run, att)

    param_c, train_t, peak_mib = training_meta_columns(repo_root, ar)

    status = sanity_map.get(run) or sanity_map.get(ar) or ""
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
            "notes": f"missing runs/{ar}/test_eval/metrics_test.json; {notes}".strip("; "),
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
    *,
    multiseed_unified: bool,
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
    )
    if multiseed_unified:
        disclaimer += (
            "**Multi-seed rows** (`n=3`): metrics are **mean ± sample std** over held-out `test_eval` scores from "
            "`results/ablation_per_seed.csv` (one value per training seed). **"
            + MD_COL_WELCH
            + "** is a two-sided Welch two-sample *p*-value (variant seeds vs **fast_dot** seeds); "
            "— on the **fast_dot** row means *reference*. "
            "**Single-seed** exploratory rows are annotated in **notes**; full Welch prose lives in "
            "`results/variant_multiseed_summary.md`.\n\n"
        )
    disclaimer += "---\n\n"
    headers = list(MD_FIELDNAMES)
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


def csv_export_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map in-memory row to CSV columns; primary BLEU/chrF/chrF++/COMET columns = means."""
    mean_alias = {
        "BLEU": "BLEU_mean",
        "chrF": "chrF_mean",
        "chrF++": "chrFpp_mean",
        "COMET": "COMET_mean",
    }
    out: dict[str, Any] = {}
    for k in CSV_FIELDNAMES:
        if k in mean_alias:
            out[k] = row.get(mean_alias[k], "")
        else:
            out[k] = row.get(k, "")
    return out


def apply_legacy_export_columns(row: dict[str, Any]) -> None:
    row[MD_COL_WELCH] = ""
    row["Welch_p_vs_fast_dot_BLEU"] = ""
    row["n_seeds"] = ""
    for k in (
        "BLEU_mean",
        "BLEU_std",
        "BLEU_n_seeds",
        "chrF_mean",
        "chrF_std",
        "chrFpp_mean",
        "chrFpp_std",
        "COMET_mean",
        "COMET_std",
    ):
        row[k] = ""


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
    p.add_argument(
        "--multiseed-csv",
        type=str,
        default="results/ablation_per_seed.csv",
        help="If this file exists and contains >=3 seeds for fast_dot, merge multi-seed means±std and Welch p vs fast_dot.",
    )
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sanity_map = load_sanity_status_by_run(repo_root)

    ms_path = repo_root / args.multiseed_csv if not Path(args.multiseed_csv).is_absolute() else Path(args.multiseed_csv)
    ms = load_multiseed_from_csv(ms_path)
    ms_ok = "fast_dot" in ms and int(ms["fast_dot"]["n"]) >= 3

    rows: list[dict[str, Any]] = []
    for run in args.runs:
        artifact_run = MULTISEED_ROW_ARTIFACT_RUN.get(run)
        disk = artifact_run or run
        met_path = repo_root / "runs" / disk / "test_eval" / "metrics_test.json"
        data = load_json(met_path)
        row = build_row(run, data, sanity_map, repo_root, artifact_run=artifact_run)
        if ms_ok:
            annotate_row_for_export(row, run, data, ms)
        else:
            apply_legacy_export_columns(row)
        rows.append(row)

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = repo_root / args.csv_out if not Path(args.csv_out).is_absolute() else Path(args.csv_out)
    md_path = repo_root / args.md_out if not Path(args.md_out).is_absolute() else Path(args.md_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        w.writeheader()
        for row in rows:
            w.writerow(csv_export_row(row))

    write_md(md_path, rows, repo_root, csv_path, multiseed_unified=ms_ok)

    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
