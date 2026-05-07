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

MD_AGGREGATION_PROVENANCE = (
    "> **Aggregation provenance**: BLEU, chrF, chrF++, COMET, BERTScore, "
    "train_time_seconds, and peak_gpu_memory_mib columns report mean ± std (n=3) "
    "for `is_aggregate=true` rows (computed from `source_run_names`). "
    "For `is_aggregate=false` rows, values are single-seed measurements (n=1) "
    "with no statistical aggregation. The Welch p column is computed against "
    "fast_dot's three-seed BLEU values per row's source seeds where applicable.\n"
    "> \n"
    "> Sources: BLEU/chrF/chrF++/COMET/BERTScore from "
    "`runs/<source_run>/test_eval/metrics_test.json`; "
    "train_time_seconds and peak_gpu_memory_mib from "
    "`runs/<source_run>/training_meta.json`. "
    "Note: `results/ablation_per_seed.csv` does not currently fill "
    "`wall_time_seconds`; the authoritative per-seed values are in "
    "`training_meta.json`."
)

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
    "var_local_window_a23_retrain",
)

# CSV columns (machine-oriented; BLEU/chrF/chrF++/COMET duplicate means where applicable).
CSV_FIELDNAMES: tuple[str, ...] = (
    "run",
    "experiment",
    "is_aggregate",
    "source_run_names",
    "representative_run",
    "metrics_source",
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
    "train_time_seconds_mean",
    "train_time_seconds_std",
    "peak_gpu_memory_mib_mean",
    "peak_gpu_memory_mib_std",
    "BERTScore_mean",
    "BERTScore_std",
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


def fmt_ms_bert(mean: float, std: float, n: int) -> str:
    return f"{mean:.5f} ± {std:.5f} (n={n})"


def _mean_sample_stdev(values: list[float]) -> tuple[float | None, float | None]:
    """Sample std (ddof=1); n<2 → std 0.0."""
    if not values:
        return None, None
    m = statistics.mean(values)
    if len(values) < 2:
        return m, 0.0
    return m, statistics.stdev(values)


def load_per_seed_wall_peak_bert(repo_root: Path, source_run_names: list[str]) -> tuple[list[float], list[float], list[float]]:
    """Authoritative wall/GPU from training_meta.json; BERTScore from each seed's test_eval/metrics_test.json."""
    walls: list[float] = []
    peaks: list[float] = []
    berts: list[float] = []
    for rn in source_run_names:
        meta = load_json(repo_root / "runs" / rn / "training_meta.json")
        if meta and meta.get("metadata_repaired") is not True:
            w = meta.get("wall_time_seconds")
            p = meta.get("peak_gpu_memory_mib")
            if w is not None:
                walls.append(float(w))
            if p is not None:
                peaks.append(float(p))
        m = load_json(repo_root / "runs" / rn / "test_eval" / "metrics_test.json")
        if m and m.get("BERTScore") is not None:
            berts.append(float(m["BERTScore"]))
    return walls, peaks, berts


def apply_aggregate_engineering_fields(
    row: dict[str, Any],
    repo_root: Path,
    *,
    experiment: str,
    n: int,
) -> None:
    """Multi-seed mean±std for train wall, peak GPU, BERTScore; legacy columns = rounded mean; MD strings set."""
    source_runs = _multiseed_source_run_names(experiment).split(";")
    walls, peaks, berts = load_per_seed_wall_peak_bert(repo_root, source_runs)

    wm, ws = _mean_sample_stdev(walls)
    pm, ps = _mean_sample_stdev(peaks)
    bm, bs = _mean_sample_stdev(berts)

    if wm is not None:
        row["train_time_seconds_mean"] = wm
        row["train_time_seconds_std"] = ws if ws is not None else ""
        row["train_time_seconds"] = fmt_ms(wm, ws or 0.0, n)
    else:
        row["train_time_seconds_mean"] = ""
        row["train_time_seconds_std"] = ""
        row["train_time_seconds"] = ""

    if pm is not None:
        row["peak_gpu_memory_mib_mean"] = pm
        row["peak_gpu_memory_mib_std"] = ps if ps is not None else ""
        row["peak_gpu_memory_mib"] = fmt_ms(pm, ps or 0.0, n)
    else:
        row["peak_gpu_memory_mib_mean"] = ""
        row["peak_gpu_memory_mib_std"] = ""
        row["peak_gpu_memory_mib"] = ""

    if bm is not None:
        row["BERTScore_mean"] = bm
        row["BERTScore_std"] = bs if bs is not None else ""
        row["BERTScore"] = fmt_ms_bert(bm, bs or 0.0, n)
    else:
        row["BERTScore_mean"] = ""
        row["BERTScore_std"] = ""
        row["BERTScore"] = ""


def apply_single_seed_engineering_fields(row: dict[str, Any], *, failed: bool = False) -> None:
    """Mirror engineering metrics into *_mean; *_std empty; MD (n=1) when not failed."""
    for col, decimals in (("train_time_seconds", 2), ("peak_gpu_memory_mib", 2), ("BERTScore", 5)):
        v = row.get(col)
        key_m = f"{col}_mean"
        key_s = f"{col}_std"
        if failed or not isinstance(v, (int, float)) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            row[key_m] = ""
            row[key_s] = ""
            continue
        fv = float(v)
        row[key_m] = fv
        row[key_s] = ""
        if col == "BERTScore":
            row[col] = f"{fv:.{decimals}f} (n=1)"
        else:
            row[col] = f"{fv:.{decimals}f} (n=1)"


def clear_engineering_aggregate_fields(row: dict[str, Any]) -> None:
    for k in (
        "train_time_seconds_mean",
        "train_time_seconds_std",
        "peak_gpu_memory_mib_mean",
        "peak_gpu_memory_mib_std",
        "BERTScore_mean",
        "BERTScore_std",
    ):
        row[k] = ""


def fmt_welch_md(p: float | None) -> str:
    if p is None:
        return ""
    s = f"{p:.4f}".rstrip("0").rstrip(".")
    return f"p={s}"


def fmt_ms_comet(mean: float, std: float, n: int) -> str:
    return f"{mean:.4f} ± {std:.4f} (n={n})"


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
    repo_root: Path,
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
        apply_aggregate_engineering_fields(row, repo_root, experiment=exp, n=n)
        _apply_aggregate_provenance(row, exp)
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
        clear_engineering_aggregate_fields(row)
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

    if is_failed:
        apply_single_seed_engineering_fields(row, failed=True)
    else:
        apply_single_seed_engineering_fields(row, failed=False)

    caveat = "(single seed; not statistically tested vs fast_dot)"
    single_variants = {
        "var_sparsemax",
        "var_local_window",
        "var_global_local",
        "var_local_window_a23_retrain",
        "head_1h_dot",
        "swap_fr_dot",
    }
    if run in single_variants:
        row["notes"] = f"{caveat} {base_notes}".strip()

    if run.startswith("var_") and run not in RUN_TO_EXPERIMENT and run not in (
        "var_sparsemax",
        "var_local_window",
        "var_global_local",
        "var_local_window_a23_retrain",
    ):
        row["notes"] = f"{caveat} {base_notes}".strip()

    _apply_single_seed_provenance(row, run)


def infer_pkg(run: str) -> str:
    if run.startswith("head") or run == "head_1h_dot":
        return "small_head"
    if run.startswith("swap") or run == "swap_fr_dot":
        return "small_swap"
    return "small_try"


def mechanism_family(run: str, _attention_type: str | None) -> str:
    if run in VARIANT_RUN_FAMILY:
        return VARIANT_RUN_FAMILY[run]
    if run == "var_local_window_a23_retrain":
        return VARIANT_RUN_FAMILY.get("var_local_window", "") or "connectivity"
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
            "Companion CSV columns **`experiment`**, **`is_aggregate`**, **`source_run_names`**, "
            "**`representative_run`**, **`metrics_source`** record aggregate vs per-run provenance "
            "for validation (`prepare_code_archive.validate`).\n\n"
        )
    disclaimer += "---\n\n"
    headers = list(MD_FIELDNAMES)
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    head = "| " + " | ".join(headers) + " |"
    lines = [
        "# Attention variants — held-out test summary",
        "",
        MD_AGGREGATION_PROVENANCE,
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


def _csv_round_or_blank(value: Any, ndigits: int) -> Any:
    if value == "" or value is None:
        return ""
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return ""


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
        elif k == "train_time_seconds":
            out[k] = _csv_round_or_blank(row.get("train_time_seconds_mean"), 2)
        elif k == "peak_gpu_memory_mib":
            out[k] = _csv_round_or_blank(row.get("peak_gpu_memory_mib_mean"), 2)
        elif k == "BERTScore":
            out[k] = _csv_round_or_blank(row.get("BERTScore_mean"), 5)
        else:
            out[k] = row.get(k, "")
    return out


def _multiseed_source_run_names(experiment: str) -> str:
    return ";".join(f"{experiment}_s{i}" for i in (1, 2, 3))


def _apply_single_seed_provenance(row: dict[str, Any], run: str) -> None:
    row["experiment"] = run
    row["is_aggregate"] = "false"
    row["source_run_names"] = run
    row["representative_run"] = run
    row["metrics_source"] = f"runs/{run}/test_eval/metrics_test.json"


def _apply_aggregate_provenance(row: dict[str, Any], experiment: str) -> None:
    row["experiment"] = experiment
    row["is_aggregate"] = "true"
    row["source_run_names"] = _multiseed_source_run_names(experiment)
    row["representative_run"] = f"{experiment}_s1"
    row["metrics_source"] = "results/ablation_per_seed.csv"


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
        "train_time_seconds_mean",
        "train_time_seconds_std",
        "peak_gpu_memory_mib_mean",
        "peak_gpu_memory_mib_std",
        "BERTScore_mean",
        "BERTScore_std",
    ):
        row[k] = ""
    _apply_single_seed_provenance(row, str(row.get("run") or ""))


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
            annotate_row_for_export(row, run, data, ms, repo_root)
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
