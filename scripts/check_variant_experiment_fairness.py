#!/usr/bin/env python3
"""
只读审计：对比多个 runs 的 resolved_config / training_meta / test_eval，
判定与参考 run（默认同列表中的 fast_dot）在公平性关键字段上是否可比。
不写任何训练产物。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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

# Full archived wall: A17 (32) + A19 `fast_add_lr3e3_s{1,2,3}` + A23 `var_local_window_a23_retrain` → **36** rows.
ARCHIVE_FAIRNESS_RUNS: tuple[str, ...] = (
    "fast_dot",
    "fast_add",
    "head_1h_dot",
    "swap_fr_dot",
    "fast_dot_s1",
    "fast_dot_s2",
    "fast_dot_s3",
    "fast_add_s1",
    "fast_add_s2",
    "fast_add_s3",
    "fast_add_lr3e3_s1",
    "fast_add_lr3e3_s2",
    "fast_add_lr3e3_s3",
    "var_bilinear",
    "var_gated_dot_additive",
    "var_sparsemax",
    "var_entmax15",
    "var_local_window",
    "var_global_local",
    "var_bilinear_s1",
    "var_bilinear_s2",
    "var_bilinear_s3",
    "var_gated_dot_additive_s1",
    "var_gated_dot_additive_s2",
    "var_gated_dot_additive_s3",
    "var_entmax15_s1",
    "var_entmax15_s2",
    "var_entmax15_s3",
    "var_local_window_fp32",
    "var_local_window_lr1e4",
    "var_local_window_window16",
    "var_local_window_a23_retrain",
    "lr_sweep_add_lr1e4",
    "lr_sweep_add_lr3e4",
    "lr_sweep_add_lr1e3",
    "lr_sweep_add_lr3e3",
)

REFERENCE_RUN_NAME = "fast_dot"

# Mismatch on these vs reference → FAIL (attention_type 除外).
FAIL_IF_MISMATCH_KEYS: tuple[str, ...] = (
    "train_path",
    "val_path",
    "test_path",
    "tokenizer_src",
    "tokenizer_tgt",
    "d_model",
    "d_ff",
    "n_layers",
    "batch_size",
    "max_steps",
    "learning_rate",
    "warmup_steps",
    "seed",
    "eval_split",
    "max_gen_len",
)

# n_heads 有意可变（如 head 消融）；记入 WARN / comparable 判定。
N_HEADS_KEY = "n_heads"

METRICS_TEST_PROTOCOL_KEYS: tuple[str, ...] = ("max_new_tokens", "batch_size")

PARAM_DELTA_WARN_FRAC = 0.01


def norm_path(s: str | None, repo_root: Path) -> str | None:
    if s is None or s == "":
        return None
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def extract_cfg(resolved_doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not resolved_doc:
        return None
    inner = resolved_doc.get("resolved_config")
    if isinstance(inner, dict):
        return inner
    return None


def infer_pkg(run_name: str, metrics_test: dict[str, Any] | None) -> str:
    if metrics_test and metrics_test.get("pkg"):
        return str(metrics_test["pkg"])
    if run_name.startswith("swap") or "_swap" in run_name:
        return "small_swap"
    if "head" in run_name.lower():
        return "small_head"
    return "small_try"


def count_parameters_build(cfg_dict: dict[str, Any], pkg: str, repo_root: Path) -> int | None:
    pkg_root = repo_root / pkg
    if not pkg_root.is_dir():
        return None
    insert = str(pkg_root)
    old_path = sys.path[:]
    try:
        if insert not in sys.path:
            sys.path.insert(0, insert)
        from model import Seq2SeqTransformer  # type: ignore

        raw = {k: v for k, v in cfg_dict.items() if not str(k).startswith("_")}
        cfg = SimpleNamespace(**raw)
        pad_idx = 0
        model = Seq2SeqTransformer(cfg, pad_idx=pad_idx)
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None
    finally:
        sys.path[:] = old_path


def collect_parameter_count(
    cfg_dict: dict[str, Any] | None,
    training_meta: dict[str, Any] | None,
    pkg: str,
    repo_root: Path,
) -> tuple[int | None, str]:
    if training_meta and training_meta.get("num_parameters") is not None:
        try:
            return int(training_meta["num_parameters"]), "training_meta.json"
        except (TypeError, ValueError):
            pass
    if cfg_dict:
        n = count_parameters_build(cfg_dict, pkg, repo_root)
        if n is not None:
            return n, "model_from_resolved_config"
    return None, "unavailable"


def compare_val(ref_v: Any, run_v: Any) -> bool:
    if ref_v is None and run_v is None:
        return True
    if ref_v is None or run_v is None:
        return False
    if isinstance(ref_v, bool):
        return bool(run_v) == ref_v
    if isinstance(ref_v, (int, float)) and isinstance(run_v, (int, float)):
        if isinstance(ref_v, float) or isinstance(run_v, float):
            return math.isclose(float(ref_v), float(run_v), rel_tol=0.0, abs_tol=1e-9)
        return int(ref_v) == int(run_v)
    return str(ref_v) == str(run_v)


def audit_one_run(
    run_name: str,
    run_dir: Path,
    repo_root: Path,
    ref_cfg: dict[str, Any],
    ref_metrics_test: dict[str, Any] | None,
    ref_run_name: str,
    training_meta: dict[str, Any] | None,
    resolved_doc: dict[str, Any] | None,
    metrics_test: dict[str, Any] | None,
    cfg_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    missing_files: list[str] = []
    if resolved_doc is None or cfg_dict is None:
        missing_files.append("resolved_config.json")
    if metrics_test is None:
        missing_files.append("test_eval/metrics_test.json")
    if training_meta is None:
        missing_files.append("training_meta.json (optional)")

    mismatched_fields: list[str] = []
    pkg = infer_pkg(run_name, metrics_test)

    # --- FAIL-level config vs ref ---
    if cfg_dict is not None:
        for key in FAIL_IF_MISMATCH_KEYS:
            rv = cfg_dict.get(key)
            rr = ref_cfg.get(key)
            if key.endswith("_path") or "tokenizer" in key:
                nv = norm_path(str(rv) if rv is not None else None, repo_root)
                nr = norm_path(str(rr) if rr is not None else None, repo_root)
                if nv != nr:
                    mismatched_fields.append(key)
            else:
                if not compare_val(rr, rv):
                    mismatched_fields.append(key)

    # --- test_eval protocol ---
    if metrics_test is not None and ref_metrics_test is not None:
        for key in METRICS_TEST_PROTOCOL_KEYS:
            if not compare_val(ref_metrics_test.get(key), metrics_test.get(key)):
                mismatched_fields.append(f"metrics_test.{key}")

    # --- n_heads (WARN tier) ---
    n_heads_note = ""
    if cfg_dict is not None and ref_cfg is not None:
        if not compare_val(ref_cfg.get(N_HEADS_KEY), cfg_dict.get(N_HEADS_KEY)):
            mismatched_fields.append(f"{N_HEADS_KEY} (!= ref; head-count ablation risk)")
            n_heads_note = "n_heads differs from reference"

    # --- run directory naming consistency ---
    notes_parts: list[str] = []
    if cfg_dict is not None:
        wn = cfg_dict.get("wandb_run_name")
        if wn is not None and str(wn) != run_name:
            notes_parts.append(f"wandb_run_name={wn!r} != directory {run_name!r} (possible mis-tagging)")
    if n_heads_note:
        notes_parts.append(n_heads_note)

    num_params, param_src = collect_parameter_count(cfg_dict, training_meta, pkg, repo_root)

    row_core = {
        "run": run_name,
        "attention_type": (cfg_dict or {}).get("attention_type", ""),
        "missing_files": missing_files,
        "mismatched_fields": mismatched_fields,
        "parameter_count": num_params,
        "parameter_count_source": param_src,
        "notes": "; ".join(notes_parts),
        "pkg_resolved": pkg,
        "training_meta_repaired": bool(
            training_meta and training_meta.get("metadata_repaired") is True
        ),
    }
    return row_core


def classify_row(
    row: dict[str, Any],
    ref_params: int | None,
    ref_run_name: str,
) -> tuple[str, bool, float | None, list[str]]:
    """Returns status, comparable_to_fast_dot, parameter_delta_percent, flags."""
    flags: list[str] = []
    missing = row["missing_files"]
    strict_missing = [m for m in missing if not m.startswith("training_meta")]

    fail_reasons: list[str] = []
    if any("resolved_config" in m or "resolved_config.json" in m for m in strict_missing):
        fail_reasons.append("missing resolved_config")
    if any("metrics_test" in m for m in strict_missing):
        fail_reasons.append("missing test_eval")

    # FAIL mismatches: exclude n_heads-only annotators (WARN tier).
    fatal_mismatch = [
        f for f in row["mismatched_fields"] if not f.startswith(N_HEADS_KEY)
    ]

    if fatal_mismatch:
        fail_reasons.append("field mismatch vs reference")

    has_fail = bool(fail_reasons)

    param_delta_pct: float | None = None
    pc = row["parameter_count"]
    if ref_params is not None and pc is not None and ref_params > 0:
        param_delta_pct = (float(pc) - float(ref_params)) / float(ref_params) * 100.0

    parameter_changed = (
        param_delta_pct is not None and abs(param_delta_pct) > PARAM_DELTA_WARN_FRAC * 100
    )
    if parameter_changed:
        flags.append("parameter_changed")

    missing_tm = any("training_meta" in m for m in missing)
    if missing_tm:
        flags.append("missing_training_meta")

    repaired_tm = bool(row.get("training_meta_repaired"))
    if repaired_tm:
        flags.append("repaired_training_meta")

    n_heads_mismatch = any(f.startswith(N_HEADS_KEY) for f in row["mismatched_fields"])

    if has_fail:
        status = "FAIL"
    elif parameter_changed or missing_tm or n_heads_mismatch or repaired_tm:
        status = "WARN"
    else:
        status = "PASS"

    comparable = (
        status != "FAIL"
        and not any("metrics_test" in m for m in strict_missing)
        and not fatal_mismatch
        and not n_heads_mismatch
    )

    extra_notes = row.get("notes") or ""
    if flags:
        flag_str = "; ".join(flags)
        row["notes"] = f"{extra_notes}; {flag_str}".strip("; ").strip() if extra_notes else flag_str

    return status, comparable, param_delta_pct, flags


def detect_signature_overlap(rows_core: list[dict[str, Any]], repo_root: Path) -> dict[str, list[str]]:
    """(train_path_norm, seed, attention_type, n_heads, d_model) -> run names."""
    buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in rows_core:
        cfg = row.get("_cfg_dict") or {}
        tp = norm_path(str(cfg.get("train_path", "")), repo_root)
        sig = (
            tp,
            cfg.get("seed"),
            cfg.get("attention_type"),
            cfg.get("n_heads"),
            cfg.get("d_model"),
        )
        buckets[sig].append(row["run"])
    overlap: dict[str, list[str]] = {}
    for sig, names in buckets.items():
        if len(names) > 1:
            for n in names:
                overlap[n] = [x for x in names if x != n]
    return overlap


def main() -> None:
    p = argparse.ArgumentParser(description="Audit cross-run experiment fairness (read-only).")
    p.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        metavar="RUN",
        help="Run directory names under runs-root",
    )
    p.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Relative to repo root unless absolute",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=str(REPO_ROOT),
    )
    p.add_argument("--output-md", type=str, default="results/variant_fairness_audit.md")
    p.add_argument("--output-json", type=str, default="results/variant_fairness_audit.json")
    p.add_argument(
        "--archive-audit",
        action="store_true",
        help="Use full 36-run archived audit list (A17 wall + A19 fast_add_lr3e3_s* + A23 var_local_window_a23_retrain).",
    )
    args = p.parse_args()

    run_list: list[str] = list(ARCHIVE_FAIRNESS_RUNS) if args.archive_audit else list(args.runs)

    repo_root = Path(args.repo_root).resolve()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = repo_root / runs_root

    ref_run_name = REFERENCE_RUN_NAME if REFERENCE_RUN_NAME in run_list else run_list[0]
    ref_dir = runs_root / ref_run_name
    ref_resolved = load_json(ref_dir / "resolved_config.json")
    ref_cfg = extract_cfg(ref_resolved) or {}
    ref_metrics_test = load_json(ref_dir / "test_eval" / "metrics_test.json")
    ref_training_meta = load_json(ref_dir / "training_meta.json")
    ref_params, _ = collect_parameter_count(
        ref_cfg,
        ref_training_meta,
        infer_pkg(ref_run_name, ref_metrics_test),
        repo_root,
    )

    rows_core: list[dict[str, Any]] = []
    for run_name in run_list:
        run_dir = runs_root / run_name
        resolved_doc = load_json(run_dir / "resolved_config.json")
        cfg_dict = extract_cfg(resolved_doc)
        training_meta = load_json(run_dir / "training_meta.json")
        metrics_test = load_json(run_dir / "test_eval" / "metrics_test.json")

        row = audit_one_run(
            run_name,
            run_dir,
            repo_root,
            ref_cfg,
            ref_metrics_test,
            ref_run_name,
            training_meta,
            resolved_doc,
            metrics_test,
            cfg_dict,
        )
        row["_cfg_dict"] = cfg_dict or {}
        rows_core.append(row)

    overlap_map = detect_signature_overlap(rows_core, repo_root)

    final_rows: list[dict[str, Any]] = []
    any_fail = False
    for row in rows_core:
        row.pop("_cfg_dict", {})
        if run_name := row["run"]:
            others = overlap_map.get(run_name)
            if others:
                prev = row.get("notes") or ""
                msg = f"signature overlap risk vs runs: {others}"
                row["notes"] = f"{prev}; {msg}".strip("; ").strip() if prev else msg

        status, comparable, delta_pct, flags = classify_row(row, ref_params, ref_run_name)
        if status == "FAIL":
            any_fail = True

        final_rows.append(
            {
                "run": row["run"],
                "attention_type": row["attention_type"],
                "status": status,
                "comparable_to_fast_dot": comparable,
                "parameter_count": row["parameter_count"],
                "parameter_delta_percent": delta_pct,
                "missing_files": row["missing_files"],
                "mismatched_fields": row["mismatched_fields"],
                "notes": row["notes"],
                "parameter_count_source": row["parameter_count_source"],
                "flags": flags,
                "training_meta_repaired": row.get("training_meta_repaired", False),
            }
        )

    out_md = repo_root / args.output_md if not Path(args.output_md).is_absolute() else Path(args.output_md)
    out_json = (
        repo_root / args.output_json
        if not Path(args.output_json).is_absolute()
        else Path(args.output_json)
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "reference_run": ref_run_name,
        "repo_root": str(repo_root),
        "runs_root": str(runs_root),
        "reference_num_parameters": ref_params,
        "parameter_warn_threshold_percent": PARAM_DELTA_WARN_FRAC * 100,
        "fail_if_mismatch_keys": list(FAIL_IF_MISMATCH_KEYS),
        "runs": final_rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Variant experiment fairness audit",
        "",
        f"Reference run: **`{ref_run_name}`** (parameter deltas vs this run when counts available).",
        "",
        "## Rules",
        "",
        "- **PASS**: key training/eval fields match reference (except `attention_type`), "
        "`test_eval/metrics_test.json` present.",
        "- **WARN**: no FAIL, but `n_heads` differs, or parameter count Δ vs reference > "
        f"{PARAM_DELTA_WARN_FRAC * 100:.0f}%, or missing `training_meta.json`, "
        "or **`training_meta.json` was repaired** (`metadata_repaired=true`; "
        "`num_parameters` only is trustworthy).",
        "- **FAIL**: missing `resolved_config.json` or `test_eval/metrics_test.json`, "
        "or mismatch on splits/tokenizers/backbone dims/training budget/seed/eval_split/"
        "`max_gen_len`, or `metrics_test` decoding batch/max_new_tokens mismatch.",
        "",
        "`parameter_changed` is an **interpretability risk**, not an automatic invalidation.",
        "",
        "## Summary table",
        "",
        "| run | attention_type | status | comparable_to_fast_dot | parameter_count | "
        "parameter_delta_percent | training_meta | missing_files | mismatched_fields | notes |",
        "|-----|----------------|--------|-------------------------|-----------------|---------------------------|---------------|---------------|---------------------|-------|",
    ]
    for r in final_rows:
        mf = "; ".join(r["mismatched_fields"]) if r["mismatched_fields"] else ""
        mis = "; ".join(r["missing_files"]) if r["missing_files"] else ""
        dp = "" if r["parameter_delta_percent"] is None else f"{r['parameter_delta_percent']:.4f}"
        pc = "" if r["parameter_count"] is None else str(r["parameter_count"])
        notes = (r.get("notes") or "").replace("|", "\\|")
        tm_note = "repaired" if r.get("training_meta_repaired") else "original"
        lines.append(
            f"| {r['run']} | {r['attention_type']} | {r['status']} | {r['comparable_to_fast_dot']} | "
            f"{pc} | {dp} | {tm_note} | {mis} | {mf} | {notes} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_json}", file=sys.stderr)
    print(f"Wrote {out_md}", file=sys.stderr)
    if any_fail:
        print("[audit] FAIL present → exit 1", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
