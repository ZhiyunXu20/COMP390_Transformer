#!/usr/bin/env python3
"""
A14 — Top attention variants × 3 seeds (review_v2 S-4).

Trains `small_try` with seeds 1,2,3 for bilinear / gated_dot_additive / (optional) entmax15.
New run dirs only: `runs/var_bilinear_s1` … (does not touch legacy `runs/var_bilinear`).

After training: `evaluate_test.py` → `runs/<name>/test_eval/metrics_test.json` (full metrics, no --eval-light).
Updates `results/ablation_per_seed.csv` (refreshes rows for these experiment labels),
writes `results/variant_multiseed_summary.md`, patches `docs/ATTENTION_VARIANTS_STATUS.md`
when all three seeds exist for a variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cross_seed_significance import (  # noqa: E402
    analyze_metric,
    collect_metric_values,
    load_csv_rows,
    stats_backend_name,
)
from cross_seed_significance import _HAS_SCIPY as HAS_SCIPY  # noqa: E402

DEFAULT_SPLITS = "data/splits/en_fr_50k_seed42"
CHRPP_KEY = "chrF++"
FAST_DOT_REF = "fast_dot"
SEEDS = (1, 2, 3)

# (experiment label, attention_type, n_heads)
VARIANT_SPECS: list[tuple[str, str, int]] = [
    ("var_bilinear", "bilinear", 4),
    ("var_gated_dot_additive", "gated_dot_additive", 4),
    ("var_entmax15", "entmax15", 4),
]

LEGACY_SINGLE_SEED_RUN: dict[str, str] = {
    "bilinear": "var_bilinear",
    "gated_dot_additive": "var_gated_dot_additive",
    "entmax15": "var_entmax15",
}

ROW_META: dict[str, dict[str, str]] = {
    "bilinear": {
        "mechanism": r"Head-specific bilinear \(q,W_h,k\) / **softmax** / **full**",
        "purpose": "Tests a low-rank bilinear interaction instead of dot-product per head.",
        "safe": (
            "A bilinear compatibility score per head is **implemented**; **3-seed** evidence vs "
            "**fast_dot** is summarized in **`results/variant_multiseed_summary.md`** "
            "(runs `var_bilinear_s{1..3}`; single-seed `runs/var_bilinear` remains a non-binding probe)."
        ),
    },
    "gated_dot_additive": {
        "mechanism": "Convex mix of dot and additive logits (learned gate) / **softmax** / **full**",
        "purpose": "Lets the model interpolate between dot-product and additive shapes without committing to one.",
        "safe": (
            "The gated dot–additive module is an **engineering option** for ablation; use "
            "**`results/variant_multiseed_summary.md`** for **3-seed** `test_eval` means and Welch vs **fast_dot**."
        ),
    },
    "entmax15": {
        "mechanism": (
            "Dot-product logits / **entmax** with α = 1.5 (bisect; falls back if deps missing) / **full**"
        ),
        "purpose": "Intermediate sparsity between softmax and sparsemax.",
        "safe": (
            "Alternative **normalizer** on a **dense** score matrix; **not** a kernel-efficient sparse "
            "mechanism. **3-seed** summaries vs **fast_dot**: **`results/variant_multiseed_summary.md`**."
        ),
    },
}

PER_SEED_FIELDS = [
    "run_name",
    "experiment",
    "attention_type",
    "n_heads",
    "seed",
    "checkpoint_kind",
    "checkpoint_path",
    "BLEU",
    "chrF",
    CHRPP_KEY,
    "COMET",
    "wall_time_seconds",
    "peak_gpu_memory_mib",
    "num_parameters",
]


def rel_under_repo(path: Path, root: Path = REPO_ROOT) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def flat_run_name(experiment: str, seed: int) -> str:
    return f"{experiment}_s{seed}"


def load_training_meta(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "training_meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        return {}


def load_metrics_test(eval_dir: Path) -> dict[str, Any]:
    p = eval_dir / "metrics_test.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        return {}


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print(f"+ cd {cwd} && {shlex.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def build_per_seed_row(
    *,
    experiment: str,
    attention_type: str,
    n_heads: int,
    seed: int,
    repo: Path,
) -> dict[str, Any]:
    run_name = flat_run_name(experiment, seed)
    run_dir = repo / "runs" / run_name
    ev_out = run_dir / "test_eval"
    best_pt = run_dir / "best.pt"
    tm = load_training_meta(run_dir)
    m = load_metrics_test(ev_out)
    bleu = m.get("BLEU")
    chrf = m.get("chrF")
    chrfpp = m.get(CHRPP_KEY)
    comet = m.get("COMET")
    return {
        "run_name": run_name,
        "experiment": experiment,
        "attention_type": attention_type,
        "n_heads": n_heads,
        "seed": seed,
        "checkpoint_kind": "best",
        "checkpoint_path": str(best_pt.resolve()) if best_pt.is_file() else "",
        "BLEU": bleu,
        "chrF": chrf,
        CHRPP_KEY: chrfpp,
        "COMET": comet,
        "wall_time_seconds": tm.get("wall_time_seconds"),
        "peak_gpu_memory_mib": tm.get("peak_gpu_memory_mib"),
        "num_parameters": tm.get("num_parameters"),
    }


def merge_ablation_csv(
    csv_path: Path,
    variant_rows: list[dict[str, Any]],
) -> None:
    if not csv_path.is_file():
        raise SystemExit(f"missing {csv_path}")
    rows = load_csv_rows(csv_path)
    fresh = [
        r
        for r in variant_rows
        if r.get("BLEU") is not None and str(r.get("BLEU")).strip() != ""
    ]
    exps_touched = {r["experiment"] for r in fresh}
    kept = [r for r in rows if r.get("experiment") not in exps_touched]
    out_rows = kept + fresh
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_SEED_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in PER_SEED_FIELDS})


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def collect_vals(rows: list[dict], experiment: str, metric: str) -> list[float]:
    return collect_metric_values(rows, experiment, metric)


def fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{float(x):.{nd}f}"


def write_variant_multiseed_summary(
    repo: Path,
    csv_path: Path,
    out_md: Path,
    specs: list[tuple[str, str, int]],
) -> None:
    rows = load_csv_rows(csv_path)
    if not any(collect_vals(rows, exp, "BLEU") for exp, _, _ in specs):
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(
            "# Variant multi-seed summary (A14)\n\n"
            "_Pending:_ no `experiment=var_bilinear` / `var_gated_dot_additive` / `var_entmax15` "
            "rows in [`results/ablation_per_seed.csv`](ablation_per_seed.csv) yet. Run:\n\n"
            "```bash\npython experiments/run_variant_multiseed.py\n```\n",
            encoding="utf-8",
        )
        print(f"Wrote stub {out_md} (no variant metrics in CSV yet)", flush=True)
        return

    lines = [
        "# Variant multi-seed summary (A14)",
        "",
        "## Methodology",
        "",
        "- **3 seeds** each (`1`, `2`, `3`) under the same protocol as **`fast_dot` / `fast_add`** "
        "in `small_try`: `max_steps=3000`, default `Config`, **`lr=3e-4`**, bf16 autocast, "
        "no extra determinism flag.",
        "- Training: **`--no-wandb`**, **`--eval-light`**, **`--no-attention-plots`** on validation BLEU subsample.",
        "- Held-out test: **`evaluate_test.py`** without `--eval-light` (BLEU + chrF++ + BERTScore + COMET).",
        "- Reference cross-seed stats for **`fast_dot`**: from `results/ablation_per_seed.csv` "
        f"(experiment=`{FAST_DOT_REF}`, `checkpoint_kind=best`).",
        "",
        "## Results",
        "",
        "| variant | mean BLEU | std BLEU | mean chrF++ | std chrF++ | mean COMET | std COMET |",
        "|---------|-----------|----------|-------------|------------|------------|-----------|",
    ]

    fd_b = collect_vals(rows, FAST_DOT_REF, "BLEU")
    fd_cp = collect_vals(rows, FAST_DOT_REF, CHRPP_KEY)
    fd_co = collect_vals(rows, FAST_DOT_REF, "COMET")
    mb, sb = mean_std(fd_b)
    mcp, scp = mean_std(fd_cp)
    mco, sco = mean_std(fd_co)
    lines.append(
        f"| fast_dot (reference) | {fmt(mb)} | {fmt(sb)} | {fmt(mcp)} | {fmt(scp)} | {fmt(mco)} | {fmt(sco)} |"
    )

    for exp, _, _ in specs:
        label = exp.replace("var_", "").replace("_", " ")
        b = collect_vals(rows, exp, "BLEU")
        cp = collect_vals(rows, exp, CHRPP_KEY)
        co = collect_vals(rows, exp, "COMET")
        mb2, sb2 = mean_std([float(x) for x in b])
        mcp2, scp2 = mean_std([float(x) for x in cp])
        co_floats = [float(x) for x in co if x is not None and str(x).strip() != ""]
        mco2, sco2 = mean_std(co_floats)
        lines.append(
            f"| {label} | {fmt(mb2)} | {fmt(sb2)} | {fmt(mcp2)} | {fmt(scp2)} | {fmt(mco2)} | {fmt(sco2)} |"
        )

    lines.extend(
        [
            "",
            "## Cross-seed Welch t-test vs fast_dot",
            "",
            "Independent samples: one **held-out test** metric per training seed (same layout as "
            "`scripts/cross_seed_significance.py`). **Δ** = mean(variant) − mean(fast_dot). "
            f"Stats backend: **{stats_backend_name()}** (SciPy present: {HAS_SCIPY}).",
            "",
        ]
    )

    for exp, _, _ in specs:
        lines.append(f"### {exp} vs {FAST_DOT_REF}")
        lines.append("")
        for metric in ("BLEU", CHRPP_KEY, "COMET"):
            va = collect_vals(rows, exp, metric)
            vb = collect_vals(rows, FAST_DOT_REF, metric)
            if len(va) < 3 or len(vb) < 3:
                lines.append(
                    f"- **{metric}**: need 3 seeds (variant n={len(va)}, fast_dot n={len(vb)})."
                )
                lines.append("")
                continue
            r = analyze_metric(metric, va, vb, exp, FAST_DOT_REF)
            lo, hi = r["ci_95_low"], r["ci_95_high"]
            ci_s = (
                f"[{fmt(lo)}, {fmt(hi)}]"
                if lo is not None and hi is not None
                else "n/a"
            )
            pv = r["p_value_two_sided"]
            pv_s = fmt(pv, nd=6) if pv is not None else "n/a"
            lines.append(
                f"- **{metric}**: Δ={fmt(r['mean_difference_a_minus_b'])} "
                f"t={fmt(r['welch_t'])} df≈{fmt(r['welch_df'])} p={pv_s} "
                f"95% CI Δ {ci_s} — {r['conclusion']}"
            )
        lines.append("")

    scipy_note = (
        "Two-sided p-values and 95% CIs: SciPy `scipy.stats.t`."
        if HAS_SCIPY
        else "SciPy unavailable; p-values/CIs computed via **mpmath** (see `scripts/cross_seed_significance.py`)."
    )
    lines.extend(["---", "", scipy_note, ""])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}", flush=True)


def patch_attention_variants_status(
    repo: Path,
    csv_path: Path,
    specs: list[tuple[str, str, int]],
) -> None:
    path = repo / "docs" / "ATTENTION_VARIANTS_STATUS.md"
    rows = load_csv_rows(csv_path)
    if not any(collect_vals(rows, exp, "BLEU") for exp, _, _ in specs):
        print("skip docs/ATTENTION_VARIANTS_STATUS.md table rows (no variant metrics yet)", flush=True)
        return
    text = path.read_text(encoding="utf-8")

    def replace_row(attn: str, new_line: str) -> None:
        nonlocal text
        pat = re.compile(rf"^\| `{re.escape(attn)}` \|.*$", re.MULTILINE)
        if not pat.search(text):
            raise SystemExit(f"could not find table row for `{attn}` in {path}")
        text = pat.sub(new_line, text, count=1)

    for exp, attn, _ in specs:
        meta = ROW_META[attn]
        b = collect_vals(rows, exp, "BLEU")
        cp = collect_vals(rows, exp, CHRPP_KEY)
        co = collect_vals(rows, exp, "COMET")
        if len(b) < 3:
            continue
        mb, sb = mean_std([float(x) for x in b])
        mcp, scp = mean_std([float(x) for x in cp])
        co_f = [float(x) for x in co if x is not None and str(x).strip() != ""]
        mco, sco = mean_std(co_f)
        r_bleu = analyze_metric("BLEU", b, collect_vals(rows, FAST_DOT_REF, "BLEU"), exp, FAST_DOT_REF)
        lo, hi = r_bleu["ci_95_low"], r_bleu["ci_95_high"]
        ci_txt = (
            f"95% CI for ΔBLEU [{fmt(lo)}, {fmt(hi)}]"
            if lo is not None and hi is not None
            else "95% CI for ΔBLEU: n/a"
        )
        p = r_bleu["p_value_two_sided"]
        p_txt = fmt(p, nd=4) if p is not None else "n/a"
        legacy = LEGACY_SINGLE_SEED_RUN[attn]
        exp_cell = (
            f"**trained (3 seeds)** (`runs/{flat_run_name(exp, 1)}` … `runs/{flat_run_name(exp, 3)}`); "
            f"held-out **test_eval** mean±std: BLEU={fmt(mb)}±{fmt(sb)}, "
            f"chrF++={fmt(mcp)}±{fmt(scp)}, COMET={fmt(mco)}±{fmt(sco)}. "
            f"Welch vs **fast_dot** (BLEU): Δ={fmt(r_bleu['mean_difference_a_minus_b'])}, "
            f"{ci_txt}, p={p_txt}, t={fmt(r_bleu['welch_t'])}, df≈{fmt(r_bleu['welch_df'])}. "
            f"Readout: **`results/variant_multiseed_summary.md`** (legacy single-seed **`runs/{legacy}`**)."
        )
        line = (
            f"| `{attn}` | {meta['mechanism']} | implemented | {exp_cell} | "
            f"{meta['purpose']} | {meta['safe']} |"
        )
        replace_row(attn, line)

    path.write_text(text, encoding="utf-8")
    print(f"Updated {path}", flush=True)


def refresh_variant_rows_from_disk(
    repo: Path,
    specs: list[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for exp, attn, nh in specs:
        for seed in SEEDS:
            out.append(
                build_per_seed_row(
                    experiment=exp,
                    attention_type=attn,
                    n_heads=nh,
                    seed=seed,
                    repo=repo,
                )
            )
    return out


def train_and_eval_one(
    *,
    repo: Path,
    py: Path,
    train_py: Path,
    eval_py: Path,
    exp: str,
    attn: str,
    n_heads: int,
    seed: int,
    args: argparse.Namespace,
    resume: bool,
) -> int:
    run_name = flat_run_name(exp, seed)
    run_dir = repo / "runs" / run_name
    test_eval = run_dir / "test_eval" / "metrics_test.json"
    best_pt = run_dir / "best.pt"
    if resume and test_eval.is_file():
        print(f"[skip] {run_name} already has test_eval", flush=True)
        return 0

    need_train = (not resume) or (not best_pt.is_file())
    if need_train:
        train_cmd = [
            str(py),
            str(train_py),
            "--train-path",
            args.train_path,
            "--val-path",
            args.val_path,
            "--test-path",
            args.test_path,
            "--tokenizer-src",
            args.tokenizer_src,
            "--tokenizer-tgt",
            args.tokenizer_tgt,
            "--output-dir",
            "runs",
            "--name",
            run_name,
            "--attention-type",
            attn,
            "--n-heads",
            str(n_heads),
            "--seed",
            str(seed),
            "--eval-split",
            "val",
            "--max-steps",
            str(args.max_steps),
            "--no-wandb",
            "--no-attention-plots",
            "--eval-light",
            "--bleu-sample-size",
            str(args.bleu_sample_size),
        ]
        rc = run_command(train_cmd, cwd=repo, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not best_pt.is_file():
        print(f"[warn] no best.pt for {run_name}", flush=True)
        return 1

    if (not resume) or (not test_eval.is_file()):
        ev_out = run_dir / "test_eval"
        eval_cmd = [
            str(py),
            str(eval_py),
            "--checkpoint",
            rel_under_repo(best_pt),
            "--test-file",
            args.test_path,
            "--tokenizer-src",
            args.tokenizer_src,
            "--tokenizer-tgt",
            args.tokenizer_tgt,
            "--output-dir",
            rel_under_repo(ev_out),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--batch-size",
            str(args.eval_batch_size),
            "--pkg",
            "small_try",
        ]
        rc = run_command(eval_cmd, cwd=repo, dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-path", type=str, default=f"{DEFAULT_SPLITS}/train.tsv")
    p.add_argument("--val-path", type=str, default=f"{DEFAULT_SPLITS}/val.tsv")
    p.add_argument("--test-path", type=str, default=f"{DEFAULT_SPLITS}/test.tsv")
    p.add_argument("--tokenizer-src", type=str, default="data/tokenizer_src_train_only.json")
    p.add_argument("--tokenizer-tgt", type=str, default="data/tokenizer_tgt_train_only.json")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--bleu-sample-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--skip-entmax15", action="store_true", help="Omit entmax15 (long wall); bilinear + gated only.")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-train from scratch even if checkpoints exist (overwrites run dir).",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="refresh CSV from disk, write summary, patch docs; no train/eval",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--csv",
        type=str,
        default="results/ablation_per_seed.csv",
    )
    args = p.parse_args()
    resume = not args.no_resume

    repo = REPO_ROOT
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo / csv_path
    if not csv_path.is_file() and not args.aggregate_only:
        raise SystemExit(f"missing {csv_path}")

    specs = list(VARIANT_SPECS)
    if args.skip_entmax15:
        specs = [s for s in specs if s[0] != "var_entmax15"]

    py = Path(sys.executable)
    train_py = repo / "small_try" / "train.py"
    eval_py = repo / "evaluate_test.py"

    if args.aggregate_only:
        if not csv_path.is_file():
            raise SystemExit(f"missing {csv_path}")
        vrows = refresh_variant_rows_from_disk(repo, specs)
        merge_ablation_csv(csv_path, vrows)
        out_md = repo / "results" / "variant_multiseed_summary.md"
        write_variant_multiseed_summary(repo, csv_path, out_md, specs)
        patch_attention_variants_status(repo, csv_path, specs)
        return 0

    for exp, attn, nh in specs:
        for seed in SEEDS:
            rc = train_and_eval_one(
                repo=repo,
                py=py,
                train_py=train_py,
                eval_py=eval_py,
                exp=exp,
                attn=attn,
                n_heads=nh,
                seed=seed,
                args=args,
                resume=resume,
            )
            if rc != 0:
                return rc

    vrows = refresh_variant_rows_from_disk(repo, specs)
    merge_ablation_csv(csv_path, vrows)
    out_md = repo / "results" / "variant_multiseed_summary.md"
    write_variant_multiseed_summary(repo, csv_path, out_md, specs)
    patch_attention_variants_status(repo, csv_path, specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
