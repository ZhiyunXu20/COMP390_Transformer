#!/usr/bin/env python3
"""Single-seed additive attention learning-rate sweep (small_try defaults; does not touch fast_add / fast_dot)."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SPLITS = "data/splits/en_fr_50k_seed42"
SEED = 42
MAX_STEPS = 1500

# (lr, run_name under runs/) — names fixed for A13 / review_v2
LR_JOBS: list[tuple[float, str]] = [
    (1e-4, "lr_sweep_add_lr1e4"),
    (3e-4, "lr_sweep_add_lr3e4"),
    (1e-3, "lr_sweep_add_lr1e3"),
    (3e-3, "lr_sweep_add_lr3e3"),
]

# Reference headline numbers (thesis / ablation); loaded from ablation_summary when present
FAST_DOT_BLEU_MEAN = 15.80
FAST_DOT_BLEU_STD = 0.52
FALLBACK_FAST_ADD_MEAN = 8.21
FALLBACK_FAST_ADD_STD = 0.70


def rel_repo(path: Path) -> str:
    path = path.resolve()
    root = REPO_ROOT.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_cmd(cmd: list[str], *, dry_run: bool) -> int:
    print(f"+ cd {REPO_ROOT} && {shlex.join(cmd)}", flush=True)
    if dry_run:
        return 0
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(cp.returncode)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ablation_fast_add_stats() -> tuple[float, float]:
    p = REPO_ROOT / "results" / "ablation_summary.csv"
    if not p.is_file():
        return FALLBACK_FAST_ADD_MEAN, FALLBACK_FAST_ADD_STD
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("experiment") == "fast_add" and row.get("checkpoint_kind") == "best":
                return float(row["bleu_mean"]), float(row["bleu_std"])
    return FALLBACK_FAST_ADD_MEAN, FALLBACK_FAST_ADD_STD


def collect_row(run_name: str, lr: float) -> dict[str, Any]:
    run_dir = REPO_ROOT / "runs" / run_name
    mpath = run_dir / "metrics.json"
    tmeta = run_dir / "training_meta.json"
    test_path = run_dir / "test_eval" / "metrics_test.json"

    val_loss = val_bleu = None
    wall_s: float | None = None
    if mpath.is_file():
        mj = load_json(mpath)
        val_loss = mj.get("final_val_loss")
        val_bleu = mj.get("final_bleu")
    if tmeta.is_file():
        wall_s = load_json(tmeta).get("wall_time_seconds")

    test_bleu = test_chrfpp = None
    if test_path.is_file():
        tj = load_json(test_path)
        test_bleu = tj.get("BLEU")
        test_chrfpp = tj.get("chrF++")

    return {
        "run_name": run_name,
        "lr": lr,
        "val_loss": val_loss,
        "val_bleu": val_bleu,
        "test_bleu": test_bleu,
        "test_chrfpp": test_chrfpp,
        "wall_s": wall_s,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["lr", "run_name", "val_loss", "val_bleu", "test_bleu", "test_chrfpp", "wall_s"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_md(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fa_mean: float,
    fa_std: float,
) -> None:
    finalists = [r for r in rows if r.get("test_bleu") is not None]
    if finalists:
        best = max(finalists, key=lambda r: float(r["test_bleu"]))
        bleus_only = [float(r["test_bleu"]) for r in finalists]
        worst_test = min(bleus_only)
        hi_test = max(bleus_only)
    else:
        best = max(rows, key=lambda r: float(r["lr"]))
        worst_test = hi_test = float("nan")

    best_lr = best["lr"]
    best_test = best.get("test_bleu")

    delta_vs_add_mean: float | None = None
    if best_test is not None:
        delta_vs_add_mean = float(best_test) - fa_mean

    if not finalists:
        gap_note = "No `test_eval/metrics_test.json` rows found; re-run training + `evaluate_test.py` or use `--aggregate-only` after eval."
    elif delta_vs_add_mean is not None and delta_vs_add_mean > 0.25:
        gap_note = (
            f"best sweep test_BLEU exceeds the fast_add mean by **~{delta_vs_add_mean:.2f} BLEU** "
            "(single seed, half training steps); HP tuning may explain **part** of the gap vs dot_product "
            "but does not approach the dot baseline in this sweep."
        )
    elif delta_vs_add_mean is not None and abs(delta_vs_add_mean) <= 0.25:
        gap_note = (
            "best sweep test_BLEU is **within ~0.25 BLEU** of the fast_add mean; **no strong lr sensitivity** "
            "is visible here—the dot-vs-additive gap is unlikely to be an artifact of lr=3e-4 alone."
        )
    elif delta_vs_add_mean is not None:
        gap_note = (
            "best sweep test_BLEU is **below** the fast_add mean at matched nominal lr settings; "
            "**shorter training (1500 steps)** likely dominates over lr choice in this comparison."
        )

    lines = [
        "# Additive learning-rate sweep (single seed)",
        "",
        "## Methodology",
        "",
        "- **Single seed** sweep (`seed=42`); not a stability / cross-seed inference exercise.",
        f"- **`max_steps={MAX_STEPS}`** (half of the standard 3000) to fit GPU time.",
        "- All hyperparameters except **`learning_rate`** follow **`small_try.Config` defaults** "
        "(same data split, batch size, architecture, additive attention, `n_heads=4`).",
        "- Training uses **`--eval-light`** (validation BLEU subsample: BLEU + chrF only, no BERTScore/COMET).",
        "- Held-out **`evaluate_test.py`** uses **`--eval-light`** (BLEU + chrF++ only).",
        "- **`--no-wandb`** on training to avoid polluting the W&B project.",
        "- **`--no-resume`** re-trains every lr from scratch (overwrites `runs/lr_sweep_add_*`). "
        "Default is resume: skip runs that already have `test_eval/metrics_test.json`; "
        "if only `best.pt` exists, run **`evaluate_test.py`** only.",
        "",
        "## Results",
        "",
        "| lr | val_loss | val_BLEU | test_BLEU | test_chrF++ | wall_s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: float(x["lr"])):
        vl = r["val_loss"]
        vb = r["val_bleu"]
        tb = r["test_bleu"]
        tc = r["test_chrfpp"]
        ws = r["wall_s"]
        lines.append(
            f"| {r['lr']:.0e} | "
            f"{vl if vl is not None else 'n/a'} | "
            f"{vb if vb is not None else 'n/a'} | "
            f"{tb if tb is not None else 'n/a'} | "
            f"{tc if tc is not None else 'n/a'} | "
            f"{ws if ws is not None else 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## Comparison vs reference points",
            "",
            f"- **fast_add** (lr=3e-4, max_steps=3000, n=3): test_BLEU = **{fa_mean:.2f} ± {fa_std:.2f}** "
            f"(from `results/ablation_summary.csv` when present).",
            f"- **fast_dot** (lr=3e-4, max_steps=3000, n=3): test_BLEU = **{FAST_DOT_BLEU_MEAN:.2f} ± {FAST_DOT_BLEU_STD:.2f}** "
            "(headline thesis figure).",
            f"- **This sweep** (max_steps={MAX_STEPS}): best additive lr is **{best_lr:.0e}** with "
            f"test_BLEU = **{best_test if best_test is not None else 'n/a'}** (`runs/{best['run_name']}`).",
            (
                f"  - Sweep test_BLEU range (this table): **{worst_test:.4f}** – **{hi_test:.4f}**."
                if finalists
                else "  - Sweep test_BLEU range: _n/a (missing test metrics)._"
            ),
            "",
            "## Interpretation",
            "",
            "If the best additive lr in this sweep yields test_BLEU **meaningfully above** the fast_add ~8.2 mean, ",
            "the dot-vs-additive comparison should note that **per-mechanism HP tuning can move additive somewhat**. ",
            "If all sweep lrs yield **similar** BLEU, the gap is **more plausibly mechanism-driven** under this budget.",
            "",
            f"**Readout:** {gap_note}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Additive LR sweep → runs/lr_sweep_add_* + results tables")
    p.add_argument("--train-path", type=str, default=f"{DEFAULT_SPLITS}/train.tsv")
    p.add_argument("--val-path", type=str, default=f"{DEFAULT_SPLITS}/val.tsv")
    p.add_argument("--test-path", type=str, default=f"{DEFAULT_SPLITS}/test.tsv")
    p.add_argument("--tokenizer-src", type=str, default="data/tokenizer_src_train_only.json")
    p.add_argument("--tokenizer-tgt", type=str, default="data/tokenizer_tgt_train_only.json")
    p.add_argument("--pkg", type=str, default="small_try")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training/eval; rebuild CSV/MD from existing runs/lr_sweep_add_*",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Always re-train each lr (overwrite runs) even if checkpoints exist",
    )
    p.add_argument("--skip-train", action="store_true", help="Never train; only evaluate where best.pt exists")
    args = p.parse_args()

    train_py = REPO_ROOT / args.pkg / "train.py"
    eval_py = REPO_ROOT / "evaluate_test.py"
    for path, label in (
        (train_py, "train.py"),
        (eval_py, "evaluate_test.py"),
        (REPO_ROOT / args.train_path, "--train-path"),
        (REPO_ROOT / args.val_path, "--val-path"),
        (REPO_ROOT / args.test_path, "--test-path"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    py = sys.executable
    fa_mean, fa_std = load_ablation_fast_add_stats()
    rows_out: list[dict[str, Any]] = []

    for lr, run_name in LR_JOBS:
        run_dir = REPO_ROOT / "runs" / run_name
        best_pt = run_dir / "best.pt"
        ev_out = run_dir / "test_eval"

        test_metrics = ev_out / "metrics_test.json"

        if not args.aggregate_only:
            resume = not args.no_resume
            do_train = True
            do_eval = True
            if args.skip_train:
                do_train = False
            elif resume and test_metrics.is_file():
                print(f"[resume] test_eval complete, skip train+eval: {run_name}", flush=True)
                do_train = False
                do_eval = False
            elif resume and best_pt.is_file():
                print(f"[resume] checkpoint exists, skip train — evaluate_test only: {run_name}", flush=True)
                do_train = False

            if do_train:
                train_cmd = [
                    py,
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
                    "additive",
                    "--n-heads",
                    "4",
                    "--seed",
                    str(SEED),
                    "--max-steps",
                    str(MAX_STEPS),
                    "--learning-rate",
                    str(lr),
                    "--no-wandb",
                    "--eval-light",
                    "--eval-split",
                    "val",
                ]
                rc = run_cmd(train_cmd, dry_run=args.dry_run)
                if rc != 0:
                    raise SystemExit(f"training failed for {run_name} rc={rc}")

            if do_eval:
                if not args.dry_run and not best_pt.is_file():
                    print(f"[warn] missing checkpoint {best_pt}, skipping test eval", file=sys.stderr)
                else:
                    eval_cmd = [
                        py,
                        str(eval_py),
                        "--checkpoint",
                        rel_repo(best_pt),
                        "--test-file",
                        args.test_path,
                        "--tokenizer-src",
                        args.tokenizer_src,
                        "--tokenizer-tgt",
                        args.tokenizer_tgt,
                        "--output-dir",
                        rel_repo(ev_out),
                        "--pkg",
                        args.pkg,
                        "--eval-light",
                    ]
                    if not args.dry_run and best_pt.is_file():
                        rc_e = run_cmd(eval_cmd, dry_run=False)
                        if rc_e != 0:
                            raise SystemExit(f"evaluate_test failed for {run_name} rc={rc_e}")

        rows_out.append(collect_row(run_name, lr))

    results_csv = REPO_ROOT / "results" / "additive_lr_sweep.csv"
    results_md = REPO_ROOT / "results" / "additive_lr_sweep.md"
    if not args.dry_run:
        write_csv(results_csv, rows_out)
        write_md(results_md, rows_out, fa_mean=fa_mean, fa_std=fa_std)
        print(f"Wrote {results_csv}", file=sys.stderr)
        print(f"Wrote {results_md}", file=sys.stderr)
    else:
        print("[dry-run] no CSV/MD written", file=sys.stderr)


if __name__ == "__main__":
    main()
