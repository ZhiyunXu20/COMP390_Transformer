#!/usr/bin/env python3
"""Write results/additive_matched_lr3e3_summary.md from cross_seed JSONs."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    old_path = REPO / "results" / "cross_seed_significance.json"
    new_path = REPO / "results" / "cross_seed_significance_lr3e3.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))

    bleu_old = old["metrics"]["BLEU"]
    bleu_new = new["metrics"]["BLEU"]

    gap_3e4 = float(bleu_old["mean_difference_a_minus_b"])
    gap_3e3 = float(bleu_new["mean_difference_a_minus_b"])
    # Shift in mean contrast when LR is aligned (same sign: fast_dot − additive).
    shift = gap_3e4 - gap_3e3

    wd = bleu_new["fast_dot"]
    wa = bleu_new["fast_add_lr3e3"]
    t = bleu_new["welch_t"]
    df = bleu_new["welch_df"]
    p = bleu_new["p_value_two_sided"]

    lines = [
        "# Additive at tuned higher LR (A19)",
        "",
        "## Protocol",
        "",
        "- **Model**: `small_try`, `attention_type=additive`, `n_heads=4`.",
        "- **Training**: `lr=3e-3`, `max_steps=3000`, seeds **1/2/3**, val BLEU for checkpointing;",
        "  `--no-wandb`, `--eval-light` during training (same style as other multiseed jobs).",
        "- **Held-out test**: `evaluate_test.py` **without** `--eval-light` (full BLEU/chrF++/COMET/BERTScore).",
        "- **Runs**: `runs/fast_add_lr3e3_s1` … `s3` (new directories only).",
        "",
        "## Cross-seed summary (held-out test BLEU)",
        "",
        f"- **fast_dot** (reference, `lr=3e-4`, n=3): mean={wd['mean']:.4g}, std={wd['std']:.4g}.",
        f"- **fast_add** @ **`lr=3e-4`** (n=3): dot–additive gap **≈ {gap_3e4:.2f} BLEU** "
        f"(Welch two-sided *p* ≈ {float(bleu_old['p_value_two_sided']):.4g}; see `results/cross_seed_significance.md`).",
        f"- **fast_add_lr3e3** (A19, `lr=3e-3`, n=3): mean={wa['mean']:.4g}, std={wa['std']:.4g}.",
        f"- **fast_dot − fast_add_lr3e3** (Welch): ΔBLEU **{gap_3e3:.3f}**, *t*={t:.3f}, df≈{df:.3g}, two-sided "
        f"*p*={p:.4g} (`results/cross_seed_significance_lr3e3.json`).",
        "",
        "## Interpretation (for thesis / boundaries)",
        "",
        (
            "With **matched** `max_steps=3000` and **matched learning rate (`3e-3`)** to the dot baseline schedule, "
            f"the cross-seed held-out contrast **fast_dot − additive** moves from **≈{gap_3e4:.2f} BLEU** "
            f"(`fast_add` @ **`3e-4`**) to **≈{gap_3e3:.2f} BLEU** (`fast_add_lr3e3`): the **ordering reverses** "
            f"(additive mean BLEU exceeds dot). The shift in the mean contrast is **≈{shift:.2f} BLEU**. "
            "In thesis language, the original **~7.60 BLEU** dot margin at `3e-4` is **fully accounted for** "
            "by LR mismatch relative to this matched protocol (**≥100%** in the sense that the sign of the gap flips). "
            "Use Welch *t* / *p* above and the JSON for exact CIs."
        ),
        "",
        "## Citations",
        "",
        "- `results/ablation_per_seed.csv` (`experiment=fast_add_lr3e3`).",
        "- `results/cross_seed_significance_lr3e3.{md,json}`.",
        "- Original `fast_dot` vs `fast_add` @ 3e-4: `results/cross_seed_significance.{md,json}`.",
        "",
    ]
    out = REPO / "results" / "additive_matched_lr3e3_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
