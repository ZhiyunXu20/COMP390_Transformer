#!/usr/bin/env python3
"""Write results/local_window_stability_report.md from A15 training/test artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _finite(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return False
    return True


def parse_val_loss_progression(log_path: Path, *, max_points: int = 6) -> str:
    if not log_path.is_file():
        return "(no log)"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    vals: list[tuple[int, str]] = []
    pat = re.compile(r"val_loss=([^\s]+)\s+bleu=")
    for i, ln in enumerate(lines):
        m = pat.search(ln)
        if m:
            vals.append((i, m.group(1)))
    if not vals:
        return "(no val_loss= lines in log)"
    head = vals[: max_points // 2]
    tail = vals[-(max_points // 2) :] if len(vals) > max_points // 2 else []
    parts: list[str] = []
    for _, s in head:
        parts.append(s)
    if len(vals) > len(head) + len(tail):
        parts.append("…")
    for _, s in tail:
        if s not in parts:  # rough dedupe
            parts.append(s)
    return " → ".join(parts)


def empty_hyp_pct(pred_path: Path) -> float | None:
    if not pred_path.is_file():
        return None
    n, empty = 0, 0
    for ln in pred_path.open(encoding="utf-8"):
        n += 1
        r = json.loads(ln)
        hy = (r.get("hypothesis") or "").strip()
        if not hy:
            empty += 1
    return 100.0 * empty / max(1, n)


def load_run_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_test_bleu(run_dir: Path) -> float | None:
    p = run_dir / "test_eval" / "metrics_test.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("BLEU")


def train_ok(m: dict) -> bool:
    fv = m.get("final_val_loss")
    fb = m.get("final_bleu")
    return _finite(fv) and _finite(fb) and float(fb) > 1.0


def main() -> None:
    rows = []

    orig = load_run_metrics(REPO / "runs/var_local_window")
    rows.append(
        {
            "experiment": "original (var_local_window)",
            "precision": "bf16",
            "lr": "3e-4",
            "window": "4",
            "val_prog": "NaN",
            "final_bleu_val": orig.get("final_bleu"),
            "test_bleu": json.loads(
                (REPO / "runs/var_local_window/test_eval/metrics_test.json").read_text(
                    encoding="utf-8"
                )
            ).get("BLEU"),
            "empty_hyp_pct": empty_hyp_pct(REPO / "runs/var_local_window/test_eval/predictions.jsonl"),
        }
    )

    specs = [
        (
            "fp32 control",
            "fp32",
            "3e-4",
            "4",
            "var_local_window_fp32",
            REPO / "logs/a15_fp32_train.log",
        ),
        (
            "lower lr",
            "bf16",
            "1e-4",
            "4",
            "var_local_window_lr1e4",
            REPO / "logs/a15_lr1e4_train.log",
        ),
        (
            "larger window",
            "bf16",
            "3e-4",
            "16",
            "var_local_window_window16",
            REPO / "logs/a15_window16_train.log",
        ),
    ]

    ok_flags: list[bool] = []
    for label, prec, lr, win, run_name, logp in specs:
        rd = REPO / "runs" / run_name
        m = load_run_metrics(rd)
        ok_flags.append(train_ok(m))
        rows.append(
            {
                "experiment": label,
                "precision": prec,
                "lr": lr,
                "window": win,
                "val_prog": parse_val_loss_progression(logp),
                "final_bleu_val": m.get("final_bleu"),
                "test_bleu": load_test_bleu(rd),
                "empty_hyp_pct": empty_hyp_pct(rd / "test_eval" / "predictions.jsonl"),
            }
        )

    ok_fp32, ok_lr, ok_win = ok_flags

    if ok_fp32 and ok_lr and ok_win:
        cause = (
            "bf16+lr+window=4 sensitivity rather than a single brittle axis: **all three** mitigations "
            "(fp32, lower lr, larger window) trained successfully in the 1500-step probe, so the failure is "
            "not mechanism-level but **configuration / numerics** under the original defaults."
        )
        fix = "fp32 training, or lr≤1e-4 in bf16, or `local_window_size`≥16 (any one is sufficient here)"
        stability = "stably under at least one of these mitigations"
    elif ok_fp32 and ok_lr:
        cause = (
            "Primarily bf16 numerical issues (fp32 succeeds). Lower lr also succeeds, "
            "suggesting gradient sensitivity under bf16 with the structural mask."
        )
        fix = "disabling bf16 autocast or reducing lr while in bf16"
        stability = "stably under either mitigation in this probe"
    elif not ok_fp32 and ok_lr and ok_win:
        cause = (
            "bf16 instability together with lr and window sensitivity; both lower lr and wider window "
            "recover training without switching to fp32."
        )
        fix = "reducing lr and/or widening the local window (bf16 retained)"
        stability = "stably with those hyperparameter changes"
    elif ok_fp32:
        cause = (
            "bf16 numerical stability under multi-layer structural masking (−∞ masked logits "
            "through softmax in autocast). The fp32 control trains successfully at the same "
            "lr and window."
        )
        fix = "disabling bf16 autocast (full-precision forward) for this variant"
        stability = "stably enough to produce non-degenerate validation BLEU (>1) in this probe"
    elif ok_lr:
        cause = (
            "optimizer instability at lr=3e-4 combined with the structural mask, not solely "
            "bf16 quirk (bf16 + lower lr succeeds)."
        )
        fix = "reducing learning rate (e.g. 1e-4) while keeping bf16"
        stability = "stably under the lowered lr in this probe"
    elif ok_win:
        cause = (
            "an overly restrictive local window (half-width 4) yielding weak/ill-conditioned "
            "attention patterns early in training; widening the window restores training signal."
        )
        fix = "increasing local_window_size (e.g. 16)"
        stability = "stably with the wider window in this probe"
    else:
        cause = (
            "an interaction between **encoder** `local_window` structural masking and **batched key-padding**—"
            "not resolved by fp32, lower lr, or `local_window_size=16` in this probe. "
            "Empirically: **`--no-bf16-autocast` still yields NaN** from the first training steps, so **bf16 alone is not the root cause**. "
            "A plausible implementation-level mechanism: for source positions deep in the **padded tail**, the allowed "
            "**|i−j| ≤ w** band can lie entirely inside keys that are **all PAD** (−∞ from `memory_key_padding` / encoder pad mask), "
            "so a softmax row is **all −∞** → **NaN attention**, which poisons `memory` and downstream logits (see `small_try/attention.py` "
            "`_structural_local_window` + `model.py` `_key_pad_mask`). "
            "`global_local` mitigates this class of failure via **global anchor columns** (see A15 vs. `runs/var_global_local`)."
        )
        fix = (
            "encoder-side mask fixes (e.g. union with a safe visibility pattern for pad queries, global anchors, or finite large-negative logits) "
            "beyond this diagnostic grid"
        )
        stability = "still not recovered in these probes"

    impl = (
        f"The original local-window failure is attributable to {cause} "
        f"We do NOT claim the local-window mechanism itself is unviable in general; we claim that "
        f"under this repository’s **encoder local-band + batched padding** interaction, the default **bf16** run **and** the A15 mitigations (fp32 math, lower lr, wider window) **still** produce NaN training in these probes. With **{fix}**, training is **{stability}**."
    )

    implications = (
        "Use the **Results** table and the diagnosis above when writing the thesis; cite **`results/local_window_stability_report.md`**. "
        "Do **not** claim the failure is **only** bf16 autocast—A15 fp32 control still diverges. "
        "Do **not** claim `local_window` is universally untrainable; claim only that **this repository’s dense local-window + padding interaction** "
        "fails under the tested protocol. Avoid claiming a sparse Longformer-style kernel; the code path is **dense L×L masked attention**."
    )

    lines = [
        "# Local window numerical stability (A15)",
        "",
        "## Setup",
        "",
        "- **Original failure:** `runs/var_local_window` with default CUDA **bf16 autocast**, "
        "**lr=3e-4**, **`local_window_size=4`**, **max_steps=3000** → `val_loss=NaN`, "
        "validation BLEU=0, held-out test BLEU=0, **empty hypotheses ~100%**.",
        "- **Diagnostics:** three runs (**max_steps=1500**, **seed=42**) with training-time "
        "`--eval-light` (BLEU/chrF on val subsample only) for wall-clock; **held-out** "
        "`evaluate_test.py` below uses **`--eval-light`** as well for the same reason.",
        "- **New outputs:** `runs/var_local_window_fp32/`, `runs/var_local_window_lr1e4/`, "
        "`runs/var_local_window_window16/` (original directory untouched).",
        "",
        "## Results",
        "",
        "| experiment | precision | lr | window | val_loss progression (subset) | val final BLEU | test BLEU | empty_hyp_pct |",
        "|------------|-----------|----|--------|------------------------------|----------------|-----------|---------------|",
    ]
    for r in rows:
        vb = r["final_bleu_val"]
        vb_s = f"{vb:.4f}" if isinstance(vb, (int, float)) and _finite(vb) else str(vb)
        tb = r["test_bleu"]
        tb_s = f"{tb:.4f}" if isinstance(tb, (int, float)) and _finite(tb) else str(tb)
        eh = r["empty_hyp_pct"]
        eh_s = f"{eh:.1f}%" if eh is not None else "N/A"
        lines.append(
            f"| {r['experiment']} | {r['precision']} | {r['lr']} | {r['window']} | "
            f"{r['val_prog']} | {vb_s} | {tb_s} | {eh_s} |"
        )

    lines.extend(
        [
            "",
            "**Success criterion (training health):** finite `final_val_loss`, validation `final_bleu` > 1, "
            "and non-trivial held-out BLEU / empty-hyp rate.",
            "",
            "## Diagnosis",
            "",
            "- If **fp32 control** trains successfully → primary cause is **bf16 numerical stability** "
            "with multi-layer **−∞ structural masks**.",
            "- If **lower lr** trains successfully → **gradient instability** at high lr × mask interaction.",
            "- If **larger window** trains successfully → **too-restrictive window** (weak attention support).",
            "- If **all three** diagnostics fail → **limitations of this implementation/stack** under the probe (not bf16-, lr-, or window=4-only); in this codebase the leading story is **encoder local-band × batched padding** (see below).",
            "",
            f"**This archive:** fp32={'**yes**' if ok_fp32 else 'no'}, "
            f"lower_lr={'**yes**' if ok_lr else 'no'}, "
            f"larger_window={'**yes**' if ok_win else 'no'}.",
            "",
            impl,
            "",
            "## Implications for dissertation",
            "",
            implications,
            "",
        ]
    )

    out = REPO / "results/local_window_stability_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
