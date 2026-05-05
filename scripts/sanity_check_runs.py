#!/usr/bin/env python3
"""Scan runs/ for silent training or generation failures (NaN loss, empty hyps, etc.)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunReport:
    run: str
    status: str
    bleu: str
    chrf: str
    val_loss: str
    empty_hyp_pct: str
    notes: list[str] = field(default_factory=list)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return x
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _is_nan_or_inf(x: Any) -> bool:
    if x is None:
        return False
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def _predictions_hyp_field(rec: dict[str, Any]) -> str:
    h = rec.get("hyp")
    if h is None:
        h = rec.get("hypothesis")
    if h is None:
        return ""
    return str(h)


def analyze_predictions(path: Path) -> tuple[float | None, float | None, int]:
    """Return (empty_pct, short_token_pct, n_lines). Unreadable → (None, None, 0)."""
    if not path.is_file():
        return None, None, 0
    n = 0
    n_empty = 0
    n_short = 0
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                hyp = _predictions_hyp_field(rec)
                n += 1
                if not hyp.strip():
                    n_empty += 1
                elif len(hyp.split()) < 2:
                    n_short += 1
    except OSError:
        return None, None, 0
    if n == 0:
        return 0.0, 0.0, 0
    return 100.0 * n_empty / n, 100.0 * n_short / n, n


def classify_run(
    *,
    run: str,
    has_metrics: bool,
    has_metrics_test: bool,
    has_training_meta: bool,
    metrics: dict[str, Any] | None,
    metrics_test: dict[str, Any] | None,
    empty_hyp_pct: float | None,
    pred_n: int,
) -> tuple[str, list[str]]:
    """Return (status, notes). Primary status by priority (single label per run)."""
    del run  # reserved for future per-run rules
    notes: list[str] = []

    if not has_metrics or not has_metrics_test or metrics is None or metrics_test is None:
        notes.append("missing metrics.json or test_eval/metrics_test.json")
        return "missing_artifacts", notes

    fvl = metrics.get("final_val_loss")
    if _is_nan_or_inf(fvl):
        notes.append("final_val_loss is NaN or non-finite")
        bleu_t0 = _safe_float(metrics_test.get("BLEU"))
        chrf_t0 = _safe_float(metrics_test.get("chrF"))
        if bleu_t0 == 0.0 and chrf_t0 == 0.0:
            notes.append("also failed_empty_outputs: test BLEU=0 and chrF=0")
        return "failed_nan", notes

    bleu_t = _safe_float(metrics_test.get("BLEU"))
    chrf_t = _safe_float(metrics_test.get("chrF"))
    bert = _safe_float(metrics_test.get("BERTScore"))
    n_ex = metrics_test.get("number_of_test_examples")
    len_ratio = _safe_float(metrics_test.get("average_length_ratio"))

    if n_ex is None or (isinstance(n_ex, (int, float)) and int(n_ex) <= 0):
        notes.append("number_of_test_examples missing or 0")
        return "missing_artifacts", notes

    if bleu_t is not None and bleu_t == 0.0 and bert == 0.0:
        notes.append("BERTScore=0 while BLEU=0 (degenerate generation metrics)")

    failed_empty = bleu_t == 0.0 and chrf_t == 0.0
    if len_ratio is not None and len_ratio == 0.0:
        notes.append("average_length_ratio==0 (suggests empty or PAD-only outputs)")
        failed_empty = True

    if empty_hyp_pct is not None and pred_n > 0 and empty_hyp_pct > 50.0:
        notes.append(f"empty_hyp_pct={empty_hyp_pct:.1f}% (>50%)")
        failed_empty = True

    if failed_empty:
        return "failed_empty_outputs", notes

    fb = metrics.get("final_bleu")
    bb = metrics.get("best_bleu_during_training")
    if fb is None or (isinstance(fb, (int, float)) and float(fb) == 0.0):
        notes.append("final_bleu missing or 0 (training-time val BLEU)")
    if bb is None or (isinstance(bb, (int, float)) and float(bb) == 0.0):
        notes.append("best_bleu_during_training missing or 0")

    if not has_training_meta:
        return "incomplete", notes

    if bleu_t is None:
        notes.append("test BLEU missing in metrics_test.json")
        return "missing_artifacts", notes
    if bleu_t <= 0:
        notes.append("test BLEU non-positive after non-empty check")
        return "failed_empty_outputs", notes

    if len_ratio is None:
        notes.append("average_length_ratio missing")
        return "warning_low_quality", notes

    if len_ratio <= 0.3:
        notes.append(f"average_length_ratio={len_ratio:.3f} (<=0.3)")
        return "warning_low_quality", notes

    if (bleu_t > 0 and bleu_t < 5.0) or len_ratio < 0.5:
        return "warning_low_quality", notes

    return "ok", notes


def _fmt_loss(x: Any) -> str:
    if x is None:
        return "missing"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def scan_run(run_dir: Path) -> RunReport:
    run = run_dir.name
    m_path = run_dir / "metrics.json"
    te_path = run_dir / "test_eval" / "metrics_test.json"
    pred_path = run_dir / "test_eval" / "predictions.jsonl"
    meta_path = run_dir / "training_meta.json"

    has_metrics = m_path.is_file()
    has_te = te_path.is_file()
    has_tm = meta_path.is_file()

    metrics: dict[str, Any] | None = None
    metrics_test: dict[str, Any] | None = None
    if has_metrics:
        try:
            metrics = json.loads(m_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = None
            has_metrics = False
    if has_te:
        try:
            metrics_test = json.loads(te_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics_test = None
            has_te = False

    empty_pct, _short_pct, pred_n = analyze_predictions(pred_path)
    status, notes = classify_run(
        run=run,
        has_metrics=has_metrics,
        has_metrics_test=has_te,
        has_training_meta=has_tm,
        metrics=metrics,
        metrics_test=metrics_test,
        empty_hyp_pct=empty_pct,
        pred_n=pred_n,
    )

    bleu_s = "-"
    chrf_s = "-"
    if metrics_test:
        b = metrics_test.get("BLEU")
        c = metrics_test.get("chrF")
        bleu_s = f"{float(b):.2f}" if b is not None else "missing"
        chrf_s = f"{float(c):.2f}" if c is not None else "missing"

    val_loss_s = _fmt_loss(metrics.get("final_val_loss") if metrics else None)
    empty_s = f"{empty_pct:.1f}" if empty_pct is not None and pred_n > 0 else "n/a"

    return RunReport(
        run=run,
        status=status,
        bleu=bleu_s,
        chrf=chrf_s,
        val_loss=val_loss_s,
        empty_hyp_pct=empty_s,
        notes=list(dict.fromkeys(notes)),
    )


def _discover_runs(runs_root: Path, only: list[str] | None) -> list[Path]:
    if not runs_root.is_dir():
        return []
    if only:
        out: list[Path] = []
        for name in only:
            d = runs_root / name
            if d.is_dir():
                out.append(d)
        return sorted(out, key=lambda p: p.name)
    return sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def build_markdown(reports: list[RunReport]) -> str:
    lines: list[str] = ["# Runs sanity report", ""]
    lines.append("## Summary table")
    lines.append("")
    lines.append("| run | status | BLEU | chrF | val_loss | empty_hyp_pct | notes |")
    lines.append("|-----|--------|------|------|----------|---------------|-------|")
    for r in reports:
        n = " ".join(r.notes).replace("|", "\\|") if r.notes else ""
        lines.append(
            f"| {r.run} | {r.status} | {r.bleu} | {r.chrf} | {r.val_loss} | {r.empty_hyp_pct} | {n} |"
        )
    lines.append("")
    lines.append("## Detailed findings (non-ok runs)")
    lines.append("")
    bad = [r for r in reports if r.status != "ok"]
    if not bad:
        lines.append("_All scanned runs classified as **ok**._")
    else:
        for r in bad:
            lines.append(f"### `{r.run}` — {r.status}")
            lines.append("")
            lines.append(f"- Test BLEU / chrF: {r.bleu} / {r.chrf}")
            lines.append(f"- Training final_val_loss (metrics.json): {r.val_loss}")
            lines.append(f"- Empty-hypothesis rate (%): {r.empty_hyp_pct}")
            if r.notes:
                lines.append("- Notes:")
                for note in r.notes:
                    lines.append(f"  - {note}")
            lines.append("")
    lines.append("## Methodology note")
    lines.append("")
    lines.append(
        "- **Training-time validation BLEU** lives in `metrics.json` (`final_bleu`, "
        "`best_bleu_during_training`, etc.) and reflects the training script’s `eval_split` "
        "(usually **val**), often on a **subsample** and with training-specific filtering."
    )
    lines.append(
        "- **Held-out test BLEU / chrF** live in `test_eval/metrics_test.json` from "
        "`evaluate_test.py` on **`test.tsv`**; these are **not comparable** to validation BLEU "
        "in `metrics.json` line-by-line."
    )
    lines.append(
        "- A run can show **healthy training metrics** yet **collapsed generation** on test "
        "(e.g. BLEU=chrF=0, high empty-hyp rate) if decoding, checkpoint selection, or "
        "numerical edge cases differ between train/val loops and final greedy decode."
    )
    lines.append(
        "- **`final_val_loss` = NaN/non-finite** usually signals numerical instability during "
        "training or validation (e.g. bf16 autocast + badly scaled logits, bad masks)."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_json(reports: list[RunReport]) -> list[dict[str, Any]]:
    return [
        {
            "run": r.run,
            "status": r.status,
            "bleu": r.bleu,
            "chrf": r.chrf,
            "val_loss": r.val_loss,
            "empty_hyp_pct": r.empty_hyp_pct,
            "notes": r.notes,
        }
        for r in reports
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=str, default="runs/", help="Directory containing run folders")
    p.add_argument("--output-md", type=str, default="results/runs_sanity_report.md")
    p.add_argument("--output-json", type=str, default="results/runs_sanity_report.json")
    p.add_argument("--runs", nargs="*", default=None, help="Optional explicit run names to scan")
    args = p.parse_args()

    root = Path.cwd()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (root / runs_root).resolve()

    reports = [scan_run(d) for d in _discover_runs(runs_root, args.runs)]

    out_md = Path(args.output_md)
    out_json = Path(args.output_json)
    if not out_md.is_absolute():
        out_md = (root / out_md).resolve()
    if not out_json.is_absolute():
        out_json = (root / out_json).resolve()

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(build_markdown(reports), encoding="utf-8")
    out_json.write_text(json.dumps(build_json(reports), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
