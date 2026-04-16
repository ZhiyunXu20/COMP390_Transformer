#!/usr/bin/env python3
"""多头基线（small_try fast_dot）与单头点积对比；可选第三列单头加性（需训练时启用）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from report_extra_metrics import (
    append_extra_conclusion_three,
    append_extra_conclusion_two,
    append_extra_rows_three_cols,
    append_extra_rows_two_cols,
)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def n_heads_of(m: dict):
    if m.get("n_heads") is not None:
        return m.get("n_heads")
    cfg = m.get("config") or {}
    return cfg.get("n_heads")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        default="/root/autodl-tmp/small_try/runs/fast_dot/metrics.json",
        help="多头 n_heads=4 + dot_product（small_try fast_dot）",
    )
    p.add_argument(
        "--one-dot",
        default="/root/autodl-tmp/small_head/runs/head_1h_dot/metrics.json",
        help="单头 n_heads=1 + dot_product",
    )
    p.add_argument(
        "--one-add",
        default=None,
        help="可选：单头 n_heads=1 + additive 的 metrics（未跑加性时可省略）",
    )
    p.add_argument("--out", default="/root/autodl-tmp/small_head/report_head.txt")
    p.add_argument(
        "--json-bundle",
        default="/root/autodl-tmp/small_head/results/bundle_head_metrics.json",
    )
    args = p.parse_args()

    baseline_p = Path(args.baseline)
    dot_p = Path(args.one_dot)
    add_p = Path(args.one_add) if args.one_add else None

    need = [("baseline (mh dot)", baseline_p), ("1h dot", dot_p)]
    if add_p is not None:
        need.append(("1h add", add_p))
    for label, path in need:
        if not path.is_file():
            raise SystemExit(f"缺少 {label} 的 metrics: {path}")

    bl = load_metrics(baseline_p)
    d1 = load_metrics(dot_p)
    a1 = load_metrics(add_p) if add_p is not None else None

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("small_head：多头（基线） vs 全模型单头（点积为主；加性为可选第三列）")
    lines.append("=" * 72)
    lines.append("")
    lines.append("基线来自 small_try runs/fast_dot（n_heads=4, dot_product）。")
    lines.append("单头实验在本目录训练（n_heads=1）；默认流水线只训点积。")
    lines.append("")

    if a1 is None:
        lines.append(f"{'指标':<34} {'mh_dot(4h)':>14} {'1h_dot':>14}")
        lines.append("-" * 64)

        def row2(name: str, key: str):
            lines.append(
                f"{name:<34} {str(bl.get(key)):>14} {str(d1.get(key)):>14}"
            )

        lines.append(
            f"{'n_heads':<34} {str(n_heads_of(bl)):>14} {str(n_heads_of(d1)):>14}"
        )
        row2("attention_type", "attention_type")
        row2("final_val_loss", "final_val_loss")
        row2("final_bleu", "final_bleu")
        row2("best_bleu_during_training", "best_bleu_during_training")
        row2("optimizer_steps", "optimizer_steps")
        append_extra_rows_two_cols(
            lines,
            bl,
            d1,
            label_w=34,
            col_a="mh_dot",
            col_b="1h_dot",
        )
        lines.append("")
        lines.append("W&B（若存在）:")
        lines.append(f"  baseline: {bl.get('wandb_url')}")
        lines.append(f"  1h_dot:   {d1.get('wandb_url')}")
        lines.append("")
        append_extra_conclusion_two(
            lines, bl, d1, name_b_minus_a="1h_dot - mh_baseline"
        )
    else:
        lines.append(
            f"{'指标':<34} {'mh_dot(4h)':>14} {'1h_dot':>14} {'1h_add':>14}"
        )
        lines.append("-" * 72)

        def row(name: str, kb: str, kc: str, kd: str):
            lines.append(
                f"{name:<34} {str(bl.get(kb)):>14} {str(d1.get(kc)):>14} {str(a1.get(kd)):>14}"
            )

        lines.append(
            f"{'n_heads':<34} {str(n_heads_of(bl)):>14} {str(n_heads_of(d1)):>14} {str(n_heads_of(a1)):>14}"
        )
        row("attention_type", "attention_type", "attention_type", "attention_type")
        row("final_val_loss", "final_val_loss", "final_val_loss", "final_val_loss")
        row("final_bleu", "final_bleu", "final_bleu", "final_bleu")
        row(
            "best_bleu_during_training",
            "best_bleu_during_training",
            "best_bleu_during_training",
            "best_bleu_during_training",
        )
        row("optimizer_steps", "optimizer_steps", "optimizer_steps", "optimizer_steps")
        append_extra_rows_three_cols(
            lines,
            bl,
            d1,
            a1,
            label_w=34,
            c0="mh_dot",
            c1="1h_dot",
            c2="1h_add",
        )
        lines.append("")
        lines.append("W&B（若存在）:")
        lines.append(f"  baseline: {bl.get('wandb_url')}")
        lines.append(f"  1h_dot:   {d1.get('wandb_url')}")
        lines.append(f"  1h_add:   {a1.get('wandb_url')}")
        lines.append("")
        append_extra_conclusion_three(lines, bl, d1, a1)

    lines.append("")
    lines.append("=" * 72)

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))

    bundle: dict = {
        "baseline_multi_head_dot": bl,
        "single_head_dot": d1,
        "report_txt": str(out.resolve()),
    }
    if a1 is not None:
        bundle["single_head_additive"] = a1
    jp = Path(args.json_bundle)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare_head_runs] 已写入合并指标: {jp}")


if __name__ == "__main__":
    main()
