#!/usr/bin/env python3
"""base_improve：两次训练 metrics 汇总（点积 vs 加性）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from report_extra_metrics import append_extra_conclusion_two, append_extra_rows_two_cols


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dot",
        default="/root/autodl-tmp/base_improve/runs/improve_dot/metrics.json",
    )
    p.add_argument(
        "--add",
        default="/root/autodl-tmp/base_improve/runs/improve_add/metrics.json",
    )
    p.add_argument("--out", default="/root/autodl-tmp/base_improve/report_improve.txt")
    p.add_argument(
        "--json-bundle",
        default="/root/autodl-tmp/base_improve/results/bundle_improve_metrics.json",
    )
    args = p.parse_args()

    dot_path = Path(args.dot)
    add_path = Path(args.add)
    if not dot_path.is_file():
        raise SystemExit(f"缺少 {dot_path}")
    if not add_path.is_file():
        raise SystemExit(f"缺少 {add_path}")

    a = load_metrics(dot_path)
    b = load_metrics(add_path)

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("base_improve：同 base_1 超参 + 训练栈加速（TF32 / compile / fused AdamW / prefetch）")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"{'指标':<30} {'dot_product':>16} {'additive':>16}")
    lines.append("-" * 64)

    def row(name: str, ka: str, kb: str | None = None):
        kb = kb or ka
        lines.append(f"{name:<30} {str(a.get(ka)):>16} {str(b.get(kb)):>16}")

    row("final_val_loss", "final_val_loss")
    row("final_bleu", "final_bleu")
    row("best_bleu_during_training", "best_bleu_during_training")
    row("optimizer_steps", "optimizer_steps")
    append_extra_rows_two_cols(
        lines,
        a,
        b,
        label_w=30,
        col_a="dot_product",
        col_b="additive",
    )
    lines.append("")
    lines.append("training_stack（应一致）:")
    lines.append(f"  dot:  {a.get('training_stack')}")
    lines.append(f"  add:  {b.get('training_stack')}")
    append_extra_conclusion_two(lines, a, b, name_b_minus_a="additive - dot")
    lines.append("=" * 64)

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))

    bundle = {
        "dot_product_metrics": a,
        "additive_metrics": b,
        "report_txt": str(out.resolve()),
    }
    jp = Path(args.json_bundle)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare_runs] 已写入 {jp}")


if __name__ == "__main__":
    main()
