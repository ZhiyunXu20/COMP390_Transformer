#!/usr/bin/env python3
"""读取 small_swap（法→英）训练的 metrics.json：可双 run 对比或仅单 run 摘要。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from report_extra_metrics import (
    append_extra_conclusion_two,
    append_extra_rows_two_cols,
    append_extra_single_run,
)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dot",
        default="/root/autodl-tmp/small_swap/runs/swap_fr_dot/metrics.json",
        help="点积注意力 run 的 metrics.json",
    )
    p.add_argument(
        "--add",
        default=None,
        help="可选：加性注意力 run；省略则只输出 dot 单 run 摘要",
    )
    p.add_argument(
        "--out",
        default="/root/autodl-tmp/small_swap/report_swap.txt",
        help="输出报告路径",
    )
    p.add_argument(
        "--json-bundle",
        default="/root/autodl-tmp/small_swap/results/bundle_swap_metrics.json",
        help="合并 metrics.json 的路径（便于归档）",
    )
    args = p.parse_args()

    dot_path = Path(args.dot)
    if not dot_path.is_file():
        raise SystemExit(f"缺少 {dot_path}，请先完成 dot_product 训练")

    a = load_metrics(dot_path)
    add_path = Path(args.add) if args.add else None
    single = add_path is None or not add_path.is_file()
    b: dict | None = None

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("small_swap：法→英（子语料 + 小模型 + 短训练）")
    lines.append("=" * 60)
    lines.append("")
    lines.append("说明：方向为法语→英语；语料列为 EN\\tFR，训练时交换为 FR 源 / EN 目标。")
    lines.append("与全数据长训的绝对 BLEU 不可直接等同；与 EN→FR 的 small_try 数值也不可横向对比。")
    lines.append("")

    if single:
        lines.append(f"单 run（dot_product） data_path: {a.get('data_path')}")
        lines.append("")
        lines.append(f"{'指标':<28} {'dot_product':>14}")
        lines.append("-" * 44)

        def row1(name: str, key: str):
            lines.append(f"{name:<28} {str(a.get(key)):>14}")

        row1("final_val_loss", "final_val_loss")
        row1("final_bleu", "final_bleu")
        row1("best_bleu_during_training", "best_bleu_during_training")
        row1("optimizer_steps", "optimizer_steps")
        append_extra_single_run(lines, a, label_w=28, col="swap_fr_dot")
        lines.append("")
        lines.append("（当前为单 run 模式：未提供有效的 --add 路径。）")
    else:
        b = load_metrics(add_path)
        lines.append(f"点积 (dot_product)  data_path: {a.get('data_path')}")
        lines.append(f"加性 (additive)      data_path: {b.get('data_path')}")
        lines.append("")
        lines.append(f"{'指标':<28} {'dot_product':>14} {'additive':>14}")
        lines.append("-" * 58)

        def row(name: str, ka: str, kb: str | None = None):
            kb = kb or ka
            va = a.get(ka)
            vb = b.get(kb)
            lines.append(f"{name:<28} {str(va):>14} {str(vb):>14}")

        row("final_val_loss", "final_val_loss")
        row("final_bleu", "final_bleu")
        row("best_bleu_during_training", "best_bleu_during_training")
        row("optimizer_steps", "optimizer_steps")
        append_extra_rows_two_cols(
            lines,
            a,
            b,
            label_w=28,
            col_a="dot_product",
            col_b="additive",
        )
        lines.append("")

        bleu_a = a.get("final_bleu")
        bleu_b = b.get("final_bleu")
        if isinstance(bleu_a, (int, float)) and isinstance(bleu_b, (int, float)):
            diff = float(bleu_b) - float(bleu_a)
            if diff > 0.5:
                winner = "加性注意力 (additive) 在本设定下验证 BLEU 更高。"
            elif diff < -0.5:
                winner = "缩放点积 (dot_product) 在本设定下验证 BLEU 更高。"
            else:
                winner = "二者验证 BLEU 接近（差距 < 0.5），可写为「相当」或结合 loss 讨论。"
            lines.append("结论（基于 final_bleu）：")
            lines.append(f"  {winner}")
            lines.append(f"  Δ(BLEU) = additive - dot = {diff:+.4f}")
        else:
            lines.append("结论：BLEU 缺失，请检查 sacrebleu 是否安装。")

        append_extra_conclusion_two(lines, a, b, name_b_minus_a="additive - dot")

    lines.append("")
    lines.append("撰写报告时可补充：数据子集规模、d_model/n_layers、max_steps、随机种子。")
    lines.append("=" * 60)

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))

    bundle: dict = {"dot_product_metrics": a, "report_txt": str(out.resolve())}
    if b is not None:
        bundle["additive_metrics"] = b
    jp = Path(args.json_bundle)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare_runs] 已写入合并指标: {jp}")


if __name__ == "__main__":
    main()
