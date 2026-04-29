#!/usr/bin/env python3
"""读取两次 small_try 训练的 metrics.json，生成 report.txt 对比结论。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from report_extra_metrics import append_extra_conclusion_two, append_extra_rows_two_cols


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dot",
        default=str(_REPO_ROOT / "runs" / "fast_dot" / "metrics.json"),
        help="点积注意力 run 的 metrics.json（相对仓库根的路径亦可）",
    )
    p.add_argument(
        "--add",
        default=str(_REPO_ROOT / "runs" / "fast_add" / "metrics.json"),
        help="加性注意力 run 的 metrics.json",
    )
    p.add_argument(
        "--out",
        default=str(_REPO_ROOT / "small_try" / "report.txt"),
        help="输出报告路径",
    )
    p.add_argument(
        "--json-bundle",
        default=str(_REPO_ROOT / "small_try" / "results" / "bundle_metrics.json"),
        help="合并两次 metrics.json 的路径（便于归档）",
    )
    args = p.parse_args()

    dot_path = Path(args.dot)
    add_path = Path(args.add)
    if not dot_path.is_file():
        raise SystemExit(f"缺少 {dot_path}，请先完成 dot_product 训练")
    if not add_path.is_file():
        raise SystemExit(f"缺少 {add_path}，请先完成 additive 训练")

    a = load_metrics(dot_path)
    b = load_metrics(add_path)

    lines = []
    lines.append("=" * 60)
    lines.append("small_try：注意力机制对比（子语料 + 小模型 + 短训练）")
    lines.append("=" * 60)
    lines.append("")
    lines.append("说明：本报告用于课程/实验结论；与全数据长训的绝对 BLEU 不可直接等同。")
    lines.append("")
    lines.append(f"点积 (dot_product)  data_path: {a.get('data_path')}")
    lines.append(f"加性 (additive)      data_path: {b.get('data_path')}")
    lines.append("")
    lines.append(f"{'指标':<28} {'dot_product':>14} {'additive':>14}")
    lines.append("-" * 58)

    def row(name: str, ka: str, kb: str = None):
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
        diff = bleu_b - bleu_a
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

    bundle = {
        "dot_product_metrics": a,
        "additive_metrics": b,
        "report_txt": str(out.resolve()),
    }
    jp = Path(args.json_bundle)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare_runs] 已写入合并指标: {jp}")


if __name__ == "__main__":
    main()
