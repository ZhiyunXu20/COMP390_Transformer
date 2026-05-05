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
    build_extra_metrics_banner_three,
    build_extra_metrics_banner_two,
    load_metrics_compare,
)


def n_heads_of(m: dict):
    if m.get("n_heads") is not None:
        return m.get("n_heads")
    cfg = m.get("config") or {}
    return cfg.get("n_heads")


_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        default=str(_REPO_ROOT / "runs" / "fast_dot" / "metrics.json"),
        help="多头 n_heads=4 + dot_product（small_try 等在仓库 runs/ 下的训练输出）",
    )
    p.add_argument(
        "--one-dot",
        default=str(_REPO_ROOT / "runs" / "head_1h_dot" / "metrics.json"),
        help="单头 n_heads=1 + dot_product",
    )
    p.add_argument(
        "--one-add",
        default=None,
        help="可选：单头 n_heads=1 + additive 的 metrics（未跑加性时可省略）",
    )
    p.add_argument(
        "--out",
        default=str(_REPO_ROOT / "small_head" / "report_head.txt"),
    )
    p.add_argument(
        "--json-bundle",
        default=str(_REPO_ROOT / "small_head" / "results" / "bundle_head_metrics.json"),
    )
    p.add_argument(
        "--prefer-test-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="优先读取各 run 的 test_eval/metrics_test.json（held-out test）；"
        "使用 --no-prefer-test-eval 则回落 metrics.json",
    )
    args = p.parse_args()

    baseline_p = Path(args.baseline)
    dot_p = Path(args.one_dot)
    add_p = Path(args.one_add) if args.one_add else None

    bl, _, prov_bl = load_metrics_compare(
        baseline_p, prefer_test_eval=args.prefer_test_eval
    )
    d1, _, prov_d1 = load_metrics_compare(dot_p, prefer_test_eval=args.prefer_test_eval)
    if add_p is not None:
        a1, _, prov_a1 = load_metrics_compare(
            add_p, prefer_test_eval=args.prefer_test_eval
        )
    else:
        a1 = None
        prov_a1 = None

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("small_head：多头（基线） vs 全模型单头（点积为主；加性为可选第三列）")
    if args.prefer_test_eval:
        if a1 is None:
            lines.append(
                f"报告标题·数据来源：mh_baseline={prov_bl}；1h_dot={prov_d1}"
            )
        else:
            lines.append(
                f"报告标题·数据来源：mh_baseline={prov_bl}；"
                f"1h_dot={prov_d1}；1h_add={prov_a1}"
            )
        lines.append(
            "（BLEU/chrF++/COMET 等指标来自 held-out test：各 run 的 test_eval/metrics_test.json）"
        )
    else:
        if a1 is None:
            lines.append(
                f"报告标题·数据来源（validation-sampled / metrics.json·eval_split）："
                f"mh_baseline={prov_bl}；1h_dot={prov_d1}"
            )
        else:
            lines.append(
                f"报告标题·数据来源（validation-sampled / metrics.json·eval_split）："
                f"mh_baseline={prov_bl}；1h_dot={prov_d1}；1h_add={prov_a1}"
            )
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
            extra_metrics_banner=build_extra_metrics_banner_two(prov_bl, prov_d1),
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
            extra_metrics_banner=build_extra_metrics_banner_three(
                prov_bl, prov_d1, prov_a1
            ),
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

    try:
        report_rel = out.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        report_rel = str(out.resolve())

    bundle: dict = {
        "baseline_multi_head_dot": bl,
        "single_head_dot": d1,
        "report_txt": report_rel,
    }
    if a1 is not None:
        bundle["single_head_additive"] = a1
    jp = Path(args.json_bundle)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare_head_runs] 已写入合并指标: {jp}")


if __name__ == "__main__":
    main()
