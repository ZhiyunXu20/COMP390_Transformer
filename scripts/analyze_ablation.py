#!/usr/bin/env python3
"""读取消融多次运行的 metrics.json：均值±标准差、配对 bootstrap、Wilcoxon（可选）、Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1) if len(xs) > 1 else 0.0
    return m, math.sqrt(v)


def _extract_test_metrics(doc: dict[str, Any]) -> dict[str, float | None]:
    test = doc.get("final_eval_on_test_split") or {}
    ex = test.get("extra_metrics") or {}
    return {
        "bleu": _to_float(test.get("final_bleu")),
        "chrf": _to_float(ex.get("chrf")),
        "chrfpp": _to_float(ex.get("chrfpp")),
        "comet": _to_float(ex.get("comet")),
    }


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _paired_bootstrap_diff(
    dot: list[float],
    add: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """配对差值 bootstrap（对索引有放回重抽样），近似 95% CI。"""
    try:
        import numpy as np
    except ImportError:
        return {"mean_diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    d = np.array(dot, dtype=np.float64) - np.array(add, dtype=np.float64)
    n = len(d)
    if n == 0:
        return {"mean_diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    obs = float(np.mean(d))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(d[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    prop_neg = np.mean(boots <= 0.0)
    prop_pos = np.mean(boots >= 0.0)
    p_two = float(2 * min(prop_neg, prop_pos))
    return {
        "mean_diff_dot_minus_additive": obs,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_approx_two_sided_bootstrap": min(1.0, p_two),
    }


def _wilcoxon_if_possible(dot: list[float], add: list[float]) -> dict[str, float | None]:
    try:
        import numpy as np
        from scipy.stats import wilcoxon
    except ImportError:
        return {"wilcoxon_statistic": None, "wilcoxon_pvalue": None}
    if len(dot) < 2:
        return {"wilcoxon_statistic": None, "wilcoxon_pvalue": None}
    d = np.asarray(dot, dtype=np.float64) - np.asarray(add, dtype=np.float64)
    if np.allclose(d, 0.0):
        return {"wilcoxon_statistic": 0.0, "wilcoxon_pvalue": 1.0}
    r = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    return {"wilcoxon_statistic": float(r.statistic), "wilcoxon_pvalue": float(r.pvalue)}


def main() -> None:
    p = argparse.ArgumentParser(description="消融结果汇总与配对显著性")
    p.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="消融输出目录（含多个 abl_* 子目录）",
    )
    p.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Markdown 报告路径（默认 <runs-dir>/ablation_summary.md）",
    )
    p.add_argument("--bootstrap-seed", type=int, default=42)
    args = p.parse_args()

    runs_dir = (ROOT / args.runs_dir).resolve() if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"目录不存在: {runs_dir}")

    out_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else runs_dir / "ablation_summary.md"
    )

    by_att_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = sub / "metrics.json"
        if not mp.is_file():
            continue
        doc = json.loads(mp.read_text(encoding="utf-8"))
        if doc.get("ablation_metrics_version") != 1:
            print(f"[跳过] 非消融 metrics: {sub}", file=sys.stderr)
            continue
        att = str(doc.get("attention_type") or "")
        sd = doc.get("seed")
        if sd is None:
            continue
        by_att_seed[(att, int(sd))] = doc

    if not by_att_seed:
        lines_err = ["# 消融汇总\n", "\n未找到任何 `ablation_metrics_version==1` 的 `metrics.json`。\n"]
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines_err), encoding="utf-8")
        print(f"已写入（空）：{out_md}", file=sys.stderr)
        raise SystemExit(1)

    attention_types = sorted({k[0] for k in by_att_seed})
    seeds = sorted({k[1] for k in by_att_seed})

    lines: list[str] = []
    lines.append("# 消融实验汇总（多 seed）\n")
    lines.append(
        "**结论提示**：单次训练的验证集/测试集分数存在随机波动；请以本节 **均值 ± 标准差** 及配对检验为准，"
        "不要将单次 run 当作最终结论。\n"
    )
    lines.append(f"- runs 目录：`{runs_dir}`")
    lines.append(f"- 注意力类型：{', '.join(attention_types)}")
    lines.append(f"- seeds：{seeds}\n")

    metric_keys = ("bleu", "chrf", "chrfpp", "comet")

    for mk in metric_keys:
        lines.append(f"## {mk.upper()}\n")
        lines.append("| attention_type | mean ± std | per-seed |")
        lines.append("|---|---|---|")
        for att in attention_types:
            vals: list[float] = []
            parts: list[str] = []
            for sd in seeds:
                doc = by_att_seed.get((att, sd))
                if doc is None:
                    parts.append(f"{sd}: —")
                    continue
                m = _extract_test_metrics(doc).get(mk)
                if m is None:
                    parts.append(f"{sd}: —")
                elif isinstance(m, float) and math.isnan(m):
                    parts.append(f"{sd}: —")
                else:
                    vals.append(m)
                    parts.append(f"{sd}: {m:.4f}")
            mu, sig = _mean_std(vals)
            spread = f"{mu:.4f} ± {sig:.4f}" if vals else "—"
            lines.append(f"| {att} | {spread} | {', '.join(parts)} |")
        lines.append("")

    # Paired dot vs additive（同一 seed；按指标分别剔除缺失）
    if "dot_product" in attention_types and "additive" in attention_types:
        lines.append("## 配对比较：dot_product − additive（同一 seed）\n")

        for mk in metric_keys:
            pd: list[float] = []
            pa: list[float] = []
            used: list[int] = []
            for sd in seeds:
                dd = by_att_seed.get(("dot_product", sd))
                aa = by_att_seed.get(("additive", sd))
                if dd is None or aa is None:
                    continue
                vd = _extract_test_metrics(dd)[mk]
                va = _extract_test_metrics(aa)[mk]
                if vd is None or va is None:
                    continue
                pd.append(vd)
                pa.append(va)
                used.append(sd)

            lines.append(f"### {mk}\n")
            lines.append(f"- 配对样本 seeds：`{used}`（n={len(pd)}）")

            if len(pd) < 2:
                lines.append("- 配对样本不足，跳过 bootstrap / Wilcoxon。\n")
                continue

            boot = _paired_bootstrap_diff(pd, pa, seed=args.bootstrap_seed)
            wc = _wilcoxon_if_possible(pd, pa)
            lines.append(
                f"- 均值差（dot − add）：**{boot['mean_diff_dot_minus_additive']:.6f}** "
                f"95% bootstrap CI [{boot['ci95_low']:.6f}, {boot['ci95_high']:.6f}]"
            )
            lines.append(
                f"- 近似双侧 p（bootstrap 均值差是否跨过 0）：{boot['p_approx_two_sided_bootstrap']:.4f}"
            )
            if wc.get("wilcoxon_pvalue") is not None:
                lines.append(
                    f"- Wilcoxon signed-rank：`stat={wc['wilcoxon_statistic']:.4g}`, "
                    f"`p={wc['wilcoxon_pvalue']:.4g}`（需 scipy）"
                )
            lines.append(
                "\n> 说明：此处为 **seed 层面配对分数** 的 bootstrap / Wilcoxon；"
                "若需 sentence-level paired bootstrap，可对同一 `predictions.jsonl` 逐句重采样（更重）。\n"
            )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out_md}")


if __name__ == "__main__":
    main()
