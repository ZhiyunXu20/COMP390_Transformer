"""从 train 写出的 metrics.json 读取 extra_metrics，供对比报告排版。"""

from __future__ import annotations

from typing import Any

EXTRA_KEYS: tuple[str, ...] = ("chrf", "chrfpp", "bertscore_f1", "comet")
EXTRA_LABELS: dict[str, str] = {
    "chrf": "extra: chrF",
    "chrfpp": "extra: chrF++",
    "bertscore_f1": "extra: BERTScore F1",
    "comet": "extra: COMET",
}


def extra_metrics_dict(m: dict[str, Any]) -> dict[str, Any]:
    ex = m.get("extra_metrics")
    if not isinstance(ex, dict):
        return {}
    return {k: ex[k] for k in EXTRA_KEYS if k in ex}


def fmt_val(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return str(v)


def append_extra_rows_two_cols(
    lines: list[str],
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    label_w: int,
    col_a: str,
    col_b: str,
    w: int = 14,
) -> None:
    """在 lines 末尾追加 extra_metrics 对照行（两列）。"""
    lines.append("")
    lines.append(
        "extra_metrics（训练结束时的最终评估；— 表示未写入或依赖未安装）:"
    )
    header = f"{'':<{label_w}} {col_a:>{w}} {col_b:>{w}}"
    lines.append(header)
    lines.append("-" * (label_w + 1 + w + 1 + w))
    ea, eb = extra_metrics_dict(a), extra_metrics_dict(b)
    for key in EXTRA_KEYS:
        va, vb = ea.get(key), eb.get(key)
        lines.append(
            f"{EXTRA_LABELS[key]:<{label_w}} {fmt_val(va):>{w}} {fmt_val(vb):>{w}}"
        )


def append_extra_rows_three_cols(
    lines: list[str],
    bl: dict[str, Any],
    d1: dict[str, Any],
    a1: dict[str, Any],
    *,
    label_w: int,
    c0: str,
    c1: str,
    c2: str,
    w: int = 12,
) -> None:
    lines.append("")
    lines.append(
        "extra_metrics（训练结束时的最终评估；— 表示未写入或依赖未安装）:"
    )
    lines.append(
        f"{'':<{label_w}} {c0:>{w}} {c1:>{w}} {c2:>{w}}"
    )
    lines.append("-" * (label_w + 1 + w * 3 + 2))
    e0, e1, e2 = (
        extra_metrics_dict(bl),
        extra_metrics_dict(d1),
        extra_metrics_dict(a1),
    )
    for key in EXTRA_KEYS:
        v0, v1, v2 = e0.get(key), e1.get(key), e2.get(key)
        lines.append(
            f"{EXTRA_LABELS[key]:<{label_w}} "
            f"{fmt_val(v0):>{w}} {fmt_val(v1):>{w}} {fmt_val(v2):>{w}}"
        )


def append_extra_conclusion_three(
    lines: list[str],
    bl: dict[str, Any],
    d1: dict[str, Any],
    a1: dict[str, Any],
) -> None:
    """三向实验：相对多头基线的差值（若三侧均有该指标）。"""
    e0 = extra_metrics_dict(bl)
    e1 = extra_metrics_dict(d1)
    e2 = extra_metrics_dict(a1)
    sub: list[str] = []
    for key in EXTRA_KEYS:
        v0, v1, v2 = e0.get(key), e1.get(key), e2.get(key)
        if not all(isinstance(x, (int, float)) for x in (v0, v1, v2)):
            continue
        v0, v1, v2 = float(v0), float(v1), float(v2)
        sub.append(
            f"{key}: 1h_dot-mh={v1 - v0:+.4f} | 1h_add-mh={v2 - v0:+.4f} | "
            f"1h_add-1h_dot={v2 - v1:+.4f}"
        )
    lines.append("")
    if sub:
        lines.append("多指标摘要（相对多头基线 mh_dot，及两单头之差）：")
        for s in sub:
            lines.append(f"  {s}")
    else:
        lines.append(
            "多指标：extra_metrics 中无三侧均可用的数值（请确认依赖已安装并完成最终评估）。"
        )


def append_extra_conclusion_two(
    lines: list[str],
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    name_b_minus_a: str = "additive - dot",
) -> None:
    """在已有 BLEU 结论后追加多指标 Δ 摘要（若两侧均有数值）。"""
    ea, eb = extra_metrics_dict(a), extra_metrics_dict(b)
    parts: list[str] = []
    for key in EXTRA_KEYS:
        va, vb = ea.get(key), eb.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            parts.append(f"Δ{key}={vb - va:+.4f}")
    if parts:
        lines.append("")
        lines.append(f"多指标摘要（{name_b_minus_a}）：")
        lines.append("  " + " | ".join(parts))
    else:
        lines.append("")
        lines.append(
            "多指标：extra_metrics 中无可用数值（请确认训练结束评估已跑且 bert-score/COMET 已安装）。"
        )


def append_extra_single_run(
    lines: list[str],
    m: dict[str, Any],
    *,
    label_w: int = 28,
    col: str = "run",
    w: int = 14,
) -> None:
    """仅一次实验时的 extra_metrics 列。"""
    lines.append("")
    lines.append("extra_metrics（最终评估；— 表示未写入或依赖未安装）:")
    lines.append(f"{'':<{label_w}} {col:>{w}}")
    lines.append("-" * (label_w + 1 + w))
    ex = extra_metrics_dict(m)
    for key in EXTRA_KEYS:
        lines.append(f"{EXTRA_LABELS[key]:<{label_w}} {fmt_val(ex.get(key)):>{w}}")
