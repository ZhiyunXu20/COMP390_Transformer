"""从 train 写出的 metrics.json 读取 extra_metrics，供对比报告排版。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def infer_run_dir_from_metrics_cli_arg(p: Path) -> Path:
    """根据 CLI 传入的 metrics 路径推断 run 根目录。"""
    p = p.resolve()
    if not p.is_file():
        raise SystemExit(f"路径不存在: {p}")
    if p.name == "metrics_test.json" and p.parent.name == "test_eval":
        return p.parent.parent
    if p.name == "metrics.json":
        return p.parent
    raise SystemExit(
        f"应为 …/metrics.json 或 …/test_eval/metrics_test.json，收到: {p}"
    )


def normalize_test_eval_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """将 test_eval/metrics_test.json 转成与训练 metrics 对齐的展示口径。"""
    bleu = raw.get("BLEU")
    extra = {
        "chrf": raw.get("chrF"),
        "chrfpp": raw.get("chrF++"),
        "bertscore_f1": raw.get("BERTScore"),
        "comet": raw.get("COMET"),
    }
    out = dict(raw)
    out["final_bleu"] = bleu
    out["best_bleu_during_training"] = bleu
    out["final_val_loss"] = "—"
    out["optimizer_steps"] = "—"
    out["extra_metrics"] = extra
    return out


def load_metrics_compare(cli_path: Path, *, prefer_test_eval: bool) -> tuple[dict[str, Any], Path, str | None]:
    """
    加载用于 compare 脚本的指标字典。

    prefer_test_eval=False：读取 cli_path 指向的文件（与旧行为一致）。
    prefer_test_eval=True：优先 runs/<name>/test_eval/metrics_test.json，
    否则回落 metrics.json（调用方须在标题标注 validation-sampled）。

    返回 (dict, 实际读取的路径, provenance)；provenance 仅在为 True 时有值：
    \"held-out test\" 或 \"validation-sampled\"。
    """
    cli_path = cli_path.resolve()
    if not prefer_test_eval:
        if not cli_path.is_file():
            raise SystemExit(f"缺少 metrics 文件: {cli_path}")
        return json.loads(cli_path.read_text(encoding="utf-8")), cli_path, None

    rd = infer_run_dir_from_metrics_cli_arg(cli_path)
    test_p = rd / "test_eval" / "metrics_test.json"
    train_p = rd / "metrics.json"
    if test_p.is_file():
        raw = json.loads(test_p.read_text(encoding="utf-8"))
        return normalize_test_eval_metrics(raw), test_p, "held-out test"
    if train_p.is_file():
        raw = json.loads(train_p.read_text(encoding="utf-8"))
        return raw, train_p, "validation-sampled"
    raise SystemExit(
        f"run 目录缺少 test_eval/metrics_test.json 与 metrics.json：{rd}"
    )


def build_extra_metrics_banner_two(prov_a: str | None, prov_b: str | None) -> str | None:
    """根据两侧数据来源生成 extra_metrics 区块说明；无 prefer-test-eval 时返回 None 用默认文案。"""
    if prov_a is None and prov_b is None:
        return None
    if prov_a == "held-out test" and prov_b == "held-out test":
        return (
            "extra_metrics（held-out test 终评，evaluate_test.py；— 表示未写入）:"
        )
    if prov_a == "validation-sampled" and prov_b == "validation-sampled":
        return (
            "extra_metrics（validation-sampled，训练结束评估；— 表示未写入）:"
        )
    return (
        "extra_metrics（两侧数据来源不一致：见报告标题；— 表示未写入）:"
    )


def build_extra_metrics_banner_three(
    p0: str | None, p1: str | None, p2: str | None
) -> str | None:
    if p0 is None and p1 is None and p2 is None:
        return None
    active = [p for p in (p0, p1, p2) if p is not None]
    if not active:
        return None
    if all(p == "held-out test" for p in active):
        return (
            "extra_metrics（held-out test 终评，evaluate_test.py；— 表示未写入）:"
        )
    if all(p == "validation-sampled" for p in active):
        return (
            "extra_metrics（validation-sampled，训练结束评估；— 表示未写入）:"
        )
    return (
        "extra_metrics（各列数据来源不一致：见报告标题；— 表示未写入）:"
    )


def build_extra_metrics_banner_single(prov: str | None) -> str | None:
    if prov is None:
        return None
    if prov == "held-out test":
        return (
            "extra_metrics（held-out test 终评，evaluate_test.py；— 表示未写入）:"
        )
    return (
        "extra_metrics（validation-sampled，训练结束评估；— 表示未写入）:"
    )


def bleu_metric_label(prov_a: str | None, prov_b: str | None) -> str:
    """prefer-test-eval 开启时结论行所用 BLEU 称谓。"""
    if prov_a is None:
        return "验证 BLEU"
    if prov_a == prov_b == "held-out test":
        return "held-out test BLEU"
    if prov_a == prov_b == "validation-sampled":
        return "validation-sampled BLEU"
    return "BLEU（两侧数据来源不同，见标题）"


def bleu_metric_label_three(
    p0: str | None, p1: str | None, p2: str | None
) -> str:
    if p0 is None:
        return "验证 BLEU"
    active = [p for p in (p0, p1, p2) if p is not None]
    if not active:
        return "验证 BLEU"
    if all(p == "held-out test" for p in active):
        return "held-out test BLEU"
    if all(p == "validation-sampled" for p in active):
        return "validation-sampled BLEU"
    return "BLEU（各列数据来源可能不同，见标题）"

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
    extra_metrics_banner: str | None = None,
) -> None:
    """在 lines 末尾追加 extra_metrics 对照行（两列）。"""
    lines.append("")
    lines.append(
        extra_metrics_banner
        if extra_metrics_banner is not None
        else "extra_metrics（训练结束时的最终评估；— 表示未写入或依赖未安装）:"
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
    extra_metrics_banner: str | None = None,
) -> None:
    lines.append("")
    lines.append(
        extra_metrics_banner
        if extra_metrics_banner is not None
        else "extra_metrics（训练结束时的最终评估；— 表示未写入或依赖未安装）:"
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
    extra_metrics_banner: str | None = None,
) -> None:
    """仅一次实验时的 extra_metrics 列。"""
    lines.append("")
    lines.append(
        extra_metrics_banner
        if extra_metrics_banner is not None
        else "extra_metrics（最终评估；— 表示未写入或依赖未安装）:"
    )
    lines.append(f"{'':<{label_w}} {col:>{w}}")
    lines.append("-" * (label_w + 1 + w))
    ex = extra_metrics_dict(m)
    for key in EXTRA_KEYS:
        lines.append(f"{EXTRA_LABELS[key]:<{label_w}} {fmt_val(ex.get(key)):>{w}}")
