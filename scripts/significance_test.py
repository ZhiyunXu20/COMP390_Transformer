#!/usr/bin/env python3
"""
对同一 test set 上两份 predictions.jsonl 做配对 bootstrap（sentence-level 同步重采样），
估计 BLEU / chrF 分数差的抽样分布、p-value 近似与置信区间。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def display_path_under_repo(path: Path) -> str:
    """写入报告时使用相对仓库根的路径，便于 clone 后在任意机器上阅读。"""
    rp = path.resolve()
    root = REPO_ROOT.resolve()
    try:
        return rp.relative_to(root).as_posix()
    except ValueError:
        return rp.as_posix()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

try:
    import sacrebleu  # noqa: F401
except ImportError:
    sacrebleu = None


def corpus_bleu(hyps: list[str], refs: list[str], *, quiet: bool = False) -> float:
    """quiet：屏蔽 SacreBLEU 在 bootstrap 内重复打印的 detokenize 提示（stderr）。"""
    from sacrebleu.metrics import BLEU

    def _run() -> float:
        return float(BLEU().corpus_score(hyps, [refs]).score)

    if quiet:
        with contextlib.redirect_stderr(io.StringIO()):
            return _run()
    return _run()


def corpus_chrf_word_order_0(hyps: list[str], refs: list[str], *, quiet: bool = False) -> float:
    from sacrebleu.metrics import CHRF

    def _run() -> float:
        return float(CHRF(word_order=0).corpus_score(hyps, [refs]).score)

    if quiet:
        with contextlib.redirect_stderr(io.StringIO()):
            return _run()
    return _run()


def load_predictions_jsonl(path: Path) -> tuple[list[str], list[str], list[str]]:
    srcs: list[str] = []
    refs: list[str] = []
    hyps: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            srcs.append(row.get("source", ""))
            refs.append(row["reference"])
            hyps.append(row["hypothesis"])
    return srcs, refs, hyps


def paired_bootstrap_both_metrics(
    hyp_a: list[str],
    hyp_b: list[str],
    refs: list[str],
    *,
    n_draws: int,
    rng: random.Random,
) -> tuple[float, float, float, float, list[float], list[float]]:
    """返回 (bleu_a, bleu_b, chrf_a, chrf_b, boot_delta_bleu, boot_delta_chrf)，delta = A−B。"""
    n = len(refs)
    if not (len(hyp_a) == len(hyp_b) == n):
        raise ValueError("hyp_a, hyp_b, refs 长度必须一致")

    ba = corpus_bleu(hyp_a, refs)
    bb = corpus_bleu(hyp_b, refs)
    ca = corpus_chrf_word_order_0(hyp_a, refs)
    cb = corpus_chrf_word_order_0(hyp_b, refs)

    boot_bleu: list[float] = []
    boot_chrf: list[float] = []
    iterator = range(n_draws)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="paired bootstrap", leave=False)

    for _ in iterator:
        idx = [rng.randrange(n) for _ in range(n)]
        ha = [hyp_a[i] for i in idx]
        hb = [hyp_b[i] for i in idx]
        rs = [refs[i] for i in idx]
        boot_bleu.append(corpus_bleu(ha, rs, quiet=True) - corpus_bleu(hb, rs, quiet=True))
        boot_chrf.append(
            corpus_chrf_word_order_0(ha, rs, quiet=True) - corpus_chrf_word_order_0(hb, rs, quiet=True)
        )

    return ba, bb, ca, cb, boot_bleu, boot_chrf


def percentile_ci(xs: list[float], alpha: float = 0.05) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    x = sorted(xs)
    n = len(x)
    lo_idx = max(0, min(n - 1, int(round((alpha / 2) * (n - 1)))))
    hi_idx = max(0, min(n - 1, int(round((1 - alpha / 2) * (n - 1)))))
    return x[lo_idx], x[hi_idx]


def two_sided_p_value_approx(_obs_delta: float, boot_deltas: list[float]) -> float:
    """Bootstrap 启发式双侧 p-value：p ≈ 2 * min(P(boot ≤ 0), P(boot ≥ 0))。"""
    if not boot_deltas:
        return float("nan")
    bs = boot_deltas
    prop_le0 = sum(1 for x in bs if x <= 0) / len(bs)
    prop_ge0 = sum(1 for x in bs if x >= 0) / len(bs)
    p = 2.0 * min(prop_le0, prop_ge0)
    return min(1.0, max(p, 0.0))


def main() -> None:
    if sacrebleu is None:
        print("需要 sacrebleu：pip install sacrebleu", file=sys.stderr)
        raise SystemExit(1)

    p = argparse.ArgumentParser(description="paired bootstrap：两份 predictions.jsonl 的 BLEU/chrF 差异显著性")
    p.add_argument("--predictions-a", type=str, required=True, help="系统 A 的 predictions.jsonl（delta = A − B）")
    p.add_argument("--predictions-b", type=str, required=True, help="系统 B")
    p.add_argument("--name-a", type=str, default="system_A")
    p.add_argument("--name-b", type=str, default="system_B")
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="bootstrap 重采样次数（越大越慢；论文常用 1000–10000）",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05, help="置信区间 (1-alpha)")
    p.add_argument(
        "--output-md",
        type=str,
        default=str(REPO_ROOT / "results" / "significance_report.md"),
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "results" / "significance_summary.json"),
    )
    args = p.parse_args()

    pa = Path(args.predictions_a).expanduser().resolve()
    pb = Path(args.predictions_b).expanduser().resolve()
    if not pa.is_file() or not pb.is_file():
        raise SystemExit("predictions 文件不存在")

    _, ref_a, hyp_a = load_predictions_jsonl(pa)
    _, ref_b, hyp_b = load_predictions_jsonl(pb)

    if len(ref_a) != len(ref_b):
        raise SystemExit(f"句数不一致: A={len(ref_a)} B={len(ref_b)}（须为同一 test set）")

    mismatches = sum(1 for i in range(len(ref_a)) if ref_a[i] != ref_b[i])
    if mismatches:
        print(
            f"警告: {mismatches}/{len(ref_a)} 行 reference 不完全一致；仍以相同行号配对 bootstrap。",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)

    s_a_bleu, s_b_bleu, s_a_c, s_b_c, boot_bleu, boot_chrf = paired_bootstrap_both_metrics(
        hyp_a,
        hyp_b,
        ref_a,
        n_draws=args.bootstrap_samples,
        rng=rng,
    )
    obs_delta_bleu = s_a_bleu - s_b_bleu
    obs_delta_c = s_a_c - s_b_c
    ci_bleu_lo, ci_bleu_hi = percentile_ci(boot_bleu, alpha=args.alpha)
    ci_c_lo, ci_c_hi = percentile_ci(boot_chrf, alpha=args.alpha)
    p_bleu = two_sided_p_value_approx(obs_delta_bleu, boot_bleu)
    p_c = two_sided_p_value_approx(obs_delta_c, boot_chrf)

    def ci_excludes_zero(ci_lo: float, ci_hi: float) -> bool:
        if math.isnan(ci_lo) or math.isnan(ci_hi):
            return False
        return (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)

    bleu_sig = ci_excludes_zero(ci_bleu_lo, ci_bleu_hi)
    chrf_sig = ci_excludes_zero(ci_c_lo, ci_c_hi)

    pa_disp = display_path_under_repo(pa)
    pb_disp = display_path_under_repo(pb)

    out_md = Path(args.output_md).expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    bleu_interp = (
        "**区间与 0 无交：** A 相对 B 的 BLEU 优势在该 bootstrap 设定下较一致。"
        if bleu_sig
        else "**区间跨过 0：** 观测 BLEU 差可能与句子级抽样噪声一致。"
    )
    chrf_interp = (
        "**区间与 0 无交：** chrF 差异方向较稳定。"
        if chrf_sig
        else "**区间跨过 0：** chrF 观测差不宜单独视为强证据。"
    )

    lines = [
        "# BLEU / chrF 配对 Bootstrap 显著性报告",
        "",
        "## 设定",
        "",
        f"- **系统 A** ({args.name_a}): `{pa_disp}`",
        f"- **系统 B** ({args.name_b}): `{pb_disp}`",
        f"- **配对句数** N = {len(ref_a)}",
        f"- **Bootstrap 次数** B = {args.bootstrap_samples}（同一套句子重采样下同时计算 BLEU 差与 chrF 差）。",
        f"- **RNG 种子** {args.seed}。",
        f"- **Δ** = score(A) − score(B)。",
        "",
        "## BLEU（SacreBLEU corpus，与评估流水线一致）",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| BLEU(A) | {s_a_bleu:.4f} |",
        f"| BLEU(B) | {s_b_bleu:.4f} |",
        f"| **Δ BLEU（观测）** | **{obs_delta_bleu:+.4f}** |",
        f"| Δ 的 bootstrap {(1 - args.alpha) * 100:.0f}% CI（百分位） | [{ci_bleu_lo:.4f}, {ci_bleu_hi:.4f}] |",
        f"| p-value（近似，双侧启发式） | {p_bleu:.4f} |",
        "",
        f"{bleu_interp}",
        "",
        "## chrF（word_order=0）",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| chrF(A) | {s_a_c:.4f} |",
        f"| chrF(B) | {s_b_c:.4f} |",
        f"| **Δ chrF（观测）** | **{obs_delta_c:+.4f}** |",
        f"| Δ 的 bootstrap {(1 - args.alpha) * 100:.0f}% CI | [{ci_c_lo:.4f}, {ci_c_hi:.4f}] |",
        f"| p-value（近似） | {p_c:.4f} |",
        "",
        f"{chrf_interp}",
        "",
        "## 汇总解读（显著 vs 随机波动）",
        "",
        "- **更可能超越噪声**：至少一种度量下 **Δ 的 CI 与 0 无交集**，且效应方向一致（同为正或同为负）。当前："
        f" BLEU [{'区间不含 0' if bleu_sig else '区间含 0'}]；chrF [{'区间不含 0' if chrf_sig else '区间含 0'}]。",
        "- **更像随机波动**：两度量 CI **均**跨过 0，或观测 Δ 很小而 CI 很宽。",
        "- **p-value**：文内数值为常见 bootstrap 启发式，**请与 CI 联合判断**；正式推断应固定假设与多重比较校正（若多次两两对比）。",
        "",
        "---",
        "",
        "*输入须为 `evaluate_test.py` 生成的完整 `predictions.jsonl`（逐句 hypothesis/reference）。*",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    summary: dict[str, Any] = {
        "predictions_a": pa_disp,
        "predictions_b": pb_disp,
        "name_a": args.name_a,
        "name_b": args.name_b,
        "n_sentences": len(ref_a),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "bleu": {
            "score_a": s_a_bleu,
            "score_b": s_b_bleu,
            "delta_observed": obs_delta_bleu,
            "ci_low": ci_bleu_lo,
            "ci_high": ci_bleu_hi,
            "p_value_approx_two_sided": p_bleu,
            "ci_excludes_zero": bleu_sig,
        },
        "chrf": {
            "score_a": s_a_c,
            "score_b": s_b_c,
            "delta_observed": obs_delta_c,
            "ci_low": ci_c_lo,
            "ci_high": ci_c_hi,
            "p_value_approx_two_sided": p_c,
            "ci_excludes_zero": chrf_sig,
        },
        "reference_mismatch_lines": mismatches,
    }
    out_js = Path(args.output_json).expanduser().resolve()
    out_js.parent.mkdir(parents=True, exist_ok=True)
    out_js.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_md}", file=sys.stderr)
    print(f"Wrote {out_js}", file=sys.stderr)


if __name__ == "__main__":
    main()
