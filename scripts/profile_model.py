#!/usr/bin/env python3
"""
比较 dot_product vs additive、不同 n_heads 下的参数量与前向耗时（small_try Seq2SeqTransformer）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
_SMALL_TRY = REPO_ROOT / "small_try"
if str(_SMALL_TRY) not in sys.path:
    sys.path.insert(0, str(_SMALL_TRY))

from attention import MultiHeadAttention  # noqa: E402
from config import Config  # noqa: E402
from model import Seq2SeqTransformer  # noqa: E402


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_all(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def multi_head_attention_param_counts(model: Seq2SeqTransformer) -> tuple[int, int]:
    """
    返回 (全部 MultiHeadAttention 块参数总和, 仅 scoring 子模块 self.attn 参数总和)。
    前者含 Wq,Wk,Wv,Wo + core；后者为 build_core_attention 产出（dot_product 时常为 0）。
    """
    block_total = 0
    scoring_total = 0
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            block_total += count_all(m)
            scoring_total += count_all(m.attn)
    return block_total, scoring_total


def build_cfg(
    *,
    attention_type: str,
    n_heads: int,
    d_model: int,
    n_layers: int,
    seq_len: int,
    d_ff: int | None,
    vocab_size: int,
) -> Config:
    cfg = Config()
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
    cfg.attention_type = attention_type  # type: ignore[assignment]
    cfg.n_heads = n_heads
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.d_ff = (4 * d_model) if d_ff is None else d_ff
    cfg.max_seq_len = max(seq_len + 8, 96)
    cfg.src_vocab_size = vocab_size
    cfg.tgt_vocab_size = vocab_size
    return cfg


def profile_one(
    cfg: Config,
    *,
    batch_size: int,
    seq_len: int,
    pad_idx: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model = Seq2SeqTransformer(cfg, pad_idx=pad_idx).to(device)
    model.eval()
    vs = cfg.src_vocab_size
    # 避开 pad_idx，减少 mask 全 padding 的退化情况
    hi = max(2, vs - 1)
    src = torch.randint(1, hi, (batch_size, seq_len), device=device)
    tgt = torch.randint(1, hi, (batch_size, seq_len), device=device)

    total_p = count_all(model)
    train_p = count_trainable(model)
    mha_p, score_p = multi_head_attention_param_counts(model)

    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(src, tgt)
        if use_cuda:
            torch.cuda.synchronize()

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(repeats):
            model(src, tgt)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed = t1 - t0
    forward_ms = (elapsed / repeats) * 1000.0
    tokens_per_forward = 2 * batch_size * seq_len
    tps = (tokens_per_forward * repeats) / elapsed if elapsed > 0 else float("nan")

    peak_mb: float | None = None
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024.0**2)

    return {
        "parameter_count": total_p,
        "trainable_parameter_count": train_p,
        "attention_module_parameter_count": mha_p,
        "attention_scoring_parameter_count": score_p,
        "forward_time_ms": forward_ms,
        "tokens_per_second": tps,
        "peak_gpu_memory_mb": peak_mb,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Attention 配置效率与复杂度 profiling（small_try）")
    p.add_argument("--attention-types", nargs="+", default=["dot_product", "additive"])
    p.add_argument("--n-heads-list", nargs="+", type=int, default=[1, 4])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=None, help="默认 4 × d_model")
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu；默认自动")
    p.add_argument("--pad-index", type=int, default=0)
    p.add_argument(
        "--output-md",
        type=str,
        default=str(REPO_ROOT / "results" / "profile_summary.md"),
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "results" / "profile_summary.json"),
    )
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    for att in args.attention_types:
        for nh in args.n_heads_list:
            cfg = build_cfg(
                attention_type=att,
                n_heads=nh,
                d_model=args.d_model,
                n_layers=args.n_layers,
                seq_len=args.seq_len,
                d_ff=args.d_ff,
                vocab_size=args.vocab_size,
            )
            row = {
                "attention_type": att,
                "n_heads": nh,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "device": str(device),
                "warmup": args.warmup,
                "repeats": args.repeats,
            }
            stats = profile_one(
                cfg,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                pad_idx=args.pad_index,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            row.update(stats)
            rows.append(row)

    out_md = Path(args.output_md).expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Attention 效率与复杂度 Profile（`small_try`）",
        "",
        "## 设定",
        "",
        f"- batch_size={args.batch_size}, seq_len={args.seq_len}, d_model={args.d_model}, n_layers={args.n_layers}",
        f"- vocab_size={args.vocab_size}, device={device}",
        f"- warmup={args.warmup}, timed repeats={args.repeats}",
        "- **attention_module_parameter_count**：每个 `MultiHeadAttention`（含 Wq/Wk/Wv/Wo + dropout 占位 + **core scoring `attn`**）参数之和（所有 encoder self-attn + decoder self-attn + decoder cross-attn）。",
        "- **attention_scoring_parameter_count**：仅 **core attention 打分模块** `MultiHeadAttention.attn`（`build_core_attention`）；dot_product 下通常为 **0**（缩放点积无额外可学习参数）。",
        "",
        "## 汇总表",
        "",
        "| attention | n_heads | params | trainable | attn block params | scoring-only params | forward ms | tok/s | peak GPU MiB |",
        "|-----------|---------|--------|-----------|-------------------|---------------------|------------|-------|--------------|",
    ]

    for r in rows:
        peak = r["peak_gpu_memory_mb"]
        peak_s = f"{peak:.2f}" if peak is not None else "n/a (CPU)"
        lines.append(
            f"| {r['attention_type']} | {r['n_heads']} | {r['parameter_count']} | "
            f"{r['trainable_parameter_count']} | {r['attention_module_parameter_count']} | "
            f"{r['attention_scoring_parameter_count']} | {r['forward_time_ms']:.4f} | "
            f"{r['tokens_per_second']:.2f} | {peak_s} |"
        )

    lines.extend(
        [
            "",
            "### 说明",
            "",
            "- **forward_time_ms**：单次 `forward(src,tgt)` 平均耗时（毫秒）。",
            "- **tokens_per_second**：按每步处理 `2 × batch × seq_len`（源 + 目标）token 估算吞吐。",
            "- 若需与论文可比，请固定 CUDA/cuDNN 版本并在独占 GPU 上重复测量。",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    out_js = Path(args.output_json).expanduser().resolve()
    out_js.parent.mkdir(parents=True, exist_ok=True)
    out_js.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_md}", file=sys.stderr)
    print(f"Wrote {out_js}", file=sys.stderr)


if __name__ == "__main__":
    main()
