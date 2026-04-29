#!/usr/bin/env python3
"""在 test 句对上抽取 decoder cross-attention：熵、头相似度、热力图、top source 位置。

用于对比 single-head vs multi-head 是否学到不同对齐模式。

用法：
  python scripts/analyze_attention.py --checkpoint small_try/runs/foo/best.pt \\
    --pkg small_try --split test --max-examples 32 --output-dir reports/attn_foo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ablation_lib import cfg_from_checkpoint_dict, import_training_stack, set_seed  # noqa: E402


def _entropy_cross_attn(attn: torch.Tensor) -> tuple[list[float], float]:
    """attn [B,H,Lt,Ls]：对每个 query 在 key 维上计算 Shannon 熵，再对 batch、query 平均。"""
    p = attn.float().clamp_min(1e-12)
    ent_q = -(p * torch.log(p)).sum(dim=-1)  # [B,H,Lt]
    per_head = ent_q.mean(dim=(0, 2))  # [H]
    overall = float(ent_q.mean().item())
    return [float(x) for x in per_head], overall


def _head_similarity_matrix(attn_bh: torch.Tensor) -> list[list[float]]:
    """attn_bh [H,Lt,Ls]：query 平均边际 [H,Ls]，head 间余弦相似度。"""
    m = attn_bh.mean(dim=1)
    m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = F.cosine_similarity(m.unsqueeze(1), m.unsqueeze(0), dim=-1)
    return sim.cpu().tolist()


def _top_source_tokens(
    attn_bh: torch.Tensor,
    src_ids: list[int],
    src_tok: Any,
    *,
    topk: int,
    pad_idx: int,
) -> list[dict[str, Any]]:
    marginal = attn_bh.float().mean(dim=(0, 1))
    k = min(topk, marginal.numel())
    vals, idx = torch.topk(marginal, k=k)
    out: list[dict[str, Any]] = []
    for prob, j in zip(vals.tolist(), idx.tolist()):
        tid = src_ids[j] if j < len(src_ids) else pad_idx
        tok_s = src_tok.id_to_token(int(tid))
        if tok_s is None:
            tok_s = str(tid)
        tok_s = tok_s.replace("Ġ", "·")
        out.append({"src_position": int(j), "token_id": int(tid), "token": tok_s, "mass": float(prob)})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-attention 分析（teacher forcing on split）")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pkg", type=str, default="small_try")
    p.add_argument("--split", choices=("test", "val", "train"), default="test")
    p.add_argument("--max-examples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk-src", type=int, default=12)
    p.add_argument(
        "--heatmap-layers",
        type=str,
        default="-1",
        help="热力图层索引，逗号分隔；-1=最后一层；空=不画图",
    )
    p.add_argument("--heatmap-examples", type=int, default=6)
    args = p.parse_args()

    repo = ROOT.resolve()
    from train_runtime import PATH_FIELDS_DEFAULT, materialize_path_fields  # noqa: PLC0415

    ckpt_path = args.checkpoint.resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg") or {}
    if not isinstance(cfg_dict, dict):
        raise SystemExit("checkpoint 缺少 cfg")

    Config, TabParallelDataset, collate_batch, load_tokenizers, tokenizer_special_ids, Seq2SeqTransformer = (
        import_training_stack(repo, args.pkg)
    )
    cfg = cfg_from_checkpoint_dict(cfg_dict, Config)
    materialize_path_fields(cfg, repo, PATH_FIELDS_DEFAULT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    src_tok, tgt_tok = load_tokenizers(cfg)
    pad_idx, bos_id, eos_id = tokenizer_special_ids(tgt_tok)

    ds = TabParallelDataset(cfg, src_tok, tgt_tok, args.split)
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, cfg.batch_size),
        shuffle=False,
        num_workers=min(getattr(cfg, "num_workers", 0), 4),
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_batch(b, pad_idx),
        persistent_workers=False,
    )

    model = Seq2SeqTransformer(cfg, pad_idx=pad_idx).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    pkg_dir = repo / args.pkg
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    from attention_plots import figure_cross_attention_heads, figure_cross_attention_mean  # noqa: E402

    n_layers = cfg.n_layers
    n_heads = cfg.n_heads

    sum_ent_layer = [torch.zeros(n_heads, dtype=torch.float64) for _ in range(n_layers)]
    sum_ent_overall = torch.zeros(n_layers, dtype=torch.float64)
    weight_queries = 0

    sum_sim = [torch.zeros(n_heads, n_heads, dtype=torch.float64) for _ in range(n_layers)]
    weight_sim = 0

    per_example_summaries: list[dict[str, Any]] = []

    heatmap_layers: list[int] = []
    if args.heatmap_layers.strip():
        raw = [int(x.strip()) for x in args.heatmap_layers.split(",") if x.strip()]
        heatmap_layers = [n_layers - 1 if x == -1 else x for x in raw]

    out_dir = args.output_dir.resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    heatmaps_drawn = 0

    with torch.no_grad():
        for src, tgt in loader:
            if global_idx >= args.max_examples:
                break
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            b, lt1 = tgt_in.shape
            logits, pack = model(src, tgt_in, output_attentions=True)
            assert isinstance(pack, dict)
            dec_cross: list[torch.Tensor] = pack["decoder_cross"]

            if logits.numel() == 0:
                continue

            nq = b * lt1
            for li, attn_l in enumerate(dec_cross):
                p = attn_l.float().clamp_min(1e-12)
                ent_q = -(p * torch.log(p)).sum(dim=-1)
                sum_ent_layer[li] += ent_q.sum(dim=(0, 2)).cpu().double()
                sum_ent_overall[li] += ent_q.sum().cpu().double()
            weight_queries += nq

            for bi in range(b):
                if global_idx >= args.max_examples:
                    break
                src_ids = [int(x) for x in src[bi].tolist()]
                tgt_row = [int(x) for x in tgt_in[bi].tolist()]
                ex_cross = [dec_cross[li][bi : bi + 1].squeeze(0) for li in range(n_layers)]

                for li in range(n_layers):
                    sim_mat = torch.tensor(
                        _head_similarity_matrix(ex_cross[li]),
                        dtype=torch.float64,
                        device="cpu",
                    )
                    sum_sim[li] += sim_mat
                weight_sim += 1

                layers_payload: dict[str, Any] = {}
                for li in range(n_layers):
                    attn_bh = ex_cross[li]
                    layers_payload[str(li)] = {
                        "entropy_per_head": _entropy_cross_attn(attn_bh.unsqueeze(0))[0],
                        "head_similarity": _head_similarity_matrix(attn_bh),
                        "top_src_tokens_marginal": _top_source_tokens(
                            attn_bh,
                            src_ids,
                            src_tok,
                            topk=args.topk_src,
                            pad_idx=pad_idx,
                        ),
                    }

                per_example_summaries.append(
                    {
                        "index": global_idx,
                        "layers": layers_payload,
                    }
                )

                if heatmaps_drawn < args.heatmap_examples and heatmap_layers:
                    for li in heatmap_layers:
                        if li < 0 or li >= n_layers:
                            continue
                        ac = dec_cross[li][bi]
                        fig_h = figure_cross_attention_heads(
                            ac,
                            src_ids,
                            tgt_row,
                            src_tok,
                            tgt_tok,
                            pad_idx,
                            layer_idx=li,
                            title_prefix=f"ex{global_idx}",
                        )
                        p_cross = fig_dir / f"cross_layer{li}_ex{global_idx}_heads.png"
                        fig_h.savefig(p_cross, dpi=120, bbox_inches="tight")
                        plt.close(fig_h)

                        fig_m = figure_cross_attention_mean(
                            ac,
                            src_ids,
                            tgt_row,
                            src_tok,
                            tgt_tok,
                            pad_idx,
                            layer_idx=li,
                            title_prefix=f"ex{global_idx}",
                        )
                        p_mean = fig_dir / f"cross_layer{li}_ex{global_idx}_mean.png"
                        fig_m.savefig(p_mean, dpi=120, bbox_inches="tight")
                        plt.close(fig_m)

                    heatmaps_drawn += 1

                global_idx += 1

    wq = float(max(1.0, weight_queries))
    mean_ent_layer = [
        [float(sum_ent_layer[li][h] / wq) for h in range(n_heads)] for li in range(n_layers)
    ]
    mean_ent_overall = [
        float(sum_ent_overall[li].item() / (wq * max(1, n_heads))) for li in range(n_layers)
    ]

    mean_sim = [(sum_sim[li] / max(1, weight_sim)).tolist() for li in range(n_layers)]

    report: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "pkg": args.pkg,
        "split": args.split,
        "n_examples_used": global_idx,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "attention_type": getattr(cfg, "attention_type", None),
        "entropy_cross_attn": {
            "per_layer_mean_over_heads": mean_ent_overall,
            "per_layer_per_head_mean": mean_ent_layer,
            "weight_queries": weight_queries,
        },
        "head_similarity_cross_attn": {
            "mean_cosine_matrix_per_layer": mean_sim,
            "n_examples_averaged": weight_sim,
            "note": "对每条样本算 head×head 余弦相似度，再在样本上平均（边际分布为对 decoder query 平均）。",
        },
        "per_example": per_example_summaries[: min(50, len(per_example_summaries))],
        "figures_dir": str(fig_dir),
    }

    json_path = out_dir / "attention_analysis.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Attention analysis (decoder cross-attention)",
        "",
        f"- Checkpoint: `{ckpt_path}`",
        f"- Package: **{args.pkg}**, split: **{args.split}**, examples: **{global_idx}**",
        f"- Heads: **{n_heads}**, layers: **{n_layers}**, attention_type: `{report.get('attention_type')}`",
        "",
        "## Mean entropy (cross-attn, averaged over queries & batch chunks)",
        "",
        "| layer | mean H (nat) |",
        "| --- | ---: |",
    ]
    for li, v in enumerate(mean_ent_overall):
        md_lines.append(f"| {li} | {v:.6f} |")
    md_lines.extend(["", "### Per layer, per head", ""])
    for li, row in enumerate(mean_ent_layer):
        md_lines.append(f"- Layer {li}: " + ", ".join(f"h{j}={x:.4f}" for j, x in enumerate(row)))
    md_lines.extend(["", "## Mean head similarity (cosine of marginal source distributions)", ""])
    for li, mat in enumerate(mean_sim):
        md_lines.append(f"### Layer {li}")
        md_lines.append("```")
        for r in mat:
            md_lines.append("  " + "  ".join(f"{x:+.3f}" for x in r))
        md_lines.append("```")

    md_lines.extend(
        [
            "",
            "## Top source tokens",
            "",
            "详见 JSON `per_example` → `layers` → `top_src_tokens_marginal`（边际对 decoder query 平均）。",
            "",
            f"## Figures (`{fig_dir}`)",
            "",
            "热力图：`cross_layer*_ex*_heads.png`（分头），`*_mean.png`（头平均）。",
        ]
    )

    md_path = out_dir / "attention_analysis.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
