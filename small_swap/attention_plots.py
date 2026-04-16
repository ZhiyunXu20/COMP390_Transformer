"""解码器交叉注意力可视化：多头网格 + 头平均图，供 W&B / 报告使用。"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tokenizers import Tokenizer


def _id_label(tok: Tokenizer, i: int, max_len: int = 14) -> str:
    t = tok.id_to_token(i)
    if t is None:
        return str(i)
    s = t.replace("Ġ", "·")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _trim_lengths(
    src_ids: Sequence[int],
    tgt_ids: Sequence[int],
    pad_idx: int,
) -> tuple[int, int]:
    """非 PAD 的有效长度（用于裁掉热力图空白）。"""
    def eff_len(ids: Sequence[int]) -> int:
        n = len(ids)
        while n > 0 and ids[n - 1] == pad_idx:
            n -= 1
        return max(n, 1)

    return eff_len(list(src_ids)), eff_len(list(tgt_ids))


def figure_cross_attention_heads(
    cross_attn: torch.Tensor,
    src_ids: Sequence[int],
    tgt_ids: Sequence[int],
    src_tok: Tokenizer,
    tgt_tok: Tokenizer,
    pad_idx: int,
    *,
    layer_idx: int,
    title_prefix: str = "",
) -> plt.Figure:
    """
    cross_attn: (H, Lt, Ls) 已取 batch=0，float32 CPU
    行：解码器位置（tgt_in）；列：编码器位置（源端）
    """
    cross_attn = cross_attn.detach().float().cpu().numpy()
    h, lt, ls = cross_attn.shape

    ls_eff, lt_eff = _trim_lengths(src_ids, tgt_ids, pad_idx)
    ls_eff = min(ls_eff, cross_attn.shape[2])
    lt_eff = min(lt_eff, cross_attn.shape[1])
    mat = cross_attn[:, :lt_eff, :ls_eff]

    xlabels = [_id_label(src_tok, int(src_ids[j])) for j in range(ls_eff)]
    ylabels = [_id_label(tgt_tok, int(tgt_ids[i])) for i in range(lt_eff)]

    ncols = min(4, h)
    nrows = int(math.ceil(h / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    for hi in range(h):
        r, c = divmod(hi, ncols)
        ax = axes[r][c]
        sns.heatmap(
            mat[hi],
            xticklabels=xlabels if lt_eff <= 24 else False,
            yticklabels=ylabels if lt_eff <= 24 else False,
            cmap="viridis",
            ax=ax,
            cbar=True,
            square=False,
        )
        ax.set_title(f"Head {hi}")
        if lt_eff <= 24:
            ax.set_xlabel("Source (encoder keys)")
            ax.set_ylabel("Target query pos.")
    # 隐藏多余子图
    for hi in range(h, nrows * ncols):
        r, c = divmod(hi, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"{title_prefix} Decoder cross-attention (layer {layer_idx})  [H={h}, tgt×src={lt_eff}×{ls_eff}]",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def figure_cross_attention_mean(
    cross_attn: torch.Tensor,
    src_ids: Sequence[int],
    tgt_ids: Sequence[int],
    src_tok: Tokenizer,
    tgt_tok: Tokenizer,
    pad_idx: int,
    *,
    layer_idx: int,
    title_prefix: str = "",
) -> plt.Figure:
    """所有头平均后的对齐矩阵，便于报告单图展示。"""
    mean = cross_attn.detach().float().mean(dim=0).cpu().numpy()
    ls_eff, lt_eff = _trim_lengths(src_ids, tgt_ids, pad_idx)
    ls_eff = min(ls_eff, mean.shape[1])
    lt_eff = min(lt_eff, mean.shape[0])
    mat = mean[:lt_eff, :ls_eff]

    xlabels = [_id_label(src_tok, int(src_ids[j])) for j in range(ls_eff)]
    ylabels = [_id_label(tgt_tok, int(tgt_ids[i])) for i in range(lt_eff)]

    fig, ax = plt.subplots(figsize=(min(14, max(8, ls_eff * 0.35)), min(12, max(6, lt_eff * 0.35))))
    sns.heatmap(
        mat,
        xticklabels=xlabels if ls_eff <= 32 else False,
        yticklabels=ylabels if lt_eff <= 32 else False,
        cmap="magma",
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Source (French) token position")
    ax.set_ylabel("Target (English) query position")
    ax.set_title(
        f"{title_prefix} Cross-attention mean over heads — layer {layer_idx}",
    )
    fig.tight_layout()
    return fig
