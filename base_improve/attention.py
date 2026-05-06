"""可切换的注意力打分：缩放点积 vs 加性（Bahdanau 风格，全序列向量化）。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import AttentionType, Config
from small_shared.attention_ops import safe_masked_softmax


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k,v: (B, H, L, Dk)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class AdditiveAttention(nn.Module):
    """score_ij = w^T tanh(Wq q_i + Wk k_j)，与点积可比、支持 (B,H,Lq,Lk) 掩码。"""

    def __init__(self, d_k: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(d_k, d_hidden, bias=True)
        self.Wk = nn.Linear(d_k, d_hidden, bias=True)
        self.v = nn.Linear(d_hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B,H,Lq,1,Hid) + (B,H,1,Lk,Hid) -> (B,H,Lq,Lk,Hid)
        q_part = self.Wq(q).unsqueeze(3)
        k_part = self.Wk(k).unsqueeze(2)
        hidden = torch.tanh(q_part + k_part)
        scores = self.v(hidden).squeeze(-1) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


def build_core_attention(cfg: Config) -> nn.Module:
    d_k = cfg.d_model // cfg.n_heads
    if cfg.attention_type == "dot_product":
        return ScaledDotProductAttention(d_k, cfg.dropout)
    if cfg.attention_type == "additive":
        return AdditiveAttention(d_k, cfg.additive_d_hidden, cfg.dropout)
    raise ValueError(f"Unknown attention_type: {cfg.attention_type}")


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        d_k = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn = build_core_attention(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, lq, _ = q.shape
        lk = k.size(1)

        q = self.w_q(q).view(b, lq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(b, lk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(b, lk, self.n_heads, self.d_k).transpose(1, 2)

        out, attn = self.attn(q, k, v, attn_mask)

        out = out.transpose(1, 2).contiguous().view(b, lq, self.d_model)
        return self.w_o(out), attn


class PositionwiseFFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])
