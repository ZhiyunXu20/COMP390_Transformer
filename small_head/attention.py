"""可切换的注意力实现：缩放点积、加性，以及若干科学目的的变体。

`local_window` / `global_local`（配置名与下文类名沿用枚举）：实现均为 **稠密 masked attention**——
对完整 ``L×L`` 打分矩阵做 softmax，仅通过 **加性结构掩码**（禁止位置 logits → −∞）体现局部带 /
全局锚点先验。**计算与存储复杂度与标准全连接自注意力同阶**（``O(L^2)`` 量级），并 **不包含**
Longformer / BigBird / ETC 等论文中的 **块稀疏内核或未物化的稀疏矩阵乘法**；若将来引入真正的
稀疏核实现，须在命名上与这里的 dense-mask 变体区分。

同一接口：forward(q, k, v, attn_mask=None) -> (out, attn)，其中 q,k,v 为 (B, H, L, Dk)。
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from small_shared.attention_ops import guard_norm_all_masked_rows, safe_masked_softmax
from small_head.config import AttentionType, Config

# sparsemax / entmax：优先使用 pip 包 entmax（含 Sparsemax、entmax_bisect）。
try:
    from entmax import entmax_bisect as _entmax_bisect
    from entmax import sparsemax as _sparsemax_entmax

    _HAS_ENTMAX = True
except ImportError:
    _HAS_ENTMAX = False
    _entmax_bisect = None  # type: ignore[misc, assignment]
    _sparsemax_entmax = None  # type: ignore[misc, assignment]


def _sparsemax_naive(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """sparsemax(z)：投影到概率单纯形，梯度稀疏；与 Martinez et al. (2016) 一致。

    数学：p_i = max(z_i - τ, 0)，其中 τ 由 sum_i p_i = 1 唯一确定。
    实现：对排序后的 z 求阈值 τ = (cumsum_{j≤k*} z_(j) - 1) / k*。
    """
    if dim < 0:
        dim += logits.ndim
    max_val = torch.amax(logits, dim=dim, keepdim=True)
    z = logits - max_val
    zs = torch.sort(z, descending=True, dim=dim)[0]
    d = z.size(dim)
    rng = torch.arange(1, d + 1, dtype=z.dtype, device=z.device)
    view = [1] * z.ndim
    view[dim] = d
    rng = rng.view(view)
    bound = 1 + rng * zs
    cumsum = torch.cumsum(zs, dim=dim)
    # k* = |{ j : 1 + j z_(j) > cumsum_j }|（与常用 vectorized 稀疏锥投影一致）
    k_star = (bound > cumsum).sum(dim=dim, keepdim=True).clamp(min=1, max=d)
    cum_prev = torch.gather(cumsum, dim, k_star - 1)
    tau = (cum_prev - 1) / k_star.to(logits.dtype)
    out = torch.relu(z - tau)
    return out


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _HAS_ENTMAX and _sparsemax_entmax is not None:
        return _sparsemax_entmax(logits, dim=dim)
    return _sparsemax_naive(logits, dim=dim)


def _entmax15(
    logits: torch.Tensor,
    dim: int = -1,
    *,
    allow_fallback: bool = False,
) -> torch.Tensor:
    """α=1.5 entmax. Raises if entmax package missing unless allow_fallback=True.

    Fallback uses sparsemax_naive — NOT entmax15 (testing / explicit opt-in only).
    """
    if _HAS_ENTMAX and _entmax_bisect is not None:
        return _entmax_bisect(logits, alpha=1.5, dim=dim)
    if allow_fallback:
        warnings.warn(
            "entmax package unavailable; falling back to sparsemax_naive. "
            "Results will NOT be entmax15. Set attention_type='sparsemax' "
            "if this is intended, or install: pip install entmax>=0.1",
            RuntimeWarning,
            stacklevel=2,
        )
        return _sparsemax_naive(logits, dim=dim)
    raise RuntimeError(
        "attention_type='entmax15' requires the entmax package. "
        "Install: pip install entmax>=0.1 . "
        "If you really want sparsemax-like fallback for testing, "
        "use _entmax15(..., allow_fallback=True) explicitly."
    )


AttnLayerKind = Literal["encoder_self", "decoder_self", "cross"]


def _structural_local_window(
    lq: int,
    lk: int,
    window_size: int,
    attn_layer: AttnLayerKind,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构造 **稠密注意力** 用的加性掩码（dense masked attention）：禁止位置 −∞，允许 0。

    与「稀疏注意力算法」无关——上层仍会形成完整 ``Q K^T`` 并经 softmax；掩码只是把 logits 加到大负数。
    形状 ``(Lq, Lk)``，广播到 scores。

    encoder_self：|i−j| ≤ w（双向局部）。
    decoder_self：因果且 i−j ≤ w（仅看过去 window 内）。
    cross：不对齐，仅用 padding；此处不加结构掩码（全 0）。
    """
    neg = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    mask = torch.full((lq, lk), float("-inf"), device=device, dtype=dtype)
    w = max(0, int(window_size))
    ii = torch.arange(lq, device=device).unsqueeze(1)
    jj = torch.arange(lk, device=device).unsqueeze(0)

    if attn_layer == "cross":
        return torch.zeros(lq, lk, device=device, dtype=dtype)

    if attn_layer == "encoder_self":
        ok = (ii - jj).abs() <= w
        mask = torch.where(ok, zero, neg)
        return mask

    # decoder_self
    ok = (jj <= ii) & ((ii - jj) <= w)
    mask = torch.where(ok, zero, neg)
    return mask


def _structural_global_local(
    lq: int,
    lk: int,
    window_size: int,
    global_tokens: int,
    attn_layer: AttnLayerKind,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构造 **全局锚点 + 局部带** 的稠密掩码（dense masked attention），命名描述结构先验而非稀疏内核。

    前 G 个 key 为全局锚点（所有 query 可见）；其余键位满足局部窗口规则。**仍为完整稠密 QK^T + softmax**，
    不是 Longformer / ETC 类线性复杂度稀疏实现。

    Encoder：attend j 当 j < G 或 |i−j| ≤ w。
    Decoder：因果下 attend j 当 j ≤ i 且 (j < G 或 i−j ≤ w)。
    Cross：仅将前 G 个 source 位置视为全局列，其余键位无额外 band（避免错误对齐假设）。
    """
    neg = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    G = max(0, min(int(global_tokens), lk))
    w = max(0, int(window_size))
    ii = torch.arange(lq, device=device).unsqueeze(1)
    jj = torch.arange(lk, device=device).unsqueeze(0)

    global_ok = jj < G

    if attn_layer == "cross":
        ok = global_ok
        return torch.where(ok, zero, neg)

    if attn_layer == "encoder_self":
        local_ok = (ii - jj).abs() <= w
        ok = global_ok | local_ok
        return torch.where(ok, zero, neg)

    local_ok = (jj <= ii) & ((ii - jj) <= w)
    ok = global_ok | local_ok
    return torch.where(ok, zero, neg)


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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class AdditiveAttention(nn.Module):
    """score_ij = v^T tanh(Wq q_i + Wk k_j) / sqrt(d_k)。"""

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


class BilinearAttention(nn.Module):
    """score_{h,i,j} = (q_{h,i}^T W_h k_{h,j}) / sqrt(d_k)。

    每层 head 一个兼容性矩阵 W_h ∈ R^{d_k×d_k}，介于点积（固定单位阵）与纯加性之间。
    """

    def __init__(self, n_heads: int, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.W = nn.Parameter(torch.empty(n_heads, d_k, d_k))
        nn.init.xavier_uniform_(self.W.view(n_heads, -1))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # einsum: bhid,hde,bhje -> bhij
        scores = torch.einsum("bhid,hde,bhje->bhij", q, self.W, k) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class GatedDotAdditiveAttention(nn.Module):
    """score = σ(α)·dot_score + (1−σ(α))·add_score（逐元素同形状相加后 softmax）。

    dot_score：缩放点积；add_score：与 AdditiveAttention 相同形式但未单独 softmax。
    α 可为标量或可学习 per-head 标量，用于检验模型是否更偏向点积或加性路径。
    """

    def __init__(
        self,
        n_heads: int,
        d_k: int,
        d_hidden: int,
        dropout: float = 0.1,
        gate_alpha_per_head: bool = True,
    ):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(d_k, d_hidden, bias=True)
        self.Wk = nn.Linear(d_k, d_hidden, bias=True)
        self.v = nn.Linear(d_hidden, 1, bias=False)
        if gate_alpha_per_head:
            self.alpha = nn.Parameter(torch.zeros(n_heads))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, _, _ = q.shape
        dot_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        q_part = self.Wq(q).unsqueeze(3)
        k_part = self.Wk(k).unsqueeze(2)
        hidden = torch.tanh(q_part + k_part)
        add_score = self.v(hidden).squeeze(-1) / math.sqrt(self.d_k)
        if self.alpha.ndim == 1:
            gate = torch.sigmoid(self.alpha).view(1, h, 1, 1)
        else:
            gate = torch.sigmoid(self.alpha).view(1, 1, 1, 1)
        scores = gate * dot_score + (1.0 - gate) * add_score
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class LocalWindowDotAttention(nn.Module):
    """缩放点积 + **稠密**局部窗口掩码（`_structural_local_window`）。

    标准 ``matmul(Q,K^T)`` + softmax；掩码仅屏蔽不允许的键位，**不是**滑动窗口稀疏乘法或未物化注意力。
    """

    def __init__(
        self,
        d_k: int,
        window_size: int,
        attn_layer: AttnLayerKind,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_k = d_k
        self.window_size = window_size
        self.attn_layer = attn_layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lq = q.size(2)
        lk = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        struct = _structural_local_window(
            lq, lk, self.window_size, self.attn_layer, q.device, q.dtype
        )
        scores = scores + struct.unsqueeze(0).unsqueeze(0)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class GlobalLocalDotAttention(nn.Module):
    """缩放点积 + global-local **稠密**掩码（`_structural_global_local`）。

    「global/local」指允许的注意力 **模式**（锚点 + 局部带），实现仍为稠密打分矩阵。
    """

    def __init__(
        self,
        d_k: int,
        window_size: int,
        global_tokens: int,
        attn_layer: AttnLayerKind,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_k = d_k
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.attn_layer = attn_layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lq = q.size(2)
        lk = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        struct = _structural_global_local(
            lq,
            lk,
            self.window_size,
            self.global_tokens,
            self.attn_layer,
            q.device,
            q.dtype,
        )
        scores = scores + struct.unsqueeze(0).unsqueeze(0)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class SparsemaxDotAttention(nn.Module):
    """缩放点积 + sparsemax 归一化（稀疏、可解释对齐）。"""

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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = _sparsemax(scores, dim=-1)
        attn = guard_norm_all_masked_rows(scores, attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class Entmax15DotAttention(nn.Module):
    """缩放点积 + entmax_{α=1.5}（entmax 包可用时）；否则退化为 sparsemax。"""

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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = _entmax15(scores, dim=-1)
        attn = guard_norm_all_masked_rows(scores, attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


def build_core_attention(cfg: Config, attn_layer: AttnLayerKind = "encoder_self") -> nn.Module:
    """构造单层 core attention（不含 Wq,Wk,Wv,Wo）。

    `attn_layer` 仅影响 `local_window` / `global_local` 的 **稠密结构掩码**（encoder_self /
    decoder_self / cross）；二者均为 dense masked attention，而非稀疏核。
    """
    d_k = cfg.d_model // cfg.n_heads
    att: AttentionType = cfg.attention_type  # type: ignore[assignment]

    if att == "dot_product":
        return ScaledDotProductAttention(d_k, cfg.dropout)
    if att == "additive":
        return AdditiveAttention(d_k, cfg.additive_d_hidden, cfg.dropout)
    if att == "bilinear":
        return BilinearAttention(cfg.n_heads, d_k, cfg.dropout)
    if att == "gated_dot_additive":
        return GatedDotAdditiveAttention(
            cfg.n_heads,
            d_k,
            cfg.additive_d_hidden,
            cfg.dropout,
            gate_alpha_per_head=cfg.gate_alpha_per_head,
        )
    if att == "local_window":
        return LocalWindowDotAttention(d_k, cfg.local_window_size, attn_layer, cfg.dropout)
    if att == "global_local":
        return GlobalLocalDotAttention(
            d_k,
            cfg.global_local_window_size,
            cfg.global_tokens,
            attn_layer,
            cfg.dropout,
        )
    if att == "sparsemax":
        return SparsemaxDotAttention(d_k, cfg.dropout)
    if att == "entmax15":
        return Entmax15DotAttention(d_k, cfg.dropout)
    raise ValueError(f"Unknown attention_type: {cfg.attention_type}")


def core_attention_param_count(cfg: Config, attn_layer: AttnLayerKind = "encoder_self") -> int:
    """Core attention 模块的参数量（不含 MultiHeadAttention 的线性投影）。"""
    m = build_core_attention(cfg, attn_layer)
    n = sum(p.numel() for p in m.parameters())
    return n


# --- 相对训练步耗时（相对 dot_product，同 batch / 设备；仅经验量级，用于实验记录）---
# bilinear：约 1.1–1.3×（head 级 W 的额外 einsum）
# gated_dot_additive：约 1.8–2.3×（双路分数）
# local_window / global_local：约 0.95–1.1×（仍为 **完整稠密** matmul；掩码仅为 logits 加法，无稀疏加速）
# sparsemax：约 1.1–1.2×（若用 entmax 自带 CUDA）；朴素实现可能更慢
# entmax15：约 1.2–1.5×（bisect）；退化为 sparsemax 时同 sparsemax


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: Config, *, attn_layer: AttnLayerKind = "encoder_self"):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        d_k = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn = build_core_attention(cfg, attn_layer)
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
