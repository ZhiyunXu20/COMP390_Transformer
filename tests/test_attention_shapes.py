"""small_try.attention：所有 attention_type 输出形状一致；padding 掩码处权重接近 0。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_SMALL_TRY = ROOT / "small_try"
if str(_SMALL_TRY) not in sys.path:
    sys.path.insert(0, str(_SMALL_TRY))

from attention import build_core_attention, core_attention_param_count  # noqa: E402
from config import Config  # noqa: E402


# 与 config.AttentionType 同步
ATTENTION_TYPES_ALL = (
    "dot_product",
    "additive",
    "bilinear",
    "gated_dot_additive",
    "local_window",
    "global_local",
    "sparsemax",
    "entmax15",
)


@pytest.fixture
def tiny_cfg() -> Config:
    cfg = Config()
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.dropout = 0.0
    cfg.additive_d_hidden = 32
    cfg.local_window_size = 2
    cfg.global_local_window_size = 2
    cfg.global_tokens = 2
    return cfg


def test_all_types_same_shapes(tiny_cfg: Config) -> None:
    B, H, L, Dk = 2, tiny_cfg.n_heads, 7, tiny_cfg.d_model // tiny_cfg.n_heads
    q = torch.randn(B, H, L, Dk)
    k = torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dk)

    ref_out = ref_attn = None
    for att in ATTENTION_TYPES_ALL:
        tiny_cfg.attention_type = att  # type: ignore[assignment]
        mod = build_core_attention(tiny_cfg, attn_layer="encoder_self")
        mod.eval()
        with torch.no_grad():
            out, attn = mod(q, k, v, attn_mask=None)
        assert out.shape == (B, H, L, Dk)
        assert attn.shape == (B, H, L, L)
        if ref_out is None:
            ref_out, ref_attn = out.shape, attn.shape
        else:
            assert out.shape == ref_out
            assert attn.shape == ref_attn


def test_padding_mask_zeros_mass(tiny_cfg: Config) -> None:
    """encoder 风格 (B,1,1,Lk)：最后一列为 PAD（-inf），对应 key 列 attention 质量应接近 0。"""
    B, H, L, Dk = 1, tiny_cfg.n_heads, 5, tiny_cfg.d_model // tiny_cfg.n_heads
    q = torch.randn(B, H, L, Dk)
    k = torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dk)
    pad = torch.zeros(B, 1, 1, L)
    pad[..., -1] = float("-inf")

    for att in ATTENTION_TYPES_ALL:
        tiny_cfg.attention_type = att  # type: ignore[assignment]
        mod = build_core_attention(tiny_cfg, attn_layer="encoder_self")
        mod.eval()
        with torch.no_grad():
            _, attn = mod(q, k, v, attn_mask=pad)
        mass_on_pad_col = attn[..., -1].abs().max().item()
        assert mass_on_pad_col < 1e-5, f"{att}: max mass on masked key col {mass_on_pad_col}"


def test_decoder_self_mask_respects_causal_and_window(tiny_cfg: Config) -> None:
    """local_window + decoder_self：未来位置必须接近 0。"""
    tiny_cfg.attention_type = "local_window"  # type: ignore[assignment]
    tiny_cfg.local_window_size = 1
    mod = build_core_attention(tiny_cfg, attn_layer="decoder_self")
    B, H, L, Dk = 1, tiny_cfg.n_heads, 6, tiny_cfg.d_model // tiny_cfg.n_heads
    q = torch.randn(B, H, L, Dk)
    k = torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dk)
    mod.eval()
    with torch.no_grad():
        _, attn = mod(q, k, v, attn_mask=None)
    # 行 i 对列 j>i 应为 0（因果 + 窗口），对所有 head
    for i in range(L):
        for j in range(i + 1, L):
            assert attn[0, :, i, j].abs().max().item() < 1e-5


def test_bilinear_extra_params_vs_dot(tiny_cfg: Config) -> None:
    tiny_cfg.attention_type = "dot_product"  # type: ignore[assignment]
    n_dot = core_attention_param_count(tiny_cfg, "encoder_self")
    tiny_cfg.attention_type = "bilinear"  # type: ignore[assignment]
    n_bi = core_attention_param_count(tiny_cfg, "encoder_self")
    d_k = tiny_cfg.d_model // tiny_cfg.n_heads
    expected = tiny_cfg.n_heads * d_k * d_k
    assert n_dot == 0
    assert n_bi == expected


def test_core_param_counts_documented(tiny_cfg: Config) -> None:
    """记录各变体 core 模块额外参数（不含 MHA 线性层）；用于实验笔记。"""
    tiny_cfg.attention_type = "gated_dot_additive"  # type: ignore[assignment]
    n_gate = core_attention_param_count(tiny_cfg, "encoder_self")
    d_k = tiny_cfg.d_model // tiny_cfg.n_heads
    dh = tiny_cfg.additive_d_hidden
    # Wq,Wk,v + alpha per head
    expected = (d_k * dh + dh) * 2 + (dh * 1) + tiny_cfg.n_heads
    assert n_gate == expected


@pytest.mark.parametrize("attn_layer", ["encoder_self", "decoder_self", "cross"])
def test_global_local_runs_all_layers(tiny_cfg: Config, attn_layer: str) -> None:
    tiny_cfg.attention_type = "global_local"  # type: ignore[assignment]
    mod = build_core_attention(tiny_cfg, attn_layer=attn_layer)  # type: ignore[arg-type]
    B, H, L, Dk = 1, tiny_cfg.n_heads, 8, tiny_cfg.d_model // tiny_cfg.n_heads
    q = torch.randn(B, H, L, Dk)
    k = torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dk)
    mod.eval()
    with torch.no_grad():
        out, attn = mod(q, k, v, None)
    assert out.shape == (B, H, L, Dk)
    assert attn.shape == (B, H, L, L)
