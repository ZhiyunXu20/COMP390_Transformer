"""Decoder self-attention: padded batches with strict masks must stay finite (A23)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from small_try.config import Config
from small_try.model import Seq2SeqTransformer


@pytest.mark.parametrize("attention_type", ["dot_product", "local_window", "global_local"])
def test_decoder_self_attn_finite_padded_batch(attention_type: str) -> None:
    cfg = Config()
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.n_layers = 1
    cfg.dropout = 0.0
    cfg.attention_type = attention_type  # type: ignore[assignment]
    cfg.local_window_size = 1
    cfg.global_local_window_size = 1
    cfg.global_tokens = 2
    cfg.max_seq_len = 32
    cfg.src_vocab_size = 128
    cfg.tgt_vocab_size = 128

    pad = 0
    model = Seq2SeqTransformer(cfg, pad_idx=pad)
    model.eval()
    B, Ls, Lt = 2, 8, 16
    src = torch.randint(1, 50, (B, Ls))
    tgt = torch.randint(1, 50, (B, Lt))
    for b in range(B):
        tgt[b, 6:] = pad

    with torch.no_grad():
        mem, mem_pad = model.encode(src)
        logits, att = model.decode(tgt, mem, mem_pad, return_attentions=True)

    assert torch.isfinite(logits).all()
    aw0 = att["decoder_self"][0]
    assert torch.isfinite(aw0).all()

    for b in range(B):
        for i in range(Lt):
            row_mass = aw0[b, :, i, :].sum(dim=-1)
            if int(tgt[b, i].item()) == pad:
                assert row_mass.abs().max().item() < 1e-5, f"pad query row {i} should have 0 mass"
            else:
                assert (row_mass - 1.0).abs().max().item() < 1e-4, (
                    f"non-pad row {i} should sum to 1, got {row_mass}"
                )


def test_local_window_core_decoder_self_matches_pipeline_window() -> None:
    """Core attention alone used to yield NaN on padded decoder rows; safe softmax + model masks fix this."""
    from small_try.attention import build_core_attention

    cfg = Config()
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.dropout = 0.0
    cfg.attention_type = "local_window"  # type: ignore[assignment]
    cfg.local_window_size = 1
    B, H, L, Dk = 1, cfg.n_heads, 12, cfg.d_model // cfg.n_heads
    q = torch.randn(B, H, L, Dk)
    k = torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dk)
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    pad_cols = torch.zeros(B, 1, 1, L)
    pad_cols[..., 7:] = float("-inf")
    pad_cols = pad_cols.expand(B, 1, L, L)
    q_pad = torch.zeros(B, 1, L, L)
    q_pad[:, :, 7:, :] = float("-inf")
    am = torch.zeros(B, 1, L, L)
    am.masked_fill_(causal.view(1, 1, L, L), float("-inf"))
    am = am + pad_cols + q_pad

    mod = build_core_attention(cfg, attn_layer="decoder_self")
    mod.eval()
    with torch.no_grad():
        out, attn = mod(q, k, v, attn_mask=am)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()
    # rows 7+ all-masked
    assert attn[:, :, 7:, :].abs().sum().item() < 1e-4


@pytest.mark.parametrize("attention_type", ["local_window", "global_local"])
def test_full_forward_finite_padded_source(attention_type: str) -> None:
    """Encoder self-attn on padded source rows must not emit NaN (training path)."""
    cfg = Config()
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.n_layers = 1
    cfg.dropout = 0.0
    cfg.attention_type = attention_type  # type: ignore[assignment]
    cfg.local_window_size = 1
    cfg.global_local_window_size = 1
    cfg.global_tokens = 2
    cfg.max_seq_len = 32
    cfg.src_vocab_size = 128
    cfg.tgt_vocab_size = 128

    pad = 0
    model = Seq2SeqTransformer(cfg, pad_idx=pad)
    model.train()
    B, Ls, Lt_in = 2, 8, 11
    src = torch.randint(1, 50, (B, Ls))
    tgt_in = torch.randint(1, 50, (B, Lt_in))
    for b in range(B):
        src[b, 5:] = pad

    logits = model(src, tgt_in)
    assert torch.isfinite(logits).all()

