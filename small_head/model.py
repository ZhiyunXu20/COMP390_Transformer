"""完整 Encoder–Decoder Transformer；注意力类型由 Config.attention_type 控制。"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from attention import MultiHeadAttention, PositionwiseFFN, PositionalEncoding
from config import Config


def _causal_square(L: int, device: torch.device) -> torch.Tensor:
    """(L,L) 上三角（不含对角）为 -inf，其余为 0。"""
    m = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    out = torch.zeros(L, L, device=device, dtype=torch.float32)
    out.masked_fill_(m, float("-inf"))
    return out


def _key_pad_mask(
    seq: torch.Tensor, pad_idx: int, tgt_len: Optional[int] = None
) -> torch.Tensor:
    """(B, L) -> (B, 1, 1, L) 的 key 侧 padding 掩码（PAD 列为 -inf）。"""
    m = seq == pad_idx
    f = torch.zeros_like(seq, dtype=torch.float32)
    f.masked_fill_(m, float("-inf"))
    return f.unsqueeze(1).unsqueeze(2)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg, attn_layer="encoder_self")
        self.ff = PositionwiseFFN(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # src_key_padding: (B,1,1,Ls) float
        q = self.norm1(x)
        attn_out, attn_w = self.self_attn(q, q, q, attn_mask=src_key_padding)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_w


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg, attn_layer="decoder_self")
        self.cross_attn = MultiHeadAttention(cfg, attn_layer="cross")
        self.ff = PositionwiseFFN(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_key_padding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # tgt_mask: (B,1,Lt,Lt) for self-attn
        # memory_key_padding: (B,1,1,Ls)
        q1 = self.norm1(x)
        sa, aw_self = self.self_attn(q1, q1, q1, attn_mask=tgt_mask)
        x = x + self.dropout(sa)

        q2 = self.norm2(x)
        ca, aw_cross = self.cross_attn(q2, memory, memory, attn_mask=memory_key_padding)
        x = x + self.dropout(ca)

        x = x + self.dropout(self.ff(self.norm3(x)))
        return x, aw_self, aw_cross


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: Config, pad_idx: int):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = pad_idx
        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model)
        self.src_pe = PositionalEncoding(cfg.d_model, max_len=cfg.max_seq_len + 4, dropout=cfg.dropout)
        self.tgt_pe = PositionalEncoding(cfg.d_model, max_len=cfg.max_seq_len + 4, dropout=cfg.dropout)

        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """src: (B, Ls)"""
        pad = _key_pad_mask(src, self.pad_idx)
        x = self.src_pe(self.src_embed(src))
        for layer in self.encoder_layers:
            x, _ = layer(x, pad)
        return x, pad

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding: torch.Tensor,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict[str, list[torch.Tensor]]]]:
        """tgt: teacher forcing input (B, Lt)"""
        b, lt = tgt.shape
        causal = _causal_square(lt, tgt.device).view(1, 1, lt, lt).expand(b, 1, lt, lt)
        tgt_pad = _key_pad_mask(tgt, self.pad_idx)
        tgt_pad_exp = tgt_pad.expand(b, 1, lt, lt)
        self_attn_mask = causal + tgt_pad_exp

        x = self.tgt_pe(self.tgt_embed(tgt))
        dec_cross: list[torch.Tensor] = []
        dec_self: list[torch.Tensor] = []
        for layer in self.decoder_layers:
            x, aw_self, aw_cross = layer(x, memory, self_attn_mask, memory_key_padding)
            if return_attentions:
                dec_self.append(aw_self)
                dec_cross.append(aw_cross)
        logits = self.lm_head(x)
        if return_attentions:
            return logits, {
                "decoder_self": dec_self,
                "decoder_cross": dec_cross,
            }
        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict[str, Any]]]:
        """tgt: [BOS..] 不含最后一个；output_attentions=True 时返回各层真实注意力权重。"""
        memory, src_pad = self.encode(src)
        if not output_attentions:
            return self.decode(tgt, memory, src_pad)
        logits, att = self.decode(tgt, memory, src_pad, return_attentions=True)
        return logits, att

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        """单条或 batch 贪心生成；src: (B, Ls)"""
        self.eval()
        memory, src_pad = self.encode(src)
        device = src.device
        b = src.size(0)
        ys = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.decode(ys, memory, src_pad)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == eos_id)
            if finished.all():
                break
        return ys


def build_logits_shifted_loss(
    logits: torch.Tensor, tgt_full: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: (B, Lt-1, V) 对应 tgt_in = tgt_full[:, :-1]
    目标: tgt_full[:, 1:]
    """
    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = tgt_full[:, 1:].contiguous().view(-1)
    return logits, labels
