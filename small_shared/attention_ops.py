"""Numerically safe masked normalization for attention logits (no NaN on all-masked rows)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def safe_masked_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Match ``F.softmax`` on usable rows; all-masked rows (−∞ only) → zeros.

    Gradients on all-masked rows are zeroed (training-safe); forward mass is 0 on those rows.
    """
    if dim < 0:
        dim += scores.ndim
    max_score = scores.amax(dim=dim, keepdim=True)
    row_ok = torch.isfinite(max_score) & (max_score > float("-inf"))
    scores_f = torch.where(row_ok.expand_as(scores), scores, torch.zeros_like(scores))
    y = F.softmax(scores_f, dim=dim)
    keep = row_ok.to(dtype=y.dtype).detach().expand_as(y)
    y = y * keep
    return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def guard_norm_all_masked_rows(
    scores: torch.Tensor, probs: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """sparsemax / entmax: zero out logits rows that were entirely masked; drop NaN/Inf."""
    if dim < 0:
        dim += scores.ndim
    max_score = scores.amax(dim=dim, keepdim=True)
    row_ok = torch.isfinite(max_score) & (max_score > float("-inf"))
    keep = row_ok.to(dtype=probs.dtype).detach().expand_as(probs)
    probs = probs * keep
    return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
