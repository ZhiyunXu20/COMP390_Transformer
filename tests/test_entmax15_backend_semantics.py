"""entmax15 raises without package; explicit fallback warns (A28)."""

from __future__ import annotations

import pytest
import torch


def test_entmax15_raises_if_entmax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from small_try import attention

    monkeypatch.setattr(attention, "_HAS_ENTMAX", False)
    monkeypatch.setattr(attention, "_entmax_bisect", None)
    with pytest.raises(RuntimeError, match="entmax package"):
        attention._entmax15(torch.randn(2, 8))


def test_entmax15_explicit_fallback_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    from small_try import attention

    monkeypatch.setattr(attention, "_HAS_ENTMAX", False)
    monkeypatch.setattr(attention, "_entmax_bisect", None)
    with pytest.warns(RuntimeWarning, match="entmax"):
        out = attention._entmax15(torch.randn(2, 8), allow_fallback=True)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_entmax15_works_when_entmax_available() -> None:
    from small_try import attention

    if not attention._HAS_ENTMAX:
        pytest.skip("entmax package not installed")
    out = attention._entmax15(torch.randn(2, 8))
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)
