"""Ensure qualified package imports do not clobber each other's Config classes."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_try_swap_head_config_modules_stay_isolated() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    st = importlib.import_module("small_try.config")
    sw = importlib.import_module("small_swap.config")
    sh = importlib.import_module("small_head.config")

    assert sw.Config().swap_parallel_columns is True
    assert st.Config().n_heads == 4
    assert sh.Config().n_heads == 4
    # reloading try default still 4 (not mutated by swap)
    c2 = st.Config()
    assert c2.n_heads == 4
    assert getattr(c2, "swap_parallel_columns", None) is not True
