"""Tests for train_runtime.configure_determinism (mocked torch.backends.cudnn)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from train_runtime import configure_determinism


def test_configure_determinism_true_sets_cudnn_and_uda():
    cudnn_ns = SimpleNamespace(benchmark=True, deterministic=False)
    with patch.object(torch.backends, "cudnn", cudnn_ns):
        with patch("train_runtime.torch.use_deterministic_algorithms") as uda:
            out = configure_determinism(42, True, cuda_available=True)
    assert cudnn_ns.benchmark is False
    assert cudnn_ns.deterministic is True
    uda.assert_called_once_with(True, warn_only=True)
    assert out["deterministic_requested"] is True
    assert out["cudnn_benchmark"] is False
    assert out["cudnn_deterministic"] is True
    assert out["use_deterministic_algorithms"] is True
    assert out["cublas_workspace_config"]
    assert out["torch_version"] == torch.__version__


def test_configure_determinism_false_cuda_enables_benchmark_no_uda():
    cudnn_ns = SimpleNamespace(benchmark=False, deterministic=False)
    with patch.object(torch.backends, "cudnn", cudnn_ns):
        with patch("train_runtime.torch.use_deterministic_algorithms") as uda:
            out = configure_determinism(42, False, cuda_available=True)
    uda.assert_not_called()
    assert cudnn_ns.benchmark is True
    assert out["deterministic_requested"] is False
    assert out["use_deterministic_algorithms"] is False
    assert out["cudnn_benchmark"] is True
    assert "torch_version" in out


def test_configure_determinism_false_cpu_skips_benchmark_toggle():
    cudnn_ns = SimpleNamespace(benchmark=False, deterministic=False)
    with patch.object(torch.backends, "cudnn", cudnn_ns):
        with patch("train_runtime.torch.use_deterministic_algorithms") as uda:
            out = configure_determinism(42, False, cuda_available=False)
    uda.assert_not_called()
    assert cudnn_ns.benchmark is False
    assert out["cudnn_benchmark"] is False
