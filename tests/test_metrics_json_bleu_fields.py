"""Disambiguation fields for metrics.json final_bleu / extra_metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from small_shared.metrics_json import FINAL_BLEU_CAVEAT, bleu_disambiguation_fields, effective_attention_backend_field


@dataclass
class _MiniCfg:
    eval_split: str = "val"
    bleu_sample_size: int = 256
    bleu_skip_identical_parallel: bool = True
    bleu_skip_similarity_threshold: float | None = 0.95


def test_bleu_disambiguation_fields_shape_and_caveat() -> None:
    d = bleu_disambiguation_fields(_MiniCfg(), {"bleu_pairs_used": 240})
    assert d["final_bleu_split"] == "val"
    assert d["final_bleu_max_samples"] == 256
    assert d["final_bleu_num_examples"] == 240
    assert d["final_bleu_skip_identical_parallel"] is True
    assert d["final_bleu_skip_similarity_threshold"] == 0.95
    assert d["final_bleu_caveat"] == FINAL_BLEU_CAVEAT
    assert d["final_extra_metrics_split"] == d["final_bleu_split"]
    assert d["final_extra_metrics_caveat"] == FINAL_BLEU_CAVEAT


def test_bleu_disambiguation_default_eval_split_without_attr() -> None:
    class Empty:
        bleu_sample_size = 128
        bleu_skip_identical_parallel = False
        bleu_skip_similarity_threshold: Any = None

    d = bleu_disambiguation_fields(Empty(), None)
    assert d["final_bleu_split"] == "val"
    assert d["final_bleu_max_samples"] == 128
    assert d["final_bleu_skip_identical_parallel"] is False
    assert d["final_bleu_num_examples"] is None


def test_effective_attention_backend_entmax15_only() -> None:
    @dataclass
    class EntCfg:
        attention_type: str = "entmax15"

    d = effective_attention_backend_field(EntCfg())
    assert d["effective_attention_backend"] in ("entmax15-bisect", "entmax15-fallback-sparsemax-naive")


def test_effective_attention_backend_omitted_when_not_entmax() -> None:
    @dataclass
    class DotCfg:
        attention_type: str = "dot_product"

    assert effective_attention_backend_field(DotCfg()) == {}
