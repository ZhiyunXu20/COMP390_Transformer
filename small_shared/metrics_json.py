"""Helpers for training `metrics.json`: backward-compatible `final_bleu` plus disambiguation fields."""

from __future__ import annotations

from typing import Any, Mapping

# Wording for new runs (`final_bleu_caveat` / `final_extra_metrics_caveat`).
FINAL_BLEU_CAVEAT = (
    "Sampled, filtered val-split BLEU intended for training-time monitoring. "
    "For held-out test results, see test_eval/metrics_test.json "
    "(full 2500 examples, no skip filtering)."
)


def effective_attention_backend_field(cfg: Any) -> dict[str, Any]:
    """Record entmax normalization backend when attention_type is entmax15 (forward-only provenance)."""
    at = getattr(cfg, "attention_type", None)
    if at != "entmax15":
        return {}
    try:
        from entmax import entmax_bisect as _entmax_bisect_check  # noqa: F401

        if _entmax_bisect_check is not None:
            return {"effective_attention_backend": "entmax15-bisect"}
    except ImportError:
        pass
    return {"effective_attention_backend": "entmax15-fallback-sparsemax-naive"}


def bleu_disambiguation_fields(
    cfg: Any,
    bleu_eval_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extra top-level keys written beside historic `final_bleu` / `extra_metrics`.

    Reflects the same sampled generation pass (split, caps, skip filters) for BLEU
    and chrF/BERT/COMET extras.
    """
    split = getattr(cfg, "eval_split", "val")
    max_samples = int(getattr(cfg, "bleu_sample_size", 256))
    skip_ip = bool(getattr(cfg, "bleu_skip_identical_parallel", True))
    skip_thr = getattr(cfg, "bleu_skip_similarity_threshold", None)
    num_ex: Any = None
    if bleu_eval_meta is not None:
        num_ex = bleu_eval_meta.get("bleu_pairs_used")

    return {
        "final_bleu_split": split,
        "final_bleu_num_examples": num_ex,
        "final_bleu_max_samples": max_samples,
        "final_bleu_skip_identical_parallel": skip_ip,
        "final_bleu_skip_similarity_threshold": skip_thr,
        "final_bleu_caveat": FINAL_BLEU_CAVEAT,
        "final_extra_metrics_split": split,
        "final_extra_metrics_num_examples": num_ex,
        "final_extra_metrics_max_samples": max_samples,
        "final_extra_metrics_skip_identical_parallel": skip_ip,
        "final_extra_metrics_skip_similarity_threshold": skip_thr,
        "final_extra_metrics_caveat": FINAL_BLEU_CAVEAT,
    }
