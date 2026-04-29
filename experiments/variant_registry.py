"""
Scientific-purpose metadata for attention variants trained via run_untrained_attention_variants.py.

Single source of truth for attention_type ↔ run_name ↔ qualitative thesis scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantMeta:
    attention_type: str
    run_name: str
    """Mechanism axis: scoring | normalization | connectivity."""
    family: str
    research_question: str
    expected_tradeoff: str
    safe_claim: str


VARIANT_REGISTRY: dict[str, VariantMeta] = {
    "bilinear": VariantMeta(
        attention_type="bilinear",
        run_name="var_bilinear",
        family="scoring",
        research_question=(
            "Does a learnable compatibility matrix improve over parameter-free dot-product scoring?"
        ),
        expected_tradeoff="more parameters, potentially better expressivity",
        safe_claim="implemented as an exploratory scoring-function variant",
    ),
    "gated_dot_additive": VariantMeta(
        attention_type="gated_dot_additive",
        run_name="var_gated_dot_additive",
        family="scoring",
        research_question=(
            "Can the model learn to interpolate between efficient dot-product scoring "
            "and nonlinear additive scoring?"
        ),
        expected_tradeoff="higher compute, potentially adaptive scoring",
        safe_claim=(
            "exploratory gated mixture of dot-product and additive logits; "
            "not positioned as a new attention primitive"
        ),
    ),
    "sparsemax": VariantMeta(
        attention_type="sparsemax",
        run_name="var_sparsemax",
        family="normalization",
        research_question="Does sparse normalization improve alignment selectivity?",
        expected_tradeoff="sparser distributions, not necessarily faster computation",
        safe_claim=(
            "replaces softmax on dense scores only; does not imply sub-quadratic or kernel-efficient attention"
        ),
    ),
    "entmax15": VariantMeta(
        attention_type="entmax15",
        run_name="var_entmax15",
        family="normalization",
        research_question="Does a smoother sparse distribution improve over sparsemax?",
        expected_tradeoff=(
            "intermediate sparsity between softmax and sparsemax; extra normalization compute vs softmax"
        ),
        safe_claim="dense score matrix; α=1.5 entmax is a normalization choice, not a sparse kernel",
    ),
    "local_window": VariantMeta(
        attention_type="local_window",
        run_name="var_local_window",
        family="connectivity",
        research_question="Is local context sufficient for short-sequence MT?",
        expected_tradeoff="structural bias toward locality; full dense matmul cost unchanged",
        safe_claim="dense masked local attention, not a kernel-efficient sparse implementation",
    ),
    "global_local": VariantMeta(
        attention_type="global_local",
        run_name="var_global_local",
        family="connectivity",
        research_question=(
            "Can global anchor tokens compensate for local-window restrictions?"
        ),
        expected_tradeoff="structural global-local pattern; full dense matmul cost unchanged",
        safe_claim="dense masked global-local pattern, not ETC proper",
    ),
}

DEFAULT_VARIANT_ORDER: tuple[str, ...] = tuple(VARIANT_REGISTRY.keys())

VARIANT_TO_RUN_NAME: dict[str, str] = {
    k: v.run_name for k, v in VARIANT_REGISTRY.items()
}


def as_dict(attention_type: str) -> dict[str, str]:
    """Flatten VariantMeta to JSON-friendly strings."""
    m = VARIANT_REGISTRY[attention_type]
    return {
        "attention_type": m.attention_type,
        "run_name": m.run_name,
        "family": m.family,
        "research_question": m.research_question,
        "expected_tradeoff": m.expected_tradeoff,
        "safe_claim": m.safe_claim,
    }
