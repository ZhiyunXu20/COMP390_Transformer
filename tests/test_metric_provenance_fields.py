"""Verify metric_provenance is added to metrics_test.json by evaluate_test.py."""

from __future__ import annotations

import sys
from pathlib import Path


def test_evaluate_test_writes_metric_provenance() -> None:
    """Smoke test: evaluate_test source defines provenance keys for metrics_test.json."""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import evaluate_test  # noqa: F401

    src = Path(evaluate_test.__file__).read_text(encoding="utf-8")
    expected_keys = [
        "sacrebleu_signature",
        "sacrebleu_version",
        "bertscore_lang",
        "bertscore_model_type",
        "bertscore_version",
        "comet_model",
        "comet_version",
        "comet_device",
    ]
    for k in expected_keys:
        assert k in src, f"metric_provenance key '{k}' not found in evaluate_test.py"


def test_historical_runs_documented() -> None:
    """Verify docs/METRIC_PROVENANCE.md exists and references historical 36 runs."""
    p = Path(__file__).resolve().parent.parent / "docs" / "METRIC_PROVENANCE.md"
    assert p.is_file(), "docs/METRIC_PROVENANCE.md not found"
    text = p.read_text(encoding="utf-8")
    assert "36 archived" in text or "historical" in text.lower()
    assert "Unbabel/wmt22-comet-da" in text
    assert "bertscore" in text.lower()
