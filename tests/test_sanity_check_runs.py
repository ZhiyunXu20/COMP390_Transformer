"""Unit tests for scripts/sanity_check_runs.py classification logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_sanity():
    path = _ROOT / "scripts" / "sanity_check_runs.py"
    name = "sanity_check_runs"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_sanity()


def _base_metrics(**over):
    m = {
        "final_val_loss": 3.0,
        "final_bleu": 10.0,
        "best_bleu_during_training": 10.0,
    }
    m.update(over)
    return m


def _base_test(**over):
    t = {
        "BLEU": 10.0,
        "chrF": 30.0,
        "BERTScore": 0.8,
        "number_of_test_examples": 100,
        "average_length_ratio": 1.0,
    }
    t.update(over)
    return t


def test_classify_failed_nan_with_empty_test_metrics():
    status, notes = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=_base_metrics(final_val_loss=float("nan")),
        metrics_test=_base_test(BLEU=0.0, chrF=0.0, BERTScore=0.0),
        empty_hyp_pct=None,
        pred_n=0,
    )
    assert status == "failed_nan"
    assert any("NaN" in n or "non-finite" in n for n in notes)
    assert any("failed_empty_outputs" in n for n in notes)


def test_classify_failed_nan_without_empty_test():
    status, notes = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=_base_metrics(final_val_loss=float("inf")),
        metrics_test=_base_test(),
        empty_hyp_pct=None,
        pred_n=0,
    )
    assert status == "failed_nan"
    assert not any("failed_empty_outputs" in n for n in notes)


def test_classify_failed_empty_outputs_zero_bleu_chrf():
    status, notes = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=_base_metrics(),
        metrics_test=_base_test(BLEU=0.0, chrF=0.0, BERTScore=0.0),
        empty_hyp_pct=0.0,
        pred_n=10,
    )
    assert status == "failed_empty_outputs"
    assert any("BERTScore=0" in n for n in notes)


def test_classify_failed_empty_high_empty_hyp_pct():
    status, _notes = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=_base_metrics(),
        metrics_test=_base_test(BLEU=8.0, chrF=25.0),
        empty_hyp_pct=60.0,
        pred_n=100,
    )
    assert status == "failed_empty_outputs"


def test_classify_missing_artifacts():
    status, _ = sc.classify_run(
        run="x",
        has_metrics=False,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=None,
        metrics_test=_base_test(),
        empty_hyp_pct=None,
        pred_n=0,
    )
    assert status == "missing_artifacts"


def test_classify_incomplete_no_training_meta():
    status, _ = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=False,
        metrics=_base_metrics(),
        metrics_test=_base_test(BLEU=12.0, chrF=35.0),
        empty_hyp_pct=0.0,
        pred_n=10,
    )
    assert status == "incomplete"


def test_classify_warning_low_quality_low_bleu():
    status, _ = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=_base_metrics(),
        metrics_test=_base_test(BLEU=3.5, chrF=20.0, average_length_ratio=0.8),
        empty_hyp_pct=0.0,
        pred_n=10,
    )
    assert status == "warning_low_quality"


def test_classify_warning_low_quality_len_ratio():
    status, _ = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=_base_metrics(),
        metrics_test=_base_test(BLEU=10.0, chrF=30.0, average_length_ratio=0.35),
        empty_hyp_pct=0.0,
        pred_n=10,
    )
    assert status == "warning_low_quality"


def test_classify_ok_high_bleu_moderate_len_ratio():
    status, _ = sc.classify_run(
        run="x",
        has_metrics=True,
        has_metrics_test=True,
        has_training_meta=True,
        metrics=_base_metrics(),
        metrics_test=_base_test(BLEU=7.5, chrF=20.0, average_length_ratio=0.55),
        empty_hyp_pct=0.0,
        pred_n=10,
    )
    assert status == "ok"


def test_analyze_predictions_hypothesis_key(tmp_path: Path):
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        '{"hypothesis": "a b"}\n{"hypothesis": ""}\n{"hypothesis": "x"}\n',
        encoding="utf-8",
    )
    empty_pct, short_pct, n = sc.analyze_predictions(pred)
    assert n == 3
    assert empty_pct == pytest.approx(100.0 / 3.0)
    assert short_pct == pytest.approx(100.0 / 3.0)


def test_analyze_predictions_hyp_key(tmp_path: Path):
    pred = tmp_path / "p.jsonl"
    pred.write_text('{"hyp": " "}\n', encoding="utf-8")
    empty_pct, _s, n = sc.analyze_predictions(pred)
    assert n == 1
    assert empty_pct == 100.0
