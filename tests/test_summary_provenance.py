"""prepare_code_archive validation for attention_variants_test_summary aggregate rows (A25)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_prepare():
    path = _ROOT / "scripts" / "prepare_code_archive.py"
    name = "prepare_code_archive"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pca = _load_prepare()


def _touch_jsonl(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n", encoding="utf-8")


def _touch_metrics(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"BLEU": 1.0, "number_of_test_examples": 1}), encoding="utf-8")


def _write_summary(tmp: Path, *, miss_predictions_for: str | None) -> None:
    results = tmp / "results"
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "attention_variants_test_summary.csv"
    fieldnames = (
        "run",
        "experiment",
        "is_aggregate",
        "source_run_names",
        "representative_run",
        "metrics_source",
        "pkg",
    )
    row = {
        "run": "fast_add_lr3e3",
        "experiment": "fast_add_lr3e3",
        "is_aggregate": "true",
        "source_run_names": "fast_add_lr3e3_s1;fast_add_lr3e3_s2;fast_add_lr3e3_s3",
        "representative_run": "fast_add_lr3e3_s1",
        "metrics_source": "results/ablation_per_seed.csv",
        "pkg": "small_try",
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)
    for name in ("fast_add_lr3e3_s1", "fast_add_lr3e3_s2", "fast_add_lr3e3_s3"):
        base = tmp / "runs" / name / "test_eval"
        _touch_metrics(base / "metrics_test.json")
        if miss_predictions_for != name:
            _touch_jsonl(base / "predictions.jsonl")


def test_aggregate_row_validates_source_runs_not_synthetic_aggregate_dir(tmp_path: Path) -> None:
    _write_summary(tmp_path, miss_predictions_for=None)
    assert (tmp_path / "runs" / "fast_add_lr3e3").is_dir() is False
    assert pca.validate(tmp_path, allow_missing_predictions=False) == 0


def test_aggregate_row_fails_if_source_predictions_missing(tmp_path: Path) -> None:
    _write_summary(tmp_path, miss_predictions_for="fast_add_lr3e3_s2")
    assert pca.validate(tmp_path, allow_missing_predictions=False) == 1


def test_legacy_csv_without_is_aggregate_still_requires_run_dir(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "attention_variants_test_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("run", "pkg"))
        w.writeheader()
        w.writerow({"run": "fast_dot", "pkg": "small_try"})
    assert pca.validate(tmp_path, allow_missing_predictions=False) == 1
