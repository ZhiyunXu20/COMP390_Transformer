"""Tests for scripts/audit_parallel_data.py split overlap audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    path = _ROOT / "scripts" / "audit_parallel_data.py"
    spec = importlib.util.spec_from_file_location("audit_parallel_data", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parser_treats_quotes_as_literal_text(tmp_path):
    p = tmp_path / "tiny.tsv"
    p.write_text(
        'hello\tbonjour\n"Euromat" works\t"Euromat" fonctionne\nbye\tau revoir\n',
        encoding="utf-8",
    )
    apd = _load_audit()
    n = apd.count_lines(str(p))
    assert n == 3, f"Expected 3 physical lines, got {n}"


def test_overlap_matrix_on_tiny_data(tmp_path):
    d = tmp_path / "splits"
    d.mkdir()
    (d / "train.tsv").write_text("a\tA\nb\tB\nc\tC\n", encoding="utf-8")
    (d / "val.tsv").write_text("a\tX\nd\tD\n", encoding="utf-8")
    (d / "test.tsv").write_text("a\tA\ne\tE\n", encoding="utf-8")
    apd = _load_audit()
    res = apd.compute_overlap(d / "train.tsv", d / "val.tsv", d / "test.tsv")
    assert res["row_counts"] == {"train": 3, "val": 2, "test": 2}
    assert res["exact_pair_overlap"]["train-test"] == 1
    assert res["source_overlap"]["train-val"] == 1
    assert res["source_overlap"]["train-test"] == 1
    assert res["source_to_multiple_targets"] == 1


def test_real_data_matches_expected_v9_numbers():
    p = _ROOT / "data" / "splits" / "en_fr_50k_seed42"
    if not p.is_dir():
        pytest.skip("real splits not present in this environment")
    apd = _load_audit()
    res = apd.compute_overlap(p / "train.tsv", p / "val.tsv", p / "test.tsv")
    assert res["row_counts"] == {"train": 45000, "val": 2500, "test": 2500}
    assert res["source_to_multiple_targets"] == 115
    assert res["target_to_multiple_sources"] == 107
    assert res["source_overlap"]["train-test"] == 8
