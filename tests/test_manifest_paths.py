"""Manifest path hygiene: repo-relative, no hard-coded /root or /home."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

MANIFESTS = (
    REPO / "small_try/results/manifest.json",
    REPO / "small_head/results/manifest_head.json",
    REPO / "small_swap/results/manifest_swap.json",
)


def _iter_json_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_json_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_json_strings(v)


@pytest.mark.parametrize("manifest_path", MANIFESTS)
def test_manifest_exists(manifest_path: Path):
    assert manifest_path.is_file(), f"missing {manifest_path.relative_to(REPO)}"


@pytest.mark.parametrize("manifest_path", MANIFESTS)
def test_manifest_no_absolute_root_or_home_paths(manifest_path: Path):
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    for s in _iter_json_strings(raw):
        assert not s.startswith("/root"), f"{manifest_path}: absolute /root path: {s!r}"
        assert not s.startswith("/home"), f"{manifest_path}: absolute /home path: {s!r}"


@pytest.mark.parametrize("manifest_path", MANIFESTS)
def test_runs_block_points_at_repo_root_runs(manifest_path: Path):
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = raw.get("runs")
    assert isinstance(runs, dict), f"{manifest_path}: missing runs dict"
    for key, val in runs.items():
        assert isinstance(val, str), f"{manifest_path}: runs.{key} must be str"
        p = Path(val)
        assert not p.is_absolute(), f"{manifest_path}: runs.{key} must be relative: {val!r}"
        assert val.startswith("runs/"), (
            f"{manifest_path}: runs.{key} must start with runs/: {val!r}"
        )
        assert (REPO / val).is_dir(), (
            f"{manifest_path}: runs.{key} -> {(REPO / val)!r} is not a directory"
        )
