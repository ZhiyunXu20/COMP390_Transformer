"""Tests for scripts/check_data_integrity.py (SHA-256 manifest / split_metadata vs TSV files)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_data_integrity.py"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_minimal_split(
    split_dir: Path,
    *,
    manifest_train_wrong: bool,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.tsv").write_bytes(b"row_train\n")
    (split_dir / "val.tsv").write_bytes(b"row_val\n")
    (split_dir / "test.tsv").write_bytes(b"row_test\n")
    ht, hv, hx = _sha256_bytes(b"row_train\n"), _sha256_bytes(b"row_val\n"), _sha256_bytes(b"row_test\n")
    bad_train = "a1e1e2954bf03d271d82b03543b1ba7aa15957ad71efe36d37f4ea6193d99514"
    m_train = bad_train if manifest_train_wrong else ht
    manifest = {"sha256": {"train.tsv": m_train, "val.tsv": hv, "test.tsv": hx}}
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    meta = {
        "train_file_sha256_for_tokenizers": ht,
        "val_file_sha256": hv,
        "test_file_sha256": hx,
    }
    (split_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _run_checker(repo: Path, split_rel: str) -> int:
    cp = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--split-subdir",
            split_rel,
            "--no-report",
        ],
        capture_output=True,
        text=True,
    )
    return cp.returncode


@pytest.mark.parametrize("wrong", [True, False])
def test_data_integrity_exit_code_matches_manifest(tmp_path: Path, wrong: bool) -> None:
    split_rel = "data/splits/test_integrity"
    split_dir = tmp_path / split_rel
    _write_minimal_split(split_dir, manifest_train_wrong=wrong)
    code = _run_checker(tmp_path, split_rel)
    assert code == (1 if wrong else 0)


def test_data_integrity_meta_mismatch_fails(tmp_path: Path) -> None:
    split_rel = "data/splits/test_integrity2"
    split_dir = tmp_path / split_rel
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.tsv").write_bytes(b"x\n")
    (split_dir / "val.tsv").write_bytes(b"y\n")
    (split_dir / "test.tsv").write_bytes(b"z\n")
    hx = _sha256_bytes(b"x\n")
    hy = _sha256_bytes(b"y\n")
    hz = _sha256_bytes(b"z\n")
    (split_dir / "manifest.json").write_text(
        json.dumps({"sha256": {"train.tsv": hx, "val.tsv": hy, "test.tsv": hz}}, indent=2),
        encoding="utf-8",
    )
    (split_dir / "split_metadata.json").write_text(
        json.dumps(
            {
                "train_file_sha256_for_tokenizers": "f" * 64,
                "val_file_sha256": hy,
                "test_file_sha256": hz,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert _run_checker(tmp_path, split_rel) == 1
