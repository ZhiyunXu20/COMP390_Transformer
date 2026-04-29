"""scripts/create_splits.py：划分无重叠、同 seed 复现、异 seed 差异。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_create_splits():
    path = ROOT / "scripts" / "create_splits.py"
    spec = importlib.util.spec_from_file_location("create_splits_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load_create_splits()


def _tiny_corpus_text() -> str:
    lines = []
    # 合法两列
    for i in range(200):
        lines.append(f"src{i}\ttgt{i}\n")
    lines.append("dup\ta\n")  # 重复 pair
    lines.append("dup\ta\n")
    lines.append("bad\n")  # 畸形
    lines.append("x\ty\tz\n")  # 非两列
    return "".join(lines)


def test_no_pair_overlap_across_splits(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text(_tiny_corpus_text(), encoding="utf-8")
    out = tmp_path / "out"
    cs.run_create_splits(
        src,
        out,
        sample_size=120,
        seed=7,
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
    )

    def pairs(path: Path) -> set[tuple[str, str]]:
        s: set[tuple[str, str]] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            a, b = line.split("\t", 1)
            s.add((a, b))
        return s

    tr = pairs(out / "train.tsv")
    va = pairs(out / "val.tsv")
    te = pairs(out / "test.tsv")
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_same_seed_is_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text(_tiny_corpus_text(), encoding="utf-8")
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    m1 = cs.run_create_splits(
        src, out1, 80, 123, 0.9, 0.05, 0.05
    )
    m2 = cs.run_create_splits(
        src, out2, 80, 123, 0.9, 0.05, 0.05
    )
    assert m1["sha256"] == m2["sha256"]
    assert (out1 / "train.tsv").read_bytes() == (out2 / "train.tsv").read_bytes()


def test_different_seed_changes_assignment(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text(_tiny_corpus_text(), encoding="utf-8")
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    cs.run_create_splits(src, out1, 100, 1, 0.9, 0.05, 0.05)
    cs.run_create_splits(src, out2, 100, 2, 0.9, 0.05, 0.05)
    t1 = (out1 / "train.tsv").read_text(encoding="utf-8")
    t2 = (out2 / "train.tsv").read_text(encoding="utf-8")
    assert t1 != t2


def test_manifest_schema(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text(_tiny_corpus_text(), encoding="utf-8")
    out = tmp_path / "out"
    cs.run_create_splits(
        src, out, 50, 0, 0.8, 0.1, 0.1
    )
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for k in (
        "source_path",
        "sample_size",
        "seed",
        "sizes",
        "sha256",
        "malformed_lines_discarded",
        "duplicate_pairs_removed",
        "creation_timestamp_utc",
    ):
        assert k in man
    assert set(man["sizes"].keys()) >= {"train", "val", "test"}
