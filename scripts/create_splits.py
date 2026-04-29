#!/usr/bin/env python3
"""从 Tab 平行语料生成可复现的 train/val/test 划分（随机抽样 + 分层比例）。

新实验请优先使用 scripts/make_splits.py（CLI 与 split_metadata.json 字段更统一）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_to_tempfile(src_path: Path) -> tuple[Path, int, int, int]:
    """流式去重：写出临时 Tab 文件；键为 sha256(src\\ntgt)，避免巨语料占用 tuple 集合内存。"""
    malformed = 0
    duplicates_removed = 0
    seen: set[bytes] = set()
    tf = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        prefix="dedupe_",
        suffix=".tsv",
    )
    tmp_path = Path(tf.name)
    try:
        with src_path.open("r", encoding="utf-8") as inf:
            for line in inf:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    malformed += 1
                    continue
                s, t = parts[0].strip(), parts[1].strip()
                dig = hashlib.sha256(f"{s}\n{t}".encode("utf-8")).digest()
                if dig in seen:
                    duplicates_removed += 1
                    continue
                seen.add(dig)
                tf.write(s + "\t" + t + "\n")
    finally:
        tf.close()

    return tmp_path, malformed, duplicates_removed, len(seen)


def reservoir_sample_pairs_from_file(
    unique_tsv: Path,
    k: int,
    seed: int,
) -> list[tuple[str, str]]:
    """无放回均匀抽样至多 k 条（Vitter reservoir），适于未知总长唯一句对流。"""
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    if k <= 0:
        return reservoir
    with unique_tsv.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            pair = (parts[0], parts[1])
            if i < k:
                reservoir.append(pair)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = pair
    return reservoir


def split_sizes(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    if n == 0:
        return 0, 0, 0
    nt = int(math.floor(train_ratio * n))
    nv = int(math.floor(val_ratio * n))
    nte = n - nt - nv
    if nte < 0:
        raise RuntimeError("split rounding produced negative test size")
    return nt, nv, nte


def split_train_val_test(
    pairs: list[tuple[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """对已定序列随机打乱后按比例切段（同一 seed 可复现）。"""
    rng = random.Random(seed + 1337)
    order = list(pairs)
    rng.shuffle(order)
    n = len(order)
    nt, nv, _ = split_sizes(n, train_ratio, val_ratio, test_ratio)
    train = order[:nt]
    val = order[nt : nt + nv]
    test = order[nt + nv :]
    return train, val, test


def write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s, t in rows:
            f.write(s + "\t" + t + "\n")


def run_create_splits(
    src: Path,
    out_dir: Path,
    sample_size: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    tmp_unique, malformed, dup_removed, n_unique = dedupe_to_tempfile(src)
    try:
        k = min(sample_size, n_unique)
        sampled = reservoir_sample_pairs_from_file(tmp_unique, k, seed)
    finally:
        tmp_unique.unlink(missing_ok=True)

    train, val, test = split_train_val_test(
        sampled, seed, train_ratio, val_ratio, test_ratio
    )

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_p = out_dir / "train.tsv"
    val_p = out_dir / "val.tsv"
    test_p = out_dir / "test.tsv"
    write_tsv(train_p, train)
    write_tsv(val_p, val)
    write_tsv(test_p, test)

    manifest: dict[str, Any] = {
        "source_path": str(src.resolve()),
        "sample_size": sample_size,
        "sampled_pairs_used": len(sampled),
        "unique_pairs_before_sample": n_unique,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "sha256": {
            "train.tsv": file_sha256(train_p),
            "val.tsv": file_sha256(val_p),
            "test.tsv": file_sha256(test_p),
        },
        "malformed_lines_discarded": malformed,
        "duplicate_pairs_removed": dup_removed,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="创建 train/val/test Tab 语料划分")
    p.add_argument("--src", type=str, required=True, help="原始 Tab 平行语料")
    p.add_argument("--out-dir", type=str, required=True, help="输出目录（含 manifest）")
    p.add_argument("--sample-size", type=int, required=True, help="去重后随机抽样条数上限")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.90)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"源文件不存在: {src}")

    run_create_splits(
        src,
        Path(args.out_dir).expanduser().resolve(),
        args.sample_size,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    print(f"完成 -> {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
