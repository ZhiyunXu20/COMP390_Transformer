#!/usr/bin/env python3
"""仅从 train.tsv 训练 source / target BPE，避免 val/test 泄漏到词表。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_special_token_strings(reference: Path | None) -> list[str]:
    """与仓库既有 tokenizer JSON 中 id 0–3 对齐（UNK, PAD, BOS, EOS）。"""
    if reference is not None and reference.is_file():
        raw = json.loads(reference.read_text(encoding="utf-8"))
        added = raw.get("added_tokens") or []
        specials = [t["content"] for t in added[:4]]
        if len(specials) >= 4:
            return specials
    return ["<UNK>", "<PAD>", "<BOS>", "<EOS>"]


def iter_column(train_file: Path, col: int) -> Iterator[str]:
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split("\t")
            if len(parts) != 2:
                continue
            text = parts[col].strip()
            if text:
                yield text


def count_valid_lines(train_file: Path) -> tuple[int, int]:
    n_src = n_tgt = 0
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split("\t")
            if len(parts) != 2:
                continue
            s, t = parts[0].strip(), parts[1].strip()
            if s:
                n_src += 1
            if t:
                n_tgt += 1
    return n_src, n_tgt


def train_bpe_from_iterator(
    lines: Iterator[str],
    out_path: Path,
    vocab_size: int,
    special_tokens: list[str],
) -> None:
    unk = special_tokens[0]
    tokenizer = Tokenizer(BPE(unk_token=unk))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(lines, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))


def main() -> None:
    p = argparse.ArgumentParser(
        description="仅使用 train.tsv 的 source/target 列分别训练两个 BPE tokenizer"
    )
    p.add_argument("--train-file", type=str, required=True, help="仅读取此文件（如 data/splits/.../train.tsv）")
    p.add_argument("--tokenizer-src-out", type=str, required=True, help="左列（source）词表输出路径")
    p.add_argument("--tokenizer-tgt-out", type=str, required=True, help="右列（target）词表输出路径")
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument(
        "--reference-tokenizer",
        type=str,
        default=None,
        help="可选：既有 tokenizer JSON，用于拷贝 UNK/PAD/BOS/EOS 字符串（默认尝试仓库 data/tokenizer_src.json）",
    )
    p.add_argument(
        "--metadata-out",
        type=str,
        default=None,
        help="tokenizer_metadata.json 路径（默认：tokenizer-src-out 所在目录下 tokenizer_metadata.json）",
    )
    p.add_argument(
        "--also-update-split-metadata",
        type=str,
        default=None,
        help="可选：已有 split_metadata.json 路径；若存在则写入 train_file_sha256_for_tokenizers 等字段",
    )
    args = p.parse_args()

    train_file = Path(args.train_file).expanduser().resolve()
    if not train_file.is_file():
        raise SystemExit(f"train 文件不存在: {train_file}")

    ref = Path(args.reference_tokenizer).expanduser().resolve() if args.reference_tokenizer else None
    if ref is None:
        cur = train_file.resolve().parent
        ref = None
        for _ in range(12):
            for cand in (cur / "tokenizer_src.json", cur / "data" / "tokenizer_src.json"):
                if cand.is_file():
                    ref = cand.resolve()
                    break
            if ref is not None:
                break
            if cur.parent == cur:
                break
            cur = cur.parent
    specials = load_special_token_strings(ref)

    train_hash = file_sha256(train_file)
    n_src_lines, n_tgt_lines = count_valid_lines(train_file)

    print(f"[src] 训练 BPE（左列），有效行约 {n_src_lines} …")
    train_bpe_from_iterator(
        iter_column(train_file, 0),
        Path(args.tokenizer_src_out).expanduser().resolve(),
        args.vocab_size,
        specials,
    )
    print(f"[tgt] 训练 BPE（右列），有效行约 {n_tgt_lines} …")
    train_bpe_from_iterator(
        iter_column(train_file, 1),
        Path(args.tokenizer_tgt_out).expanduser().resolve(),
        args.vocab_size,
        specials,
    )

    src_out = Path(args.tokenizer_src_out).expanduser().resolve()
    tgt_out = Path(args.tokenizer_tgt_out).expanduser().resolve()
    meta_path = (
        Path(args.metadata_out).expanduser().resolve()
        if args.metadata_out
        else src_out.parent / "tokenizer_metadata.json"
    )

    payload: dict[str, Any] = {
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_file),
        "train_file_sha256": train_hash,
        "vocab_size": args.vocab_size,
        "tokenizer_src_out": str(src_out),
        "tokenizer_tgt_out": str(tgt_out),
        "special_tokens_ids_0_to_3": specials,
        "train_lines_non_empty_src_col": n_src_lines,
        "train_lines_non_empty_tgt_col": n_tgt_lines,
        "reference_tokenizer_used_for_specials": str(ref) if ref else None,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"metadata -> {meta_path}")

    split_meta_path = args.also_update_split_metadata
    if split_meta_path:
        smp = Path(split_meta_path).expanduser().resolve()
        if smp.is_file():
            prev = json.loads(smp.read_text(encoding="utf-8"))
            prev["train_file_sha256_for_tokenizers"] = train_hash
            prev["tokenizer_metadata_path"] = str(meta_path)
            prev["tokenizer_src_json"] = str(src_out)
            prev["tokenizer_tgt_json"] = str(tgt_out)
            smp.write_text(json.dumps(prev, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"已更新 split_metadata -> {smp}")
        else:
            print(f"警告: 未找到 split_metadata（跳过）: {smp}", flush=True)


if __name__ == "__main__":
    main()
