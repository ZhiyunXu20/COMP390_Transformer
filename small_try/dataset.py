"""平行句对：默认读 train/val/test.tsv；可选单文件 legacy 模式（与旧 corpus_50k.tsv 兼容）。"""

from __future__ import annotations

import os
from typing import Literal

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from config import Config
from tokens import BOS_ID, EOS_ID, PAD_ID


def load_tokenizers(cfg: Config) -> tuple[Tokenizer, Tokenizer]:
    if not os.path.isfile(cfg.tokenizer_src):
        raise FileNotFoundError(cfg.tokenizer_src)
    if not os.path.isfile(cfg.tokenizer_tgt):
        raise FileNotFoundError(cfg.tokenizer_tgt)
    return Tokenizer.from_file(cfg.tokenizer_src), Tokenizer.from_file(cfg.tokenizer_tgt)


def _split_tsv_path(cfg: Config, split: Literal["train", "val", "test"]) -> str:
    key = {"train": "train_path", "val": "val_path", "test": "test_path"}[split]
    p = getattr(cfg, key, None)
    if isinstance(p, str) and p.strip():
        return p.strip()
    raise ValueError(f"config.{key} 为空；请在 Config 或命令行指定划分文件")


def _legacy_split_bucket(
    valid_pair_index: int,
    train_r: float,
    val_r: float,
    test_r: float,
) -> Literal["train", "val", "test"]:
    s = train_r + val_r + test_r
    if s <= 0:
        raise ValueError("legacy split ratios must sum to a positive value")
    train_r, val_r, test_r = train_r / s, val_r / s, test_r / s
    scale = 1000
    t_end = int(round(scale * train_r))
    v_end = int(round(scale * (train_r + val_r)))
    m = valid_pair_index % scale
    if m < t_end:
        return "train"
    if m < v_end:
        return "val"
    return "test"


class TabParallelDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        src_tok: Tokenizer,
        tgt_tok: Tokenizer,
        split: Literal["train", "val", "test"],
    ):
        self.cfg = cfg
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.split = split
        self.src_lines: list[str] = []
        self.tgt_lines: list[str] = []

        use_splits = getattr(cfg, "use_split_files", True)
        if use_splits:
            path = _split_tsv_path(cfg, split)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[{split}] 数据文件不存在: {path}\n"
                    "请先运行: python scripts/make_splits.py ... 或在 Config 中指向已有 train.tsv / val.tsv / test.tsv"
                )
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 2:
                        continue
                    self.src_lines.append(parts[0].strip())
                    self.tgt_lines.append(parts[1].strip())
            print(f"[{split}] 加载 {len(self.src_lines)} 条句对 <- {path}")
            return

        dp = getattr(cfg, "data_path", None)
        if not isinstance(dp, str) or not dp.strip():
            raise ValueError(
                "use_split_files=False 时请在 Config.data_path 指定单个 Tab 语料（如 corpus_50k.tsv）"
            )
        if not os.path.isfile(dp):
            raise FileNotFoundError(
                f"[{split}] legacy 数据文件不存在: {dp}\n"
                "或设置 use_split_files=True 并使用 scripts/make_splits.py 生成的划分文件。"
            )
        test_r = float(getattr(cfg, "test_ratio", 0.05))
        train_r = 1.0 - float(cfg.val_ratio) - test_r
        if train_r <= 0:
            raise ValueError("legacy 模式需要 val_ratio + test_ratio < 1")

        pi = 0
        with open(dp, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                parts = raw.split("\t")
                if len(parts) != 2:
                    continue
                s0, t0 = parts[0].strip(), parts[1].strip()
                if not s0 or not t0 or s0 == t0:
                    continue
                bucket = _legacy_split_bucket(pi, train_r, float(cfg.val_ratio), test_r)
                pi += 1
                if bucket != split:
                    continue
                self.src_lines.append(s0)
                self.tgt_lines.append(t0)

        print(
            f"[{split}] 加载 {len(self.src_lines)} 条句对 (legacy 单文件 bucket <- val_ratio/test_ratio) <- {dp}"
        )

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        src_ids = self.src_tok.encode(self.src_lines[i]).ids
        tgt_raw = self.tgt_tok.encode(self.tgt_lines[i]).ids

        budget = self.cfg.max_seq_len
        inner = min(len(tgt_raw), max(0, budget - 2))
        tgt_ids = [BOS_ID] + tgt_raw[:inner] + [EOS_ID]

        src_ids = src_ids[:budget]

        return src_ids, tgt_ids


def collate_batch(
    batch: list[tuple[list[int], list[int]]],
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_list, tgt_list = zip(*batch)
    src_tensors = [torch.tensor(x, dtype=torch.long) for x in src_list]
    tgt_tensors = [torch.tensor(x, dtype=torch.long) for x in tgt_list]
    src_pad = torch.nn.utils.rnn.pad_sequence(
        src_tensors, batch_first=True, padding_value=pad_idx
    )
    tgt_pad = torch.nn.utils.rnn.pad_sequence(
        tgt_tensors, batch_first=True, padding_value=pad_idx
    )
    return src_pad, tgt_pad


def tokenizer_special_ids(tgt_tok: Tokenizer) -> tuple[int, int, int]:
    pad, bos, eos = PAD_ID, BOS_ID, EOS_ID
    for i, name in ((pad, "PAD"), (bos, "BOS"), (eos, "EOS")):
        if tgt_tok.id_to_token(i) is None:
            raise RuntimeError(f"目标分词器缺少 id={i} ({name})")
    return pad, bos, eos
