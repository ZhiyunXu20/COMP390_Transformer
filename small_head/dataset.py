"""与 base_1 相同逻辑；仅路径由 small_head/config 指定。"""

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


class TabParallelDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        src_tok: Tokenizer,
        tgt_tok: Tokenizer,
        split: Literal["train", "val"],
    ):
        self.cfg = cfg
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.split = split
        self.src_lines: list[str] = []
        self.tgt_lines: list[str] = []

        period = max(1, int(round(1.0 / cfg.val_ratio)))
        with open(cfg.data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                is_val = idx % period == 0
                if split == "val" and not is_val:
                    continue
                if split == "train" and is_val:
                    continue
                self.src_lines.append(parts[0])
                self.tgt_lines.append(parts[1])

        print(f"[{split}] 加载 {len(self.src_lines)} 条句对 (period={period})")

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
