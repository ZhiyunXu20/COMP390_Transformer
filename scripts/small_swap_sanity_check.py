#!/usr/bin/env python3
"""随机抽样 small_swap 验证/测试句对，核对 FR→EN 方向；可选加载 checkpoint 打印 greedy 译文。

用法示例::

    python scripts/small_swap_sanity_check.py --split val --seed 42 --n 5
    python scripts/small_swap_sanity_check.py --split test --checkpoint runs/swap_fr_dot/best.pt --n 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _repo_hint(text: str) -> str:
    """粗粒度脚本提示（非严格语种识别）：法语常见字母 vs 拉丁字母为主的英语。"""
    t = text.strip()
    if not t:
        return "(empty)"
    fr_markers = sum(1 for c in t.lower() if c in "éèêëàâôîïûùçœæ")
    ratio = fr_markers / max(1, len(t))
    ascii_ratio = sum(1 for c in t if ord(c) < 128) / max(1, len(t))
    bits = []
    if fr_markers >= 2 or ratio > 0.03:
        bits.append("likely_fr_chars")
    if ascii_ratio > 0.92 and fr_markers == 0:
        bits.append("likely_en_ascii_heavy")
    return ", ".join(bits) if bits else "ambiguous"


def main() -> None:
    p = argparse.ArgumentParser(description="small_swap FR→EN 数据与解码抽样核对")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--n", type=int, default=5, help="抽样条数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default=None, help="可选：best.pt 等，用于 greedy 解码")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    pkg = _REPO / "small_swap"
    sys.path.insert(0, str(pkg))

    from config import Config
    from dataset import TabParallelDataset, collate_batch, load_tokenizers, tokenizer_special_ids
    from model import Seq2SeqTransformer
    from train_runtime import PATH_FIELDS_DEFAULT, infer_repo_root, materialize_path_fields

    cfg = Config()
    repo_root = infer_repo_root(pkg / "train.py")
    materialize_path_fields(cfg, repo_root, PATH_FIELDS_DEFAULT)

    if not getattr(cfg, "swap_parallel_columns", False):
        print("警告: Config.swap_parallel_columns 应为 True（FR→EN）", file=sys.stderr)

    src_tok, tgt_tok = load_tokenizers(cfg)
    cfg.src_vocab_size = src_tok.get_vocab_size()
    cfg.tgt_vocab_size = tgt_tok.get_vocab_size()
    pad_idx, bos_id, eos_id = tokenizer_special_ids(tgt_tok)

    ds = TabParallelDataset(cfg, src_tok, tgt_tok, args.split)
    n = len(ds)
    if n == 0:
        raise SystemExit(f"[{args.split}] 数据集为空")

    rng = random.Random(args.seed)
    indices = rng.sample(range(n), min(args.n, n))

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model: Seq2SeqTransformer | None = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = (repo_root / ckpt_path).resolve()
        if not ckpt_path.is_file():
            raise SystemExit(f"checkpoint 不存在: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_cfg = {k: v for k, v in ckpt["cfg"].items() if not str(k).startswith("_")}
        mcfg = SimpleNamespace(**raw_cfg)
        materialize_path_fields(mcfg, repo_root, PATH_FIELDS_DEFAULT)
        mcfg.tokenizer_src = str(cfg.tokenizer_src)
        mcfg.tokenizer_tgt = str(cfg.tokenizer_tgt)
        mcfg.src_vocab_size = src_tok.get_vocab_size()
        mcfg.tgt_vocab_size = tgt_tok.get_vocab_size()
        if not hasattr(mcfg, "max_gen_len") or mcfg.max_gen_len is None:
            mcfg.max_gen_len = getattr(cfg, "max_gen_len", 64)
        model = Seq2SeqTransformer(mcfg, pad_idx=pad_idx).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

    print(f"repo_root={repo_root}")
    print(f"split={args.split} size={n} sample_indices={indices}")
    print(f"tokenizer_src={cfg.tokenizer_src}")
    print(f"tokenizer_tgt={cfg.tokenizer_tgt}")
    print(f"swap_parallel_columns={getattr(cfg, 'swap_parallel_columns', None)}")
    print("-" * 72)

    for k, idx in enumerate(indices):
        src_ids, tgt_ids = ds[idx]
        src_s = src_tok.decode([t for t in src_ids if t != pad_idx])
        ref_s = tgt_tok.decode([t for t in tgt_ids if t not in (pad_idx, bos_id, eos_id)])
        print(f"\n### Sample {k + 1} (dataset index {idx})")
        print(f"source (encoder input, expect FR): {_repo_hint(src_s)}")
        print(src_s[:800])
        print(f"reference (decoder target, expect EN): {_repo_hint(ref_s)}")
        print(ref_s[:800])

        if model is not None:
            src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = model.greedy_decode(src_t, bos_id, eos_id, cfg.max_gen_len)
            hyp_ids = gen[0].tolist()
            while hyp_ids and hyp_ids[0] == bos_id:
                hyp_ids.pop(0)
            while hyp_ids and hyp_ids[-1] == eos_id:
                hyp_ids.pop(-1)
            hyp_s = tgt_tok.decode(hyp_ids)
            print(f"prediction (greedy EN tgt_tok): {_repo_hint(hyp_s)}")
            print(hyp_s[:800])

    if model is None:
        print("\n(未传 --checkpoint：仅核对语料方向；加 checkpoint 可核对译文是否为英语。)")

    # 可选：单 batch DataLoader 冒烟（与训练一致）
    if model is not None:
        subset = torch.utils.data.Subset(ds, indices[:1])
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate_batch(b, pad_idx),
        )
        src_b, tgt_b = next(iter(loader))
        src_b = src_b.to(device)
        with torch.no_grad():
            _ = model(src_b, tgt_b[:, :-1].to(device))
        print("\n[smoke] one forward + greedy OK")


if __name__ == "__main__":
    main()
