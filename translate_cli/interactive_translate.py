#!/usr/bin/env python3
"""
英->法 交互翻译：用 --pkg 指定训练时使用的代码目录（small_try / small_head / …）并加载权重。
在仓库根目录运行；-c 可为相对路径（相对仓库根）。

  python translate_cli/interactive_translate.py -c runs/<run>/best.pt --pkg small_try
  echo 'Hello.' | python translate_cli/interactive_translate.py -c runs/fast_dot/best.pt --pkg small_try
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer


def load_model_and_tokenizers(
    checkpoint: Path,
    device: torch.device,
    *,
    codebase_root: Path,
):
    if str(codebase_root) not in sys.path:
        sys.path.insert(0, str(codebase_root))

    from dataset import load_tokenizers, tokenizer_special_ids  # noqa: E402
    from model import Seq2SeqTransformer  # noqa: E402

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model" not in ckpt or "cfg" not in ckpt:
        raise ValueError("checkpoint 需包含 'model' 与 'cfg'（本仓库 train.py 保存格式）")

    raw = {k: v for k, v in ckpt["cfg"].items() if not str(k).startswith("_")}
    cfg = SimpleNamespace(**raw)

    src_tok, tgt_tok = load_tokenizers(cfg)
    pad_idx, bos_id, eos_id = tokenizer_special_ids(tgt_tok)

    model = Seq2SeqTransformer(cfg, pad_idx=pad_idx).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, src_tok, tgt_tok, cfg, pad_idx, bos_id, eos_id


@torch.no_grad()
def translate_line(
    text: str,
    model,
    src_tok: Tokenizer,
    tgt_tok: Tokenizer,
    cfg,
    pad_idx: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
) -> str:
    text = text.strip()
    if not text:
        return ""
    ids = src_tok.encode(text).ids
    budget = getattr(cfg, "max_seq_len", 128)
    ids = ids[:budget]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    max_gen = getattr(cfg, "max_gen_len", 96)
    gen = model.greedy_decode(src, bos_id, eos_id, max_gen)
    hyp_ids = gen[0].tolist()
    while hyp_ids and hyp_ids[0] == bos_id:
        hyp_ids.pop(0)
    while hyp_ids and hyp_ids[-1] == eos_id:
        hyp_ids.pop(-1)
    hyp_ids = [t for t in hyp_ids if t not in (pad_idx,)]
    return tgt_tok.decode(hyp_ids)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="英->法 终端翻译（加载 Seq2SeqTransformer checkpoint）")
    p.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="best.pt / last.pt（相对仓库根或绝对路径）",
    )
    p.add_argument(
        "--pkg",
        type=str,
        default="small_try",
        choices=("small_try", "small_head", "small_swap", "base_improve", "base_1"),
        help="训练时使用的代码目录（checkpoint 在仓库 runs/ 下时必须指定以加载 dataset/model）",
    )
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    args = p.parse_args()
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = _REPO_ROOT / "runs" / "best.pt"
    else:
        ckpt = Path(ckpt)
        if not ckpt.is_absolute():
            ckpt = (_REPO_ROOT / ckpt).resolve()

    if not ckpt.is_file():
        print(f"找不到 checkpoint: {ckpt}", file=sys.stderr)
        print(
            "示例：python translate_cli/interactive_translate.py -c runs/<run_name>/best.pt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"加载: {ckpt}", file=sys.stderr)
    print(f"设备: {device}", file=sys.stderr)

    codebase_root = (_REPO_ROOT / args.pkg).resolve()
    if not codebase_root.is_dir():
        print(f"找不到代码目录: {codebase_root}", file=sys.stderr)
        raise SystemExit(1)

    bundle = load_model_and_tokenizers(ckpt, device, codebase_root=codebase_root)
    model, src_tok, tgt_tok, cfg, pad_idx, bos_id, eos_id = bundle

    def one(line: str) -> None:
        out = translate_line(
            line, model, src_tok, tgt_tok, cfg, pad_idx, bos_id, eos_id, device
        )
        print(out)

    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                one(line)
        return

    print("英->法翻译，输入英文回车；空行退出。", file=sys.stderr)
    while True:
        try:
            line = input("EN> ")
        except (EOFError, KeyboardInterrupt):
            print("", file=sys.stderr)
            break
        if not line.strip():
            break
        one(line)


if __name__ == "__main__":
    main()
