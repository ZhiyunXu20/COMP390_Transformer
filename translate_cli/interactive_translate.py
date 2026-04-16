#!/usr/bin/env python3
"""
英->法 交互翻译：根据 checkpoint 自动选择 small_try / small_head / small_swap / base_improve / base_1 代码目录并加载权重。
用法:
  python interactive_translate.py --checkpoint /path/to/best.pt
  echo 'Hello .' | python interactive_translate.py -c ...   # 非交互一行
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer


def resolve_codebase_root(checkpoint: Path) -> Path:
    """checkpoint 路径须位于 .../small_try/...、.../small_head/...、.../small_swap/...、.../base_improve/... 或 .../base_1/... 下。"""
    p = checkpoint.resolve()
    for name in ("small_try", "small_head", "small_swap", "base_improve", "base_1"):
        if name in p.parts:
            i = p.parts.index(name)
            return Path(*p.parts[: i + 1])
    raise ValueError(
        f"无法从 {checkpoint} 推断训练代码目录，请将 .pt 放在 small_try/small_head/small_swap/base_improve/base_1/runs/ 下"
    )


def load_model_and_tokenizers(checkpoint: Path, device: torch.device):
    root = resolve_codebase_root(checkpoint)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

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


def main() -> None:
    p = argparse.ArgumentParser(description="英->法 终端翻译（加载 Seq2SeqTransformer checkpoint）")
    p.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=Path("/root/autodl-tmp/small_try/runs/fast_dot/best.pt"),
        help="best.pt 或 last.pt",
    )
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        print(f"找不到 checkpoint: {args.checkpoint}", file=sys.stderr)
        raise SystemExit(1)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"加载: {args.checkpoint}", file=sys.stderr)
    print(f"设备: {device}", file=sys.stderr)

    bundle = load_model_and_tokenizers(args.checkpoint, device)
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
