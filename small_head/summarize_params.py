#!/usr/bin/env python3
"""打印各实验配置下 Seq2SeqTransformer 总参数量（用于架构可比性说明）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import Config
from model import Seq2SeqTransformer


def total_params(model: Seq2SeqTransformer) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    pad_idx = 1
    rows = []
    for label, nh, att in [
        ("multi_head_dot_baseline", 4, "dot_product"),
        ("single_head_dot", 1, "dot_product"),
        ("single_head_additive", 1, "additive"),
    ]:
        cfg = Config()
        cfg.n_heads = nh
        cfg.attention_type = att  # type: ignore[assignment]
        m = Seq2SeqTransformer(cfg, pad_idx=pad_idx)
        n = total_params(m)
        d_k = cfg.d_model // cfg.n_heads
        rows.append(
            {
                "label": label,
                "n_heads": nh,
                "attention_type": att,
                "d_k": d_k,
                "total_parameters": n,
            }
        )
        print(f"{label}: n_heads={nh} d_k={d_k} attn={att} total_params={n:,}")

    out = Path(__file__).resolve().parent / "results" / "param_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
