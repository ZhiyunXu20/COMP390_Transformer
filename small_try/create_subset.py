#!/usr/bin/env python3
"""从全量 EN-FR.txt 截取前 N 条有效句对，供 small_try 使用（不修改原文件）。"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SRC = "/root/autodl-tmp/data/EN-FR.txt"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "corpus_50k.tsv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC, help="原始 Tab 语料")
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="输出子语料路径（默认 small_try/data/corpus_50k.tsv）",
    )
    p.add_argument("--lines", type=int, default=50_000, help="有效句对条数上限")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.src, "r", encoding="utf-8") as inf, open(
        out_path, "w", encoding="utf-8"
    ) as outf:
        for line in inf:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            outf.write(parts[0] + "\t" + parts[1] + "\n")
            n += 1
            if n >= args.lines:
                break

    print(f"已写入 {n} 条句对 -> {out_path}")


if __name__ == "__main__":
    main()
