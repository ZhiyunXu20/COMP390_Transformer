#!/usr/bin/env python3
"""从 run_all_small 完整日志中提取：流水线顺序、每轮完成行、对比报告与结论（去掉 tqdm/wandb 噪声）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

STEP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[\d+/\d+\]")
RUNALL = re.compile(r"\[run_all_small\]")
DONE = re.compile(r"^完成。metrics")
SEP = re.compile(r"^={20,}$")
EPOCH = re.compile(r"^(epoch \d+:|val:\s+\d+%|\s*$)")
WB = re.compile(r"^wandb:")


def keep_line(line: str) -> bool:
    s = line.rstrip("\n")
    if RUNALL.search(s) or "验证指标:" in s:
        return True
    if s.startswith("torch ") and len(s) < 120:
        return True
    if "========" in s and ("流水线" in s or "成功" in s):
        return True
    if STEP.match(s):
        return True
    if DONE.match(s):
        return True
    if s.startswith("[compare_runs]") or ("MANIFEST" in s and "->" in s):
        return True
    if s.startswith(("small_try：", "small_head：", "small_swap：")):
        return True
    if SEP.match(s):
        return True
    if s.startswith("说明：") or s.startswith("点积 ") or s.startswith("加性 "):
        return True
    if s.startswith("单 run（") or "data_path:" in s:
        return True
    if re.match(r"^指标\s+", s) or re.match(r"^extra:", s):
        return True
    if "extra_metrics（" in s:
        return True
    if s.startswith("结论（") or s.startswith("多指标摘要"):
        return True
    if "在本设定下" in s or s.strip().startswith("缩放点积"):
        return True
    if "wandb.ai" in s and "http" in s:
        return False
    if s.startswith("  Δ") or s.startswith("  "):
        if any(
            k in s
            for k in (
                "dot_product",
                "additive",
                "mh_dot",
                "1h_dot",
                "chrF",
                "BERTScore",
                "COMET",
            )
        ):
            return True
    if re.match(
        r"^\s*(final_val_loss|final_bleu|best_bleu|best_bleu_during_training|optimizer_steps)\b",
        s,
    ):
        return True
    if re.match(r"^-{20,}$", s):
        return True
    if "撰写报告时可补充" in s:
        return True
    if "multi_head" in s or "single_head" in s and "n_heads" in s:
        return True
    if "已写入" in s and "param_counts" in s:
        return True
    if s.startswith("ERROR") or s.startswith("Traceback"):
        return True
    if EPOCH.match(s) or WB.match(s):
        return False
    if "it/s" in s or "█" in s[:5] if s else False:
        return False
    return False


def summarize(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = [
        f"# 流水线摘要（自 {path.name} 自动提取）",
        f"# 源文件: {path.resolve()}",
        "",
    ]
    for line in lines:
        if keep_line(line):
            out.append(line.rstrip())
    # 去掉连续空行
    compact: list[str] = []
    prev_empty = False
    for ln in out:
        empty = not ln.strip()
        if empty and prev_empty:
            continue
        compact.append(ln)
        prev_empty = empty
    return "\n".join(compact) + "\n"


def main() -> int:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "").resolve()
    if not p.is_file():
        print("用法: python scripts/summarize_pipeline_log.py <完整.log>", file=sys.stderr)
        return 2
    sys.stdout.write(summarize(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
