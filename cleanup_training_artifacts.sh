#!/usr/bin/env bash
# 删除各实验包下的训练产物：runs、wandb_cache、logs、报告与 results 内归档 json。
# 保留源码、requirements、plan*.txt、data/ 子语料等。
# 用法：bash /root/autodl-tmp/cleanup_training_artifacts.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$ROOT/wandb_cache_attention_small" "$ROOT/wandb_cache_attention_small_smoke" 2>/dev/null || true
for name in small_try small_head small_swap base_1 base_improve; do
  d="$ROOT/$name"
  [[ -d "$d" ]] || continue
  rm -rf "$d/wandb_cache" "$d/runs" "$d/logs"
  mkdir -p "$d/runs" "$d/results" "$d/logs"
  rm -f "$d/report.txt" "$d/report_head.txt" "$d/report_swap.txt" "$d/report_improve.txt" 2>/dev/null || true
  rm -f "$d/results"/* 2>/dev/null || true
  echo "cleaned $d"
done
echo "done."
