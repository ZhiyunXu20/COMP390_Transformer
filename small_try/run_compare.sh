#!/usr/bin/env bash
# 一键：生成子语料 → 训练两种注意力 → 生成 report.txt
# 用法：在仓库根目录：bash small_try/run_compare.sh
# 输出目录默认为仓库根下 runs/<run_name>/（见 Config.output_dir）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ">>> [1/4] 生成子语料（约 5 万句，若已存在可注释本步）"
python create_subset.py --lines 50000 --out data/corpus_50k.tsv

echo ">>> [2/4] 训练 dot_product -> ${REPO}/runs/fast_dot/"
python train.py --attention dot_product --name fast_dot --max-steps 3000

echo ">>> [3/4] 训练 additive -> ${REPO}/runs/fast_add/"
python train.py --attention additive --name fast_add --max-steps 3000

echo ">>> [4/4] 生成 report.txt"
python compare_runs.py \
  --dot "${REPO}/runs/fast_dot/metrics.json" \
  --add "${REPO}/runs/fast_add/metrics.json" \
  --out "${SCRIPT_DIR}/report.txt"

echo "完成。请查看 report.txt 与各 run 目录下 metrics.json / best.pt"
