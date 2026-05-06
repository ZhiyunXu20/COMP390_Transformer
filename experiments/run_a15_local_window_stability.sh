#!/usr/bin/env bash
# A15: local_window 数值稳定性诊断（勿修改 runs/var_local_window/）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-/root/miniconda3/bin/python}"
mkdir -p logs

BASE=(
  small_try/train.py
  --train-path data/splits/en_fr_50k_seed42/train.tsv
  --val-path data/splits/en_fr_50k_seed42/val.tsv
  --test-path data/splits/en_fr_50k_seed42/test.tsv
  --tokenizer-src data/tokenizer_src_train_only.json
  --tokenizer-tgt data/tokenizer_tgt_train_only.json
  --output-dir runs
  --attention-type local_window
  --seed 42
  --max-steps 1500
  --eval-split val
  --no-wandb
  --eval-light
  --no-attention-plots
)

echo "### A15 exp1 fp32"
"$PY" "${BASE[@]}" \
  --name var_local_window_fp32 \
  --local-window-size 4 \
  --no-bf16-autocast \
  2>&1 | tee logs/a15_fp32_train.log

echo "### A15 exp2 lr1e4 bf16"
"$PY" "${BASE[@]}" \
  --name var_local_window_lr1e4 \
  --local-window-size 4 \
  --lr 1e-4 \
  2>&1 | tee logs/a15_lr1e4_train.log

echo "### A15 exp3 window16 bf16"
"$PY" "${BASE[@]}" \
  --name var_local_window_window16 \
  --local-window-size 16 \
  2>&1 | tee logs/a15_window16_train.log

EV=(evaluate_test.py
  --test-file data/splits/en_fr_50k_seed42/test.tsv
  --tokenizer-src data/tokenizer_src_train_only.json
  --tokenizer-tgt data/tokenizer_tgt_train_only.json
  --max-new-tokens 64
  --batch-size 32
  --pkg small_try
  --eval-light
)

for run in var_local_window_fp32 var_local_window_lr1e4 var_local_window_window16; do
  echo "### A15 evaluate_test $run"
  "$PY" "${EV[@]}" \
    --checkpoint "runs/${run}/best.pt" \
    --output-dir "runs/${run}/test_eval" \
    2>&1 | tee "logs/a15_${run}_eval.log"
done

echo "### A15 generate report"
"$PY" experiments/generate_local_window_stability_report.py
