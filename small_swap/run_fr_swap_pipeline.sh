#!/usr/bin/env bash
# small_swap：法→英；split 语料 + dot_product → 单 run 报告 + MANIFEST（与 small_try 同一划分协议）
# 加性注意力：train.py --attention additive --allow-additive-attention

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache}"
mkdir -p "$WANDB_DIR" logs results

TRAIN_PATH="${TRAIN_PATH:-data/splits/en_fr_50k_seed42/train.tsv}"
VAL_PATH="${VAL_PATH:-data/splits/en_fr_50k_seed42/val.tsv}"
TEST_PATH="${TEST_PATH:-data/splits/en_fr_50k_seed42/test.tsv}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

on_err() {
  log "ERROR: 流水线在 line $1 失败，退出码 $2"
  exit "$2"
}
trap 'on_err ${LINENO} $?' ERR

log "======== small_swap（法→英）流水线开始 ========"
log "ROOT=$ROOT REPO=$REPO"
log "划分: train=$TRAIN_PATH val=$VAL_PATH test=$TEST_PATH（相对仓库根）"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-默认}"

for rel in "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH"; do
  if [[ ! -f "$REPO/$rel" ]]; then
    log "ERROR: 缺少划分文件: $REPO/$rel"
    log "请先运行: python scripts/make_splits.py ...（见 docs/REPRODUCIBILITY.md）"
    exit 1
  fi
done

python - << 'PY' || exit 1
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), end=" ")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("(CPU)")
PY

for m in sacrebleu wandb tqdm matplotlib seaborn tokenizers; do
  python -c "import importlib; importlib.import_module('$m')" || { log "错误: 缺少依赖: $m"; exit 1; }
done

MAXS="${TRAIN_MAX_STEPS:-3000}"
EXTRA=()
[[ -n "${TRAIN_VAL_EVERY:-}" ]] && EXTRA+=(--val-every "${TRAIN_VAL_EVERY}")
[[ "${EVAL_LIGHT:-}" == 1 ]] && EXTRA+=(--eval-light)
BATCH_OPT=()
[[ -n "${TRAIN_BATCH_SIZE:-}" ]] && BATCH_OPT+=(--batch-size "$TRAIN_BATCH_SIZE")

log "[1/3] dot_product → $REPO/runs/swap_fr_dot/（W&B project=attention-small-2, group=swap-fr-en, max_steps=$MAXS）"
python train.py \
  --attention dot_product \
  --name swap_fr_dot \
  --max-steps "$MAXS" \
  --train-path "$TRAIN_PATH" \
  --val-path "$VAL_PATH" \
  --test-path "$TEST_PATH" \
  "${EXTRA[@]}" \
  "${BATCH_OPT[@]}"

log "[2/3] report_swap.txt（单 run；若曾训 additive 可传 --add 路径再跑 compare_runs）"
python compare_runs.py \
  --dot "$REPO/runs/swap_fr_dot/metrics.json" \
  --out "$ROOT/report_swap.txt" \
  --json-bundle "$ROOT/results/bundle_swap_metrics.json" \
  --prefer-test-eval

log "[3/3] manifest_swap.json（仓库相对路径）"
python "$REPO/scripts/write_pipeline_manifest.py" small_swap \
  --repo-root "$REPO" \
  --small-swap-dir "$ROOT" \
  --train-path "$TRAIN_PATH" \
  --val-path "$VAL_PATH" \
  --test-path "$TEST_PATH"

log "======== small_swap 全部成功 ========"
log "查看: report_swap.txt | results/bundle_swap_metrics.json | $REPO/runs/swap_fr_dot/"
