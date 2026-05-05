#!/usr/bin/env bash
# small_head：仅单头点积训练；多头基线引用 small_try fast_dot metrics。
# 与 small_try 使用同一 split + train_only tokenizer 协议。
# 加性注意力仍可在代码中启用：train.py --attention additive --allow-additive-attention

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache}"
mkdir -p "$WANDB_DIR" logs results

BASELINE_METRICS="${BASELINE_METRICS:-$REPO/runs/fast_dot/metrics.json}"

TRAIN_PATH="${TRAIN_PATH:-data/splits/en_fr_50k_seed42/train.tsv}"
VAL_PATH="${VAL_PATH:-data/splits/en_fr_50k_seed42/val.tsv}"
TEST_PATH="${TEST_PATH:-data/splits/en_fr_50k_seed42/test.tsv}"
TOKENIZER_SRC="${TOKENIZER_SRC:-data/tokenizer_src_train_only.json}"
TOKENIZER_TGT="${TOKENIZER_TGT:-data/tokenizer_tgt_train_only.json}"
TOKENIZER_METADATA="${TOKENIZER_METADATA:-data/tokenizer_metadata.json}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

on_err() {
  log "ERROR: 流水线在 line $1 失败，退出码 $2"
  exit "$2"
}
trap 'on_err ${LINENO} $?' ERR

log "======== small_head 流水线开始 ========"
log "ROOT=$ROOT REPO=$REPO"
log "BASELINE_METRICS=$BASELINE_METRICS"
log "划分: train=$TRAIN_PATH val=$VAL_PATH test=$TEST_PATH（相对仓库根）"
log "tokenizer: src=$TOKENIZER_SRC tgt=$TOKENIZER_TGT meta=$TOKENIZER_METADATA"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-默认}"

if [[ ! -f "$BASELINE_METRICS" ]]; then
  log "错误: 缺少多头基线 metrics: $BASELINE_METRICS"
  log "请先完成 small_try 的 fast_dot，或设置 BASELINE_METRICS 指向有效 metrics.json"
  exit 1
fi

for rel in "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$TOKENIZER_SRC" "$TOKENIZER_TGT"; do
  if [[ ! -f "$REPO/$rel" ]]; then
    log "ERROR: 缺少文件: $REPO/$rel"
    log "请先运行 scripts/make_splits.py 与 scripts/train_tokenizers_from_train_split.py（见 docs/REPRODUCIBILITY.md）"
    exit 1
  fi
done
if [[ ! -f "$REPO/$TOKENIZER_METADATA" ]]; then
  log "WARN: 未找到 $REPO/$TOKENIZER_METADATA（可选；manifest 仍会记录预期路径）"
fi

python - << 'PY' || exit 1
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), end=" ")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("(CPU)")
PY

for m in sacrebleu wandb tqdm matplotlib seaborn tokenizers; do
  python -c "import importlib; importlib.import_module('$m')" || { log "错误: 缺少 Python 依赖: $m"; exit 1; }
done

log "[0/3] 参数量摘要 -> results/param_counts.json"
python summarize_params.py

MAXS="${TRAIN_MAX_STEPS:-3000}"
EXTRA=()
[[ -n "${TRAIN_VAL_EVERY:-}" ]] && EXTRA+=(--val-every "${TRAIN_VAL_EVERY}")
[[ "${EVAL_LIGHT:-}" == 1 ]] && EXTRA+=(--eval-light)
BATCH_OPT=()
[[ -n "${TRAIN_BATCH_SIZE:-}" ]] && BATCH_OPT+=(--batch-size "$TRAIN_BATCH_SIZE")

log "[1/3] 单头 + dot_product -> ${REPO}/runs/head_1h_dot/（W&B project=attention-small-2, group=head-ablation, max_steps=$MAXS）"
python train.py \
  --attention dot_product \
  --n-heads 1 \
  --name head_1h_dot \
  --max-steps "$MAXS" \
  --train-path "$TRAIN_PATH" \
  --val-path "$VAL_PATH" \
  --test-path "$TEST_PATH" \
  --tokenizer-src "$TOKENIZER_SRC" \
  --tokenizer-tgt "$TOKENIZER_TGT" \
  "${EXTRA[@]}" \
  "${BATCH_OPT[@]}"

log "[2/3] 对比 report_head.txt（基线 + 单头点积，两列）"
python compare_head_runs.py \
  --baseline "$BASELINE_METRICS" \
  --one-dot "$REPO/runs/head_1h_dot/metrics.json" \
  --out "$ROOT/report_head.txt" \
  --json-bundle "$ROOT/results/bundle_head_metrics.json" \
  --prefer-test-eval

log "[3/3] manifest_head.json（仓库相对路径）"
python "$REPO/scripts/write_pipeline_manifest.py" small_head \
  --repo-root "$REPO" \
  --small-head-dir "$ROOT" \
  --train-path "$TRAIN_PATH" \
  --val-path "$VAL_PATH" \
  --test-path "$TEST_PATH" \
  --tokenizer-src "$TOKENIZER_SRC" \
  --tokenizer-tgt "$TOKENIZER_TGT" \
  --tokenizer-metadata "$TOKENIZER_METADATA" \
  --baseline-metrics "$BASELINE_METRICS"

log "======== small_head 全部成功 ========"
log "查看: report_head.txt | results/bundle_head_metrics.json | results/manifest_head.json"
