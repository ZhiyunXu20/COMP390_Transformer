#!/usr/bin/env bash
# small_try 全流程：split 语料（与 Config 一致）→ dot → additive → 报告 + 合并 JSON + MANIFEST
# 遇错立即退出 (set -euo pipefail)。日志请用 nohup 重定向或 tee。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# W&B project=attention-small-2，group=try-en-fr（见 config）
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache}"
mkdir -p "$WANDB_DIR" logs results

TRAIN_PATH="${TRAIN_PATH:-data/splits/en_fr_50k_seed42/train.tsv}"
VAL_PATH="${VAL_PATH:-data/splits/en_fr_50k_seed42/val.tsv}"
TEST_PATH="${TEST_PATH:-data/splits/en_fr_50k_seed42/test.tsv}"
TOKENIZER_SRC="${TOKENIZER_SRC:-data/tokenizer_src_train_only.json}"
TOKENIZER_TGT="${TOKENIZER_TGT:-data/tokenizer_tgt_train_only.json}"
TOKENIZER_METADATA="${TOKENIZER_METADATA:-data/tokenizer_metadata.json}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "======== small_try 流水线开始 ========"
log "PWD=$ROOT REPO=$REPO"
log "划分: train=$TRAIN_PATH val=$VAL_PATH test=$TEST_PATH（相对仓库根）"
log "tokenizer: src=$TOKENIZER_SRC tgt=$TOKENIZER_TGT meta=$TOKENIZER_METADATA"
log "CUDA: ${CUDA_VISIBLE_DEVICES:-默认}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true

log "[1/5] 校验划分与 train_only 词表文件存在"
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

MAXS="${TRAIN_MAX_STEPS:-3000}"
EXTRA=()
[[ -n "${TRAIN_VAL_EVERY:-}" ]] && EXTRA+=(--val-every "${TRAIN_VAL_EVERY}")
[[ "${EVAL_LIGHT:-}" == 1 ]] && EXTRA+=(--eval-light)
BATCH_OPT=()
[[ -n "${TRAIN_BATCH_SIZE:-}" ]] && BATCH_OPT+=(--batch-size "$TRAIN_BATCH_SIZE")

COMMON_TRAIN=(
  --train-path "$TRAIN_PATH"
  --val-path "$VAL_PATH"
  --test-path "$TEST_PATH"
  --tokenizer-src "$TOKENIZER_SRC"
  --tokenizer-tgt "$TOKENIZER_TGT"
)

log "[2/5] 训练 dot_product → runs/fast_dot/（W&B project=attention-small-2, max_steps=$MAXS）"
python train.py \
  --attention dot_product \
  --name fast_dot \
  --max-steps "$MAXS" \
  "${COMMON_TRAIN[@]}" \
  "${EXTRA[@]}" \
  "${BATCH_OPT[@]}"

log "[3/5] 训练 additive → runs/fast_add/"
python train.py \
  --attention additive \
  --name fast_add \
  --max-steps "$MAXS" \
  "${COMMON_TRAIN[@]}" \
  "${EXTRA[@]}" \
  "${BATCH_OPT[@]}"

log "[4/5] 生成 report.txt 与 bundle_metrics.json（metrics 在仓库根 runs/，见 train_runtime）"
python compare_runs.py \
  --dot "$REPO/runs/fast_dot/metrics.json" \
  --add "$REPO/runs/fast_add/metrics.json" \
  --out report.txt \
  --json-bundle results/bundle_metrics.json \
  --prefer-test-eval

log "[5/5] 写入流水线清单 manifest.json（仓库相对路径）"
python "$REPO/scripts/write_pipeline_manifest.py" small_try \
  --repo-root "$REPO" \
  --small-try-dir "$ROOT" \
  --train-path "$TRAIN_PATH" \
  --val-path "$VAL_PATH" \
  --test-path "$TEST_PATH" \
  --tokenizer-src "$TOKENIZER_SRC" \
  --tokenizer-tgt "$TOKENIZER_TGT" \
  --tokenizer-metadata "$TOKENIZER_METADATA"

log "======== 全部成功 ========"
log "查看: report.txt | results/bundle_metrics.json | $REPO/runs/fast_*/"
