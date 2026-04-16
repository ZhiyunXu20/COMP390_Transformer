#!/usr/bin/env bash
# small_try 全流程：子语料 → dot 训练 → additive 训练 → 报告 + 合并 JSON
# 遇错立即退出 (set -euo pipefail)。日志请用 nohup 重定向或 tee。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# W&B project=attention-small，group=try-en-fr（见 config）
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache}"
mkdir -p "$WANDB_DIR" logs results

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "======== small_try 流水线开始 ========"
log "PWD=$ROOT"
log "CUDA: ${CUDA_VISIBLE_DEVICES:-默认}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true

log "[1/5] 生成子语料 corpus_50k.tsv（5 万句）"
python create_subset.py --lines 50000 --out data/corpus_50k.tsv

MAXS="${TRAIN_MAX_STEPS:-3000}"
EXTRA=()
[[ -n "${TRAIN_VAL_EVERY:-}" ]] && EXTRA+=(--val-every "${TRAIN_VAL_EVERY}")
[[ "${EVAL_LIGHT:-}" == 1 ]] && EXTRA+=(--eval-light)
BATCH_OPT=()
[[ -n "${TRAIN_BATCH_SIZE:-}" ]] && BATCH_OPT+=(--batch-size "$TRAIN_BATCH_SIZE")

log "[2/5] 训练 dot_product → runs/fast_dot/（W&B project=attention-small, max_steps=$MAXS）"
python train.py --attention dot_product --name fast_dot --max-steps "$MAXS" "${EXTRA[@]}" "${BATCH_OPT[@]}"

log "[3/5] 训练 additive → runs/fast_add/"
python train.py --attention additive --name fast_add --max-steps "$MAXS" "${EXTRA[@]}" "${BATCH_OPT[@]}"

log "[4/5] 生成 report.txt 与 bundle_metrics.json"
python compare_runs.py \
  --dot runs/fast_dot/metrics.json \
  --add runs/fast_add/metrics.json \
  --out report.txt \
  --json-bundle results/bundle_metrics.json

log "[5/5] 写入流水线清单 manifest.json"
small_try_root="$ROOT" python - << 'PY'
import json, time, os
from pathlib import Path
root = Path(os.environ["small_try_root"])
m = {
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "data": str(root / "data/corpus_50k.tsv"),
    "report": str(root / "report.txt"),
    "bundle": str(root / "results/bundle_metrics.json"),
    "runs": {
        "dot": str(root / "runs/fast_dot"),
        "add": str(root / "runs/fast_add"),
    },
}
(root / "results" / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("MANIFEST ->", root / "results/manifest.json")
PY

log "======== 全部成功 ========"
log "查看: report.txt | results/bundle_metrics.json | runs/fast_*/metrics.json"
