#!/usr/bin/env bash
# small_swap：法→英；子语料 → 仅 dot_product → 单 run 报告 + MANIFEST
# 加性注意力：train.py --attention additive --allow-additive-attention

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache}"
mkdir -p "$WANDB_DIR" logs results data

CORPUS_SRC="${CORPUS_SRC:-/root/autodl-tmp/small_try/data/corpus_50k.tsv}"
CORPUS_LOCAL="$ROOT/data/corpus_50k.tsv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

on_err() {
  log "ERROR: 流水线在 line $1 失败，退出码 $2"
  exit "$2"
}
trap 'on_err ${LINENO} $?' ERR

log "======== small_swap（法→英）流水线开始 ========"
log "ROOT=$ROOT"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-默认}"

if [[ ! -f "$CORPUS_SRC" ]]; then
  log "本地无 $CORPUS_SRC，尝试从 EN-FR.txt 生成子语料…"
  python create_subset.py --lines 50000 --out "$CORPUS_LOCAL"
else
  log "使用语料: $CORPUS_SRC -> $CORPUS_LOCAL"
  cp -f "$CORPUS_SRC" "$CORPUS_LOCAL"
fi
test -f "$CORPUS_LOCAL" || { log "错误: 缺少 $CORPUS_LOCAL"; exit 1; }

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

log "[1/3] dot_product → runs/swap_fr_dot/（W&B project=attention-small, group=swap-fr-en, max_steps=$MAXS）"
python train.py --attention dot_product --name swap_fr_dot --max-steps "$MAXS" --data-path "$CORPUS_LOCAL" "${EXTRA[@]}" "${BATCH_OPT[@]}"

log "[2/3] report_swap.txt（单 run；若曾训 additive 可传 --add 路径再跑 compare_runs）"
python compare_runs.py \
  --dot "$ROOT/runs/swap_fr_dot/metrics.json" \
  --out "$ROOT/report_swap.txt" \
  --json-bundle "$ROOT/results/bundle_swap_metrics.json"

log "[3/3] manifest_swap.json"
small_swap_root="$ROOT" python - << 'PY'
import json, os, time
from pathlib import Path
root = Path(os.environ["small_swap_root"])
m = {
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "direction": "fr_en",
    "data": str(root / "data/corpus_50k.tsv"),
    "report": str(root / "report_swap.txt"),
    "bundle": str(root / "results/bundle_swap_metrics.json"),
    "runs": {
        "dot": str(root / "runs/swap_fr_dot"),
    },
}
(root / "results" / "manifest_swap.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("MANIFEST_SWAP ->", root / "results/manifest_swap.json")
PY

log "======== small_swap 全部成功 ========"
log "查看: report_swap.txt | results/bundle_swap_metrics.json | runs/swap_fr_dot/"
