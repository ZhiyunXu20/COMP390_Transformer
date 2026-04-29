#!/usr/bin/env bash
# small_head：仅单头点积训练；多头基线引用 small_try fast_dot metrics。
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
DATA_TSV="${DATA_TSV:-$REPO/small_try/data/corpus_50k.tsv}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

on_err() {
  log "ERROR: 流水线在 line $1 失败，退出码 $2"
  exit "$2"
}
trap 'on_err ${LINENO} $?' ERR

log "======== small_head 流水线开始 ========"
log "ROOT=$ROOT"
log "BASELINE_METRICS=$BASELINE_METRICS"
log "DATA_TSV=$DATA_TSV"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-默认}"

if [[ ! -f "$BASELINE_METRICS" ]]; then
  log "错误: 缺少多头基线 metrics: $BASELINE_METRICS"
  log "请先完成 small_try 的 fast_dot，或设置 BASELINE_METRICS 指向有效 metrics.json"
  exit 1
fi
if [[ ! -f "$DATA_TSV" ]]; then
  log "错误: 缺少训练数据: $DATA_TSV"
  log "请在 small_try 下运行 create_subset 或恢复 corpus_50k.tsv"
  exit 1
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
python train.py --attention dot_product --n-heads 1 --name head_1h_dot --max-steps "$MAXS" "${EXTRA[@]}" "${BATCH_OPT[@]}"

log "[2/3] 对比 report_head.txt（基线 + 单头点积，两列）"
python compare_head_runs.py \
  --baseline "$BASELINE_METRICS" \
  --one-dot "$REPO/runs/head_1h_dot/metrics.json" \
  --out "$ROOT/report_head.txt" \
  --json-bundle "$ROOT/results/bundle_head_metrics.json"

log "[3/3] manifest_head.json"
small_head_root="$ROOT" REPO="$REPO" BASELINE_METRICS="$BASELINE_METRICS" python - << 'PY'
import json, os, time
from pathlib import Path
root = Path(os.environ["small_head_root"])
repo = Path(os.environ["REPO"])
bl = Path(os.environ["BASELINE_METRICS"])
m = {
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "baseline_metrics_path": str(bl.resolve()),
    "data_path": str(repo / "small_try/data/corpus_50k.tsv"),
    "report": str(root / "report_head.txt"),
    "bundle": str(root / "results/bundle_head_metrics.json"),
    "param_counts": str(root / "results/param_counts.json"),
    "runs": {
        "baseline_mh_dot": str(bl.parent),
        "head_1h_dot": str(repo / "runs/head_1h_dot"),
    },
}
(root / "results" / "manifest_head.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("MANIFEST_HEAD ->", root / "results/manifest_head.json")
PY

log "======== small_head 全部成功 ========"
log "查看: report_head.txt | results/bundle_head_metrics.json | results/manifest_head.json"
