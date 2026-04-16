#!/usr/bin/env bash
# base_improve：全量 EN-FR、与 base_1 相同 max_steps；点积 + 加性 串联；遇错退出。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache}"
mkdir -p "$WANDB_DIR" logs results

DATA="${DATA:-/root/autodl-tmp/data/EN-FR.txt}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

on_err() {
  log "ERROR: line $1 exit $2"
  exit "$2"
}
trap 'on_err ${LINENO} $?' ERR

log "======== base_improve 流水线开始 ========"
log "ROOT=$ROOT"
log "DATA=$DATA"

test -f "$DATA" || { log "错误: 缺少语料 $DATA"; exit 1; }

python - << 'PY' || exit 1
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), end=" ")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("(CPU)")
PY

for m in sacrebleu wandb tqdm matplotlib seaborn tokenizers; do
  python -c "import importlib; importlib.import_module('$m')" || { log "错误: 缺少 $m"; exit 1; }
done

log "[1/3] dot_product → runs/improve_dot/  W&B project=en-fr-attention-improve"
python train.py --attention dot_product --name improve_dot

log "[2/3] additive → runs/improve_add/"
python train.py --attention additive --name improve_add

log "[3/3] report_improve.txt + BUNDLE"
python compare_runs.py \
  --dot "$ROOT/runs/improve_dot/metrics.json" \
  --add "$ROOT/runs/improve_add/metrics.json" \
  --out "$ROOT/report_improve.txt" \
  --json-bundle "$ROOT/results/bundle_improve_metrics.json"

base_improve_root="$ROOT" python - << 'PY'
import json, os, time
from pathlib import Path
root = Path(os.environ["base_improve_root"])
m = {
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "data": "/root/autodl-tmp/data/EN-FR.txt",
    "note": "base_improve: same hparams as base_1, training stack optimizations only",
    "report": str(root / "report_improve.txt"),
    "bundle": str(root / "results/bundle_improve_metrics.json"),
    "runs": {
        "dot": str(root / "runs/improve_dot"),
        "add": str(root / "runs/improve_add"),
    },
}
(root / "results" / "manifest_improve.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("manifest_improve ->", root / "results/manifest_improve.json")
PY

log "======== base_improve 全部成功 ========"
