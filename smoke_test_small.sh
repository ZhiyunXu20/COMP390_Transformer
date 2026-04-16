#!/usr/bin/env bash
# 快速试跑整条 small 流水线（少步数、W&B 离线），用于检查依赖与脚本是否报错。
# 正式训练请勿使用；正式训练用 launch_small_training_nohup.sh 或 run_all_small_training.sh。
#
# 用法：bash /root/autodl-tmp/smoke_test_small.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAIN_MAX_STEPS="${SMOKE_MAX_STEPS:-80}"
export TRAIN_VAL_EVERY="${SMOKE_VAL_EVERY:-20}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache_attention_small_smoke}"
mkdir -p "$WANDB_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [smoke] $*"; }

log "本轮试跑使用 EVAL_LIGHT=1（仅 BLEU/chrF++）；不污染当前 shell 的 EVAL_LIGHT"
log "WANDB_MODE=$WANDB_MODE WANDB_DIR=$WANDB_DIR"
log "TRAIN_MAX_STEPS=$TRAIN_MAX_STEPS TRAIN_VAL_EVERY=$TRAIN_VAL_EVERY"
log "开始试跑 run_all_small_training（若失败会 set -e 退出）"

EVAL_LIGHT=1 bash "$ROOT/run_all_small_training.sh"
log "smoke 试跑成功。"
