#!/usr/bin/env bash
# 正式训练：nohup 后台运行完整 small 流水线；日志写入 logs/run_all_small_<时间>.log。
# 任一步失败则进程非 0 退出（子脚本为 set -e）；指标见各 runs/*/metrics.json 与 report / results。
#
# 用法：
#   bash /root/autodl-tmp/launch_small_training_nohup.sh
#
# 前置建议：在交互终端执行一次 `wandb login`，以便在线同步 project「attention-small」。
# 默认 WANDB_MODE=online（正式同步）；仅离线调试：export WANDB_MODE=offline
# 可选：export TRAIN_BATCH_SIZE=256 等在显存有余时加大 batch（覆盖三阶段 train 的 --batch-size）
# HF 下载默认走国内镜像（与 run_all_small_training.sh 一致）；覆盖: export HF_ENDPOINT=https://huggingface.co
# 默认 HF_HUB_DISABLE_XET=1（弱网避免 XET SSL 问题）；需 XET: export HF_HUB_DISABLE_XET=0
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# 正式后台训练不应继承此前试跑留在 shell 里的 EVAL_LIGHT=1
unset EVAL_LIGHT || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$ROOT/logs"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="$ROOT/logs/run_all_small_${STAMP}.log"
PIDFILE="$ROOT/logs/run_all_small_${STAMP}.pid"

unset TRAIN_MAX_STEPS
unset TRAIN_VAL_EVERY
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache_attention_small}"
mkdir -p "$WANDB_DIR" "$ROOT/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日志: $LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WANDB_MODE=$WANDB_MODE HF_ENDPOINT=$HF_ENDPOINT WANDB_DIR=$WANDB_DIR project=attention-small"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-未设置，使用各 config 默认 192；可 export TRAIN_BATCH_SIZE=256 提速}（EVAL_LIGHT 已 unset → 全量指标）"

nohup bash "$ROOT/run_all_small_training.sh" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已启动 PID=$(cat "$PIDFILE")"
echo "查看进度: tail -f $LOG"
echo "停止: kill \$(cat $PIDFILE)"
