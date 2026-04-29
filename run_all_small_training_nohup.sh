#!/usr/bin/env bash
# nohup 后台启动 run_all_small_training.sh：完整 stdout/stderr 写入时间戳日志；
# 流水线遇错即退出（run_all_small_training.sh 与各子脚本均为 set -euo pipefail）。
# W&B project 由各包 Config.project_name 指定（当前默认 attention-small-2）。
#
# 用法：
#   cd /path/to/repo && bash run_all_small_training_nohup.sh
#
# 可选环境变量（与 run_all_small_training.sh 一致）：
#   WANDB_MODE=offline   # 不上传云端，仅本地离线 run
#   TRAIN_MAX_STEPS TRAIN_BATCH_SIZE CUDA_VISIBLE_DEVICES 等
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/run_all_small_training_${TS}.log"
PIDFILE="$ROOT/logs/run_all_small_training_${TS}.pid"

# pipefail：pipeline 失败码取自 run_all_small_training.sh，而非 tee
nohup bash -c 'set -euo pipefail
bash "'"$ROOT"'/run_all_small_training.sh" 2>&1 | tee -a "'"$LOG"'"
code=${PIPESTATUS[0]}
echo "[$(date -Is)] pipeline_exit_code=${code}" >> "'"$LOG"'"
exit "${code}"
' </dev/null >/dev/null 2>&1 &

echo $! > "$PIDFILE"

echo "Started background PID=$(cat "$PIDFILE")"
echo "Log file: $LOG"
echo "Tail: tail -f \"$LOG\""
