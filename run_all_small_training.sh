#!/usr/bin/env bash
# 一键跑完 small_try → small_head → small_swap 全流程训练与报告。
# 所有训练 run 写入同一 W&B project：attention-small-2（各包 config.project_name）；
# 用 wandb_group 区分：try-en-fr | head-ablation | swap-fr-en。
#
# 用法：
#   cd /root/autodl-tmp && bash run_all_small_training.sh
#   nohup bash run_all_small_training.sh > logs/run_all_small.log 2>&1 &
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES   默认使用当前可见 GPU
#   WANDB_DIR              未设置时默认为 $ROOT/wandb_cache_attention_small（三阶段共用一处离线缓存）
#   TRAIN_MAX_STEPS        默认 3000；试跑可设 80 等（run_fast_full / head / swap 内 train 均读取）
#   TRAIN_BATCH_SIZE       可选，传给各 train.py --batch-size，显存有余时加大可缩短墙钟时间
#   TRAIN_VAL_EVERY        可选，覆盖验证间隔（与 TRAIN_MAX_STEPS 配合试跑）
#   WANDB_MODE             launch 默认 online；离线调试 export WANDB_MODE=offline
#   EVAL_LIGHT=1           各 train 加 --eval-light，跳过 BERTScore/COMET（仅 smoke_test_small 使用）
#   HF_ENDPOINT            未设置时默认 https://hf-mirror.com（HF 模型下载）；官方: export HF_ENDPOINT=https://huggingface.co
#   HF_HUB_DISABLE_XET     未设置时默认 1，避免 XET（cas-bridge）弱网 SSL 断连；需 XET 时: export HF_HUB_DISABLE_XET=0
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f /etc/network_turbo ]]; then
  # shellcheck source=/dev/null
  source /etc/network_turbo
fi
_turbo_no_proxy="localhost,127.0.0.1,hf-mirror.com,huggingface.co,hf.co,xethub.hf.co"
export NO_PROXY="${_turbo_no_proxy}${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${_turbo_no_proxy}${no_proxy:+,${no_proxy}}"
# 正式训练默认全量验证（BERTScore/COMET）。仅当本进程环境中 EVAL_LIGHT 恰好为 1 时各子脚本才加 --eval-light。
# 其它取值或未设置一律清除，避免 shell 里误留的 EVAL_LIGHT=0 等与试跑残留干扰。
if [[ "${EVAL_LIGHT:-}" != "1" ]]; then
  unset EVAL_LIGHT || true
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb_cache_attention_small}"
mkdir -p "$WANDB_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [run_all_small] $*"; }

log "仓库根: $ROOT"
log "HF_ENDPOINT=$HF_ENDPOINT"
log "WANDB_MODE=$WANDB_MODE WANDB_DIR=$WANDB_DIR"
log "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-<未设置，使用各 config.batch_size>}"
log "验证指标: $([[ "${EVAL_LIGHT:-}" == 1 ]] && echo 'EVAL_LIGHT=1 仅 BLEU/chrF++' || echo '全量（BERTScore/COMET，与各 config 一致）')"
log "W&B project（统一）: attention-small-2"
log "顺序: small_try (EN→FR dot+add) → small_head (单头) → small_swap (FR→EN dot)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

bash "$ROOT/small_try/run_fast_full.sh"
bash "$ROOT/small_head/run_head_pipeline.sh"
bash "$ROOT/small_swap/run_fr_swap_pipeline.sh"

log "======== 全部 small 流水线成功 ========"
log "W&B: 打开项目 attention-small-2，按 Group 筛选各子实验。"
