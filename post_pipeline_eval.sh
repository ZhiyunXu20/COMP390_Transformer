#!/usr/bin/env bash
# 训练流水线（run_all_small_training.sh）之后的评测：对 runs/*/best.pt 跑 held-out test，
# 并可选配对 bootstrap、效率 profile。
#
# 用法（仓库根目录）：
#   bash post_pipeline_eval.sh
#
# 环境变量（可选）：
#   TEST_FILE                 默认 data/splits/en_fr_50k_seed42/test.tsv
#   MAX_NEW_TOKENS            默认 64（传给 evaluate_test.py --max-new-tokens）
#   EVAL_BATCH_SIZE           默认 32
#   EVAL_CPU=1                加 --cpu
#   SKIP_IF_EXISTS=1          若某 run 已有 runs/<name>/test_eval/metrics_test.json 则跳过该 run
#   SKIP_MISSING_RUNS=1       缺少 runs/<name>/best.pt 时跳过而非报错（默认报错）
#   RUN_SIGNIFICANCE=1       在 test 评测完成后跑配对 bootstrap（须两端同为 EN→FR）
#   SIG_BOOTSTRAP_SAMPLES      传给 significance_test.py（默认 2000；要大样本可设 10000）
#   SIG_PRED_A / SIG_PRED_B   覆盖配对比较的 predictions.jsonl（默认 fast_dot vs fast_add）
#   SIG_NAME_A / SIG_NAME_B    significance_test 显示名称
#   RUN_PROFILE=1            运行 scripts/profile_model.py（合成 batch，与 checkpoint 无关）
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TEST_FILE="${TEST_FILE:-data/splits/en_fr_50k_seed42/test.tsv}"
MAX_NEW="${MAX_NEW_TOKENS:-64}"
BS="${EVAL_BATCH_SIZE:-32}"

# EN→FR（small_try / small_head）：左列英右列法 → tokenizer_src=英词表 tokenizer_tgt=法词表
TOK_EN_FR_SRC="${TOK_EN_FR_SRC:-data/tokenizer_src_train_only.json}"
TOK_EN_FR_TGT="${TOK_EN_FR_TGT:-data/tokenizer_tgt_train_only.json}"
# FR→EN（small_swap）：配置中为 tokenizer_src=法语 tokenizer_tgt=英语
TOK_FR_EN_SRC="${TOK_FR_EN_SRC:-data/tokenizer_tgt_train_only.json}"
TOK_FR_EN_TGT="${TOK_FR_EN_TGT:-data/tokenizer_src_train_only.json}"

CPU_FLAG=()
if [[ "${EVAL_CPU:-0}" == "1" ]]; then
  CPU_FLAG+=(--cpu)
fi

log() { echo "[$(date -Is)] [post_pipeline_eval] $*"; }

require_ckpt() {
  local ckpt="$1"
  local name="$2"
  if [[ ! -f "$ckpt" ]]; then
    if [[ "${SKIP_MISSING_RUNS:-0}" == "1" ]]; then
      log "SKIP（无 checkpoint）: $name -> $ckpt"
      return 1
    fi
    log "ERROR: 缺少 checkpoint: $ckpt"
    exit 2
  fi
  return 0
}

run_eval_test() {
  local run_name="$1"
  local pkg="$2"
  local tok_src="$3"
  local tok_tgt="$4"

  local ckpt="$ROOT/runs/$run_name/best.pt"
  local out="$ROOT/runs/$run_name/test_eval"

  if ! require_ckpt "$ckpt" "$run_name"; then
    return 0
  fi

  if [[ "${SKIP_IF_EXISTS:-0}" == "1" ]] && [[ -f "$out/metrics_test.json" ]]; then
    log "SKIP（已有 metrics_test.json）: $run_name -> $out"
    return 0
  fi

  log "evaluate_test: $run_name (pkg=$pkg) -> $out"
  python evaluate_test.py \
    --checkpoint "$ckpt" \
    --test-file "$TEST_FILE" \
    --tokenizer-src "$tok_src" \
    --tokenizer-tgt "$tok_tgt" \
    --output-dir "$out" \
    --pkg "$pkg" \
    --max-new-tokens "$MAX_NEW" \
    --batch-size "$BS" \
    "${CPU_FLAG[@]}"
}

log "仓库根: $ROOT"
log "TEST_FILE=$TEST_FILE MAX_NEW_TOKENS=$MAX_NEW EVAL_BATCH_SIZE=$BS"

if [[ ! -f "$(pwd)/evaluate_test.py" ]]; then
  log "ERROR: 请在仓库根目录执行（缺少 evaluate_test.py）"
  exit 3
fi

if [[ ! -f "$ROOT/$TEST_FILE" ]]; then
  log "ERROR: test 文件不存在: $ROOT/$TEST_FILE"
  exit 4
fi

# 四条默认 run（与 run_all_small_training 产出一致）
run_eval_test fast_dot small_try "$TOK_EN_FR_SRC" "$TOK_EN_FR_TGT"
run_eval_test fast_add small_try "$TOK_EN_FR_SRC" "$TOK_EN_FR_TGT"
run_eval_test head_1h_dot small_head "$TOK_EN_FR_SRC" "$TOK_EN_FR_TGT"

# FR→EN（勿与 EN→FR 结果混做 significance_test：reference 语种不同）
run_eval_test swap_fr_dot small_swap "$TOK_FR_EN_SRC" "$TOK_FR_EN_TGT"

if [[ "${RUN_SIGNIFICANCE:-0}" == "1" ]]; then
  PA="${SIG_PRED_A:-$ROOT/runs/fast_dot/test_eval/predictions.jsonl}"
  PB="${SIG_PRED_B:-$ROOT/runs/fast_add/test_eval/predictions.jsonl}"
  NA="${SIG_NAME_A:-fast_dot}"
  NB="${SIG_NAME_B:-fast_add}"
  if [[ ! -f "$PA" ]] || [[ ! -f "$PB" ]]; then
    log "ERROR: RUN_SIGNIFICANCE=1 但缺少 predictions: $PA 或 $PB"
    exit 6
  fi
  mkdir -p "$ROOT/results"
  OUT_MD="${SIG_OUT_MD:-$ROOT/results/significance_${NA}_vs_${NB}.md}"
  OUT_JS="${SIG_OUT_JSON:-$ROOT/results/significance_${NA}_vs_${NB}.json}"
  BS_BOOT="${SIG_BOOTSTRAP_SAMPLES:-2000}"
  log "significance_test: $NA vs $NB (bootstrap_samples=$BS_BOOT)"
  python scripts/significance_test.py \
    --predictions-a "$PA" \
    --predictions-b "$PB" \
    --name-a "$NA" \
    --name-b "$NB" \
    --bootstrap-samples "$BS_BOOT" \
    --output-md "$OUT_MD" \
    --output-json "$OUT_JS"
fi

if [[ "${RUN_PROFILE:-0}" == "1" ]]; then
  log "profile_model.py（合成 batch 测速）"
  python scripts/profile_model.py \
    --batch-size "${PROFILE_BATCH_SIZE:-8}" \
    --seq-len "${PROFILE_SEQ_LEN:-64}" \
    --d-model "${PROFILE_D_MODEL:-256}" \
    --n-layers "${PROFILE_N_LAYERS:-4}"
fi

log "完成。test 输出目录示例: runs/fast_dot/test_eval/（metrics_test.json、predictions.jsonl）"
