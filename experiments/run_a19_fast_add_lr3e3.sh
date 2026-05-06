#!/usr/bin/env bash
# A19: additive @ lr=3e-3, max_steps=3000, seeds 1–3 (matched to fast_dot lr); new runs only.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"
mkdir -p logs

BASE_TRAIN=(
  small_try/train.py
  --train-path data/splits/en_fr_50k_seed42/train.tsv
  --val-path data/splits/en_fr_50k_seed42/val.tsv
  --test-path data/splits/en_fr_50k_seed42/test.tsv
  --tokenizer-src data/tokenizer_src_train_only.json
  --tokenizer-tgt data/tokenizer_tgt_train_only.json
  --output-dir runs
  --attention-type additive
  --n-heads 4
  --lr 3e-3
  --max-steps 3000
  --eval-split val
  --no-wandb
  --eval-light
  --no-attention-plots
)

EV=(
  evaluate_test.py
  --test-file data/splits/en_fr_50k_seed42/test.tsv
  --tokenizer-src data/tokenizer_src_train_only.json
  --tokenizer-tgt data/tokenizer_tgt_train_only.json
  --max-new-tokens 64
  --batch-size 32
  --pkg small_try
)

for s in 1 2 3; do
  name="fast_add_lr3e3_s${s}"
  echo "### A19 train $name"
  "$PY" "${BASE_TRAIN[@]}" --name "$name" --seed "$s" 2>&1 | tee "logs/a19_${name}_train.log"
  echo "### A19 evaluate_test $name"
  "$PY" "${EV[@]}" \
    --checkpoint "runs/${name}/best.pt" \
    --output-dir "runs/${name}/test_eval" \
    2>&1 | tee "logs/a19_${name}_eval.log"
done

echo "### A19 append ablation_per_seed + cross-seed lr3e3"
"$PY" experiments/append_a19_ablation_rows.py
"$PY" scripts/cross_seed_significance.py \
  --csv results/ablation_per_seed.csv \
  --experiment-a fast_dot \
  --experiment-b fast_add_lr3e3 \
  --output-md results/cross_seed_significance_lr3e3.md \
  --output-json results/cross_seed_significance_lr3e3.json
"$PY" experiments/write_additive_matched_lr3e3_summary.py

echo "A19 done."
