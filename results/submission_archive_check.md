# Submission archive check

- **repository**: `/root/autodl-tmp`
- **--strict**: `True`
- **overall**: **WARN**

## Python package / script directories

- **status**: **PASS**
- [PASS] `small_try/`: 9 Python files
- [PASS] `small_head/`: 9 Python files
- [PASS] `small_swap/`: 9 Python files
- [PASS] `base_1/`: 7 Python files
- [PASS] `base_improve/`: 8 Python files
- [PASS] `scripts/`: 23 Python files
- [PASS] `experiments/`: 3 Python files
- [PASS] `tests/`: 7 Python files
- [PASS] `translate_cli/`: 1 Python files

## data/splits and tokenizers

- **status**: **PASS**
- [PASS] `data/splits/en_fr_50k_seed42/train.tsv`
- [PASS] `data/splits/en_fr_50k_seed42/val.tsv`
- [PASS] `data/splits/en_fr_50k_seed42/test.tsv`
- [PASS] `data/splits/en_fr_50k_seed42/split_metadata.json`
- [PASS] `data/tokenizer_metadata.json`
- [PASS] `data/tokenizer*.json` count=5

## 10 reported runs — test_eval + core JSON

- **status**: **PASS**
- [PASS] `runs/fast_dot/metrics.json`
- [PASS] `runs/fast_dot/resolved_config.json`
- [PASS] `runs/fast_dot/training_meta.json`
- [PASS] `runs/fast_dot/test_eval/metrics_test.json`
- [PASS] `runs/fast_dot/test_eval/predictions.jsonl`
- [PASS] `runs/fast_dot/test_eval/examples.md`
- [PASS] `runs/fast_add/metrics.json`
- [PASS] `runs/fast_add/resolved_config.json`
- [PASS] `runs/fast_add/training_meta.json`
- [PASS] `runs/fast_add/test_eval/metrics_test.json`
- [PASS] `runs/fast_add/test_eval/predictions.jsonl`
- [PASS] `runs/fast_add/test_eval/examples.md`
- [PASS] `runs/head_1h_dot/metrics.json`
- [PASS] `runs/head_1h_dot/resolved_config.json`
- [PASS] `runs/head_1h_dot/training_meta.json`
- [PASS] `runs/head_1h_dot/test_eval/metrics_test.json`
- [PASS] `runs/head_1h_dot/test_eval/predictions.jsonl`
- [PASS] `runs/head_1h_dot/test_eval/examples.md`
- [PASS] `runs/swap_fr_dot/metrics.json`
- [PASS] `runs/swap_fr_dot/resolved_config.json`
- [PASS] `runs/swap_fr_dot/training_meta.json`
- [PASS] `runs/swap_fr_dot/test_eval/metrics_test.json`
- [PASS] `runs/swap_fr_dot/test_eval/predictions.jsonl`
- [PASS] `runs/swap_fr_dot/test_eval/examples.md`
- [PASS] `runs/var_bilinear/metrics.json`
- [PASS] `runs/var_bilinear/resolved_config.json`
- [PASS] `runs/var_bilinear/training_meta.json`
- [PASS] `runs/var_bilinear/test_eval/metrics_test.json`
- [PASS] `runs/var_bilinear/test_eval/predictions.jsonl`
- [PASS] `runs/var_bilinear/test_eval/examples.md`
- [PASS] `runs/var_gated_dot_additive/metrics.json`
- [PASS] `runs/var_gated_dot_additive/resolved_config.json`
- [PASS] `runs/var_gated_dot_additive/training_meta.json`
- [PASS] `runs/var_gated_dot_additive/test_eval/metrics_test.json`
- [PASS] `runs/var_gated_dot_additive/test_eval/predictions.jsonl`
- [PASS] `runs/var_gated_dot_additive/test_eval/examples.md`
- [PASS] `runs/var_sparsemax/metrics.json`
- [PASS] `runs/var_sparsemax/resolved_config.json`
- [PASS] `runs/var_sparsemax/training_meta.json`
- [PASS] `runs/var_sparsemax/test_eval/metrics_test.json`
- [PASS] `runs/var_sparsemax/test_eval/predictions.jsonl`
- [PASS] `runs/var_sparsemax/test_eval/examples.md`
- [PASS] `runs/var_entmax15/metrics.json`
- [PASS] `runs/var_entmax15/resolved_config.json`
- [PASS] `runs/var_entmax15/training_meta.json`
- [PASS] `runs/var_entmax15/test_eval/metrics_test.json`
- [PASS] `runs/var_entmax15/test_eval/predictions.jsonl`
- [PASS] `runs/var_entmax15/test_eval/examples.md`
- [PASS] `runs/var_local_window/metrics.json`
- [PASS] `runs/var_local_window/resolved_config.json`
- [PASS] `runs/var_local_window/training_meta.json`
- [PASS] `runs/var_local_window/test_eval/metrics_test.json`
- [PASS] `runs/var_local_window/test_eval/predictions.jsonl`
- [PASS] `runs/var_local_window/test_eval/examples.md`
- [PASS] `runs/var_global_local/metrics.json`
- [PASS] `runs/var_global_local/resolved_config.json`
- [PASS] `runs/var_global_local/training_meta.json`
- [PASS] `runs/var_global_local/test_eval/metrics_test.json`
- [PASS] `runs/var_global_local/test_eval/predictions.jsonl`
- [PASS] `runs/var_global_local/test_eval/examples.md`

## Checkpoints — must be gitignored (not shipped)

- **status**: **PASS**
- [PASS] no unchecked checkpoint paths (or no checkpoints; `.gitignore` covers `*.pt` etc.)

## results/* summaries

- **status**: **PASS**
- [PASS] `results/attention_variants_test_summary.md`
- [PASS] `results/attention_variants_test_summary.csv`
- [PASS] `results/cross_seed_significance.md`
- [PASS] `results/cross_seed_significance.json`
- [PASS] `results/variant_fairness_audit.md`
- [PASS] `results/variant_fairness_audit.json`
- [PASS] `results/runs_sanity_report.md`
- [PASS] `results/runs_sanity_report.json`
- [PASS] `results/ablation_summary.md`
- [PASS] `results/ablation_summary.csv`
- [PASS] `results/ablation_per_seed.csv`
- [PASS] `results/significance_fast_dot_vs_fast_add.md`
- [PASS] `results/significance_fast_dot_vs_fast_add.json`

## required docs/*.md

- **status**: **PASS**
- [PASS] `docs/RESEARCH_AUDIT.md`
- [PASS] `docs/ATTENTION_VARIANTS_STATUS.md`
- [PASS] `docs/THESIS_RESULT_BOUNDARIES.md`
- [PASS] `docs/REPRODUCIBILITY.md`
- [PASS] `docs/CODE_ARCHIVE_CHECKLIST.md`
- [PASS] `docs/ENVIRONMENT.md`

## repository root files

- **status**: **PASS**
- [PASS] `README.md`
- [PASS] `requirements.txt`
- [PASS] `.gitignore`

## forbidden / noisy paths (policy)

- **status**: **WARN**
- [WARN] large `data/EN-FR.txt` (~898 MiB); exclude from submission zip
- [WARN] `.log` (consider omitting): `logs/post_pipeline_significance_only_20260429_182445.log`
- [WARN] `.log` (consider omitting): `logs/run_all_small_training_20260429_100558.log`
- [WARN] `.log` (consider omitting): `logs/run_all_small_training_20260429_105557.log`
- [WARN] `.log` (consider omitting): `logs/variants_run.log`
- [WARN] `.log` (consider omitting): `results/fast_add_resume.log`
- [WARN] `.log` (consider omitting): `results/fast_add_train_eval.log`
- [WARN] `.log` (consider omitting): `small_head/tea_debug.log`
- [WARN] `.log` (consider omitting): `small_swap/tea_debug.log`
- [WARN] `.log` (consider omitting): `small_try/tea_debug.log`
- [WARN] `.log` (consider omitting): `wandb_cache_attention_small/wandb/debug-internal.log`
- [WARN] `.log` (consider omitting): `wandb_cache_attention_small/wandb/debug.log`
- [WARN] `.log` (consider omitting): `wandb_cache_attention_small/wandb/run-20260429_100607-w1e1wtda/files/output.log`
- [WARN] (suppressing further `.log` listings)

## prepare_code_archive.validate (test_eval pairs + CSV)

- **status**: **PASS**
- [PASS] `prepare_code_archive.validate` OK

## estimated archive size (non-ignored files)

- **status**: **PASS**
- [PASS] ~50.3 MiB for files not matched by `git check-ignore` (local `du -sh runs/` may be much larger if checkpoints exist but are ignored)
