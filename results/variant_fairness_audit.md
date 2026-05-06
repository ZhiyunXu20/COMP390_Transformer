# Variant experiment fairness audit

Reference run: **`fast_dot`** (parameter deltas vs this run when counts available).

## Rules

- **PASS**: key training/eval fields match reference (except `attention_type`), `test_eval/metrics_test.json` present.
- **WARN**: no FAIL, but `n_heads` differs, or parameter count Δ vs reference > 1%, or missing `training_meta.json`, or **`training_meta.json` was repaired** (`metadata_repaired=true`; `num_parameters` only is trustworthy).
- **FAIL**: missing `resolved_config.json` or `test_eval/metrics_test.json`, or mismatch on splits/tokenizers/backbone dims/training budget/seed/eval_split/`max_gen_len`, or `metrics_test` decoding batch/max_new_tokens mismatch.

`parameter_changed` is an **interpretability risk**, not an automatic invalidation.

## Summary table

| run | attention_type | status | comparable_to_fast_dot | parameter_count | parameter_delta_percent | training_meta | missing_files | mismatched_fields | notes |
|-----|----------------|--------|-------------------------|-----------------|---------------------------|---------------|---------------|---------------------|-------|
| fast_dot | dot_product | PASS | True | 30442800 | 0.0000 | original |  |  |  |
| fast_add | additive | WARN | True | 30845232 | 1.3219 | original |  |  | signature overlap risk vs runs: ['lr_sweep_add_lr1e4', 'lr_sweep_add_lr3e4', 'lr_sweep_add_lr1e3', 'lr_sweep_add_lr3e3']; parameter_changed |
| head_1h_dot | dot_product | WARN | False | 30442800 | 0.0000 | repaired |  | n_heads (!= ref; head-count ablation risk) | n_heads differs from reference; repaired_training_meta |
| swap_fr_dot | dot_product | FAIL | False | 30442800 | 0.0000 | repaired |  | train_path; tokenizer_src; tokenizer_tgt | repaired_training_meta |
| fast_dot_s1 | dot_product | FAIL | False | 30442800 | 0.0000 | original |  | seed |  |
| fast_dot_s2 | dot_product | FAIL | False | 30442800 | 0.0000 | original |  | seed |  |
| fast_dot_s3 | dot_product | FAIL | False | 30442800 | 0.0000 | original |  | seed |  |
| fast_add_s1 | additive | FAIL | False | 30845232 | 1.3219 | original |  | seed | signature overlap risk vs runs: ['fast_add_lr3e3_s1']; parameter_changed |
| fast_add_s2 | additive | FAIL | False | 30845232 | 1.3219 | original |  | seed | signature overlap risk vs runs: ['fast_add_lr3e3_s2']; parameter_changed |
| fast_add_s3 | additive | FAIL | False | 30845232 | 1.3219 | original |  | seed | signature overlap risk vs runs: ['fast_add_lr3e3_s3']; parameter_changed |
| fast_add_lr3e3_s1 | additive | FAIL | False | 30845232 | 1.3219 | original |  | learning_rate; seed | signature overlap risk vs runs: ['fast_add_s1']; parameter_changed |
| fast_add_lr3e3_s2 | additive | FAIL | False | 30845232 | 1.3219 | original |  | learning_rate; seed | signature overlap risk vs runs: ['fast_add_s2']; parameter_changed |
| fast_add_lr3e3_s3 | additive | FAIL | False | 30845232 | 1.3219 | original |  | learning_rate; seed | signature overlap risk vs runs: ['fast_add_s3']; parameter_changed |
| var_bilinear | bilinear | PASS | True | 30639408 | 0.6458 | original |  |  |  |
| var_gated_dot_additive | gated_dot_additive | WARN | True | 30845280 | 1.3221 | original |  |  | parameter_changed |
| var_sparsemax | sparsemax | PASS | True | 30442800 | 0.0000 | original |  |  |  |
| var_entmax15 | entmax15 | PASS | True | 30442800 | 0.0000 | original |  |  |  |
| var_local_window | local_window | PASS | True | 30442800 | 0.0000 | original |  |  | signature overlap risk vs runs: ['var_local_window_fp32', 'var_local_window_lr1e4', 'var_local_window_window16', 'var_local_window_a23_retrain'] |
| var_global_local | global_local | PASS | True | 30442800 | 0.0000 | original |  |  |  |
| var_bilinear_s1 | bilinear | FAIL | False | 30639408 | 0.6458 | original |  | seed |  |
| var_bilinear_s2 | bilinear | FAIL | False | 30639408 | 0.6458 | original |  | seed |  |
| var_bilinear_s3 | bilinear | FAIL | False | 30639408 | 0.6458 | original |  | seed |  |
| var_gated_dot_additive_s1 | gated_dot_additive | FAIL | False | 30845280 | 1.3221 | original |  | seed | parameter_changed |
| var_gated_dot_additive_s2 | gated_dot_additive | FAIL | False | 30845280 | 1.3221 | original |  | seed | parameter_changed |
| var_gated_dot_additive_s3 | gated_dot_additive | FAIL | False | 30845280 | 1.3221 | original |  | seed | parameter_changed |
| var_entmax15_s1 | entmax15 | FAIL | False | 30442800 | 0.0000 | original |  | seed |  |
| var_entmax15_s2 | entmax15 | FAIL | False | 30442800 | 0.0000 | original |  | seed |  |
| var_entmax15_s3 | entmax15 | FAIL | False | 30442800 | 0.0000 | original |  | seed |  |
| var_local_window_fp32 | local_window | FAIL | False | 30442800 | 0.0000 | original |  | max_steps | signature overlap risk vs runs: ['var_local_window', 'var_local_window_lr1e4', 'var_local_window_window16', 'var_local_window_a23_retrain'] |
| var_local_window_lr1e4 | local_window | FAIL | False | 30442800 | 0.0000 | original |  | max_steps; learning_rate | signature overlap risk vs runs: ['var_local_window', 'var_local_window_fp32', 'var_local_window_window16', 'var_local_window_a23_retrain'] |
| var_local_window_window16 | local_window | FAIL | False | 30442800 | 0.0000 | original |  | max_steps | signature overlap risk vs runs: ['var_local_window', 'var_local_window_fp32', 'var_local_window_lr1e4', 'var_local_window_a23_retrain'] |
| var_local_window_a23_retrain | local_window | PASS | True | 30442800 | 0.0000 | original |  |  | signature overlap risk vs runs: ['var_local_window', 'var_local_window_fp32', 'var_local_window_lr1e4', 'var_local_window_window16'] |
| lr_sweep_add_lr1e4 | additive | FAIL | False | 30845232 | 1.3219 | original |  | max_steps; learning_rate | signature overlap risk vs runs: ['fast_add', 'lr_sweep_add_lr3e4', 'lr_sweep_add_lr1e3', 'lr_sweep_add_lr3e3']; parameter_changed |
| lr_sweep_add_lr3e4 | additive | FAIL | False | 30845232 | 1.3219 | original |  | max_steps | signature overlap risk vs runs: ['fast_add', 'lr_sweep_add_lr1e4', 'lr_sweep_add_lr1e3', 'lr_sweep_add_lr3e3']; parameter_changed |
| lr_sweep_add_lr1e3 | additive | FAIL | False | 30845232 | 1.3219 | original |  | max_steps; learning_rate | signature overlap risk vs runs: ['fast_add', 'lr_sweep_add_lr1e4', 'lr_sweep_add_lr3e4', 'lr_sweep_add_lr3e3']; parameter_changed |
| lr_sweep_add_lr3e3 | additive | FAIL | False | 30845232 | 1.3219 | original |  | max_steps; learning_rate | signature overlap risk vs runs: ['fast_add', 'lr_sweep_add_lr1e4', 'lr_sweep_add_lr3e4', 'lr_sweep_add_lr1e3']; parameter_changed |
