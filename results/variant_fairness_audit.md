# Variant experiment fairness audit

Reference run: **`fast_dot`** (parameter deltas vs this run when counts available).

## Rules

- **PASS**: key training/eval fields match reference (except `attention_type`), `test_eval/metrics_test.json` present.
- **WARN**: no FAIL, but `n_heads` differs, or parameter count Δ vs reference > 1%, or missing `training_meta.json`.
- **FAIL**: missing `resolved_config.json` or `test_eval/metrics_test.json`, or mismatch on splits/tokenizers/backbone dims/training budget/seed/eval_split/`max_gen_len`, or `metrics_test` decoding batch/max_new_tokens mismatch.

`parameter_changed` is an **interpretability risk**, not an automatic invalidation.

## Summary table

| run | attention_type | status | comparable_to_fast_dot | parameter_count | parameter_delta_percent | missing_files | mismatched_fields | notes |
|-----|----------------|--------|-------------------------|-----------------|---------------------------|---------------|---------------------|-------|
| fast_dot | dot_product | PASS | True | 30442800 | 0.0000 |  |  |  |
| fast_add | additive | WARN | True | 30845232 | 1.3219 |  |  | parameter_changed |
| head_1h_dot | dot_product | WARN | False | 30442800 | 0.0000 | training_meta.json (optional) | n_heads (!= ref; head-count ablation risk) | n_heads differs from reference; missing_training_meta |
