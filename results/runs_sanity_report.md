# Runs sanity report

## Summary table

| run | status | BLEU | chrF | val_loss | empty_hyp_pct | notes |
|-----|--------|------|------|----------|---------------|-------|
| fast_add | ok | 8.16 | 26.13 | 3.7453 | 0.0 |  |
| fast_add_s1 | ok | 7.76 | 25.87 | 3.7525 | 0.0 |  |
| fast_add_s2 | ok | 7.85 | 26.54 | 3.7849 | 0.0 |  |
| fast_add_s3 | ok | 9.01 | 28.64 | 3.6355 | 0.0 |  |
| fast_dot | ok | 16.07 | 37.67 | 3.2555 | 0.0 |  |
| fast_dot_s1 | ok | 16.28 | 37.75 | 3.2466 | 0.0 |  |
| fast_dot_s2 | ok | 15.24 | 36.69 | 3.2940 | 0.0 |  |
| fast_dot_s3 | ok | 15.89 | 37.80 | 3.2521 | 0.0 |  |
| head_1h_dot | ok | 14.57 | 35.86 | 3.3663 | 0.0 |  |
| lr_sweep_add_lr1e3 | ok | 12.71 | 33.88 | 3.3536 | 0.0 |  |
| lr_sweep_add_lr1e4 | warning_low_quality | 2.89 | 15.93 | 4.8463 | 0.0 |  |
| lr_sweep_add_lr3e3 | ok | 15.48 | 37.32 | 3.0549 | 0.0 |  |
| lr_sweep_add_lr3e4 | ok | 5.20 | 21.27 | 4.1712 | 0.0 |  |
| swap_fr_dot | ok | 9.76 | 32.22 | 4.6244 | 0.0 |  |
| var_bilinear | ok | 15.17 | 37.16 | 3.2848 | 0.0 |  |
| var_bilinear_s1 | ok | 14.99 | 36.89 | 3.2833 | 0.0 |  |
| var_bilinear_s2 | ok | 12.82 | 33.14 | 3.3566 | 0.0 |  |
| var_bilinear_s3 | ok | 15.37 | 37.33 | 3.3002 | 0.0 |  |
| var_entmax15 | ok | 13.66 | 34.09 | 3.3921 | 0.0 |  |
| var_entmax15_s1 | ok | 13.91 | 35.24 | 3.3587 | 0.0 |  |
| var_entmax15_s2 | ok | 12.85 | 33.45 | 3.4497 | 0.0 |  |
| var_entmax15_s3 | ok | 14.36 | 35.52 | 3.3747 | 0.0 |  |
| var_gated_dot_additive | ok | 14.90 | 36.52 | 3.3073 | 0.0 |  |
| var_gated_dot_additive_s1 | ok | 14.46 | 35.81 | 3.3330 | 0.0 |  |
| var_gated_dot_additive_s2 | ok | 15.08 | 36.84 | 3.2486 | 0.0 |  |
| var_gated_dot_additive_s3 | ok | 15.79 | 37.87 | 3.2776 | 0.0 |  |
| var_global_local | ok | 7.30 | 24.12 | 3.9100 | 0.0 |  |
| var_local_window | failed_nan | 0.00 | 0.00 | nan | 100.0 | final_val_loss is NaN or non-finite also failed_empty_outputs: test BLEU=0 and chrF=0 |
| var_local_window_fp32 | failed_nan | 0.00 | 0.00 | nan | 100.0 | final_val_loss is NaN or non-finite also failed_empty_outputs: test BLEU=0 and chrF=0 |
| var_local_window_lr1e4 | failed_nan | 0.00 | 0.00 | nan | 100.0 | final_val_loss is NaN or non-finite also failed_empty_outputs: test BLEU=0 and chrF=0 |
| var_local_window_window16 | failed_nan | 0.00 | 0.00 | nan | 100.0 | final_val_loss is NaN or non-finite also failed_empty_outputs: test BLEU=0 and chrF=0 |
| var_sparsemax | ok | 7.65 | 26.54 | 3.8158 | 0.0 |  |

## Detailed findings (non-ok runs)

### `lr_sweep_add_lr1e4` — warning_low_quality

- Test BLEU / chrF: 2.89 / 15.93
- Training final_val_loss (metrics.json): 4.8463
- Empty-hypothesis rate (%): 0.0

### `var_local_window` — failed_nan

- Test BLEU / chrF: 0.00 / 0.00
- Training final_val_loss (metrics.json): nan
- Empty-hypothesis rate (%): 100.0
- Notes:
  - final_val_loss is NaN or non-finite
  - also failed_empty_outputs: test BLEU=0 and chrF=0

### `var_local_window_fp32` — failed_nan

- Test BLEU / chrF: 0.00 / 0.00
- Training final_val_loss (metrics.json): nan
- Empty-hypothesis rate (%): 100.0
- Notes:
  - final_val_loss is NaN or non-finite
  - also failed_empty_outputs: test BLEU=0 and chrF=0

### `var_local_window_lr1e4` — failed_nan

- Test BLEU / chrF: 0.00 / 0.00
- Training final_val_loss (metrics.json): nan
- Empty-hypothesis rate (%): 100.0
- Notes:
  - final_val_loss is NaN or non-finite
  - also failed_empty_outputs: test BLEU=0 and chrF=0

### `var_local_window_window16` — failed_nan

- Test BLEU / chrF: 0.00 / 0.00
- Training final_val_loss (metrics.json): nan
- Empty-hypothesis rate (%): 100.0
- Notes:
  - final_val_loss is NaN or non-finite
  - also failed_empty_outputs: test BLEU=0 and chrF=0

## Methodology note

- **Training-time validation BLEU** lives in `metrics.json` (`final_bleu`, `best_bleu_during_training`, etc.) and reflects the training script’s `eval_split` (usually **val**), often on a **subsample** and with training-specific filtering.
- **Held-out test BLEU / chrF** live in `test_eval/metrics_test.json` from `evaluate_test.py` on **`test.tsv`**; these are **not comparable** to validation BLEU in `metrics.json` line-by-line.
- A run can show **healthy training metrics** yet **collapsed generation** on test (e.g. BLEU=chrF=0, high empty-hyp rate) if decoding, checkpoint selection, or numerical edge cases differ between train/val loops and final greedy decode.
- **`final_val_loss` = NaN/non-finite** usually signals numerical instability during training or validation (e.g. bf16 autocast + badly scaled logits, bad masks).

