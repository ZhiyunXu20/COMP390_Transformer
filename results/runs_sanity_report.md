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
| swap_fr_dot | ok | 9.76 | 32.22 | 4.6244 | 0.0 |  |
| var_bilinear | ok | 15.17 | 37.16 | 3.2848 | 0.0 |  |
| var_entmax15 | ok | 13.66 | 34.09 | 3.3921 | 0.0 |  |
| var_gated_dot_additive | ok | 14.90 | 36.52 | 3.3073 | 0.0 |  |
| var_global_local | ok | 7.30 | 24.12 | 3.9100 | 0.0 |  |
| var_local_window | failed_nan | 0.00 | 0.00 | nan | 100.0 | final_val_loss is NaN or non-finite also failed_empty_outputs: test BLEU=0 and chrF=0 |
| var_sparsemax | ok | 7.65 | 26.54 | 3.8158 | 0.0 |  |

## Detailed findings (non-ok runs)

### `var_local_window` — failed_nan

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

