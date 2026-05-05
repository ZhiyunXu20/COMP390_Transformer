# Attention variants — held-out test summary

**All reported scores are from independent held-out test evaluation, not training-time validation metrics.**

Sources are limited to `runs/<run>/test_eval/metrics_test.json` only (never `runs/<run>/metrics.json`).

**status** comes from `results/runs_sanity_report.json` (re-run `python scripts/sanity_check_runs.py` to refresh).

Training-time fields (`parameter_count`, `train_time_seconds`, `peak_gpu_memory_mib`) are read from `training_meta.json` and are **blank when `metadata_repaired=true`**.

---


| run | pkg | attention_type | mechanism_family | BLEU | chrF | chrF++ | COMET | BERTScore | number_of_test_examples | parameter_count | train_time_seconds | peak_gpu_memory_mib | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fast_dot | small_try | dot_product | scoring | 16.0667 | 37.6701 | 36.0358 | 0.505479 | 0.878385 | 2500 | 30442800 | 790.929 | 10359 | ok | n_heads=4 |
| fast_add | small_try | additive | scoring | 8.16327 | 26.1255 | 24.6448 | 0.444405 | 0.853527 | 2500 | 30845232 | 1983.87 | 51823.5 | ok | n_heads=4 |
| head_1h_dot | small_head | dot_product | scoring | 14.5673 | 35.856 | 34.186 | 0.493225 | 0.874409 | 2500 |  |  |  | ok | n_heads=1 |
| swap_fr_dot | small_swap | dot_product | scoring | 9.76292 | 32.2171 | 30.3822 | 0.503693 | 0.791393 | 2500 |  |  |  | ok | n_heads=4 |
| var_bilinear | small_try | bilinear | scoring | 15.1704 | 37.1568 | 35.369 | 0.505744 | 0.877564 | 2500 | 30639408 | 805.03 | 10482.4 | ok | n_heads=4; single seed; ranking below \|Δ\| = 1 BLEU is unreliable due to seed variance |
| var_gated_dot_additive | small_try | gated_dot_additive | scoring | 14.905 | 36.5162 | 34.9101 | 0.499885 | 0.876579 | 2500 | 30845280 | 2091.08 | 52348.6 | ok | n_heads=4; single seed; ranking below \|Δ\| = 1 BLEU is unreliable due to seed variance |
| var_sparsemax | small_try | sparsemax | normalization | 7.64874 | 26.5404 | 25.0512 | 0.445171 | 0.854192 | 2500 | 30442800 | 1155.48 | 10386.7 | ok | n_heads=4; single seed; ranking below \|Δ\| = 1 BLEU is unreliable due to seed variance |
| var_entmax15 | small_try | entmax15 | normalization | 13.6606 | 34.0908 | 32.6658 | 0.496477 | 0.872806 | 2500 | 30442800 | 9014.79 | 10379.1 | ok | n_heads=4; single seed; ranking below \|Δ\| = 1 BLEU is unreliable due to seed variance |
| var_local_window | small_try | local_window | connectivity | 0 | 0 | 0 | 0.356851 | 0 | 2500 | 30442800 | 1610.36 | 10374.9 | failed_nan | n_heads=4; val_loss=NaN; outputs empty; numerical-stability failure, not mechanism comparison.; single seed; ranking below \|Δ\| = 1 BLEU is unreliable due to seed variance |
| var_global_local | small_try | global_local | connectivity | 7.30136 | 24.1162 | 22.7096 | 0.436818 | 0.849364 | 2500 | 30442800 | 1069.04 | 10374.9 | ok | n_heads=4; single seed; ranking below \|Δ\| = 1 BLEU is unreliable due to seed variance |

_Companion CSV: `results/attention_variants_test_summary.csv`._

### Context (external baselines — not our runs)

For **scale context only** (do **not** rank this table against these numbers): Vaswani et al. 2017 report **38.1 / 41.8 BLEU** (EN–FR base/big, ~4.5M pairs, beam 4, length penalty 0.6); a typical IWSLT'17-style EN–FR tutorial is often **30+ BLEU** on ~225k pairs with beam. This project uses **~50k pairs**, **greedy** decoding, short training (**3000 steps**), and `max_seq_len=96`, so **~15.8 BLEU** (multi-seed mean on `fast_dot`) is in a plausible range versus those references—not evidence of a broken stack. **Do not** claim competitive parity with Vaswani 2017 on BLEU alone.

---
