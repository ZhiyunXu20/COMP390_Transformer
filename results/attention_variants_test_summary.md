# Attention variants — held-out test summary

**All reported scores are from independent held-out test evaluation, not training-time validation metrics.**

Sources are limited to `runs/<run>/test_eval/metrics_test.json` only (never `runs/<run>/metrics.json`).

**status** comes from `results/runs_sanity_report.json` (re-run `python scripts/sanity_check_runs.py` to refresh).

Training-time fields (`parameter_count`, `train_time_seconds`, `peak_gpu_memory_mib`) are read from `training_meta.json` and are **blank when `metadata_repaired=true`**.

**Multi-seed rows** (`n=3`): metrics are **mean ± sample std** over held-out `test_eval` scores from `results/ablation_per_seed.csv` (one value per training seed). **Welch p vs fast_dot (BLEU)** is a two-sided Welch two-sample *p*-value (variant seeds vs **fast_dot** seeds); — on the **fast_dot** row means *reference*. **Single-seed** exploratory rows are annotated in **notes**; full Welch prose lives in `results/variant_multiseed_summary.md`.

---


| run | pkg | attention_type | mechanism_family | BLEU | chrF | chrF++ | COMET | n_seeds | Welch p vs fast_dot (BLEU) | BERTScore | number_of_test_examples | parameter_count | train_time_seconds | peak_gpu_memory_mib | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fast_dot | small_try | dot_product | scoring | 15.80 ± 0.52 (n=3) | 37.42 ± 0.63 (n=3) | 35.79 ± 0.58 (n=3) | 0.5089 ± 0.0032 (n=3) | 3 | — | 0.878385 | 2500 | 30442800 | 790.929 | 10359 | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/fast_dot`: BLEU=16.07, chrF++=36.04, COMET=0.5055. n_heads=4 |
| fast_add | small_try | additive | scoring | 8.21 ± 0.70 (n=3) | 27.02 ± 1.45 (n=3) | 25.44 ± 1.30 (n=3) | 0.4542 ± 0.0074 (n=3) | 3 | p=0.0002 | 0.853527 | 2500 | 30845232 | 1983.87 | 51823.5 | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/fast_add`: BLEU=8.16, chrF++=24.64, COMET=0.4444. n_heads=4 |
| head_1h_dot | small_head | dot_product | scoring | 14.57 (n=1) | 35.86 (n=1) | 34.19 (n=1) | 0.4932 (n=1) | 1 |  | 0.874409 | 2500 |  |  |  | ok | (single seed; not statistically tested vs fast_dot) n_heads=1 |
| swap_fr_dot | small_swap | dot_product | scoring | 9.76 (n=1) | 32.22 (n=1) | 30.38 (n=1) | 0.5037 (n=1) | 1 |  | 0.791393 | 2500 |  |  |  | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |
| var_bilinear | small_try | bilinear | scoring | 14.39 ± 1.38 (n=3) | 35.79 ± 2.30 (n=3) | 34.20 ± 2.20 (n=3) | 0.4983 ± 0.0082 (n=3) | 3 | p=0.2118 | 0.877564 | 2500 | 30639408 | 805.03 | 10482.4 | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/var_bilinear`: BLEU=15.17, chrF++=35.37, COMET=0.5057. n_heads=4 |
| var_gated_dot_additive | small_try | gated_dot_additive | scoring | 15.11 ± 0.66 (n=3) | 36.84 ± 1.03 (n=3) | 35.19 ± 0.99 (n=3) | 0.5066 ± 0.0054 (n=3) | 3 | p=0.2323 | 0.876579 | 2500 | 30845280 | 2091.08 | 52348.6 | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/var_gated_dot_additive`: BLEU=14.90, chrF++=34.91, COMET=0.4999. n_heads=4 |
| var_sparsemax | small_try | sparsemax | normalization | 7.65 (n=1) | 26.54 (n=1) | 25.05 (n=1) | 0.4452 (n=1) | 1 |  | 0.854192 | 2500 | 30442800 | 1155.48 | 10386.7 | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |
| var_entmax15 | small_try | entmax15 | normalization | 13.70 ± 0.78 (n=3) | 34.74 ± 1.12 (n=3) | 33.16 ± 1.06 (n=3) | 0.4940 ± 0.0048 (n=3) | 3 | p=0.0227 | 0.872806 | 2500 | 30442800 | 9014.79 | 10379.1 | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/var_entmax15`: BLEU=13.66, chrF++=32.67, COMET=0.4965. n_heads=4 |
| var_local_window | small_try | local_window | connectivity | 0 | 0 | 0 | 0.356851 | failed_nan |  | 0 | 2500 | 30442800 | 1610.36 | 10374.9 | failed_nan | (single seed; not statistically tested vs fast_dot) n_heads=4; val_loss=NaN; outputs empty; numerical-stability failure, not mechanism comparison. |
| var_global_local | small_try | global_local | connectivity | 7.30 (n=1) | 24.12 (n=1) | 22.71 (n=1) | 0.4368 (n=1) | 1 |  | 0.849364 | 2500 | 30442800 | 1069.04 | 10374.9 | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |
| var_local_window_a23_retrain | small_try | local_window |  | 16.82 (n=1) | 38.37 (n=1) | 36.69 (n=1) | 0.5231 (n=1) | 1 |  | 0.879923 | 2500 | 30442800 | 1110.32 | 10700.4 | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |

_Companion CSV: `results/attention_variants_test_summary.csv`._

### Context (external baselines — not our runs)

For **scale context only** (do **not** rank this table against these numbers): Vaswani et al. 2017 report **38.1 / 41.8 BLEU** (EN–FR base/big, ~4.5M pairs, beam 4, length penalty 0.6); a typical IWSLT'17-style EN–FR tutorial is often **30+ BLEU** on ~225k pairs with beam. This project uses **~50k pairs**, **greedy** decoding, short training (**3000 steps**), and `max_seq_len=96`, so **~15.8 BLEU** (multi-seed mean on `fast_dot`) is in a plausible range versus those references—not evidence of a broken stack. **Do not** claim competitive parity with Vaswani 2017 on BLEU alone.

---
