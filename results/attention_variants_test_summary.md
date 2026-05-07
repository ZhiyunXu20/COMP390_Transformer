# Attention variants — held-out test summary

> **Aggregation provenance**: BLEU, chrF, chrF++, COMET, BERTScore, train_time_seconds, and peak_gpu_memory_mib columns report mean ± std (n=3) for `is_aggregate=true` rows (computed from `source_run_names`). For `is_aggregate=false` rows, values are single-seed measurements (n=1) with no statistical aggregation. The Welch p column is computed against fast_dot's three-seed BLEU values per row's source seeds where applicable.
> 
> Sources: BLEU/chrF/chrF++/COMET/BERTScore from `runs/<source_run>/test_eval/metrics_test.json`; train_time_seconds and peak_gpu_memory_mib from `runs/<source_run>/training_meta.json`. Note: `results/ablation_per_seed.csv` does not currently fill `wall_time_seconds`; the authoritative per-seed values are in `training_meta.json`.

**All reported scores are from independent held-out test evaluation, not training-time validation metrics.**

Sources are limited to `runs/<run>/test_eval/metrics_test.json` only (never `runs/<run>/metrics.json`).

**status** comes from `results/runs_sanity_report.json` (re-run `python scripts/sanity_check_runs.py` to refresh).

Training-time fields (`parameter_count`, `train_time_seconds`, `peak_gpu_memory_mib`) are read from `training_meta.json` and are **blank when `metadata_repaired=true`**.

**Multi-seed rows** (`n=3`): metrics are **mean ± sample std** over held-out `test_eval` scores from `results/ablation_per_seed.csv` (one value per training seed). **Welch p vs fast_dot (BLEU)** is a two-sided Welch two-sample *p*-value (variant seeds vs **fast_dot** seeds); — on the **fast_dot** row means *reference*. **Single-seed** exploratory rows are annotated in **notes**; full Welch prose lives in `results/variant_multiseed_summary.md`.

Companion CSV columns **`experiment`**, **`is_aggregate`**, **`source_run_names`**, **`representative_run`**, **`metrics_source`** record aggregate vs per-run provenance for validation (`prepare_code_archive.validate`).

---


| run | pkg | attention_type | mechanism_family | BLEU | chrF | chrF++ | COMET | n_seeds | Welch p vs fast_dot (BLEU) | BERTScore | number_of_test_examples | parameter_count | train_time_seconds | peak_gpu_memory_mib | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fast_dot | small_try | dot_product | scoring | 15.80 ± 0.52 (n=3) | 37.42 ± 0.63 (n=3) | 35.79 ± 0.58 (n=3) | 0.5089 ± 0.0032 (n=3) | 3 | — | 0.87839 ± 0.00108 (n=3) | 2500 | 30442800 | 718.89 ± 8.08 (n=3) | 10374.90 ± 0.16 (n=3) | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/fast_dot`: BLEU=16.07, chrF++=36.04, COMET=0.5055. n_heads=4 |
| fast_add | small_try | additive | scoring | 8.21 ± 0.70 (n=3) | 27.02 ± 1.45 (n=3) | 25.44 ± 1.30 (n=3) | 0.4542 ± 0.0074 (n=3) | 3 | p=0.0002 | 0.85553 ± 0.00223 (n=3) | 2500 | 30845232 | 1943.83 ± 31.59 (n=3) | 51823.74 ± 0.11 (n=3) | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/fast_add`: BLEU=8.16, chrF++=24.64, COMET=0.4444. n_heads=4 |
| fast_add_lr3e3 | small_try | additive | scoring | 21.56 ± 0.42 (n=3) | 44.69 ± 0.47 (n=3) | 42.70 ± 0.46 (n=3) | 0.5776 ± 0.0061 (n=3) | 3 | p=0.0002 | 0.89013 ± 0.00114 (n=3) | 2500 | 30845232 | 1819.36 ± 19.38 (n=3) | 51824.12 ± 0.03 (n=3) | ok | 3 seeds from `results/ablation_per_seed.csv`; representative `runs/fast_add_lr3e3_s1/test_eval` metrics (notes blurbs): BLEU=21.88, chrF++=43.09, COMET=0.5735. n_heads=4 |
| head_1h_dot | small_head | dot_product | scoring | 14.57 (n=1) | 35.86 (n=1) | 34.19 (n=1) | 0.4932 (n=1) | 1 |  | 0.87441 (n=1) | 2500 |  |  |  | ok | (single seed; not statistically tested vs fast_dot) n_heads=1 |
| swap_fr_dot | small_swap | dot_product | scoring | 9.76 (n=1) | 32.22 (n=1) | 30.38 (n=1) | 0.5037 (n=1) | 1 |  | 0.79139 (n=1) | 2500 |  |  |  | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |
| var_bilinear | small_try | bilinear | scoring | 14.39 ± 1.38 (n=3) | 35.79 ± 2.30 (n=3) | 34.20 ± 2.20 (n=3) | 0.4983 ± 0.0082 (n=3) | 3 | p=0.2118 | 0.87584 ± 0.00397 (n=3) | 2500 | 30639408 | 721.15 ± 14.33 (n=3) | 10486.00 ± 0.80 (n=3) | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/var_bilinear`: BLEU=15.17, chrF++=35.37, COMET=0.5057. n_heads=4 |
| var_gated_dot_additive | small_try | gated_dot_additive | scoring | 15.11 ± 0.66 (n=3) | 36.84 ± 1.03 (n=3) | 35.19 ± 0.99 (n=3) | 0.5066 ± 0.0054 (n=3) | 3 | p=0.2323 | 0.87759 ± 0.00190 (n=3) | 2500 | 30845280 | 2008.04 ± 10.97 (n=3) | 52347.88 ± 0.20 (n=3) | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/var_gated_dot_additive`: BLEU=14.90, chrF++=34.91, COMET=0.4999. n_heads=4 |
| var_sparsemax | small_try | sparsemax | normalization | 7.65 (n=1) | 26.54 (n=1) | 25.05 (n=1) | 0.4452 (n=1) | 1 |  | 0.85419 (n=1) | 2500 | 30442800 | 1155.48 (n=1) | 10386.65 (n=1) | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |
| var_entmax15 | small_try | entmax15 | normalization | 13.70 ± 0.78 (n=3) | 34.74 ± 1.12 (n=3) | 33.16 ± 1.06 (n=3) | 0.4940 ± 0.0048 (n=3) | 3 | p=0.0227 | 0.87300 ± 0.00241 (n=3) | 2500 | 30442800 | 8748.75 ± 194.55 (n=3) | 10378.50 ± 0.62 (n=3) | ok | 3 seeds from `results/ablation_per_seed.csv`; legacy single-seed `runs/var_entmax15`: BLEU=13.66, chrF++=32.67, COMET=0.4965. n_heads=4 |
| var_local_window | small_try | local_window | connectivity | 0 | 0 | 0 | 0.356851 | failed_nan |  | 0 | 2500 | 30442800 | 1610.36 | 10374.9 | failed_nan | (single seed; not statistically tested vs fast_dot) n_heads=4; val_loss=NaN; outputs empty; numerical-stability failure, not mechanism comparison. |
| var_global_local | small_try | global_local | connectivity | 7.30 (n=1) | 24.12 (n=1) | 22.71 (n=1) | 0.4368 (n=1) | 1 |  | 0.84936 (n=1) | 2500 | 30442800 | 1069.04 (n=1) | 10374.93 (n=1) | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |
| var_local_window_a23_retrain | small_try | local_window | connectivity | 16.82 (n=1) | 38.37 (n=1) | 36.69 (n=1) | 0.5231 (n=1) | 1 |  | 0.87992 (n=1) | 2500 | 30442800 | 1110.32 (n=1) | 10700.42 (n=1) | ok | (single seed; not statistically tested vs fast_dot) n_heads=4 |

_Companion CSV: `results/attention_variants_test_summary.csv`._

### Context (external baselines — not our runs)

For **scale context only** (do **not** rank this table against these numbers): Vaswani et al. 2017 report **38.1 / 41.8 BLEU** (EN–FR base/big, ~4.5M pairs, beam 4, length penalty 0.6); a typical IWSLT'17-style EN–FR tutorial is often **30+ BLEU** on ~225k pairs with beam. This project uses **~50k pairs**, **greedy** decoding, short training (**3000 steps**), and `max_seq_len=96`, so **~15.8 BLEU** (multi-seed mean on `fast_dot`) is in a plausible range versus those references—not evidence of a broken stack. **Do not** claim competitive parity with Vaswani 2017 on BLEU alone.

---
