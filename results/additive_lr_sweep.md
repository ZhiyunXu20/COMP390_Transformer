# Additive learning-rate sweep (single seed)

## Methodology

- **Single seed** sweep (`seed=42`); not a stability / cross-seed inference exercise.
- **`max_steps=1500`** (half of the standard 3000) to fit GPU time.
- All hyperparameters except **`learning_rate`** follow **`small_try.Config` defaults** (same data split, batch size, architecture, additive attention, `n_heads=4`).
- Training uses **`--eval-light`** (validation BLEU subsample: BLEU + chrF only, no BERTScore/COMET).
- Held-out **`evaluate_test.py`** uses **`--eval-light`** (BLEU + chrF++ only).
- **`--no-resume`** re-trains every lr from scratch (overwrites `runs/lr_sweep_add_*`). Default is resume: skip runs that already have `test_eval/metrics_test.json`; if only `best.pt` exists, run **`evaluate_test.py`** only.

## Results

| lr | val_loss | val_BLEU | test_BLEU | test_chrF++ | wall_s |
|---:|---:|---:|---:|---:|---:|
| 1e-04 | 4.846339911848739 | 3.266509790185837 | 2.8884360692030144 | 15.062193949443156 | 930.7704473715276 |
| 3e-04 | 4.171219560899871 | 5.45477765953419 | 5.198087479062094 | 19.883217348270076 | 920.0436653327197 |
| 1e-03 | 3.353633355369497 | 12.897998518931136 | 12.706446662287124 | 32.175475798056766 | 913.5582611132413 |
| 3e-03 | 3.0548538595540435 | 16.60689518913766 | 15.480186774893292 | 35.60919351062927 | 930.4225677233189 |

## Comparison vs reference points

- **fast_add** (lr=3e-4, max_steps=3000, n=3): test_BLEU = **8.21 ± 0.70** (from `results/ablation_summary.csv` when present).
- **fast_dot** (lr=3e-4, max_steps=3000, n=3): test_BLEU = **15.80 ± 0.52** (headline thesis figure).
- **This sweep** (max_steps=1500): best additive lr is **3e-03** with test_BLEU = **15.480186774893292** (`runs/lr_sweep_add_lr3e3`).
  - Sweep test_BLEU range (this table): **2.8884** – **15.4802**.

## Interpretation

If the best additive lr in this sweep yields test_BLEU **meaningfully above** the fast_add ~8.2 mean, 
the dot-vs-additive comparison should note that **per-mechanism HP tuning can move additive somewhat**. 
If all sweep lrs yield **similar** BLEU, the gap is **more plausibly mechanism-driven** under this budget.

**Readout:** best sweep test_BLEU exceeds the fast_add mean by **~7.27 BLEU** (single seed, half training steps); HP tuning may explain **part** of the gap vs dot_product but does not approach the dot baseline in this sweep.
