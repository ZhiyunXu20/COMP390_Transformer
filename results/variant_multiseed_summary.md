# Variant multi-seed summary (A14)

## Methodology

- **3 seeds** each (`1`, `2`, `3`) under the same protocol as **`fast_dot` / `fast_add`** in `small_try`: `max_steps=3000`, default `Config`, **`lr=3e-4`**, bf16 autocast, no extra determinism flag.
- Training: **`--no-wandb`**, **`--eval-light`**, **`--no-attention-plots`** on validation BLEU subsample.
- Held-out test: **`evaluate_test.py`** without `--eval-light` (BLEU + chrF++ + BERTScore + COMET).
- Reference cross-seed stats for **`fast_dot`**: from `results/ablation_per_seed.csv` (experiment=`fast_dot`, `checkpoint_kind=best`).

## Results

| variant | mean BLEU | std BLEU | mean chrF++ | std chrF++ | mean COMET | std COMET |
|---------|-----------|----------|-------------|------------|------------|-----------|
| fast_dot (reference) | 15.8030 | 0.5225 | 35.7896 | 0.5793 | 0.5089 | 0.0032 |
| bilinear | 14.3936 | 1.3796 | 34.1990 | 2.1971 | 0.4983 | 0.0082 |
| gated dot additive | 15.1097 | 0.6647 | 35.1901 | 0.9863 | 0.5066 | 0.0054 |
| entmax15 | 13.7033 | 0.7764 | 33.1552 | 1.0611 | 0.4940 | 0.0048 |

## Cross-seed Welch t-test vs fast_dot

Independent samples: one **held-out test** metric per training seed (same layout as `scripts/cross_seed_significance.py`). **Δ** = mean(variant) − mean(fast_dot). Stats backend: **scipy** (SciPy present: True).

### var_bilinear vs fast_dot

- **BLEU**: Δ=-1.4094 t=-1.6548 df≈2.5622 p=0.211795 95% CI Δ [-4.4017, 1.5829] — var_bilinear is not significantly different from fast_dot at α=0.05 (two-sided).
- **chrF++**: Δ=-1.5907 t=-1.2126 df≈2.2767 p=0.336058 95% CI Δ [-6.6260, 3.4447] — var_bilinear is not significantly different from fast_dot at α=0.05 (two-sided).
- **COMET**: Δ=-0.0106 t=-2.0932 df≈2.6180 p=0.140526 95% CI Δ [-0.0281, 0.0069] — var_bilinear is not significantly different from fast_dot at α=0.05 (two-sided).

### var_gated_dot_additive vs fast_dot

- **BLEU**: Δ=-0.6933 t=-1.4204 df≈3.7887 p=0.232289 95% CI Δ [-2.0789, 0.6922] — var_gated_dot_additive is not significantly different from fast_dot at α=0.05 (two-sided).
- **chrF++**: Δ=-0.5995 t=-0.9078 df≈3.2331 p=0.426475 95% CI Δ [-2.6180, 1.4189] — var_gated_dot_additive is not significantly different from fast_dot at α=0.05 (two-sided).
- **COMET**: Δ=-0.0022 t=-0.6169 df≈3.2798 p=0.577497 95% CI Δ [-0.0133, 0.0088] — var_gated_dot_additive is not significantly different from fast_dot at α=0.05 (two-sided).

### var_entmax15 vs fast_dot

- **BLEU**: Δ=-2.0998 t=-3.8864 df≈3.5033 p=0.022749 95% CI Δ [-3.6875, -0.5120] — var_entmax15 is significantly worse than fast_dot at α=0.05 (two-sided).
- **chrF++**: Δ=-2.6345 t=-3.7744 df≈3.0948 p=0.030855 95% CI Δ [-4.8178, -0.4511] — var_entmax15 is significantly worse than fast_dot at α=0.05 (two-sided).
- **COMET**: Δ=-0.0149 t=-4.4662 df≈3.5251 p=0.014752 95% CI Δ [-0.0246, -0.0051] — var_entmax15 is significantly worse than fast_dot at α=0.05 (two-sided).

---

Two-sided p-values and 95% CIs: SciPy `scipy.stats.t`.
