# Cross-seed significance: fast_dot vs fast_add_lr3e3

## Methodology note

This is cross-**SEED** evidence using per-seed **final test** BLEU / chrF++ / COMET as **independent samples** (one value per training seed). It **complements** but does **NOT** replace the sentence-level paired bootstrap in `scripts/significance_test.py`, which measures **within–prediction-set** resampling variation on a **single** trained model. **Both** perspectives should be reported when discussing dot vs additive.

## BLEU: results

- **fast_dot**: mean=15.803, std=0.52249, n=3, values=[16.27755073167496, 15.243098254602273, 15.8884560205611]
- **fast_add_lr3e3**: mean=21.565, std=0.418071, n=3, values=[21.88070162614213, 21.723361128228106, 21.090847128615547]
- mean difference: -5.76193 (fast_dot minus fast_add_lr3e3)
- Welch t: -14.9141, df ≈ 3.81641
- 95% CI for difference: [-6.85525, -4.66862]
- p-value (two-sided): 0.000159323
- Conclusion: fast_dot is significantly worse than fast_add_lr3e3 at α=0.05 (two-sided).

## chrF++: results

- **fast_dot**: mean=35.7896, std=0.579269, n=3, values=[36.13274834172182, 35.12084196151532, 36.1153549930606]
- **fast_add_lr3e3**: mean=42.6973, std=0.459442, n=3, values=[43.09400931636172, 42.80406660576032, 42.19391525771166]
- mean difference: -6.90768 (fast_dot minus fast_add_lr3e3)
- Welch t: -16.1824, df ≈ 3.80285
- 95% CI for difference: [-8.11748, -5.69789]
- p-value (two-sided): 0.000119953
- Conclusion: fast_dot is significantly worse than fast_add_lr3e3 at α=0.05 (two-sided).

## COMET: results

- **fast_dot**: mean=0.508885, std=0.00324447, n=3, values=[0.5067629693210125, 0.5072731954693794, 0.5126202752172947]
- **fast_add_lr3e3**: mean=0.577574, std=0.00608667, n=3, values=[0.5734998296558858, 0.5845706348121166, 0.5746511001765728]
- mean difference: -0.0686884 (fast_dot minus fast_add_lr3e3)
- Welch t: -17.2488, df ≈ 3.05165
- 95% CI for difference: [-0.0812411, -0.0561357]
- p-value (two-sided): 0.000384066
- Conclusion: fast_dot is significantly worse than fast_add_lr3e3 at α=0.05 (two-sided).

## All metrics summary

| Metric | Δ (A-B) | 95% CI | t | df | p |
|--------|---------|--------|---|---|---|
| BLEU | -5.76193 | [-6.855, -4.669] | -14.9141 | 3.81641 | 0.0001593 |
| chrF++ | -6.90768 | [-8.117, -5.698] | -16.1824 | 3.80285 | 0.00012 |
| COMET | -0.0686884 | [-0.08124, -0.05614] | -17.2488 | 3.05165 | 0.0003841 |

Two-sided p-values and 95% CIs: **SciPy** `scipy.stats.t`.
