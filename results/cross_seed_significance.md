# Cross-seed significance: fast_dot vs fast_add

## Methodology note

This is cross-**SEED** evidence using per-seed **final test** BLEU / chrF++ / COMET as **independent samples** (one value per training seed). It **complements** but does **NOT** replace the sentence-level paired bootstrap in `scripts/significance_test.py`, which measures **within–prediction-set** resampling variation on a **single** trained model. **Both** perspectives should be reported when discussing dot vs additive.

## BLEU: results

- **fast_dot**: mean=15.803, std=0.52249, n=3, values=[16.27755073167496, 15.243098254602273, 15.8884560205611]
- **fast_add**: mean=8.20571, std=0.696089, n=3, values=[7.7623095148713634, 7.846808746410006, 9.00799750430904]
- mean difference: 7.59733 (fast_dot minus fast_add)
- Welch t: 15.1189, df ≈ 3.71064
- 95% CI for difference: [6.15818, 9.03648]
- p-value (two-sided): 0.000180849
- Conclusion: fast_dot is significantly better than fast_add at α=0.05 (two-sided).

## chrF++: results

- **fast_dot**: mean=35.7896, std=0.579269, n=3, values=[36.13274834172182, 35.12084196151532, 36.1153549930606]
- **fast_add**: mean=25.4362, std=1.29622, n=3, values=[24.48966397535564, 24.905376769927237, 26.913589234420794]
- mean difference: 10.3534 (fast_dot minus fast_add)
- Welch t: 12.6307, df ≈ 2.7682
- 95% CI for difference: [7.61674, 13.0901]
- p-value (two-sided): 0.00157495
- Conclusion: fast_dot is significantly better than fast_add at α=0.05 (two-sided).

## COMET: results

- **fast_dot**: mean=0.508885, std=0.00324447, n=3, values=[0.5067629693210125, 0.5072731954693794, 0.5126202752172947]
- **fast_add**: mean=0.454179, std=0.00740017, n=3, values=[0.44794571356773377, 0.4522346744120121, 0.46235767346024514]
- mean difference: 0.0547061 (fast_dot minus fast_add)
- Welch t: 11.7267, df ≈ 2.74149
- 95% CI for difference: [0.0390355, 0.0703768]
- p-value (two-sided): 0.00201403
- Conclusion: fast_dot is significantly better than fast_add at α=0.05 (two-sided).

## All metrics summary

| Metric | Δ (A-B) | 95% CI | t | df | p |
|--------|---------|--------|---|---|---|
| BLEU | 7.59733 | [6.158, 9.036] | 15.1189 | 3.71064 | 0.0001808 |
| chrF++ | 10.3534 | [7.617, 13.09] | 12.6307 | 2.7682 | 0.001575 |
| COMET | 0.0547061 | [0.03904, 0.07038] | 11.7267 | 2.74149 | 0.002014 |

SciPy unavailable (import failed); using **mpmath** Student-t CDF (regularized incomplete beta) and bisection for `ppf`.
