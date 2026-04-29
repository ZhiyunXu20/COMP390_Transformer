# Attention variants — held-out test summary

**All reported scores are from independent held-out test evaluation, not training-time validation metrics.**

Sources are limited to `runs/<run>/test_eval/metrics_test.json` only (never `runs/<run>/metrics.json`).

---


| run | attention_type | family | bleu | chrf | comet | bertscore_f1 | num_examples | checkpoint | notes | status |
|---|---|---|---|---|---|---|---|---|---|---|
| fast_dot | dot_product | scaled dot-product / softmax / dense full | 16.0667 | 37.6701 | 0.505479 | 0.878385 | 2500 | /root/autodl-tmp/runs/fast_dot/best.pt | pkg=small_try; n_heads=4 | ok |
| fast_add | additive | additive scoring / softmax / dense full | 8.16327 | 26.1255 | 0.444405 | 0.853527 | 2500 | /root/autodl-tmp/runs/fast_add/best.pt | pkg=small_try; n_heads=4 | ok |
| head_1h_dot | dot_product | scaled dot-product / softmax / dense full | 14.5673 | 35.856 | 0.493225 | 0.874409 | 2500 | /root/autodl-tmp/runs/head_1h_dot/best.pt | pkg=small_head; n_heads=1 | ok |
| var_bilinear |  |  |  |  |  |  |  |  | missing runs/var_bilinear/test_eval/metrics_test.json | missing_test_eval |
| var_gated_dot_additive |  |  |  |  |  |  |  |  | missing runs/var_gated_dot_additive/test_eval/metrics_test.json | missing_test_eval |
| var_sparsemax |  |  |  |  |  |  |  |  | missing runs/var_sparsemax/test_eval/metrics_test.json | missing_test_eval |
| var_entmax15 |  |  |  |  |  |  |  |  | missing runs/var_entmax15/test_eval/metrics_test.json | missing_test_eval |
| var_local_window |  |  |  |  |  |  |  |  | missing runs/var_local_window/test_eval/metrics_test.json | missing_test_eval |
| var_global_local |  |  |  |  |  |  |  |  | missing runs/var_global_local/test_eval/metrics_test.json | missing_test_eval |

_Companion CSV: `results/attention_variants_test_summary.csv`._
