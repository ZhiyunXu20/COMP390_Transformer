# Attention ablation summary

Seeds: `1, 2, 3`；训练目录：`runs/fast_dot_s1` … `runs/fast_add_s3`（flat）。
同一 `train.tsv` / `val.tsv` / `test.tsv`；`max_steps=3000`（legacy 未指定时为 Config 默认）。
Test 指标在独立 `test.tsv` 上由 `evaluate_test.py` 计算。

## Aggregate (mean ± std over seeds)

| experiment | checkpoint | mean BLEU | std BLEU | mean chrF++ | std chrF++ | mean COMET | std COMET | chrF (mean±std) | train time (s) | peak GPU (MiB) | num params |
|------------|------------|-----------|----------|-------------|------------|------------|-----------|-----------------|----------------|----------------|------------|
| fast_add | best | 8.2057 | 0.6961 | 25.4362 | 1.2962 | 0.4542 | 0.0074 | 27.0167 ± 1.4461 | 1943.8 ± 31.6 | 51823.7 ± 0.1 | 30845232 |
| fast_dot | best | 15.8030 | 0.5225 | 35.7896 | 0.5793 | 0.5089 | 0.0032 | 37.4169 ± 0.6269 | 718.9 ± 8.1 | 10374.9 ± 0.2 | 30442800 |

## Per-seed detail

完整行级结果见 `results/ablation_per_seed.csv`。
