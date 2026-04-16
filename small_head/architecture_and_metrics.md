# small_head：架构与可比性说明

## 与 small_try 多头基线的关系

在 **与 small_try 相同的** 数据子集、步数、优化器与模型宽度（`d_model=256`, `n_layers=4`, `d_ff=1024`）下，将 **多头基线**（`n_heads=4`，点积注意力，即 `small_try/runs/fast_dot`）与 **全模型单头**（`n_heads=1`，默认 `dot_product`）对比。

**不必重训基线。** 多头点积基线若已在 `small_try` 完成且 `metrics.json` 存在，与本流水线设定一致，可直接作为 baseline 合并进报告。仅在以下情况需要重训：

- 更换了 `data_path`、`max_steps`、`seed` 等与 small_try 不一致的设定；

本流水线默认 **引用** `small_try/runs/fast_dot/metrics.json`，**只新训** `head_1h_dot`（默认）。若需单头加性实验，使用 `train.py --attention additive --allow-additive-attention` 并可在 `compare_head_runs.py` 传入 `--one-add`。
