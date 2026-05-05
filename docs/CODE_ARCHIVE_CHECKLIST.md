# Code archive checklist（提交包内容清单）

用于对照 `scripts/check_submission_archive.py` 与 `scripts/prepare_code_archive.py` 打包前的自检。

## Files that MUST be in the archive

- 源代码：**所有** `small_try/`、`small_head/`、`small_swap/`、`base_1/`、`base_improve/`、`scripts/`、`experiments/`、`tests/`、`translate_cli/` 下的 **`.py`** 文件（及同目录内被代码引用的非 Python 配置若已纳入 git）。
- **划分与元数据**：`data/splits/en_fr_50k_seed42/train.tsv`、`val.tsv`、`test.tsv`、`split_metadata.json`。
- **词表**：`data/tokenizer_*.json`、`data/tokenizer_metadata.json`（以 `.gitignore` 未排除且已跟踪为准）。
- **每个纳入汇总的 run 目录**（至少包含下列 **10 个报告用 run**：`fast_dot`、`fast_add`、`head_1h_dot`、`swap_fr_dot`、`var_bilinear`、`var_gated_dot_additive`、`var_sparsemax`、`var_entmax15`、`var_local_window`、`var_global_local`）：
  - `metrics.json`、`resolved_config.json`、`training_meta.json`（含事后 **repaired** 的 meta）。
  - `test_eval/metrics_test.json`、`test_eval/predictions.jsonl`、`test_eval/examples.md`。
- **汇总与审计**：`results/` 下本次提交需要的 **`.md` / `.json` / `.csv`**（脚本会校验一组核心文件名；见 `check_submission_archive.py` 内列表）。
- **文档**：`docs/*.md`（至少包含审计与可复现主文档，见校验脚本）。
- **仓库根**：`README.md`、`requirements.txt`、`.gitignore`。

## Files that MUST NOT be in the archive

- **任意模型检查点**：`*.pt`、`*.pth`、`*.ckpt`、`*.safetensors`（应被 `.gitignore` 排除；打包脚本亦按 git ignore 规则跳过）。
- **`wandb/`**、各包下 **`wandb_cache*`** 等本地实验缓存目录。
- **完整粗语料**（体积过大时）：如 `data/EN-FR.txt` 或完整 DCEP 平行人；若必须保留占位，请在 README 说明获取方式而非提交全量文件。
- **噪音日志**：任意 **`.log`**（如流水线 `nohup` 输出）；`results/*.log` 亦不建议纳入最终审稿包。
- **`__pycache__/`**、**`.pytest_cache/`**、**`.git/`**。

## Optional inclusions

- 体积较小、无敏感信息的 **`logs/`**。
- **W&B 导出的可视化 PNG**（若审稿需要且版权允许）。

## Predictions intentionally excluded（说明）

- **`predictions_val_sample.jsonl`**（训练 run 目录下、来自训练脚本 val 子样本）：可能与 **held-out test** 结论混读，是否纳入由 `.gitignore` / 打包策略决定；**不得以缺失为由判定 held-out test 失效**——以 **`test_eval/predictions.jsonl`** 为准。
