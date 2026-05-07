# 可复现运行说明（路径与 CLI）

所有命令均在**仓库根目录**执行（下文记为 `$REPO`）。路径默认为相对仓库根的 POSIX 字符串；也可用绝对路径。可选环境变量：`REPO_ROOT` 或 `AUTODL_REPO_ROOT`（覆盖仓库根推断）。

**历史绝对路径与 duplicate 计数口径**：见 **`docs/PROVENANCE_CAVEATS.md`**（V9-A31：`/root/autodl-tmp` 出处、`manifest.json` vs `split_metadata.json` 去重数字不可混用等）。

## Data integrity caveat（A20）

**历史问题**：`data/splits/en_fr_50k_seed42/manifest.json` 中的 **`sha256` 块**曾与盘上 **`train.tsv` / `val.tsv` / `test.tsv` 的真实 SHA-256** 不一致（而外审指出 **`scripts/check_submission_archive.py --strict` 此前未拦截该问题**）。**`split_metadata.json`** 内的 **`train_file_sha256_for_tokenizers`** 自始与真实 **`train.tsv`** 一致（该字段在划分之后、词表训练步骤写入；其 **`creation_time`** 略晚于 **`manifest.json`** 的 **`creation_timestamp_utc`** 属正常流程时序）。

**当前政策**：以**不修改 `.tsv` 本体**为前提，将 **`manifest.json` 的 `sha256`** 更新为对三个 TSV 的实测摘要，并标注 **`regenerated_from_actual_files`** / **`regenerated_at`**；在 **`split_metadata.json`** 中补全 **`val_file_sha256`**、**`test_file_sha256`** 与 manifest 对齐。

**持续校验**：`python scripts/check_data_integrity.py` 比对 **manifest** 与 **split_metadata** 中的记录与盘上文件；**`python scripts/check_submission_archive.py --strict`** 在 strict 模式下会调用该校验，失败即 **exit 1**。报告见 **`results/data_integrity_report.md`**。

## Code archive contents

用于论文补充材料或审稿人可下载的**代码归档**（不含可重新训练产生的权重）。打包前用 `python scripts/prepare_code_archive.py` 校验 held-out `predictions.jsonl` 与 `metrics_test.json` 是否成对，以及 `results/attention_variants_test_summary.csv`：**单行** run 需有 `runs/<run>/test_eval/metrics_test.json`；**`is_aggregate=true`** 行则需 `source_run_names` 中每个 seed run 的 `test_eval/metrics_test.json` 与 `predictions.jsonl` 均存在（不要求存在合成的 `runs/<aggregate>/`）。`--output-zip path.zip` 在通过校验后生成 zip。归档内容意向如下（最终 zip 以脚本逻辑为准：包含未被 `.gitignore` 排除的仓库文件，并始终排除 `.git/`）。

**Include**

- 源代码目录：`small_try/`、`small_head/`、`small_swap/`、`base_1/`、`base_improve/`、`scripts/`、`experiments/`、`tests/`、`translate_cli/`
- 固定划分与元数据：`data/splits/en_fr_50k_seed42/*.tsv`、`manifest.json`、`split_metadata.json`
- 词表：`data/tokenizer_*.json`、`data/tokenizer_metadata.json`
- 各 run 的训练与配置摘要：`runs/<run>/metrics.json`、`resolved_config.json`、`training_meta.json`（若存在）
- 各 run 的 **held-out test** 评估产物：`runs/<run>/test_eval/metrics_test.json`、`examples.md`、`predictions.jsonl`
- 汇总与审计：`results/*.md`、`results/*.csv`、`results/*.json`
- 文档：所有 `docs/*.md`，以及仓库根 `README.md`、各子包 `requirements.txt`、`.gitignore`

**Exclude**

- 所有模型检查点：`*.pt`、`*.pth`、`*.ckpt`、`*.safetensors`（与 `.gitignore` 一致）
- W&B 本地缓存：`wandb/`、`wandb_cache*/`
- 过大的完整平行语料（如 `data/EN-FR.txt`，可按需从发布渠道获取）
- 通用日志：`*.log`、`logs/` 等
- Python 缓存：`__pycache__/`、`.pytest_cache/`

## Result tables must be regenerated after new runs

在刷新或新增 `runs/` 后，派生 **`results/*.md` / CSV / JSON** 时应按下列顺序执行（**A2 → A5**；不要手改汇总表中的数字）：

**A2 — Runs 健康与状态**（`results/runs_sanity_report.{md,json}`；汇总表 `status` 列依赖此输出）：

```bash
cd "$REPO"
python scripts/sanity_check_runs.py
```

**A3 — 修补缺失的 `training_meta.json`（若需要）**（仅写入保守字段如 `num_parameters`；默认针对 `head_1h_dot`、`swap_fr_dot`）：

```bash
cd "$REPO"
python scripts/repair_missing_training_meta.py
```

**A4 — Attention 变体 test 汇总 + 公平性审计**

```bash
cd "$REPO"
python scripts/summarize_attention_variants.py \
  --runs fast_dot fast_add head_1h_dot swap_fr_dot var_bilinear var_gated_dot_additive \
    var_sparsemax var_entmax15 var_local_window var_global_local

python scripts/check_variant_experiment_fairness.py \
  --runs fast_dot fast_add head_1h_dot swap_fr_dot var_bilinear var_gated_dot_additive \
    var_sparsemax var_entmax15 var_local_window var_global_local
```

**A5 — Cross-seed Welch（dot vs additive，per-seed held-out test 指标）**（`results/cross_seed_significance.{md,json}`；输入默认 `results/ablation_per_seed.csv`）：

```bash
cd "$REPO"
python scripts/cross_seed_significance.py
```

不要编辑 `runs/*/test_eval/metrics_test.json`；仅重跑上述脚本以刷新派生汇总。

## Existing runs determinism caveat

本 archive 中既有 run 在训练时**未**统一开启严格确定性开关（例如 `cudnn.benchmark=True`，且未启用 `use_deterministic_algorithms`）。**fast_dot** 跨 seed 的 BLEU 标准差约 **0.52** 反映 **随机种子、dropout、数据打乱与 cuDNN 非确定性** 的共同方差。今后若以 **`--deterministic`** 启动新训练，相关设置会写入该次 run 的 **`resolved_config.json`**（`determinism` 字段）、**`metrics.json`** 与 **`training_meta.json`**。

## Determinism flag for future runs

Existing runs in this archive were trained without `--deterministic`. The cross-seed standard deviation of 0.52 BLEU on fast_dot (n=3) reflects combined variance from random initialization, dropout, data shuffling, and cuDNN nondeterminism. To reduce this variance in future experiments, train with e.g. `python small_try/train.py --deterministic ...`（`small_head` / `small_swap` / `base_1` / `base_improve` 同样提供该标志）。`--deterministic` 可能使训练略慢（常见幅度约 **10%** 以内，视 GPU 与 batch 而定）。

## 数据划分（train / val / test）

从完整平行语料生成独立测试集与 `split_metadata.json`：

```bash
cd "$REPO"
python scripts/make_splits.py \
  --input data/EN-FR.txt \
  --output-dir data/splits/en_fr_50k_seed42 \
  --sample-size 50000 \
  --seed 42 \
  --train-ratio 0.90 \
  --val-ratio 0.05 \
  --test-ratio 0.05
```

脚本会：跳过空行与格式错误行、去掉 `source == target`、对句对去重、用固定 seed 打乱后按比例划分；若 `--sample-size` 小于去重后的总数，则在唯一句对上做**均匀随机抽样**（水库抽样），而非取文件前 N 行。`--sample-size 0` 表示使用去重后的全部句对。

仓库内 `scripts/create_splits.py` 为早期等价脚本；新实验请以 `make_splits.py` 为准。

默认配置（`small_try` / `small_head` / `small_swap` 的 `config.py`）使用划分文件，`use_split_files=True`。若仍需单一 `corpus_50k.tsv`：在配置中设 `use_split_files=False` 并指定 `data_path`，此时按 `val_ratio` / `test_ratio` 对**有效句对**做确定性分桶（模 1000），不再使用周期 `idx % period`。

## 词表（仅从 train.tsv，防泄漏）

训练前应用 **仅读取 `train.tsv`** 的脚本生成 BPE，避免 val/test 进入词表：

```bash
cd "$REPO"
python scripts/train_tokenizers_from_train_split.py \
  --train-file data/splits/en_fr_50k_seed42/train.tsv \
  --tokenizer-src-out data/tokenizer_src_train_only.json \
  --tokenizer-tgt-out data/tokenizer_tgt_train_only.json \
  --vocab-size 30000
```

默认写出 `data/tokenizer_metadata.json`（可用 `--metadata-out` 覆盖），内含 `train_file_sha256`。可选 `--also-update-split-metadata data/splits/en_fr_50k_seed42/split_metadata.json`，向划分元数据追加 `train_file_sha256_for_tokenizers` 等字段。

默认 `config.py` 已指向 `tokenizer_*_train_only.json`；若尚未生成这些文件，可先运行上述命令，或通过 `--tokenizer-src` / `--tokenizer-tgt` 临时指向旧版 `data/tokenizer_src.json`。

## Test 集最终评估（`evaluate_test.py`）

训练时请将 **`eval_split=val`**，仅用验证集做 early stopping / 周期性 BLEU；**不要用 test.tsv 做训练中途早停**。对保留的 `test.tsv` 在训练结束后运行仓库根目录的 **`evaluate_test.py`**（不做 identical/high-similarity 跳过；BLEU/chrF/COMET/BERTScore 等见输出的 `metrics_test.json`）。Independent test evaluation is stored as local artifacts by default; W&B logging is optional and secondary（见 `--wandb-eval`）。

```bash
cd "$REPO"
python evaluate_test.py \
  --checkpoint runs/<run_name>/best.pt \
  --test-file data/splits/en_fr_50k_seed42/test.tsv \
  --tokenizer-src data/tokenizer_src_train_only.json \
  --tokenizer-tgt data/tokenizer_tgt_train_only.json \
  --output-dir runs/<run_name>/test_eval \
  --max-new-tokens 64 \
  --batch-size 32 \
  --pkg small_try
```

输出：`predictions.jsonl`、`metrics_test.json`、`examples.md`。若在训练脚本中将 `eval_split` 设为 `test`，日志会打印警告。

训练结束后每个 run 目录会写出 **`training_meta.json`**（墙钟时间、`num_parameters`、峰值 GPU 显存等），供 `experiments/run_ablation.py` 汇总。

### Experiment fairness audit

Read-only cross-run check: same splits / train-only tokenizers / backbone & budget / seed / eval protocol (plus optional parameter-count deltas vs `fast_dot`). Writes **`results/variant_fairness_audit.md`** and **`results/variant_fairness_audit.json`**; **exit 1** if any run **FAIL** (missing configs or critical mismatch).

```bash
cd "$REPO"
python scripts/check_variant_experiment_fairness.py \
  --archive-audit
```

省略 `--runs` 时使用脚本内置默认列表（含 `fast_dot`、`fast_add`、`head_1h_dot`、若干 `var_*` 等）；可用 `--runs-root`、`--output-md`、`--output-json` 自定义路径。

### 两系统显著性检验（`scripts/significance_test.py`）

对两份 **`evaluate_test.py`** 生成的 **`predictions.jsonl`**（同一 test、行对齐）做配对 bootstrap，输出 ΔBLEU / ΔchrF、近似 p-value、`results/significance_report.md` 解读：

```bash
cd "$REPO"
python scripts/significance_test.py \
  --predictions-a runs/exp_a/test_eval_best_val/predictions.jsonl \
  --predictions-b runs/exp_b/test_eval_best_val/predictions.jsonl \
  --name-a dot_h4 --name-b add_h4
```

## 多 seed ablation（`experiments/run_ablation.py`）

**默认（flat）**：`fast_dot` / `fast_add`（各 `dot_product` / `additive`，`n_heads=4`），seeds **`1 2 3`**，目录 **`runs/fast_dot_s1` … `runs/fast_add_s3`**；`--max-steps` 默认 **3000**；每轮训练后对 **`best.pt`** 调用 **`evaluate_test.py`** → **`runs/<run>/test_eval/`**。汇总 **`results/ablation_summary.csv`** / **`ablation_summary.md`**（含 BLEU / chrF++ / COMET 等跨 seed 均值与标准差）及 **`ablation_per_seed.csv`**。

```bash
cd "$REPO"
python experiments/run_ablation.py --dry-run
python experiments/run_ablation.py \
  --train-path data/splits/en_fr_50k_seed42/train.tsv \
  --val-path data/splits/en_fr_50k_seed42/val.tsv \
  --test-path data/splits/en_fr_50k_seed42/test.tsv \
  --seeds 1 2 3 \
  --max-steps 3000
```

GPU 空闲后若只需补跑 **`fast_add_*`**（不重训 `fast_dot_*`）：

```bash
python experiments/run_ablation.py --experiments fast_add --seeds 1 2 3 --max-steps 3000
```

完成训练与 test 评估后，可用 **`--aggregate-only`** 仅从磁盘刷新汇总（不重训）：

```bash
python experiments/run_ablation.py --aggregate-only --seeds 1 2 3
```

**Legacy**：加 **`--legacy-layout`** 恢复旧版 **`runs/<experiment>/seed_<seed>/`**（dot_h4 / add_h4 / 单头等），seeds 默认 `42 43 44`，并对 **`best.pt`** 与 **`last.pt`** 各跑一次 test 评估。

`--dry-run` 仅打印命令；`--skip-add-h1` 仅作用于 legacy；`--no-eval` 只训练不测评；`--wandb` 打开 W&B。

## 模型效率 Profile（`scripts/profile_model.py`）

对 **`dot_product` / `additive`** 与 **`n_heads ∈ {1,4}`**（默认）做合成 batch 前向测速，输出 **`results/profile_summary.md`** 与 **`profile_summary.json`**（总参数、可训练参数、`attention_module_parameter_count`、仅打分网络 `attention_scoring_parameter_count`、耗时、tok/s、CUDA 峰值显存）。

```bash
cd "$REPO"
python scripts/profile_model.py \
  --batch-size 8 --seq-len 64 --d-model 256 --n-layers 4
```

## 共用训练 CLI（`small_try` / `small_head` / `small_swap`）

三组 `train.py` 均注册同一套参数（来自根目录 `train_runtime.py`），命令行会覆盖各包 `config.py` 中的默认值：

| 参数 | 说明 |
|------|------|
| `--data-path` | 训练语料 TSV（未单独指定 `--train-path` 时等同 `train_path`） |
| `--train-path` / `--val-path` / `--test-path` | 划分后的 TSV |
| `--tokenizer-src` / `--tokenizer-tgt` | 词表 JSON |
| `--output-dir` | 输出父目录（实际 run 目录为 `<output-dir>/<run_name>/`） |
| `--attention-type`（或 `--attention`） | 注意力类型 |
| `--n-heads` | 注意力头数 |
| `--seed` | 随机种子 |
| `--max-steps` | 优化器步数上限 |
| `--batch-size` | batch size |
| `--no-wandb` | 关闭 W&B |

查看完整帮助：

```bash
cd "$REPO"
python small_try/train.py --help
python small_head/train.py --help
python small_swap/train.py --help
```

**`local_window` / `global_local`：** 代码中为 **dense masked attention**（完整 `L×L` 注意力打分 + softmax，仅用加性掩码表达局部带或「全局锚点 + 局部带」先验）。**不是** Longformer / ETC 等稀疏核或未物化的块稀疏注意力实现；勿在论文中写成「稀疏 / 线性复杂度注意力模块」，除非另行实现并单独命名。

### small_try（英→法）

默认：`config.py` 中 `train_path` 等为 `data/splits/en_fr_50k_seed42/...`，`output_dir` 为 `runs`。

示例（显式写出常用参数，输出到 `runs/fast_dot/`）：

```bash
cd "$REPO"
python small_try/train.py \
  --data-path data/splits/en_fr_50k_seed42/train.tsv \
  --tokenizer-src data/tokenizer_src_train_only.json \
  --tokenizer-tgt data/tokenizer_tgt_train_only.json \
  --output-dir runs \
  --attention-type dot_product \
  --n-heads 4 \
  --seed 42 \
  --max-steps 3000 \
  --batch-size 192 \
  --name fast_dot \
  --no-wandb
```

### small_head（单头等）

依赖多头基线 metrics 时，可先完成 `small_try` 的对应 run，再在流水线或手动对比中使用 `runs/fast_dot/metrics.json`。

### small_swap（法→英）

与 `small_try` 共用划分与词表路径；训练脚本内通过配置启用列交换，CLI 与上表一致。

```bash
cd "$REPO"
python small_swap/train.py \
  --output-dir runs \
  --attention-type dot_product \
  --name swap_fr_dot \
  --no-wandb
```

## base_1（独立 Encoder–Decoder 栈）

仓库中的 **`base_1/`** 是与 `small_try` **分叉**的另一套完整 Transformer 训练目录；**仅**支持缩放点积与加性注意力，**不表示** `small_try` 里的全部注意力变体都已并入该路径。适用范围与论文叙事对齐说明见 **`base_1/README.md`**。

## 流水线脚本（可选）

在对应子目录下执行，脚本内部用 `$REPO` 解析路径，不依赖固定机器路径：

- `small_try/run_compare.sh`
- `small_head/run_head_pipeline.sh`
- `small_swap/run_fr_swap_pipeline.sh`

## 指标对比

```bash
cd "$REPO"
python small_try/compare_runs.py --dot runs/fast_dot/metrics.json --add runs/fast_add/metrics.json
python small_head/compare_head_runs.py --baseline runs/fast_dot/metrics.json --one-dot runs/head_1h_dot/metrics.json
python small_swap/compare_runs.py --dot runs/swap_fr_dot/metrics.json
```

## 交互翻译（`translate_cli`）

在仓库根目录执行；checkpoint 可为相对路径（相对仓库根）。需提供 `--pkg` 以匹配训练时使用的代码包：

```bash
cd "$REPO"
echo 'Hello.' | python translate_cli/interactive_translate.py \
  -c runs/fast_dot/best.pt \
  --pkg small_try
```

未指定 `-c` 时默认尝试加载 `runs/best.pt`。

更多说明见 `translate_cli/README.txt`。
