# 可复现运行说明（路径与 CLI）

所有命令均在**仓库根目录**执行（下文记为 `$REPO`）。路径默认为相对仓库根的 POSIX 字符串；也可用绝对路径。可选环境变量：`REPO_ROOT` 或 `AUTODL_REPO_ROOT`（覆盖仓库根推断）。

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

训练时请将 **`eval_split=val`**，仅用验证集做 early stopping / 周期性 BLEU；**不要用 test.tsv 做训练中途早停**。对保留的 `test.tsv` 在训练结束后运行仓库根目录的 **`evaluate_test.py`**（不做 identical/high-similarity 跳过；BLEU/chrF/COMET/BERTScore 等见输出的 `metrics_test.json`）：

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

批量运行 dot/add × head数 × seed∈{42,43,44}，输出目录 **`runs/<experiment>/seed_<seed>/`**；每次训练后对 **`best.pt`**（验证 BLEU 最优）与 **`last.pt`**（最后一轮）分别调用 `evaluate_test.py`，汇总 **`results/ablation_summary.csv`** / **`ablation_summary.md`**（及明细 **`ablation_per_seed.csv`**）。

```bash
cd "$REPO"
python experiments/run_ablation.py \
  --train-path data/splits/en_fr_50k_seed42/train.tsv \
  --val-path data/splits/en_fr_50k_seed42/val.tsv \
  --test-path data/splits/en_fr_50k_seed42/test.tsv \
  --max-steps 3000
```

`--dry-run` 仅打印命令；`--skip-add-h1` 跳过 additive + 单头组合；`--no-eval` 只训练不测评；`--wandb` 打开 W&B。

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
