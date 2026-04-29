# 科研工程审计报告（仓库快照）

**角色**：严格审稿视角（可复现性、声称与实现一致性、指标语义）。  
**范围**：`small_try`、`small_head`、`small_swap`、`base_1`、`conclusion`（若存在）、`scripts`、`data`。  
**说明**：本文件仅基于当前仓库源码与配置可读内容；**不修改代码**。

---

## 1. 本项目实际实现了什么（不夸大）

- **序列到序列 Transformer（Encoder–Decoder）**：嵌入 + 位置编码（`small_try/model.py` `Seq2SeqTransformer`）、每层 Pre-LN（`EncoderLayer` / `DecoderLayer` 中先 `LayerNorm` 再子层）。
- **三类注意力位置**：编码器自注意力、解码器因果自注意力、解码器对 encoder memory 的交叉注意力；均由 `MultiHeadAttention` 实例承担（`small_try/model.py` `EncoderLayer.self_attn`、`DecoderLayer.self_attn`、`DecoderLayer.cross_attn`）。
- **注意力类型可配置**：`small_try/attention.py` 中 `build_core_attention(cfg, attn_layer)` 支持 `dot_product`、`additive` 及扩展类型（`bilinear`、`gated_dot_additive`、`local_window`、`global_local`、`sparsemax`、`entmax15`）。`base_1/attention.py` **仅**实现 `dot_product` 与 `additive`（`build_core_attention` 无 `attn_layer` 参数）。
- **训练脚本**（以 `small_try/train.py` 为代表）：AdamW + warmup（`get_warmup_lambda`）、Teacher forcing CE（`build_logits_shifted_loss`）、周期性在 `eval_split` 上做 greedy + `mt_eval.evaluate_generation_corpus`；训练结束再做一次 `force_heavy=True` 的生成评估并写入 `metrics.json`（`save_metrics_json`）。
- **评估**：`mt_eval.py` 提供 BLEU（SacreBLEU）、chrF/chrF++、`score_bertscore_f1`、`score_comet`；`compute_extra_metrics` 内对 **BERTScore/COMET** 有 **heavy 步频门控**（见 §5）。
- **数据**：平行句对 TSV + SentencePiece tokenizer JSON（`dataset.py` `load_tokenizers`）；划分可由 `scripts/create_splits.py` 生成 manifest（见 §3）。
- **消融/终评**：`scripts/ablation_lib.py` `evaluate_checkpoint_on_test` 在 **test** DataLoader 上跑评估并可写 `predictions.jsonl`。

未在源码层面验证的内容（故**不写入“已实现”**）：外部集群调度、W&B 云端可见性、任意具体 checkpoint 数值结论。

---

## 2. `dot_product` / `additive` 是否接入三类注意力

### `small_try` / `small_head` / `small_swap`（同源结构）

- `EncoderLayer.forward`：`self.self_attn(q,q,q, attn_mask=src_key_padding)`（`small_try/model.py`）。
- `DecoderLayer.forward`：自注意力 `self.self_attn(..., attn_mask=tgt_mask)`；交叉注意力 `self.cross_attn(q2, memory, memory, attn_mask=memory_key_padding)`（同上）。
- **同一套 `MultiHeadAttention`** 内部只构造 **一个** `build_core_attention(cfg, attn_layer)` 的结果（`small_try/attention.py` `MultiHeadAttention.__init__`），即 **encoder / decoder-self / cross 共用同一种 `attention_type` 对应的 core**（仅 `attn_layer` 字符串用于 `local_window`/`global_local` 的结构掩码）。

因此：**是**——当 `cfg.attention_type` 为 `dot_product` 或 `additive` 时，三种位置的注意力核心均为对应实现（`ScaledDotProductAttention` 或 `AdditiveAttention`，见 `build_core_attention` 分支 `411:444:small_try/attention.py`）。

### `base_1`

- `base_1/model.py` 中三类注意力同样经由 `MultiHeadAttention`，`build_core_attention(cfg)` 仅在 `dot_product`/`additive` 间切换（`base_1/attention.py` `69:75`）。
- **注意**：`base_1` 的 `MultiHeadAttention` **无** `attn_layer` 参数（与 `small_*` 分叉）。

---

## 3. 数据集与 train/val/test 划分是否“科学”

**实现了什么**

- `scripts/create_splits.py`：`dedupe_to_tempfile`（按 `sha256(src+"\n"+tgt)` 去重）→ `reservoir_sample_pairs_from_file` 抽样子集 → `split_train_val_test`（`seed+1337` 打乱后按比例切段）→ 写 `train.tsv`/`val.tsv`/`test.tsv` + `manifest.json`。
- 示例 manifest：`data/splits/en_fr_50k_seed42/manifest.json` 记录 `sha256`、样本量、比例、`creation_timestamp_utc` 等。

**局限性（审稿意见）**

- 划分是 **随机打乱 + 比例切分**，**无**显式领域/难度分层；若语料存在显著主题聚类，邻近句泄漏风险需自行评估。
- `TabParallelDataset.__getitem__` 对 **tgt** 截断为 `budget-2`（保留 BOS/EOS 位）、src 截断为 `max_seq_len`（`small_try/dataset.py` `71:76`），长短句分布被裁剪改变——需在论文中说明。
- **重现依赖**：默认路径指向 `data/splits/...`；若本地缺少文件则训练直接 `FileNotFoundError`（`dataset.py` `47:52`）。

---

## 4. `metrics.json` 里的 BLEU 等：validation 还是 test？

- `small_try/train.py`：`eval_ds = TabParallelDataset(cfg, ..., cfg.eval_split)`（约 `212:213`）；默认 `Config.eval_split == "val"`（`small_try/config.py` `74:75`）。
- 训练结束打印的「最终评估」仍使用该 `eval_loader`（`421:449:small_try/train.py`），写入 `final_bleu`、`final_val_loss`（`save_metrics_json`）。
- **结论**：默认 **`metrics.json` 中的 `final_bleu` / `final_extra_metrics` 对应的是 `eval_split`（默认 val），不是 test**。  
  若 CLI `--eval-split test`，才会在 test 上算这些“final”数字（见 `train_runtime.apply_shared_cli_to_config` 对 `eval_split` 的覆盖）。

**与之对照**：`scripts/ablation_lib.evaluate_checkpoint_on_test` **固定** `TabParallelDataset(..., "test")`（`91:91`），其产出写入合并 metrics 的 `final_eval_on_test_split`（`merge_ablation_metrics_json`）。

---

## 5. BLEU、chrF、COMET、BERTScore：哪些会出现在“最终结果”里？

依据 `mt_eval.compute_extra_metrics`（`343:417:mt_eval.py`）与训练脚本调用方式：

| 指标 | 何时计算 | 备注 |
|------|-----------|------|
| **BLEU** | 每次 `evaluate_generation_corpus` | `_bleu_score_and_signature`；核心分数不受 heavy 门控 |
| **chrF / chrF++** | `use_chrf` 且 hyps 非空 | **不受** `run_heavy` 限制（与 BERT/COMET 不同） |
| **BERTScore** | `use_bertscore`且 `run_heavy` 为真 | `run_heavy = force_heavy or heavy_every is None or (optimizer_step % heavy_every == 0)` |
| **COMET** | `use_comet`且 `run_heavy` 且 `len(srcs)==len(hyps)` | 同上 |

因此：

- **训练结束那一次**：`small_try/train.py` 传入 `force_heavy=True`（`447:447`），**若** `eval_use_bertscore`/`eval_use_comet` 为真且依赖可用，`metrics.json` 的 `extra_metrics` **可以**含 BERTScore/COMET。
- **训练中周期性验证**：`optimizer_step` 例如 120、240… 当 `120 % eval_heavy_metrics_every_optimizer_steps != 0`（默认 `eval_heavy_metrics_every_optimizer_steps=2000`，`small_try/config.py` `95`）时，`run_heavy` 为 **False** → **BERTScore/COMET 常为 `not_run`/`None`**，仅 BLEU/chrF 系列稳定出现。
- **`--eval-light`**：`small_try/train.py` 关闭 BERT/COMET（`164:166`）。

**predictions.jsonl**：`evaluate_generation_corpus` 在 `force_heavy=True` 时总会写出预测文件；若未传 `predictions_jsonl_path`，路径为 **`Path("predictions.jsonl")`（当前工作目录）**（`509:512:mt_eval.py`）。`ablation_lib` 显式传入 `run_dir / "predictions.jsonl"`（`133:133`）。

---

## 6. `small_head` 单头实验的解释风险

- **公平性**：单头时 `d_k = d_model // n_heads`（`small_try/attention.py` `467`）。若仅将 `n_heads` 改为 1 而保持 `d_model`，则 **每头维度变大**，与多头基线的 **per-head 宽度不同**，比较的是「不同分解方式的 MHA」，而非「仅 head 数变化」。
- **默认配置**：`small_head/config.py` 仍默认 `n_heads: int = 4`（`39:46`）；单头需用户在 CLI/配置中显式改为 `1`，否则并非单头实验。
- **注意力可视化**：W&B 热力图路径依赖 `decoder_cross`（`small_try/train.py` `363:376`）；单头时图为 1 列 head，解释「分工」受限——与多头对比时需声明。

---

## 7. <span style="color:red">**重点：`small_swap` 与 `small_try` 的 final BLEU「完全相同」问题**</span>

<span style="color:red">若你在报告中观察到 **`small_swap`（FR→EN）某次运行的 `final_bleu` 与 `small_try`（EN→FR）`dot_product` 的 `final_bleu` 数值完全相同**，这在科学上应视为 **高风险信号**，而非「两个方向等价」的证据。</span>

**原因要点（基于实现）**

- **任务定义不同**：`small_swap/dataset.py` 在 `swap_parallel_columns=True` 时交换列（`62:64`），且 **tokenizer 路径对调**（`small_swap/config.py` `36:37`）；参考译文的语言与字符串空间与 `small_try` 不同。
- **评估语言配置不同**：例如 `eval_bertscore_lang` 在 `small_try` 默认为 `"fr"`、`small_swap` 为 `"en"`（`small_try/config.py` `96`、`small_swap/config.py` `99`）。BLEU 虽不依赖该字段，但说明两套 pipeline **并非同一评估模板**。
- **更可能解释**：报表复制错误、混淆了 `metrics.json` 与消融 merged JSON、或两次运行实际上读了同一文件/同一 checkpoint；**数值 bitwise 相同**应优先排查 **人为与路径错误**，而非模型结论。

**建议（不留代码，仅审计动作）**：核对两次运行的 `metrics.json` / `resolved_config.json` 是否同源；确认 `eval_split`、划分路径、`wandb_run_name`；对 FR→EN 报告 **独立** 的 test 终评（`ablation_lib` 或 `--eval-split test`）。

---

## 8. 硬编码路径、可复现性、缺失资源

| 风险 | 证据 |
|------|------|
| **相对路径依赖仓库根** | `Config` 默认 `data/splits/...`、`data/tokenizer_*.json`；`train_runtime.materialize_path_fields` 将字段解析为绝对路径（`train_runtime.py` `38:47`）。启动目录改变且未走 `materialize` 时仍可能出错。 |
| **环境变量覆盖根目录** | `infer_repo_root` 可用 `REPO_ROOT`/`AUTODL_REPO_ROOT`（`train_runtime.py` `15:18`）。 |
| **缺失数据/划分** | `TabParallelDataset` 文件不存在即抛错（`dataset.py`）。`data/EN-FR.txt` 在 `.gitignore` 中，需要本地生成或自备才能跑 `create_splits`。 |
| **Checkpoint** | `.gitignore` 忽略 `**/*.pt`；仓库克隆后 **不含** 训练权重。 |
| **`predictions.jsonl` 落地位置** | 未传路径时写到 **进程 cwd** 的 `predictions.jsonl`（见 §5），易导致「找不到预测文件」或混入他人运行产物。 |
| **`conclusion/` 目录** | 当前快照中 **不存在**该目录；`base_1/model.py` 文档仍引用 `conclusion/base/transformer architecture.svg`（`base_1/model.py` 文件头注释），与仓库现状 **不一致**，属文档漂移。 |
| **`base_1` 与 `small_*` 分叉** | `base_1` 仍为旧版 `MultiHeadAttention(cfg)` / 二元 `attention_type`，与 `small_try` 扩展注意力 **不是**同一代码路径。 |

---

## 9. 优先级建议（P0 / P1 / P2）

### P0（影响结论可信度）

1. **统一报告口径**：明确论文表格中的 BLEU/chrF/COMET 来自 **`metrics.json`（默认 val）** 还是 **`ablation_lib` / `--eval-split test`**；禁止混用而不标注。
2. **审慎对待 cross-direction 数值**：对 §7 中红色标注的「完全相同」进行溯源核查。
3. **`small_head` 单头 vs 多头**：若主张「仅 head 数变化」，需固定可比维度（例如匹配参数量或固定 `d_k`），并在文中写明当前实现实际改变的是 `d_k`（见 §6）。

### P1（可复现性与工程卫生）

1. **预测文件路径**：训练结束若写 `predictions.jsonl`，应固定到 **run 目录**（当前依赖 cwd 的行为见 `mt_eval.py`）。
2. **Heavy 指标间隔**：默认 `eval_heavy_metrics_every_optimizer_steps=2000` 与 `val_every=120` 组合，导致训练中大部分时间 **无** BERT/COMET；若期望监控曲线，应调整步频或在文档中说明。
3. **同步文档**：移除或更新 `base_1/model.py` 对已删除 `conclusion/` 路径的引用。

### P2（增强与扩展）

1. **划分策略**：考虑按长度桶或哈希文档 ID 的分层划分，降低朴素随机带来的泄漏争议。
2. **脚本清单**：`scripts/` 已含 `analyze_predictions.py`、`analyze_attention.py`、`analyze_ablation.py` 等；建议在论文方法附录对照 **命令行与输入输出**，避免「口头流程」与仓库不一致。
3. **`base_1` 维护策略**：若不再用于论文主实验，标注 deprecated，避免审稿人比对错误目录。

---

**文档生成说明**：审计基于读取 `small_try/model.py`、`small_try/attention.py`、`small_try/train.py`、`small_try/dataset.py`、`small_try/config.py`、`small_swap/dataset.py`、`small_swap/config.py`、`base_1/model.py`、`base_1/attention.py`、`mt_eval.py`、`train_runtime.py`、`scripts/create_splits.py`、`scripts/ablation_lib.py`、`data/splits/en_fr_50k_seed42/manifest.json`。未执行训练或下载权重。
