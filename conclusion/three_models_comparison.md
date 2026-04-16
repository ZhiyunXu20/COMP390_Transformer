# 三套代码对比：base_old（归档）、base_1、small_try

> **Small 系列完整说明（思路、流水线、W&B、指标归档）见 [small_project_report.md](./small_project_report.md)。**

说明：`base_old`（原 `Base`）为早期四文件快照，已归档、不再维护；`base_1` 由 `NEW_TRY` 重命名而来，承载完整 Seq2Seq Transformer 实现；`small_try` 为快速对比实验（小模型 + 子集 + 短步数）。**`small_head` / `small_swap`** 与 `small_try` 同属「small 注意力实验」，细节见上述报告。

---

## 1. 架构与实现要点

| 项目 | base_old（归档） | base_1 | small_try |
|------|------|----------------------|----------|
| 总体结构 | **非完整 Transformer**：`Transformer.forward` 仅 `src_embed`/`tgt_embed` → `out_proj`，**无 Encoder/Decoder 栈、无位置编码、无注意力计算图**；`MultiHeadAttention` 已实现但未接入主前向 | **标准 Encoder–Decoder**：`Seq2SeqTransformer`，`n_layers` 层编码器（自注意力+FFN）与解码器（掩码自注意力+交叉注意力+FFN），双端位置编码，`lm_head` 输出词表 logits | 与 base_1 **同构**，仅 `config` 中 `d_model/n_layers/n_heads/d_ff` 更小 |
| 注意力 | 类中有 `ScaledDotProductAttention`，主模型未使用 | 点积或 **加性注意力**（`attention_type`），与多头拼接一致 | 同 base_1 |
| 残差与归一化 | 未实现层堆叠 | **Pre-LN**（先 `LayerNorm` 再子层再残差），与论文原图「Add & Norm」**画法顺序**可能不同，但模块集合与数据流与经典 Transformer 一致 | 同 base_1 |
| 位置编码 | 无 | 正弦 `PositionalEncoding`（`attention.py`） | 同 base_1 |
| 与 `conclusion/base/transformer architecture.svg` 的关系 | **不符合**：该图为标准 Enc–Dec 堆叠示意图；base_old **缺少图中主体数据路径** | **语义符合**：嵌入、编码器栈、解码器栈（含 Masked 与 Encoder–Decoder 子层）、输出线性层均具备；仅 **Pre-LN vs 图中常见 Post-LN** 为实现细节差异（已在 `base_1/model.py` 文档说明） | 同 base_1（结构一致，规模缩小） |

---

## 2. 超参数与训练设置（摘自各 `config`）

| 超参数 | base_old (`TransformerConfig`) | base_1 (`Config`) | small_try (`Config`) |
|--------|---------------------------|---------------------|---------------------|
| `d_model` | 512 | 512 | 256 |
| `n_layers` | 6 | 6 | 4 |
| `n_heads` | 8 | 8 | 4 |
| `d_ff` | 2048 | 2048 | 1024 |
| `dropout` | 0.1 | 0.1 | 0.1 |
| `max_seq_len` | 128 | 128 | 96 |
| `batch_size` | 128 | 96 | 192 |
| `grad_accum_steps` | 2 | 2 | 1 |
| 等效 batch（约） | 256 | 192 | 192 |
| `learning_rate` | 5e-4 | 5e-4 | 3e-4 |
| `warmup_steps` | 4000 | 4000 | 200 |
| `label_smoothing` | 0.1 | 0.1 | 0.1 |
| `weight_decay` | 0.01 | 0.01 | 0.01 |
| `max_steps` | 200000 | 200000 | 3000 |
| `epochs` | 20 | 20 | 100（常与 `max_steps` 联合，先到先停） |
| 验证 | 无独立验证集逻辑 | `val_ratio=0.005`，`val_every=2000`（optimizer 步），BLEU 子样本 | `val_ratio=0.05`，`val_every=120`，BLEU 子样本 |
| 注意力变体 | 配置项未接入主模型 | `dot_product` / `additive` | 同 base_1 |
| 输出目录 / W&B | `train.py` 写死 project `en-fr-transformer` | `output_dir=/root/autodl-tmp/base_1/runs`，`project_name=en-fr-attention` | `small_try/runs`；小型实验统一 **`project_name=attention-small`**（`wandb_group` 区分子实验，见 [small_project_report.md](./small_project_report.md)） |

---

## 3. 训练数据规模

| 项目 | 数据路径 / 构造 | 规模（量级） |
|------|-----------------|--------------|
| base_old | `data/EN-FR.txt`（tab 分隔），`TranslationDataset` 全量读入 | 当前语料约 **372.9 万** 行句对（与磁盘 `wc -l` 一致）；无 train/val 划分 |
| base_1 | 同上路径；`TabParallelDataset` 按行号周期划分 train/val（`val_ratio=0.005`） | 总句对同上；约 **0.5%** 验证、**99.5%** 训练（周期 `period≈200`） |
| small_try | `small_try/data/corpus_50k.tsv`（由 `create_subset.py` 生成 5 万句） | **5 万** 行；配置下约 **4.75 万 train / 0.25 万 val**（`val_ratio=0.05`） |

---

## 4. base_old 的典型问题（简表）

| 类别 | 说明 |
|------|------|
| **模型未完成** | Encoder/Decoder 未接线，训练未执行真实自/交叉注意力。 |
| **注意力可视化** | `output_attentions=True` 时曾为均匀假注意力。 |
| **显存** | 大 batch + 长序列 + 全量语料易 OOM。 |

更细的工程项已在 `base_1` / small 侧处理（如 sacrebleu API、W&B 生命周期、tokenizer 与 `tokens.py` 对齐等）。

---

*说明：`base_1` 目录由 `NEW_TRY` 重命名；`base_1/config.output_dir` 为 `/root/autodl-tmp/base_1/runs`。*
