# small_try 模型架构摘要

> 与 `small_head` / `small_swap` 并列的 **small 项目总览**（流水线、W&B、指标归档）见 [small_project_report.md](./small_project_report.md)。

本文档以 **`/root/autodl-tmp/small_try`** 为基准，概括其 Encoder–Decoder Transformer、可切换注意力、训练与评测方式，便于与 `conclusion/base/transformer architecture.svg` 对照阅读。

---

## 1. 总体结构（与经典图示的对应）

| 图示（SVG）中的块 | small_try 中的实现 |
|-------------------|-------------------|
| 输入嵌入 + 位置编码 | `src_embed` / `tgt_embed` + `PositionalEncoding`（`attention.py`） |
| 编码器 ×N | `EncoderLayer` × `n_layers`：多头自注意力 + 前馈（`model.py`） |
| 解码器 ×N | `DecoderLayer`：掩码自注意力 + **交叉注意力** + 前馈 |
| 输出层 | `lm_head`：`d_model → tgt_vocab_size` |
| Encoder–Decoder 连线 | 解码器中的 **cross-attention**（`decoder_cross` 权重即可视化对象） |

残差结构为 **Pre-LN**（先 `LayerNorm` 再子层），与部分教材插图的 Post-LN 画法不同，但模块组成与数据流一致。

---

## 2. 两种注意力是否「可比」（架构判断）

**结论：在控制变量的意义上，二者具有可比性**——除「注意力打分方式」外，**多头拆分维度、`W_q/W_k/W_v/W_o`、层数、FFN、位置编码、损失与数据管线均相同**。

| 项目 | `dot_product` | `additive` |
|------|----------------|------------|
| 接口 | 均为 `(Q,K,V)` → `(output, attn)`，形状 `(B,H,Lq,Lk)` 上 softmax | 同左 |
| 打分 | 缩放点积 \(QK^\top/\sqrt{d_k}\) | Bahdanau 风格：\(w^\top \tanh(W_q q + W_k k)\)，隐维 `additive_d_hidden` |
| 接入方式 | `MultiHeadAttention` 共用同一套线性投影；仅 **`build_core_attention(cfg)`** 切换核心模块（`attention.py`） |
| 作用范围 | **三种注意力**（编码器自注意力、解码器自注意力、编码器–解码器交叉注意力）**全部**使用同一类打分 |

**需注意的差异（不影响「是否可比」，但影响解释）：**

1. **参数量与容量**：加性分支额外引入 `W_q/W_k/v`（在 **per-head `d_k`** 上），总参数与优化动态与点积版不完全相同。  
2. **计算与显存**：加性在 \(L_q\times L_k\) 上显式打分，长序列时更慢、更吃显存。  
3. **公平对比建议**：固定 `seed`、`max_steps`、数据与超参；若报告「谁更好」，宜说明二者在上述意义上是 **打分机制消融**，而非仅替换 cross-attn 一层。

---

## 3. 默认超参数（`small_try/config.py`）

| 项 | 取值 |
|----|------|
| `d_model` / `n_layers` / `n_heads` / `d_ff` | 256 / 4 / 4 / 1024 |
| `max_seq_len` | 96 |
| `dropout` | 0.1 |
| `additive_d_hidden` | 256 |
| `batch_size` × `grad_accum_steps` | 192 × 1 |
| `max_steps` | 3000（常与 `epochs` 联用，先到先停） |
| `learning_rate` / `warmup_steps` | 3e-4 / 200 |
| `label_smoothing` | 0.1 |
| 数据 | 子语料约 5 万句（`corpus_50k.tsv`），`val_ratio=0.05` |
| 验证 | `val_every`（optimizer 步）、`bleu_sample_size` 条贪心 BLEU |
| W&B | `project_name=attention-small`，`wandb_group=try-en-fr` |

---

## 4. 训练目标与日志

- **训练损失**：对 **teacher forcing** 下「预测下一目标 token」的 **CrossEntropy（ignore PAD + label smoothing）**，见 `build_logits_shifted_loss`。  
- **验证**：`val_loss`（token 平均 CE）+ **BLEU / chrF / BERTScore / COMET**（见仓库根 `mt_eval.py`）。  
- **W&B**：`train_loss`、`lr`、`val_loss`、`bleu`；可选 **解码器最后一层 cross-attention** 热力图（`attention/cross_L*`）。

---

## 5. 流水线（可选）

- 单包：`run_fast_full.sh` → 子语料 → `dot_product` / `additive` → `compare_runs.py` → `report.txt`、`results/bundle_metrics.json`、`manifest.json`。
- 三包串联：仓库根 `run_all_small_training.sh`（再可用 `launch_small_training_nohup.sh` 后台跑）。

---

## 6. 与 base_1 的关系

small_try **复用同一套 `Seq2SeqTransformer` + 注意力切换思路**，通过 **更小模型、子集、短步数** 做快速对比；严肃长训与全量数据见 `base_1`。

---

*文件用途：结题/报告中的「模型与实验设置」短摘要；细节以仓库内 `small_try/config.py`、`model.py`、`train.py` 为准。*
