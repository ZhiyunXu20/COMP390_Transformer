# base_1：完整 Encoder–Decoder Transformer（受限注意力）

本目录是一套**自包含**的训练栈（`model.py`、`attention.py`、`train.py`、`dataset.py` 等），与论文主实验常用的 **`small_try` / `small_head` / `small_swap` 不是同一代码路径**，也不要求与之逐行对齐。

## 注意力支持范围

- **`Config.attention_type` 仅支持 `dot_product` 与 `additive`**（见 `attention.py` 中的 `build_core_attention`；无按层注入的 `attn_layer` 参数）。
- **不包含** `small_try` 中的扩展类型（例如 bilinear、gated_dot_additive、local_window、global_local、sparsemax、entmax15 等）。

因此：**不能**把 `base_1` 理解为「`small_try` 里全部注意力变体都已迁移到这套大模型 Encoder–Decoder 实现」。若叙事或实验涉及那些变体，请以对应的 **`small_*` 包**为准；本目录仅保留经典两种打分形式作为对照或独立实验。

## 结构与文档

模型注释中的「图中」指 Vaswani et al. 论文中的标准 Transformer 示意图约定；**不依赖**仓库内任何单独的示意图资源目录。
