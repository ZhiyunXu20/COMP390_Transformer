# Thesis result boundaries（论文可用表述边界）

本文档约束：**论文正文与图表能在何种证据下声称什么**，并与仓库内脚本、指标语义对齐。详细工程审计见 **`docs/RESEARCH_AUDIT.md`**；注意力变体代码与稠密/稀疏界限见 **`docs/ATTENTION_VARIANTS_STATUS.md`**；探索变体元数据见 **`experiments/variant_registry.py`**。

---

## 1. Completed core experiments（已完成的核心实验）

以下为仓库默认流水线与已提交的 baseline run 所**直接支撑**的结论类型（须在文中写明任务、划分、`evaluate_test` vs 训练日志等指标来源）。

### dot_product vs additive

- **任务**：以 `small_try` 为代表的小规模 EN→FR Transformer；同一划分与词表前提下对比 **`attention_type=dot_product`** 与 **`additive`**。
- **证据载体**：例如 `runs/fast_dot`、`runs/fast_add` 的配置与评估产物；**held-out test** 应以仓库根目录 **`evaluate_test.py`** 写出的 **`runs/<run>/test_eval/metrics_test.json`** 为准（见 §3 禁止项）。

### Single-head vs multi-head

- **任务**：在可比栈（如 `small_head` 与 `small_try`）下对比 **`n_heads=1`** 与 **`n_heads>1`**（默认 baseline 多为 4 头）。
- **叙事限制**：单头时 **`d_k = d_model // n_heads`** 变大，与多头基线的 **per-head 维度不同**；比较的是「不同 head 分解」而非「仅 head 数目」的孤立效应（见 **`docs/RESEARCH_AUDIT.md`** §6）。
- **证据示例**：`runs/head_1h_dot`（单头）与 `runs/fast_dot`（多头）等需在文中注明上述公平性 caveat。

### FR→EN swap（`small_swap`）

- **任务**：法→英方向的平行语料与 **`small_swap`** pipeline（列交换、tokenizer 路径与 `small_try` **不对齐**）。
- **必须写的 caveat**：`small_swap` 与 `small_try` **并非同一评估模板**（如 BERTScore 语言字段等）；若观察到与 EN→FR run **数值 bitwise 相同**的 BLEU，应视为 **高风险信号**，优先排查报表/路径错误，而非跨方向等价结论（见 **`docs/RESEARCH_AUDIT.md`** §7）。
- **证据示例**：`runs/swap_fr_dot`；正式叙事中应明确「第二方向」与主方向的 **配置差异**，而非暗示两套 pipeline 完全镜像。

---

## 2. Implemented but not yet experimentally validated variants（已实现、尚未完成系统性实验验证）

以下 **`attention_type`** 在 **`small_try` / `small_head` / `small_swap`** 的 `attention.py` 中**已实现**，并可由 **`experiments/run_untrained_attention_variants.py`** 调度到 **`runs/var_*`**；**默认 baseline 提交快照中不要求**已具备与 dot/add 同等强度的多 seed、完整 test 终评汇总。

| Variant | 备注 |
|---------|------|
| `bilinear` | 探索性打分形式 |
| `gated_dot_additive` | 探索性门控混合打分 |
| `sparsemax` | 归一化改为 sparsemax（稠密分数矩阵上） |
| `entmax15` | 归一化为 entmax α=1.5（稠密分数矩阵上） |
| `local_window` | **稠密**结构掩码表达局部先验，非稀疏核 |
| `global_local` | **稠密**结构掩码表达全局锚点 + 局部带，非 ETC 类实现 |

汇总 **仅读 test_eval** 的表格可由 **`scripts/summarize_attention_variants.py`** 生成（`results/attention_variants_test_summary.*`）。

---

## 3. Claims that must NOT be made（禁止作出的声称）

- **不要**将 **`local_window` / `global_local`** 表述为 **Longformer、BigBird、ETC** 等意义上的 **真正稀疏 / 次线性复杂度注意力**；实现为 **完整 L×L 打分矩阵 + 掩码 + softmax（或替代归一化）**。
- **不要**声称实现了 **hard attention**（不可微、离散选址一类机制）；仓库内为连续权重注意力及其变体。
- **不要**声称实现了 **CV 意义上的 spatial / channel attention**；对象为序列 token 注意力。
- **不要**将 **`runs/<run>/metrics.json`** 中的 **`final_bleu`** 等直接当作 **test 集最终结果** 报道；默认 **`eval_split=val`** 时其为 **验证集**语义（见 **`docs/RESEARCH_AUDIT.md`** §4）。Test 报告须基于 **`evaluate_test.py`** → **`test_eval/metrics_test.json`** 或等价独立 test 流水线。
- **不要**引用历史口径下的 **BLEU 66/76** 一类极高数值 **除非**：明确绑定 **旧协议 / 旧语料 / 旧预处理**，并与当前 SacreBLEU + 本仓库划分 **分段呈现**，避免读者误认为与本文同一实验设定可比。

---

## Recommended thesis wording（推荐论文措辞）

This project implements and evaluates core attention variants under a controlled small-scale Transformer setting, and provides additional implemented exploratory variants for future systematic evaluation.
