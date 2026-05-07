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
- **参数量公平性（已对齐）**：`runs/head_1h_dot` 使用单头、**`d_k = d_model`**（256），总参数量与多头 dot baseline **一致**（30,442,800；见 **`results/variant_fairness_audit.md`**）。叙事上仍是比较 **不同的 multi-head 分解**（1 个宽头 vs 4 个窄头），而非仅孤立改变 head 数目。
- **证据示例**：`runs/head_1h_dot`（单头）与 `runs/fast_dot`（多头）需在文中注明 head 数与上述解释性边界。

### FR→EN swap（`small_swap`）

- **任务**：法→英方向的平行语料与 **`small_swap`** pipeline（列交换、tokenizer 路径与 `small_try` **不对齐**）。
- **必须写的 caveat**：`small_swap` 与 `small_try` **并非同一评估模板**（如 BERTScore 语言字段等）；若观察到与 EN→FR run **数值 bitwise 相同**的 BLEU，应视为 **高风险信号**，优先排查报表/路径错误，而非跨方向等价结论（见 **`docs/RESEARCH_AUDIT.md`** §7）。
- **证据示例**：`runs/swap_fr_dot`；正式叙事中应明确「第二方向」与主方向的 **配置差异**，而非暗示两套 pipeline 完全镜像。

---

## Data split overlap (source/target level)

仓库以**逐物理行制表符分隔**（raw line-delimited TSV）的方式存储与加载 split：`small_try/dataset.py`、`small_head/dataset.py`、`small_swap/dataset.py` 均逐行 `line.rstrip("\n").split("\t")` 读取，`scripts/make_splits.py` 也按逐行 TSV 写入。注意：默认 quote-aware `csv.reader` 不适用于本仓库 split，因为 DCEP 自然语言文本中含未按 CSV 规范转义的双引号（如 `"Euromat"`），会错误地跨物理行合并记录。审计应使用 `line.rstrip("\n").split("\t")` 或 `csv.reader(delimiter="\t", quoting=csv.QUOTE_NONE)`。

按真实 loader 口径，split 行数为：**train = 45000, val = 2500, test = 2500**。无空 src/tgt 行；无 src==tgt 行。

Pair-level (source, target) overlap matrix:

| overlap type | train-val | train-test | val-test |
|---|---:|---:|---:|
| exact (src, tgt) pair | 0 | 0 | 0 |
| source string | 10 | 8 | 1 |
| target string | 8 | 9 | 1 |

Alignment noise (1-to-many):

- source strings with >1 distinct target: **115**
- target strings with >1 distinct source: **107**

Implication: 8 test source strings (~0.32% of test) appear in train. No exact pair leak across splits, but DCEP's 1-to-many alignment noise (115 source strings with multiple targets, 107 target strings with multiple sources) means a small fraction of test source strings share a string with train. This may slightly inflate held-out test BLEU.

**Forbidden claim**: "Held-out test BLEU is unaffected by training data" or "Splits are perfectly disjoint at the string level."

**Required Threats to Validity wording (verbatim recommended)**: "Although pair-level deduplication ensures no exact (source, target) pair appears in both train and test, 8 source strings (~0.32% of test) appear in both splits, and the corpus contains 115 source-strings with multiple distinct targets and 107 target-strings with multiple distinct sources, reflecting alignment noise inherent to DCEP. We did not filter cross-split source/target overlaps, and held-out test BLEU may be slightly inflated."

---

## 2. Exploratory single-seed results（探索性单-seed 结果）

以下 **`attention_type`** 在 **`small_try`** 等路径中**已实现**，并以 **`runs/var_*`** 等形式完成了训练与 held-out test 终评。**主汇总表**为 **`results/attention_variants_test_summary.md`**（A18：在存在 **`results/ablation_per_seed.csv`** 时，`scripts/summarize_attention_variants.py --multiseed-csv` 将 **`fast_dot` / `fast_add` / `fast_add_lr3e3`（A19：additive 使用调优后的 `lr=3e-3`，dot 保持其 baseline `lr=3e-4`）/ A14 多 seed 变体** 的 test 指标合并为 **mean ± std (n=3)**，并给出 **Welch *p* vs `fast_dot` (BLEU)**；**`sparsemax` / `local_window` / `global_local`** 等仍以 **单 seed** 呈现于该表）。完整 Welch 叙事（Δ、95% CI、*t*、df，逐指标）见 **`results/variant_multiseed_summary.md`**；**A19** 与 **`fast_dot`** 的专门对照另见 **`results/cross_seed_significance_lr3e3.{md,json}`**。

| Variant | 备注 |
|---------|------|
| `bilinear` | 探索性打分形式；`var_bilinear` |
| `gated_dot_additive` | 探索性门控混合打分；`var_gated_dot_additive` |
| `sparsemax` | 归一化改为 sparsemax（稠密分数矩阵上）；`var_sparsemax` |
| `entmax15` | 归一化为 entmax α=1.5（稠密分数矩阵上）；`var_entmax15` |
| `local_window` | **稠密**结构掩码；**修复前** archive 中 **`var_local_window`** 及 A15 探针（fp32 / 降 lr / 加宽窗口）训练数值失败（NaN），见 **`results/local_window_stability_report.md`**。**修复后（A23）** 单 seed 重训 **`var_local_window_a23_retrain`**：held-out **test BLEU≈16.82**（`test_eval/metrics_test.json`）；勿再将机制叙述为「必然不可训练」而忽略补丁与重训结果 |
| `global_local` | **稠密**结构掩码；`var_global_local` |

Exploratory variants differ in statistical strength: **`bilinear`**, **`gated_dot_additive`**, and **`entmax15`** have **3-seed** held-out test rows in **`results/attention_variants_test_summary.md`** (Welch *p* vs **`fast_dot`** on BLEU in that table). **`results/variant_multiseed_summary.md`** 保留 **逐指标** Welch 全文（含 CI / *t* / df）。The dominant **~7.6 BLEU** dot-vs-additive gap remains **3-seed / Welch**-supported (**`results/cross_seed_significance.*`** 与统一汇总表一致)；other **`var_*`** rows are still mostly **single-seed** unless the table marks **n=3**.

汇总 **仅读 `test_eval`** 的表格由 **`scripts/summarize_attention_variants.py`** 生成；默认传入 **`--multiseed-csv results/ablation_per_seed.csv`** 以合并 **A14 多 seed** 与 **fast_dot / fast_add** 三 seed（**`results/attention_variants_test_summary.*`**）。**`results/variant_multiseed_summary.md`** 为 Welch **详细**附录，**非**日常交叉引用的唯一入口。

---

## 3. Claims that must NOT be made（禁止作出的声称）

- **不要**将 **`local_window` / `global_local`** 表述为 **Longformer、BigBird、ETC** 等意义上的 **真正稀疏 / 次线性复杂度注意力**；实现为 **完整 L×L 打分矩阵 + 掩码 + softmax（或替代归一化）**。
- **不要**声称实现了 **hard attention**（不可微、离散选址一类机制）；仓库内为连续权重注意力及其变体。
- **不要**声称实现了 **CV 意义上的 spatial / channel attention**；对象为序列 token 注意力。
- **不要**将 **`runs/<run>/metrics.json`** 中的 **`final_bleu`** 等直接当作 **test 集最终结果** 报道；默认 **`eval_split=val`** 时其为 **验证集**语义（见 **`docs/RESEARCH_AUDIT.md`** §4）。Test 报告须基于 **`evaluate_test.py`** → **`test_eval/metrics_test.json`** 或等价独立 test 流水线。
- **不要**引用历史口径下的 **BLEU 66/76** 一类极高数值 **除非**：明确绑定 **旧协议 / 旧语料 / 旧预处理**，并与当前 SacreBLEU + 本仓库划分 **分段呈现**，避免读者误认为与本文同一实验设定可比。
- **不要**在 **仅单 seed、且 |ΔBLEU| < 1.0** 的探索性对比中声称一方「显著更好」，**除非**该对比在 **`results/attention_variants_test_summary.md`** 中已标明 **n=3** 且 **Welch *p* vs `fast_dot` (BLEU)** 支持该结论，或你在 **`results/variant_multiseed_summary.md`** 中核对同一变体的完整 Welch 报告；**`sparsemax` / `global_local`** 以及修复前的 **`var_local_window`** 失败行仍以 **单 seed（或失败）** 呈现，不得写成已确立排序。（**A23** 后 **`var_local_window_a23_retrain`** 有可用单 seed test 行，但仍不提供 n=3/Welch。）
- **不要**将 **additive** 表述为 **「本质上劣于 dot-product」**：**在共享 lr=3e-4 下 additive 显著低于 dot-product，但在 tuned 高学习率（lr=3e-3）下 additive 显著高于 dot-product baseline 5.76 BLEU（A19）**。可辩护的结论是 **learning-rate / 优化敏感性**，**不是**机制层面的普适排名。
- Do not describe additive attention as intrinsically worse than dot-product attention. Under shared **lr=3e-4**, additive underperforms dot-product; under the **tuned higher lr=3e-3** protocol (A19), additive significantly outperforms the dot-product baseline by **5.76 BLEU**. The defensible conclusion is **learning-rate / optimization sensitivity**, not a **universal mechanism ranking**.
- **不要**把 **仅 cross-seed Welch** 或 **仅句子级 bootstrap** 之一说成已穷尽所有不确定性；**dot vs additive** 主结论应 **同时引用** `results/cross_seed_significance.*`（训练 seed 间）与 `scripts/significance_test.py` / `results/significance_fast_dot_vs_fast_add.md`（同一 test 句子上重采样）。
- **不要**将 **dot vs additive** 的差距说成 **「纯机制 / purely mechanism-driven」** 而不同时说明 **A19** 与历史 LR 设定：**共享 `lr=3e-4` 时** dot 优于 additive（均值差约 **+7.60 BLEU**，Welch 显著）；**在相同 `max_steps=3000` 下对 additive 使用较高调参学习率 `lr=3e-3`、dot 仍保持基线 `3e-4` 时**，held-out test 上 **fast_dot − additive** 均值约为 **−5.76 BLEU**（additive 更高；Welch **t≈−14.91**，双尾 **p≈1.6×10⁻⁴**，见 **`results/cross_seed_significance_lr3e3.json`**）。这表明 **学习率/优化敏感性可充分解释并反转**在 `3e-4` 对比下 **原本观察到的**排序；**不**等于证明两机制内在等价。综述见 **`results/additive_matched_lr3e3_summary.md`**。补充语境仍可引用单 seed、较短预算的 **`results/additive_lr_sweep.*`**。

### Forbidden claims

- Do not claim **"every column in the main summary table is mean ± std for all multi-seed rows"** without naming exceptions from older docs. **Aggregate rows report mean ± std** for **BLEU, chrF, chrF++, COMET, BERTScore, train_time_seconds, peak_gpu_memory_mib** (n=3); **single-seed** rows report **n=1** values **without** a standard deviation on those columns.
- Do not claim **`local_window`** failed **only** because of bf16 autocast. **A15** (`results/local_window_stability_report.md`) shows **`runs/var_local_window_fp32`** with **`--no-bf16-autocast`** still diverges (NaN / BLEU=0), alongside bf16 runs at **lower lr** and **`local_window_size=16`** (1500-step probes). **Pre-fix** runs **`var_local_window`**, **`var_local_window_fp32`**, **`var_local_window_lr1e4`**, **`var_local_window_window16`** all NaN under the then-current implementation. **Post-fix (A23):** **`runs/var_local_window_a23_retrain`** with the `safe_masked_softmax` + encoder/decoder padding-row mask patch reaches **held-out test BLEU≈16.82** ( **`test_eval/metrics_test.json`**; single seed, `max_steps=3000`). A **supported** claim is that the **dense local-window mechanism is trainable under our setup** once the all-masked-row NaN is patched—not that fp32 alone fixes pre-patch code. Do not claim the **idea** of local windows is globally unviable—only that **this repository’s pre-A23 implementation + protocol** failed under the A15 grid. Do not claim true Longformer / BigBird / ETC sparse-kernel implementation: `local_window` and `global_local` remain dense L×L attention with structural masks on logits.
- Do not treat **`bilinear` vs `gated_dot_additive` vs `fast_dot`** as statistically ordered on **< ~1 BLEU** gaps **unless** you cite **`results/attention_variants_test_summary.md`** (n=3 means ± std and **Welch *p* vs fast_dot (BLEU)**) and, for exact CI/t/df, **`results/variant_multiseed_summary.md`**. For **`sparsemax`** and **`global_local`**, and for **pre-fix `local_window`** rows, keep the original “no ranking” caveat; **`var_local_window_a23_retrain`** is **single-seed descriptive** only.
- Do not directly compare swap_fr_dot BLEU with EN→FR runs as if they were the same task.
- Do not present the main run's single-seed BLEU 16.07 as the headline number; use 15.80 ± 0.52 (n=3) instead.
- Do not describe the dot–additive gap as **purely mechanism-driven** without citing **A19** and the **`lr=3e-4`** evidence: under **shared `lr=3e-4`**, dot leads additive by **≈ +7.60 BLEU** (Welch); under **A19** (**same `max_steps=3000`**, **additive at tuned `lr=3e-3`**, **dot baseline stays `lr=3e-4`**), **fast_dot − additive** is **≈ −5.76 BLEU** (additive higher), **Welch t ≈ −14.9**, two-sided **p ≈ 1.6×10⁻⁴** (**`results/cross_seed_significance_lr3e3.json`**). **A higher learning rate for additive is sufficient to reverse the ranking** seen at `3e-4`; that shows strong **optimization / LR confounding**, not **intrinsic mechanism equivalence**. See **`results/additive_matched_lr3e3_summary.md`**. The older single-seed **additive LR sweep** (**`results/additive_lr_sweep.md`**, **`results/additive_lr_sweep.csv`**, shorter budget) remains **context only**, not a substitute for this n=3 A19 run.

---

## Statistical claims supported by current evidence

统计结论以 **`results/cross_seed_significance.md`**（及同目录 JSON）为权威表述；以下为该文件的摘要，便于论文交叉引用。

本仓库对 **dot_product vs additive**（`fast_dot` vs `fast_add`）的统计支撑是 **双轨** 的，二者 **互补、不可替代**：

1. **Cross-seed（训练 seed 变异）**：对 **`results/ablation_per_seed.csv`** 中 **n=3 vs n=3** 的 per-seed **held-out test** 指标做 **Welch t 检验**（脚本 **`scripts/cross_seed_significance.py`**，输出 **`results/cross_seed_significance.{md,json}`**）。当前快照量级示意为：
   - **BLEU**：Δ ≈ **7.60**，Welch **t ≈ 15.1**，**df ≈ 3.7**，双尾 **p ≪ 0.05**（详见 JSON，统计后端可能为 SciPy 或 mpmath）。
   - **chrF++**：Δ ≈ **10.35**，**t ≈ 12.6**，**df ≈ 2.8**。
   - **COMET**：Δ ≈ **0.055**，**t ≈ 11.7**，**df ≈ 2.7**。

2. **Sentence-level paired bootstrap（单模型 test 抽样变异）**：对 **主 run**（如 `runs/fast_dot` vs `runs/fast_add`）的 **`test_eval/predictions.jsonl`** 做 **配对重采样**（**`scripts/significance_test.py`**，报告示例 **`results/significance_fast_dot_vs_fast_add.md`**）。当前快照：**ΔBLEU ≈ +7.90**，bootstrap **95% CI ≈ [7.50, 8.33]**（与脚本输出一致为准）。

**单 seed 的 `var_*` 探索性变体之间的优劣**：对 **在 `attention_variants_test_summary` 中仍标注单 seed** 的变体，仅作 **描述性** 呈现。**`bilinear` / `gated_dot_additive` / `entmax15`** 在该表中为 **n=3**， headline *p* 见 **Welch** 列；更长的 **Welch** 段落见 **`results/variant_multiseed_summary.md`**。不得与 dot/add 主结论的 **双轨** 统计混为一谈。

---

## External reference points (context only)

These external reference points are cited for **SCALE CONTEXT only**, not for direct ranking against our runs.

- **Vaswani et al. 2017** (*Attention is All You Need*), Table 2: WMT'14 EN–FR **base** model = **38.1 BLEU**; **big** model = **41.8 BLEU**. Trained on ~4.5M sentence pairs, beam search width = 4, length penalty 0.6.
- **Typical fairseq IWSLT'17 EN–FR tutorial** result: **30+ BLEU** on ~225k pairs with beam search.
- **Our setup**: 50k subset (~1.1% of WMT'14 scale), 3000 training steps, **greedy** decoding, `max_seq_len` = 96. Scores are **expected to be significantly lower** than the references above.

**Therefore:** our **15.80 ± 0.52 BLEU** on `fast_dot` (n=3, held-out test) is **consistent with the expected operational range** for a small-data + greedy setup—**not** a sign of a broken model. **Direct numerical comparison with Vaswani et al. 2017 is INVALID** due to different data scale, decoding, and training budget.

---

## Recommended thesis wording（推荐论文措辞）

This project implements and evaluates core attention variants under a controlled small-scale Transformer setting, and provides additional implemented exploratory variants for future systematic evaluation.
