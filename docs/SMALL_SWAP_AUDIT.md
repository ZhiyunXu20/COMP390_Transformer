# small_swap BLEU 审计（与 small_try dot_product final_bleu 完全相同）

本文档对应「命令 9」审计结论：**静态代码审查未发现 small_swap 错误复用 small_try 的 metrics 或错误拼接 BLEU 语料的缺陷**。若在报告中看到两个方向的 **`metrics.json` 内 `final_bleu`** **数值完全一致**，应优先从 **文件路径、运行目录、人工拷贝与打印精度** 等层面核对，而不是推断「FR→EN 与 EN→FR 可互换」。

**Held-out test 口径**：课程/论文主表若以 `evaluate_test.py` 为准，**`runs/swap_fr_dot/test_eval/metrics_test.json`** 中 BLEU 约为 **9.76**（与 **`results/attention_variants_test_summary.md`** 一致）；请勿与 EN→FR 的 **`fast_dot`** test BLEU（约 **16.07**）混为同一任务下的横向排名。

---

## 1. 写入 `final_bleu` / `best_bleu` / 汇总文件的代码位置

| 产物 | 写入位置 | 说明 |
|------|----------|------|
| `metrics.json` 中的 `final_bleu`、`best_bleu_during_training` | `small_try/train.py`、`small_swap/train.py`、`small_head/train.py` 的 `save_metrics_json()` | 训练结束时由本包内 **`evaluate_generation_corpus`** 返回值写入；路径为 **`{repo_root}/runs/<run_name>/metrics.json`** |
| `bundle_swap_metrics.json` | `small_swap/compare_runs.py` | 读取 **`--dot`**（及可选 **`--add`**）指向的 **`metrics.json`**，合并后写入 **`--json-bundle`** |
| `report_swap.txt` | `small_swap/compare_runs.py` | 同上，读入的 metrics 仅来自 CLI 传入路径 |
| 流水线调用 | `small_swap/run_fr_swap_pipeline.sh` | 训练：`python train.py ... --name swap_fr_dot`；汇总：`compare_runs.py --dot "$REPO/runs/swap_fr_dot/metrics.json"` |

**未发现**任何 Python 代码把 `small_try` 的 `metrics.json` 自动复制或合并进 `small_swap` 的输出目录。

---

## 2. small_swap 是否错误复用了 small_try 的 metrics？

- **`compare_runs.py`** 只 **`json.loads` 用户给出的文件路径**，默认 `--dot` 为仓库根目录下的 **`runs/swap_fr_dot/metrics.json`**（与 `compare_runs.py` 内 `Path(__file__).parent.parent / "runs" / ...` 一致）。
- **`small_swap/train.py`** 中 `infer_repo_root` 将 **`output_dir`** 解析为仓库根下的 **`runs`**，`run_name` 来自 **`--name`**（流水线为 **`swap_fr_dot`**），故 **`metrics.json` 落在 `<repo>/runs/swap_fr_dot/metrics.json`**。
- **`small_try`** 若使用 **`--name fast_dot`**（或其它名称），则写入 **不同子目录**，除非两次运行使用了 **相同的 `--name`**，否则 **不会覆盖同一文件**。

因此：**逻辑上 small_swap 不会「静默」读取 small_try 的 metrics**；若两者数值相同，更可能是 **打开了错误的文件**、**手动复制了 JSON**、或 **两次运行写入了同一 `runs/<同名>/metrics.json`**。

**自检**：合格的 **`small_swap`** 的 `metrics.json` 应包含 **`"translation_direction": "fr_en"`** 与 **`"swap_parallel_columns": true`**（见 `small_swap/train.py` 中 `save_metrics_json` 的 payload）。若缺失或与论文叙述不符，说明该文件并非 swap 训练产出或已被篡改。

---

## 3. `evaluate_generation_corpus` 在 swap 下的语义（French source / English reference）

共享实现见仓库根目录 **`mt_eval.py`** 中的 **`collect_greedy_predictions`**：

- **源句解码**：`src_str = src_tok.decode(src_ids)` —— 使用 **`tokenizer_src`**。
- **参考译文解码**：`ref_str = tgt_tok.decode(ref_ids_clean)` —— 使用 **`tokenizer_tgt`**。
- **模型假设**：`hyp_str = tgt_tok.decode(hyp_ids)` —— 同样 **`tokenizer_tgt`**。

**small_swap** 在 **`small_swap/config.py`** 中约定：

- **`tokenizer_src`** = `data/tokenizer_tgt_train_only.json`（在划分脚本命名下对应 **法语列** 训练的分词器）；
- **`tokenizer_tgt`** = `data/tokenizer_src_train_only.json`（对应 **英语列**）；
- **`swap_parallel_columns: True`**，且在 **`small_swap/dataset.py`** 的 **`TabParallelDataset`** 中将 TSV 两列 **`EN \\t FR`** 交换为 **encoder 输入 = 法语、decoder 目标 = 英语**。

因此在 swap 模式下，**BLEU 的比较对象是「英语参考」与「英语假设」**，与 **`eval_bertscore_lang: "en"`** 一致；**未发现**把法语当作 reference 来算 BLEU 的代码路径。

---

## 4. 随机抽样核对（source=法语，reference=英语，prediction=英语）

仓库提供脚本（需已生成划分 TSV 与 tokenizer 文件）：

```bash
# 仅核对语料方向（不需要 GPU）
python scripts/small_swap_sanity_check.py --split val --n 5 --seed 42

# 带 greedy 解码（需要 checkpoint）
python scripts/small_swap_sanity_check.py --split test --n 5 --checkpoint runs/swap_fr_dot/best.pt
```

脚本会打印粗粒度字符启发（`likely_fr_chars` / `likely_en_ascii_heavy`）；最终以人工扫一眼句子为准。

---

## 5. 为何「final_bleu」可能与 small_try dot_product **完全相同**（无代码 bug 时的合理解释）

1. **小数打印或截图精度**：论文表格若只保留两位小数，两种方向、不同运行可能在展示上「撞数」。
2. **文件混淆**：误将 **`runs/fast_dot/metrics.json`** 与 **`runs/swap_fr_dot/metrics.json`** 当作两份独立结果，实则其一被覆盖或未重新训练。
3. **同名 run 目录**：若 **`small_try`** 某次也使用 **`--name swap_fr_dot`**（或与 swap 共用同名路径），后一次训练会覆盖 **`metrics.json`**，造成「两个实验数值一致」的假象。
4. **统计上**：在 **短训、小样本 BLEU（如 `bleu_sample_size`）** 下，两条独立得到的 corpus BLEU **理论上可以非常接近甚至相等**，尤其在数值分辨率不高时；这不等于任务或方向相同。

更正式的论述亦见 **`docs/RESEARCH_AUDIT.md`**（将「两方向 BLEU 完全相同」标为高风险信号）。

---

## 6. 重新评估命令（独立 test，避免与训练 val 混淆）

在仓库根目录执行（路径按你本地 checkpoint / 划分文件调整）：

```bash
python evaluate_test.py \
  --pkg small_swap \
  --checkpoint runs/swap_fr_dot/best.pt \
  --test-file data/splits/en_fr_50k_seed42/test.tsv \
  --tokenizer-src data/tokenizer_tgt_train_only.json \
  --tokenizer-tgt data/tokenizer_src_train_only.json \
  --output-dir eval_out/small_swap_test_eval
```

产出：`eval_out/small_swap_test_eval/metrics_test.json`、`predictions.jsonl`、`examples.md`。  
与 **`small_try`** 对比时，请使用 **`--pkg small_try`** 及 **对应 checkpoint**，并 **分别打开两份 `metrics_test.json`**，勿混用路径。

**完整性校验（可选）**：

```bash
sha256sum runs/swap_fr_dot/metrics.json runs/fast_dot/metrics.json
python -c "import json; print(json.load(open('runs/swap_fr_dot/metrics.json'))['translation_direction'])"
```

第二行应输出 **`fr_en`**（swap）；若为 **`KeyError`** 或缺失该字段，则说明该文件不是当前 **`small_swap/train.py`** 写出的版本。
