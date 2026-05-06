# Attention small-scale experiments（EN→FR / FR→EN）

本仓库包含 `small_try`、`small_head`、`small_swap` 等小模型注意力对比实验代码，以及 `runs/` 下的冻结训练产出与 `results/` 派生汇总。

## Reproducibility note: cross-commit experiments

Different runs in this archive were trained at different commits with `git_dirty=True` at the time of training. The **eight** commits recorded across the **36** archived `metrics.json` rows (full **A17+A19+A23** wall via `scripts/check_variant_experiment_fairness.py --archive-audit`) are `2c605436fe` (main runs), `97a3bfcb31` (fast_dot multi-seed), `1ec4ccc52a` (fast_add multi-seed), `a750583efd` (single-seed `var_*` exploratory variants), **`e09f93ad44`** (A13 additive LR sweep: `lr_sweep_add_*`, `max_steps=1500`), **`8a9e118647`** (A14 variant multi-seed + A15 `local_window` stability probes; `max_steps=1500` on those probes), **`e6cd4aa7bb73`** (A19: `fast_add_lr3e3_s{1,2,3}`, `lr=3e-3`, `max_steps=3000`), and **`bd1b53ebac69`** (A23: `var_local_window_a23_retrain` post `safe_masked_softmax` fix; same training protocol as original `var_local_window` — proved it was a softmax mask bug, not a mechanism issue). **Core training/evaluation files are *not* byte-identical across all of these commits** (for example `small_try/attention.py` and `small_try/train.py` change between the older pair and the newer pair; see **§ v4 / v5 / v6** in the integrity doc for `a750`→`e09`→`8a9`→`e6cd`→`bd1b53` deltas). A13, A14, A15, **A19**, and **A23** add runs beyond the original 16-run wall; **[docs/RUN_COMMIT_INTEGRITY.md](docs/RUN_COMMIT_INTEGRITY.md)** has the full **36-run** table, **eight-commit** blob-hash matrix, diff summaries, and safe-vs-caveat notes. **`results/variant_fairness_audit.md`** lists the expanded fairness audit (**`--archive-audit`**, 36 runs).

## How to verify the archive

```bash
python scripts/check_submission_archive.py --strict
python -m pytest -q
python scripts/sanity_check_runs.py
```

`check_submission_archive.py` 会写出 **`results/submission_archive_check.md`**；`--no-strict` 时仅作报告不因缺项退出 1。可选用 **`--max-archive-size-mib N`** 对「拟打包的非 gitignore 文件总体积」做 WARN。

## Documentation and reproducibility

- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) — 数据划分、词表、训练入口、结果表重算顺序（A2–A5）、归档打包。
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — Python / PyTorch / 依赖与网络说明。
- [docs/CODE_ARCHIVE_CHECKLIST.md](docs/CODE_ARCHIVE_CHECKLIST.md) — 提交包应包含/排除清单（与 `check_submission_archive.py` 一致）。
- [docs/THESIS_RESULT_BOUNDARIES.md](docs/THESIS_RESULT_BOUNDARIES.md) — 论文可声称边界与禁止表述。
- [docs/ATTENTION_VARIANTS_STATUS.md](docs/ATTENTION_VARIANTS_STATUS.md) — `attention_type` 实现与实验状态。
- [results/attention_variants_test_summary.md](results/attention_variants_test_summary.md) — held-out test 指标汇总（仅 `test_eval/metrics_test.json`）。

## Layout（概要）

| Path | Role |
|------|------|
| `small_try/` | EN→FR；dot vs additive 主对比 |
| `small_head/` | 单头 vs 多头（dot） |
| `small_swap/` | FR→EN（列交换） |
| `runs/<run>/` | 配置、指标、checkpoint（checkpoint 默认不纳入 zip） |
| `runs/<run>/test_eval/` | held-out `metrics_test.json`、`predictions.jsonl` |
| `scripts/` | 划分、汇总、健全性检查、归档 |

## License / citation

若本工作衍生自课程或论文项目，请在正文中单独写明任务设定、划分与指标口径；勿将不同 commit / 方向的 BLEU 混排为同一排行榜。
