# Attention small-scale experiments（EN→FR / FR→EN）

本仓库包含 `small_try`、`small_head`、`small_swap` 等小模型注意力对比实验代码，以及 `runs/` 下的冻结训练产出与 `results/` 派生汇总。

## Reproducibility note: cross-commit experiments

Different runs in this archive were trained at different commits with `git_dirty=True` at the time of training. The four commits recorded in `metrics.json` are `2c605436fe` (main runs), `97a3bfcb31` (fast_dot multi-seed), `1ec4ccc52a` (fast_add multi-seed), and `a750583efd` (var_* exploratory variants). **Core training/evaluation files are *not* byte-identical across all of these commits** (for example `small_try/attention.py` and `small_try/train.py` change between the older pair and the newer pair). See **[docs/RUN_COMMIT_INTEGRITY.md](docs/RUN_COMMIT_INTEGRITY.md)** for per-run metadata, blob-hash matrices, diff summaries, and safe-vs-caveat comparisons.

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
