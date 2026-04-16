# Small 全流程正式训练 — 执行说明与指标落盘

## 流水线顺序（`run_all_small_training.sh`）

1. **small_try**：子语料 `corpus_50k.tsv` → **dot_product**（`runs/fast_dot/`）→ **additive**（`runs/fast_add/`）→ `report.txt`、`results/bundle_metrics.json`
2. **small_head**：依赖 **small_try** 的 `runs/fast_dot/metrics.json` 作多头基线 → **单头 dot**（`runs/head_1h_dot/`）→ `report_head.txt`、`results/bundle_head_metrics.json`
3. **small_swap**：法→英 → **dot**（`runs/swap_fr_dot/`）→ `report_swap.txt`、`results/bundle_swap_metrics.json`

任一步命令非 0 退出则**整条流水线失败**（各脚本 `set -euo pipefail`）。

## 显存与速度

- 各 `config.py` 默认 **`batch_size=192`**、`grad_accum_steps=1`，注释按大显存 GPU 设定。
- 若显存仍有余、希望提高吞吐：启动前 **`export TRAIN_BATCH_SIZE=256`**（或 320 等），会传给三阶段所有 `train.py --batch-size`。
- 若 OOM：减小 `TRAIN_BATCH_SIZE` 或改回仅用 config 默认值（不要 export）。
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** 已在各流水线脚本中设置，减轻碎片。

## 指标（正式训练默认「全量」）

- **不设置 `EVAL_LIGHT`**（`launch_small_training_nohup.sh` 会 `unset EVAL_LIGHT`）：验证含 **BLEU、chrF++、BERTScore、COMET**（与各包 `eval_bertscore_lang` 一致；重指标按 `eval_heavy_metrics_every_optimizer_steps` 降频，**训练结束一次最终验证会 `force_heavy=True`** 保证终值）。
- **W&B**：`project_name=attention-small`，用 **`wandb_group`** 区分：`try-en-fr` | `head-ablation` | `swap-fr-en`。默认 **`WANDB_MODE=online`**，需已 `wandb login`。
- **本地 JSON**：每 run 目录 **`metrics.json`**；各阶段 **`results/bundle_*.json`** 汇总。

## 日志与后台

```bash
bash /root/autodl-tmp/launch_small_training_nohup.sh
# 终端会打印 LOG 路径与 PID；例如：
# tail -f /root/autodl-tmp/logs/run_all_small_YYYYMMDD_HHMMSS.log
```

## 环境变量速查

| 变量 | 含义 |
|------|------|
| `WANDB_DIR` | 默认 `$ROOT/wandb_cache_attention_small`，三阶段共用 |
| `WANDB_MODE` | 默认 `online` |
| `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` | 见根目录 shell 与 `mt_eval.py` |
| `TRAIN_MAX_STEPS` / `TRAIN_VAL_EVERY` | 覆盖训练步数与验证间隔 |
| `TRAIN_BATCH_SIZE` | 可选，统一加大/减小 batch |

---

*本文档随流水线脚本更新；以仓库内 `run_all_small_training.sh`、`launch_small_training_nohup.sh` 为准。*
