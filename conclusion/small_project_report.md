# Small 注意力实验项目：思路、流水线与产出说明

本文档概括仓库中 **`small_try` / `small_head` / `small_swap`** 三套代码的定位、科学问题、工程流水线、W&B 组织方式及归档指标；与 `three_models_comparison.md`（与 `base_1` 对比表）互为补充。

---

## 1. 目标与分工

| 目录 | 翻译方向 | 核心科学问题 | W&B `wandb_group` |
|------|----------|--------------|-------------------|
| `small_try` | 英→法（EN-FR） | 在**同一小模型与子语料**上对比 **点积注意力** vs **加性注意力**，控制其它超参一致 | `try-en-fr` |
| `small_head` | 英→法 | 在相同数据与宽度下，对比 **多头基线**（引用 `small_try` 的 `fast_dot`）与 **全模型单头（n_heads=1）** | `head-ablation` |
| `small_swap` | 法→英（FR-EN） | 与 `small_try` **同规模**、**列交换**得到反向平行语料，观察 **方向** 对指标的影响（与 EN→FR 数值不可直接横向对比） | `swap-fr-en` |

共同点：**小 Transformer**（如 `d_model=256`, `n_layers=4`, 子集约 5 万句、`max_steps` 默认 3000 级），用于**快速迭代**与注意力机制对比；严肃长训与全量数据见 `base_1`。

---

## 2. 实验思路（简述）

1. **small_try**：先训 `dot_product`（`runs/fast_dot`），再训 `additive`（`runs/fast_add`）。`compare_runs.py` 汇总两次 `metrics.json`，生成 `report.txt` 与 `results/bundle_metrics.json`。
2. **small_head**：**不重复训多头基线**；读取 `small_try/runs/fast_dot/metrics.json` 作为基线，仅训单头点积 `runs/head_1h_dot`。`compare_head_runs.py` 写 `report_head.txt`、`results/bundle_head_metrics.json`；`summarize_params.py` 写 `results/param_counts.json`。
3. **small_swap**：从 `small_try/data/corpus_50k.tsv` 复制语料（或本地生成子集），训 `swap_fr_dot`；`compare_runs.py`（单 run 模式）写 `report_swap.txt` 等。

所有训练 run 默认写入 **同一 W&B Project：`attention-small`**，用 **`wandb_group`** 区分上述三组；本地离线目录可通过环境变量 `WANDB_DIR` 统一（如根目录 `wandb_cache_attention_small`）。

---

## 3. 一键流水线与脚本

| 脚本 | 作用 |
|------|------|
| `run_all_small_training.sh` | 顺序调用 `small_try/run_fast_full.sh` → `small_head/run_head_pipeline.sh` → `small_swap/run_fr_swap_pipeline.sh` |
| `smoke_test_small.sh` | 少步数试跑；默认 `WANDB_MODE=offline`、`EVAL_LIGHT=1`（跳过 BERTScore/COMET）、独立 `wandb_cache_attention_small_smoke` |
| `launch_small_training_nohup.sh` | **正式训练**后台启动，日志在 `logs/run_all_small_<时间戳>.log`，并写 `.pid` |

环境变量（可选）：

- `TRAIN_MAX_STEPS` / `TRAIN_VAL_EVERY`：试跑时缩短步数与验证间隔。
- `EVAL_LIGHT=1`：各 `train.py` 增加 `--eval-light`，仅 BLEU/chrF++，**正式跑请勿设置**（需完整指标时保持默认）。

---

## 4. 指标与归档（「完美记录」所指）

每个 run 目录下 **`metrics.json`** 含：验证损失、BLEU、（非 eval-light 时）chrF++/BERTScore/COMET 等扩展字段、W&B run URL、`config` 快照。

各阶段 **`results/`** 内：

- `small_try`：`bundle_metrics.json`、`manifest.json`
- `small_head`：`bundle_head_metrics.json`、`manifest_head.json`、`param_counts.json`
- `small_swap`：`bundle_swap_metrics.json`、`manifest_swap.json`

根目录 **`report_extra_metrics.py`** 为对比脚本提供额外结论行；**`mt_eval.py`** 为验证/终评统一评测入口。

---

## 5. W&B 使用要点

1. 交互终端执行 **`wandb login`**，再跑正式流水线或 `launch_small_training_nohup.sh`。
2. 网页端打开 Project **`attention-small`**，按 **Group** 筛选 `try-en-fr` / `head-ablation` / `swap-fr-en`。
3. 仅离线时设置 `WANDB_MODE=offline`，事后可用 `wandb sync <offline-run-dir>` 上传。

---

## 6. BERTScore / COMET：要手动下载到哪里？

**一般不需要**事先把权重放到某个自定义目录。`mt_eval.py` 在第一次计算时会：

- **BERTScore**：通过 `bert-score` 包按语言码（如 `fr`）拉取对应 **BERT 类模型**，缓存走 **Hugging Face Hub 默认路径**。
- **COMET**：通过 `unbabel-comet` 的 `download_model("Unbabel/wmt22-comet-da")` 拉取 **COMET 评测模型**（同样多依赖 HF Hub / 包内逻辑）。

**默认缓存位置（Linux 常见）**

| 用途 | 典型路径 | 可调环境变量 |
|------|----------|----------------|
| Hugging Face Hub | `~/.cache/huggingface/hub` | `HF_HOME`（根目录，其下会有 `hub`）、或 `HF_HUB_CACHE` |
| 旧版 Transformers | `~/.cache/huggingface/transformers` | `TRANSFORMERS_CACHE`、`HF_HOME` |

在训练前可统一指定大盘，例如：

```bash
export HF_HOME=/data/hf_cache
mkdir -p "$HF_HOME"
```

之后首次成功跑通一次验证/终评，权重会写进上述目录，**同机重复训练无需再下**。

**国内/弱网环境**

- 仓库已默认 **`HF_ENDPOINT=https://hf-mirror.com`**（`run_all_small_training.sh`、`launch_small_training_nohup.sh`、`smoke_test_small.sh` 与 `mt_eval.py` 在未预先设置时生效）。若需走 **官网**，请先执行：`export HF_ENDPOINT=https://huggingface.co`。
- 镜像只改 **Hub API** 时，大权重仍可能经 **HF XET**（如 `cas-bridge.xethub.hf.co`）拉取，弱网下易出现 **`SSLEOFError` / `UNEXPECTED_EOF_WHILE_READING`**。仓库默认 **`HF_HUB_DISABLE_XET=1`**，让大文件尽量走 **传统 CDN**；若你网络稳定且希望用 XET，可 `export HF_HUB_DISABLE_XET=0`。
- 亦可在一台能访问 **huggingface.co** 的机器上先跑通一次，再把整个 **`HF_HOME` 目录**打包拷到训练机 **相同路径**。
- 若暂时无法拉取模型，可用试跑脚本里的思路：流水线设 **`EVAL_LIGHT=1`** 或 `train.py` 传 **`--eval-light`**，只算 BLEU/chrF++，不依赖 BERTScore/COMET（正式对比建议仍争取把重指标跑通）。

**依赖包（需已安装）**：`pip install bert-score unbabel-comet`（版本过旧可能导致 COMET 报 `Model ... not supported`，可升级 `unbabel-comet` 与 `huggingface_hub`）。

**BERTScore 与 `lang`（按任务区分，不是全程 `en`）**

| 包 | 翻译方向 | `config.eval_bertscore_lang` | 含义 |
|----|----------|-------------------------------|------|
| `small_try` / `small_head` | 英→法 | **`fr`** | 假设/参考均为法语 |
| `small_swap` | 法→英 | **`en`** | 假设/参考均为英语 |

自检脚本 `test_mt_eval_metrics.py` 仅用 **`lang=en` + 小模型** 测镜像是否通；**不能**代表 swap 以外任务的语言设置。

**BERTScore 模型体积**：按 `lang` 使用 bert-score 默认模型时，单语大模型（如 en 的 RoBERTa-large）约 **1.4GB**，首下会像「卡住」实为下载中。换轻量模型可设 **`BERTSCORE_MODEL_TYPE`**，但须与 **`eval_bertscore_lang` 语义一致**（例如法语句勿用纯英文 DistilBERT）。

**代码侧（仓库已做）**

- `mt_eval.py` 为 Hugging Face 设置默认 **`HF_HUB_DOWNLOAD_TIMEOUT=120`**（秒）、**`HF_HUB_DISABLE_XET=1`**；弱网可自行 `export HF_HUB_DOWNLOAD_TIMEOUT=300`，或按需关闭 XET 禁用项（见上文「国内/弱网」）。
- `score_bertscore_f1` 支持参数 **`model_type`** 或环境变量 **`BERTSCORE_MODEL_TYPE`**，便于换小模型。
- **COMET** 在 `download_model` / `load_from_checkpoint` 阶段若失败会 **打印告警并返回 None**，不再因未捕获异常而 **中断整次训练**（`predict` 阶段仍单独 try/except）。

**自检脚本**（装包并网络/缓存就绪后执行）：

```bash
python3 /root/autodl-tmp/scripts/test_mt_eval_metrics.py
```

---

## 7. 与 `base_1` / `base_improve` 的关系

- **`base_1`**：全数据、大模型步数、标准 Seq2Seq 训练配置。
- **`base_improve`**：与 `base_1` 同数值超参，叠加训练侧加速（如 compile 等），**不属于** small 三件套。
- **Small 三件套**：独立子目录与 `output_dir`，仅复用仓库级 `data/tokenizer_*.json` 等；结论对比表见 `three_models_comparison.md`。

---

*若目录名或 project 名在后续重构中有变，以各包 `config.py` 与根目录 `run_all_small_training.sh` 为准。BERTScore/COMET 缓存说明见上文 §6。*
