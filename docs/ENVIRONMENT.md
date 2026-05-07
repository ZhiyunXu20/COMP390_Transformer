# Environment（复现环境说明）

## Two-tier environment policy

This repository ships **two** complementary dependency descriptions:

1. **`requirements.txt`** — **Portable install constraints**. Lower-bound pins are set for reproducibility-critical libraries (e.g. sacrebleu, tokenizers, torch). Upper bounds are intentionally relaxed where safe so modern Python environments can resolve installs. **This is what users should run:**
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **`environment_freeze_h800.txt`** — **Exact training-host freeze**. This is a `pip freeze` snapshot from the H800 training node where the **36** archived runs were trained. Use it only to understand the **historical** training environment; **do not** `pip install -r` this file (it includes machine-local paths and CUDA-specific wheels).

### Known version differences between the two files

These differences are **intentional**:

| Package | requirements.txt | environment_freeze_h800.txt | Note |
|---------|------------------|----------------------------|------|
| scipy | `>=1.9` | 1.17.1 | Welch *t*-test API stable across versions; `cross_seed_significance.py` has an **mpmath** fallback |
| numpy | `>=1.24` | 1.26.4 | Training host pinned a 1.26.x line |
| torch | `>=2.0,<3` | 2.7.0+cu128 (cp312 wheel URL in freeze) | CUDA **12.8** build in freeze; install a matching PyTorch build for your GPU |

For scientific reproducibility, what matters is **not** matching every pip-freeze line: it is **determinism flags** (e.g. `--deterministic`), **SacreBLEU signature** alignment when reporting BLEU, and the **per-run** `git_commit` / seed protocol recorded under `runs/`.

## Python and PyTorch versions used

以下自仓库内 **`environment_freeze_h800.txt`**（H800 训练机上的 `pip freeze`）整理；未单独列出 interpreter 行时，可根据 PyTorch wheel 推断 **`cp312` → Python 3.12**。

| 项 | 版本 / 说明 |
|----|----------------|
| **Python** | 3.12（由 `torch-...-cp312-...whl'` 推断；以本机 `python --version` 为准） |
| **PyTorch** | 2.7.0+cu128（见 freeze 中 `torch @ .../torch-2.7.0%2Bcu128-...whl`） |
| **CUDA（PyTorch 打包）** | **12.8**（`cu128`） |
| **GPU** | NVIDIA **H800**（与 freeze 文件名一致；其他 GPU 可运行但需匹配 CUDA 的 PyTorch 构建） |

## Major dependencies

| 包 | 用途 |
|----|------|
| **torch** | 训练与推理 |
| **tokenizers** | Hugging Face 分词器（训练数据管线） |
| **sacrebleu** | BLEU / chrF（`mt_eval.py`） |
| **sentencepiece** | 与 BPE/SentencePiece 词表相关依赖（经 tokenizers / 语料脚本） |
| **bert-score** | BERTScore（可选 heavy 指标） |
| **unbabel-comet** | COMET（可选 heavy 指标） |
| **matplotlib** / **seaborn** | 注意力可视化、绘图 |
| **numpy** | 数组与 DataLoader worker 种子 |
| **scipy** | **`scripts/cross_seed_significance.py`** 在 SciPy 可用时使用 `scipy.stats.t`；不可用时回退 **mpmath**（见同脚本输出中的 `stats_backend`） |
| **mpmath** | Welch p 值 / t 分位数后备（无 SciPy 或与 NumPy 不兼容时） |
| **wandb** | 实验日志（可 `--no-wandb` 关闭） |
| **pytest** | 测试（`tests/`） |

## How to install

```bash
cd /path/to/repo
pip install -r requirements.txt
```

- **entmax**（`sparsemax` / `entmax15` 注意力）已列入 `requirements.txt`。
- **COMET**：首次在 valid 配置下跑到 COMET 时，可能从 Hugging Face 拉取模型缓存；见下文网络说明。
- 若仅需对齐论文中的 **冻结环境**，可参考 **`environment_freeze_h800.txt`** 手工对齐版本（不推荐在异构机器上逐条硬编码 conda 本地路径）。

## Network access requirements

- **COMET**、**BERTScore** 的部分后端依赖 Hugging Face Hub 或首次下载权重；需外网或可访问的镜像。
- **W&B** 上传需 api key（或本地离线 `--no-wandb`）。
