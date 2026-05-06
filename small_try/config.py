"""快速对比实验配置：小模型 + 子语料 + 少步数。勿与 base_1 全规模配置混用。

路径默认为相对仓库根目录的 POSIX 风格字符串，由 train.py 解析为绝对路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AttentionType = Literal[
    "dot_product",
    "additive",
    "bilinear",
    "gated_dot_additive",
    "local_window",
    "global_local",
    "sparsemax",
    "entmax15",
]
EvalSplit = Literal["train", "val", "test"]


@dataclass
class Config:
    # 相对仓库根的路径（train.py 通过 train_runtime.materialize_path_fields 解析为绝对路径）
    train_path: str = "data/splits/en_fr_50k_seed42/train.tsv"
    val_path: str = "data/splits/en_fr_50k_seed42/val.tsv"
    test_path: str = "data/splits/en_fr_50k_seed42/test.tsv"
    data_path: str = "data/splits/en_fr_50k_seed42/train.tsv"

    tokenizer_src: str = "data/tokenizer_src_train_only.json"
    tokenizer_tgt: str = "data/tokenizer_tgt_train_only.json"

    src_vocab_size: int = 30000
    tgt_vocab_size: int = 30000

    max_seq_len: int = 96

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    additive_d_hidden: int = 256

    # local_window / global_local：结构掩码超参（稠密 masked attention；非稀疏核）
    local_window_size: int = 4
    global_local_window_size: int = 4
    global_tokens: int = 4
    # gated_dot_additive：gate = sigmoid(alpha)；True 时每 head 一个 alpha
    gate_alpha_per_head: bool = True

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    batch_size: int = 192
    grad_accum_steps: int = 1
    epochs: int = 100
    max_steps: int = 3000
    warmup_steps: int = 200

    val_every: int = 120
    bleu_sample_size: int = 256
    max_gen_len: int = 64

    val_ratio: float = 0.05
    test_ratio: float = 0.05
    use_split_files: bool = True

    attention_type: AttentionType = "dot_product"

    # 验证损失与 BLEU 使用的划分（训练 DataLoader 固定 train）
    eval_split: EvalSplit = "val"

    seed: int = 42
    num_workers: int = 8
    output_dir: str = "runs"
    project_name: str = "attention-small-2"
    wandb_group: str | None = "try-en-fr"
    use_wandb: bool = True
    wandb_run_name: str | None = None

    wandb_log_attention: bool = True
    wandb_attention_every: int = 300
    attention_viz_decoder_layer: int = -1

    bleu_skip_identical_parallel: bool = True
    bleu_skip_similarity_threshold: float | None = 0.98

    eval_use_chrf: bool = True
    eval_use_bertscore: bool = True
    eval_use_comet: bool = True
    eval_heavy_metrics_every_optimizer_steps: int = 2000
    eval_bertscore_lang: str = "fr"
    eval_bertscore_model_type: str = "distilbert-base-multilingual-cased"
    eval_bertscore_device: str = "cpu"
    eval_comet_gpus: int = 0
    eval_comet_model: str = "Unbabel/wmt22-comet-da"

    # CUDA 上默认 bf16 autocast；诊断实验可关闭（fp32 训练）
    use_bf16_autocast: bool = True
