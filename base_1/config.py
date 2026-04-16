"""实验配置：路径、模型、训练与注意力类型。"""

from dataclasses import dataclass
from typing import Literal

AttentionType = Literal["dot_product", "additive"]


@dataclass
class Config:
    # --- 数据（沿用你现有的语料与已训练好的 BPE）---
    data_path: str = "/root/autodl-tmp/data/EN-FR.txt"
    tokenizer_src: str = "/root/autodl-tmp/data/tokenizer_src.json"
    tokenizer_tgt: str = "/root/autodl-tmp/data/tokenizer_tgt.json"

    # 词表大小在 train 中从 tokenizer 读取后写回；此处为占位
    src_vocab_size: int = 32000
    tgt_vocab_size: int = 32000

    max_seq_len: int = 128  # 含 BOS/EOS 的总长度上限

    # --- Transformer（Base 量级，适配 H800 单卡）---
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # 加性注意力内部瓶颈维（Bahdanau 风格 MLP）
    additive_d_hidden: int = 512

    # --- 训练 ---
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    batch_size: int = 96
    grad_accum_steps: int = 2
    epochs: int = 20
    max_steps: int = 200_000  # 任一条件先到即停
    warmup_steps: int = 4000

    # 验证：每 val_every 个 optimizer 更新步做一次验证（非 micro-step）
    val_every: int = 2000
    # BLEU 子样本条数（控制验证耗时）
    bleu_sample_size: int = 256
    max_gen_len: int = 96

    # 确定性划分：约 val_ratio 的样本作为验证集
    val_ratio: float = 0.005

    # --- 注意力对比实验 ---
    attention_type: AttentionType = "dot_product"

    # --- 运行 ---
    seed: int = 42
    num_workers: int = 6
    output_dir: str = "/root/autodl-tmp/base_1/runs"
    project_name: str = "en-fr-attention"
    use_wandb: bool = True
    wandb_run_name: str | None = None

    # W&B：解码器 cross-attention 热力图（真实权重，多头 + 平均）
    wandb_log_attention: bool = True
    wandb_attention_every: int = 2000
    attention_viz_decoder_layer: int = -1

    bleu_skip_identical_parallel: bool = True
    bleu_skip_similarity_threshold: float | None = 0.98

    eval_use_chrf: bool = True
    eval_use_bertscore: bool = True
    eval_use_comet: bool = True
    eval_heavy_metrics_every_optimizer_steps: int = 2000
    eval_bertscore_lang: str = "fr"
    eval_comet_model: str = "Unbabel/wmt22-comet-da"
