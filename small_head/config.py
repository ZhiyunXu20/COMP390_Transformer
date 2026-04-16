"""small_head：以 small_try 多头为基线，对比「全模型 n_heads=1」单头（默认仅点积）。"""

from dataclasses import dataclass
from typing import Literal

AttentionType = Literal["dot_product", "additive"]


@dataclass
class Config:
    # 与 small_try 共用子语料（指向 small_try/data）
    data_path: str = "/root/autodl-tmp/small_try/data/corpus_50k.tsv"
    tokenizer_src: str = "/root/autodl-tmp/data/tokenizer_src.json"
    tokenizer_tgt: str = "/root/autodl-tmp/data/tokenizer_tgt.json"

    src_vocab_size: int = 32000
    tgt_vocab_size: int = 32000

    max_seq_len: int = 96

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4  # 训练时用 --n-heads 1 覆盖为单头
    d_ff: int = 1024
    dropout: float = 0.1

    additive_d_hidden: int = 256

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

    attention_type: AttentionType = "dot_product"

    seed: int = 42
    num_workers: int = 8
    output_dir: str = "/root/autodl-tmp/small_head/runs"
    project_name: str = "attention-small"
    use_wandb: bool = True
    wandb_run_name: str | None = None
    wandb_group: str | None = "head-ablation"

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
