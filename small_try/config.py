"""快速对比实验配置：小模型 + 子语料 + 少步数。勿与 base_1 全规模配置混用。"""

from dataclasses import dataclass
from typing import Literal

AttentionType = Literal["dot_product", "additive"]


@dataclass
class Config:
    # 子语料（由 create_subset.py 生成）；若不存在请先运行该脚本
    data_path: str = "/root/autodl-tmp/small_try/data/corpus_50k.tsv"
    tokenizer_src: str = "/root/autodl-tmp/data/tokenizer_src.json"
    tokenizer_tgt: str = "/root/autodl-tmp/data/tokenizer_tgt.json"

    src_vocab_size: int = 32000
    tgt_vocab_size: int = 32000

    max_seq_len: int = 96

    # 小 Transformer：几分钟～十几分钟级可跑完对比
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    additive_d_hidden: int = 256

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    # H800 级显存：小模型可开大 batch，提高吞吐、缩短墙钟时间
    batch_size: int = 192
    grad_accum_steps: int = 1
    epochs: int = 100
    max_steps: int = 3000
    warmup_steps: int = 200

    val_every: int = 120
    bleu_sample_size: int = 256
    max_gen_len: int = 64

    # 子集上略提高验证比例，BLEU 更稳
    val_ratio: float = 0.05

    attention_type: AttentionType = "dot_product"

    seed: int = 42
    num_workers: int = 8
    output_dir: str = "/root/autodl-tmp/small_try/runs"
    project_name: str = "attention-small"
    wandb_group: str | None = "try-en-fr"
    use_wandb: bool = True
    wandb_run_name: str | None = None

    # W&B 注意力热力图（解码器 cross-attention，真实权重）
    wandb_log_attention: bool = True
    wandb_attention_every: int = 300  # 按 global_step；步数少时可略稀
    attention_viz_decoder_layer: int = -1  # -1 表示最后一层

    # BLEU：语料中英/法列相同或极似（数字行）易虚高；评估时可跳过（与 base_improve 一致）
    bleu_skip_identical_parallel: bool = True
    bleu_skip_similarity_threshold: float | None = 0.98

    # 生成指标：chrF++ 每次验证；BERTScore/COMET 按 optimizer 步降频（省显存与时间）
    eval_use_chrf: bool = True
    eval_use_bertscore: bool = True
    eval_use_comet: bool = True
    eval_heavy_metrics_every_optimizer_steps: int = 2000
    eval_bertscore_lang: str = "fr"
    # 与 scripts/test_mt_eval_metrics_dummy 一致；CPU 上跑，避免与 seq2seq 争 GPU 致 OOM/假死
    eval_bertscore_model_type: str = "distilbert-base-multilingual-cased"
    eval_bertscore_device: str = "cpu"
    eval_comet_gpus: int = 0
    eval_comet_model: str = "Unbabel/wmt22-comet-da"
