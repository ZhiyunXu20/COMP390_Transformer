#!/usr/bin/env python3
"""训练入口：注意力类型可切换，支持验证 BLEU、checkpoint、W&B。"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_plots import figure_cross_attention_heads, figure_cross_attention_mean
from config import Config
from dataset import TabParallelDataset, collate_batch, load_tokenizers, tokenizer_special_ids
from model import Seq2SeqTransformer, build_logits_shifted_loss

try:
    import wandb
except ImportError:
    wandb = None

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from mt_eval import evaluate_generation_corpus, flatten_extra_for_log
from small_shared.metrics_json import bleu_disambiguation_fields, effective_attention_backend_field
from train_runtime import configure_determinism, make_worker_init_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_warmup_lambda(cfg: Config):
    def lr_lambda(step: int):
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        return 1.0

    return lr_lambda


@torch.no_grad()
def validation_loss(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    pad_idx: int,
    device: torch.device,
    desc: str = "val",
) -> float:
    ce_sum = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    model.eval()
    total, ntok = 0.0, 0
    for src, tgt in tqdm(loader, desc=desc, leave=False):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        tgt_in = tgt[:, :-1]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(src, tgt_in)
            logits, labels = build_logits_shifted_loss(logits, tgt)
            loss_sum = ce_sum(logits, labels)
        n = (labels != pad_idx).sum().item()
        total += loss_sum.item()
        ntok += n
    model.train()
    return total / max(1, ntok)


def save_metrics_json(
    path: Path,
    cfg: Config,
    *,
    determinism: dict | None = None,
    final_val_loss: float,
    final_bleu: float | None,
    best_bleu: float,
    optimizer_step: int,
    global_step: int,
    wandb_url: str | None,
    run_display_name: str,
    bleu_eval_meta: dict | None,
    final_extra_metrics: dict[str, float | None] | None,
) -> None:
    def jsonable(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        return str(obj)

    payload = {
        "determinism": jsonable(determinism) if determinism is not None else None,
        "run": "base_1",
        "attention_type": cfg.attention_type,
        "final_val_loss": final_val_loss,
        "final_bleu": final_bleu,
        "best_bleu_during_training": best_bleu,
        "optimizer_steps": optimizer_step,
        "global_steps": global_step,
        "bleu_evaluation": jsonable(bleu_eval_meta) if bleu_eval_meta else None,
        "extra_metrics": jsonable(final_extra_metrics) if final_extra_metrics else None,
        "config": jsonable(cfg.__dict__),
    }
    payload.update(bleu_disambiguation_fields(cfg, bleu_eval_meta))
    payload.update(effective_attention_backend_field(cfg))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EN-FR attention comparison training")
    p.add_argument(
        "--attention",
        type=str,
        default="dot_product",
        choices=("dot_product", "additive"),
        help="注意力打分类型",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--name", type=str, default=None, help="运行子目录名 / wandb run name")
    p.add_argument("--max-steps", type=int, default=None, help="全局 batch 步数上限（覆盖 config.max_steps）")
    p.add_argument(
        "--val-every",
        type=int,
        default=None,
        help="每多少次 optimizer.step 验证一次（覆盖 config.val_every）",
    )
    p.add_argument(
        "--no-attention-plots",
        action="store_true",
        help="关闭 W&B 解码器 cross-attention 热力图",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="更严格的可复现模式（cuDNN deterministic、确定性算法 warn_only）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.attention_type = args.attention  # type: ignore[assignment]
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_wandb:
        cfg.use_wandb = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.name:
        cfg.wandb_run_name = args.name
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.val_every is not None:
        cfg.val_every = args.val_every
    if args.no_attention_plots:
        cfg.wandb_log_attention = False

    set_seed(cfg.seed)
    det_state = configure_determinism(
        cfg.seed, args.deterministic, cuda_available=torch.cuda.is_available()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_float32_matmul_precision("high")

    src_tok, tgt_tok = load_tokenizers(cfg)
    cfg.src_vocab_size = src_tok.get_vocab_size()
    cfg.tgt_vocab_size = tgt_tok.get_vocab_size()
    pad_idx, bos_id, eos_id = tokenizer_special_ids(tgt_tok)

    run_name = cfg.wandb_run_name or f"{cfg.attention_type}_{int(time.time())}"
    out = Path(cfg.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    if cfg.use_wandb and wandb is not None:
        wandb.init(
            project=cfg.project_name,
            name=run_name,
            config={**cfg.__dict__},
        )
    elif cfg.use_wandb and wandb is None:
        print("wandb 未安装，跳过日志", file=sys.stderr)

    train_ds = TabParallelDataset(cfg, src_tok, tgt_tok, "train")
    val_ds = TabParallelDataset(cfg, src_tok, tgt_tok, "val")

    g_train = torch.Generator()
    g_train.manual_seed(cfg.seed)
    tl_kw = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": lambda b: collate_batch(b, pad_idx),
        "persistent_workers": cfg.num_workers > 0,
        "generator": g_train,
    }
    if cfg.num_workers > 0:
        tl_kw["worker_init_fn"] = make_worker_init_fn(cfg.seed)
    train_loader = DataLoader(train_ds, **tl_kw)

    g_val = torch.Generator()
    g_val.manual_seed(cfg.seed)
    vl_kw = {
        "batch_size": min(cfg.batch_size, 32),
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": lambda b: collate_batch(b, pad_idx),
        "persistent_workers": cfg.num_workers > 0,
        "generator": g_val,
    }
    if cfg.num_workers > 0:
        vl_kw["worker_init_fn"] = make_worker_init_fn(cfg.seed)
    val_loader = DataLoader(val_ds, **vl_kw)

    model = Seq2SeqTransformer(cfg, pad_idx=pad_idx).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_lambda(cfg))

    scaler_enabled = device.type == "cuda"
    global_step = 0
    optimizer_step = 0
    best_bleu = -1.0
    accum = 0

    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for src, tgt in pbar:
            if global_step >= cfg.max_steps:
                break
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_in = tgt[:, :-1]

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=scaler_enabled
            ):
                logits = model(src, tgt_in)
                logits_flat, labels = build_logits_shifted_loss(logits, tgt)
                loss = criterion(logits_flat, labels) / cfg.grad_accum_steps

            loss.backward()
            accum += 1

            if accum % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1
                accum = 0

                if optimizer_step % cfg.val_every == 0:
                    vloss = validation_loss(model, val_loader, pad_idx, device)
                    bleu_diag: dict = {}
                    bleu, extra_m = evaluate_generation_corpus(
                        model,
                        val_loader,
                        src_tok,
                        tgt_tok,
                        pad_idx,
                        bos_id,
                        eos_id,
                        device,
                        max_samples=cfg.bleu_sample_size,
                        max_gen_len=cfg.max_gen_len,
                        skip_identical_parallel=cfg.bleu_skip_identical_parallel,
                        similarity_threshold=cfg.bleu_skip_similarity_threshold,
                        optimizer_step=optimizer_step,
                        use_chrf=cfg.eval_use_chrf,
                        use_bertscore=cfg.eval_use_bertscore,
                        use_comet=cfg.eval_use_comet,
                        heavy_every=cfg.eval_heavy_metrics_every_optimizer_steps,
                        bertscore_lang=cfg.eval_bertscore_lang,
                        comet_model=cfg.eval_comet_model,
                        force_heavy=False,
                        out_diag=bleu_diag,
                    )
                    msg = f"val_loss={vloss:.4f}"
                    if bleu is not None:
                        msg += f" bleu={bleu:.2f}"
                        if bleu > best_bleu:
                            best_bleu = bleu
                            ck = out / "best.pt"
                            torch.save(
                                {
                                    "model": model.state_dict(),
                                    "cfg": cfg.__dict__,
                                    "bleu": bleu,
                                    "optimizer_step": optimizer_step,
                                },
                                ck,
                            )
                    if extra_m.get("chrf") is not None:
                        msg += f" chrf={extra_m['chrf']:.2f}"
                    if extra_m.get("bertscore_f1") is not None:
                        msg += f" bertscore_f1={extra_m['bertscore_f1']:.4f}"
                    if extra_m.get("comet") is not None:
                        msg += f" comet={extra_m['comet']:.4f}"
                    print(msg)
                    if cfg.use_wandb and wandb is not None:
                        log = {"val_loss": vloss, "optimizer_step": optimizer_step}
                        if bleu is not None:
                            log["bleu"] = bleu
                        for k, v in bleu_diag.items():
                            log[f"bleu_meta/{k}"] = v
                        log.update(flatten_extra_for_log(extra_m))
                        wandb.log(log)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item() * cfg.grad_accum_steps))

            if (
                cfg.use_wandb
                and wandb is not None
                and cfg.wandb_log_attention
                and global_step % cfg.wandb_attention_every == 0
            ):
                li = (
                    cfg.attention_viz_decoder_layer
                    if cfg.attention_viz_decoder_layer >= 0
                    else cfg.n_layers - 1
                )
                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        with torch.amp.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=scaler_enabled
                        ):
                            _, att_dict = model(
                                src[:1], tgt_in[:1], output_attentions=True
                            )
                        cross_h = att_dict["decoder_cross"][li][0].float().cpu()
                    fig_h = figure_cross_attention_heads(
                        cross_h,
                        src[0].tolist(),
                        tgt_in[0].tolist(),
                        src_tok,
                        tgt_tok,
                        pad_idx,
                        layer_idx=li,
                        title_prefix=f"step={global_step}",
                    )
                    fig_m = figure_cross_attention_mean(
                        att_dict["decoder_cross"][li][0].float(),
                        src[0].tolist(),
                        tgt_in[0].tolist(),
                        src_tok,
                        tgt_tok,
                        pad_idx,
                        layer_idx=li,
                        title_prefix=f"step={global_step}",
                    )
                    wandb.log(
                        {
                            f"attention/cross_L{li}_all_heads": wandb.Image(fig_h),
                            f"attention/cross_L{li}_mean": wandb.Image(fig_m),
                        }
                    )
                    plt.close(fig_h)
                    plt.close(fig_m)
                finally:
                    if was_training:
                        model.train()

            if cfg.use_wandb and wandb is not None and global_step % 50 == 0:
                wandb.log(
                    {
                        "train_loss": loss.item() * cfg.grad_accum_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    }
                )

            if global_step >= cfg.max_steps:
                break

        if global_step >= cfg.max_steps:
            break

    last = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "optimizer_step": optimizer_step,
            "global_step": global_step,
        },
        last,
    )
    print("最终评估…", file=sys.stderr)
    final_vloss = validation_loss(model, val_loader, pad_idx, device, desc="final_val")
    final_bleu_diag: dict = {}
    final_bleu, final_extra = evaluate_generation_corpus(
        model,
        val_loader,
        src_tok,
        tgt_tok,
        pad_idx,
        bos_id,
        eos_id,
        device,
        max_samples=cfg.bleu_sample_size,
        max_gen_len=cfg.max_gen_len,
        skip_identical_parallel=cfg.bleu_skip_identical_parallel,
        similarity_threshold=cfg.bleu_skip_similarity_threshold,
        optimizer_step=optimizer_step,
        use_chrf=cfg.eval_use_chrf,
        use_bertscore=cfg.eval_use_bertscore,
        use_comet=cfg.eval_use_comet,
        heavy_every=cfg.eval_heavy_metrics_every_optimizer_steps,
        bertscore_lang=cfg.eval_bertscore_lang,
        comet_model=cfg.eval_comet_model,
        force_heavy=True,
        out_diag=final_bleu_diag,
    )
    print(
        f"完成。checkpoint: {last} best_bleu={best_bleu} "
        f"final_val_loss={final_vloss:.4f} final_bleu={final_bleu}",
        file=sys.stderr,
    )
    wb_url = None
    if cfg.use_wandb and wandb is not None and wandb.run is not None:
        wb_url = getattr(wandb.run, "url", None) or wandb.run.get_url()
        log_final: dict = {
            "final_val_loss": float(final_vloss),
            "best_bleu_during_train": float(best_bleu),
        }
        if final_bleu is not None:
            log_final["final_bleu"] = float(final_bleu)
        for k, v in final_bleu_diag.items():
            log_final[f"bleu_meta/{k}"] = v
        log_final.update(flatten_extra_for_log(final_extra))
        wandb.log(log_final)
        wandb.finish()
    save_metrics_json(
        out / "metrics.json",
        cfg,
        determinism=det_state,
        final_val_loss=final_vloss,
        final_bleu=final_bleu,
        best_bleu=best_bleu,
        optimizer_step=optimizer_step,
        global_step=global_step,
        wandb_url=wb_url,
        run_display_name=run_name,
        bleu_eval_meta=final_bleu_diag,
        final_extra_metrics=final_extra,
    )
    print(f"metrics -> {out / 'metrics.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
