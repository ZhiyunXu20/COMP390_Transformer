#!/usr/bin/env python3
"""small_try 快速对比训练：结束后写出 metrics.json 供 compare_runs.py 汇总报告。"""

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

from config import Config
from attention_plots import figure_cross_attention_heads, figure_cross_attention_mean
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
from train_runtime import (
    PATH_FIELDS_DEFAULT,
    apply_shared_cli_to_config,
    git_commit_and_dirty,
    infer_repo_root,
    materialize_path_fields,
    register_shared_cli_arguments,
    save_resolved_config_json,
)


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
    final_val_loss: float,
    final_bleu: float | None,
    best_bleu: float,
    optimizer_step: int,
    global_step: int,
    wandb_url: str | None = None,
    run_display_name: str | None = None,
    bleu_eval_meta: dict | None = None,
    final_extra_metrics: dict[str, float | None] | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> None:
    def jsonable(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        return str(obj)

    payload = {
        "attention_type": cfg.attention_type,
        "n_heads": cfg.n_heads,
        "eval_split": getattr(cfg, "eval_split", "val"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "final_val_loss": final_val_loss,
        "final_bleu": final_bleu,
        "best_bleu_during_training": best_bleu,
        "optimizer_steps": optimizer_step,
        "global_steps": global_step,
        "data_path": cfg.data_path,
        "train_path": getattr(cfg, "train_path", cfg.data_path),
        "val_path": getattr(cfg, "val_path", None),
        "test_path": getattr(cfg, "test_path", None),
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "max_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "wandb_project": cfg.project_name,
        "wandb_url": wandb_url,
        "run_name": run_display_name,
        "bleu_evaluation": jsonable(bleu_eval_meta) if bleu_eval_meta else None,
        "extra_metrics": jsonable(final_extra_metrics) if final_extra_metrics else None,
        "config": jsonable(cfg.__dict__),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="small_try 快速对比训练（EN→FR；dot + additive）")
    register_shared_cli_arguments(p)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument(
        "--no-attention-plots",
        action="store_true",
        help="关闭 W&B 上的解码器 cross-attention 热力图",
    )
    p.add_argument(
        "--eval-light",
        action="store_true",
        help="仅 BLEU/chrF++，跳过 BERTScore/COMET（无网或快速试跑）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.no_attention_plots:
        cfg.wandb_log_attention = False
    if args.eval_light:
        cfg.eval_use_bertscore = False
        cfg.eval_use_comet = False
    apply_shared_cli_to_config(cfg, args)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.name:
        cfg.wandb_run_name = args.name
    if args.val_every is not None:
        cfg.val_every = args.val_every

    repo_root = infer_repo_root(Path(__file__))
    materialize_path_fields(cfg, repo_root, PATH_FIELDS_DEFAULT)
    if cfg.d_model % cfg.n_heads != 0:
        print(
            f"错误: d_model={cfg.d_model} 必须能被 n_heads={cfg.n_heads} 整除",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if getattr(cfg, "eval_split", "val") == "test":
        print(
            "警告: eval_split=test：训练中周期性验证/BLEU 与 best.pt 选择将基于 test.tsv，易造成泄漏。"
            " 请将 eval_split 设为 val，并使用仓库根目录 evaluate_test.py 对独立 test.tsv 做最终评估。",
            file=sys.stderr,
        )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    src_tok, tgt_tok = load_tokenizers(cfg)
    cfg.src_vocab_size = src_tok.get_vocab_size()
    cfg.tgt_vocab_size = tgt_tok.get_vocab_size()
    pad_idx, bos_id, eos_id = tokenizer_special_ids(tgt_tok)

    run_name = cfg.wandb_run_name or f"{cfg.attention_type}_{int(time.time())}"
    out = Path(cfg.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    save_resolved_config_json(out, cfg, repo_root, argv=sys.argv)

    if cfg.use_wandb and wandb is not None:
        wb_kw: dict = {"project": cfg.project_name, "name": run_name, "config": {**cfg.__dict__}}
        wg = getattr(cfg, "wandb_group", None)
        if wg:
            wb_kw["group"] = wg
        wandb.init(**wb_kw)
    elif cfg.use_wandb and wandb is None:
        print("wandb 未安装，跳过", file=sys.stderr)

    train_ds = TabParallelDataset(cfg, src_tok, tgt_tok, "train")
    eval_ds = TabParallelDataset(cfg, src_tok, tgt_tok, cfg.eval_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_batch(b, pad_idx),
        persistent_workers=cfg.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=min(cfg.batch_size, 256),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_batch(b, pad_idx),
        persistent_workers=cfg.num_workers > 0,
    )

    train_wall_t0 = time.monotonic()

    model = Seq2SeqTransformer(cfg, pad_idx=pad_idx).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

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

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=scaler_enabled):
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
                    vloss = validation_loss(model, eval_loader, pad_idx, device)
                    bleu_diag: dict = {}
                    bleu, extra_m = evaluate_generation_corpus(
                        model,
                        eval_loader,
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
                        bertscore_model_type=getattr(
                            cfg, "eval_bertscore_model_type", None
                        ),
                        bertscore_device=getattr(cfg, "eval_bertscore_device", None),
                        comet_model=cfg.eval_comet_model,
                        comet_gpus=getattr(cfg, "eval_comet_gpus", None),
                        force_heavy=False,
                        out_diag=bleu_diag,
                    )
                    msg = f"val_loss={vloss:.4f}"
                    if bleu is not None:
                        msg += f" bleu={bleu:.2f}"
                        if bleu > best_bleu:
                            best_bleu = bleu
                            torch.save(
                                {
                                    "model": model.state_dict(),
                                    "cfg": cfg.__dict__,
                                    "bleu": bleu,
                                    "optimizer_step": optimizer_step,
                                },
                                out / "best.pt",
                            )
                    if extra_m.get("chrf") is not None:
                        msg += f" chrf={extra_m['chrf']:.2f}"
                    if extra_m.get("bertscore_f1") is not None:
                        msg += f" bertscore_f1={extra_m['bertscore_f1']:.4f}"
                    if extra_m.get("comet") is not None:
                        msg += f" comet={extra_m['comet']:.4f}"
                    if bleu_diag:
                        msg += (
                            f" | BLEU样本={bleu_diag.get('bleu_pairs_used')} "
                            f"跳过同文={bleu_diag.get('bleu_skipped_identical_parallel')} "
                            f"高相似跳过={bleu_diag.get('bleu_skipped_high_similarity')}"
                        )
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

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "optimizer_step": optimizer_step,
            "global_step": global_step,
        },
        out / "last.pt",
    )

    print("最终评估（用于报告）…")
    final_vloss = validation_loss(model, eval_loader, pad_idx, device, desc="final_val")
    final_bleu_diag: dict = {}
    final_bleu, final_extra = evaluate_generation_corpus(
        model,
        eval_loader,
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
        bertscore_model_type=getattr(cfg, "eval_bertscore_model_type", None),
        bertscore_device=getattr(cfg, "eval_bertscore_device", None),
        comet_model=cfg.eval_comet_model,
        comet_gpus=getattr(cfg, "eval_comet_gpus", None),
        force_heavy=True,
        out_diag=final_bleu_diag,
    )
    print(
        f"完成。metrics -> {out / 'metrics.json'} | final_val_loss={final_vloss:.4f} "
        f"final_bleu={final_bleu} best_bleu={best_bleu}"
    )
    wb_url = None
    if cfg.use_wandb and wandb is not None and wandb.run is not None:
        wb_url = wandb.run.get_url()
        log_final = {
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
    gc, gd = git_commit_and_dirty(repo_root)
    save_metrics_json(
        out / "metrics.json",
        cfg,
        final_val_loss=final_vloss,
        final_bleu=final_bleu,
        best_bleu=best_bleu,
        optimizer_step=optimizer_step,
        global_step=global_step,
        wandb_url=wb_url,
        run_display_name=run_name,
        bleu_eval_meta=final_bleu_diag,
        final_extra_metrics=final_extra,
        git_commit=gc,
        git_dirty=gd,
    )

    peak_bytes: int | None = None
    if device.type == "cuda":
        peak_bytes = int(torch.cuda.max_memory_allocated())
    wall_s = time.monotonic() - train_wall_t0
    tm_path = out / "training_meta.json"
    tm_path.write_text(
        json.dumps(
            {
                "wall_time_seconds": wall_s,
                "num_parameters": num_parameters,
                "peak_gpu_memory_bytes": peak_bytes,
                "peak_gpu_memory_mib": (peak_bytes / (1024 * 1024)) if peak_bytes is not None else None,
                "run_directory": str(out.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"training_meta -> {tm_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
