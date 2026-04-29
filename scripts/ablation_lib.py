"""消融实验共用：加载 best checkpoint 并在 test 划分上终评。"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader


def import_training_stack(repo_root: Path, pkg_name: str) -> tuple[Any, Callable[..., Any], Any, Any, Any, Any]:
    pkg_dir = repo_root / pkg_name
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"未知配置包: {pkg_dir}")
    s = str(pkg_dir)
    if s not in sys.path:
        sys.path.insert(0, s)
    from config import Config
    from dataset import TabParallelDataset, collate_batch, load_tokenizers, tokenizer_special_ids
    from model import Seq2SeqTransformer

    return Config, TabParallelDataset, collate_batch, load_tokenizers, tokenizer_special_ids, Seq2SeqTransformer


def cfg_from_checkpoint_dict(cfg_dict: dict[str, Any], ConfigCls: Any) -> Any:
    cfg = ConfigCls()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_checkpoint_on_test(
    repo_root: Path,
    pkg_name: str,
    run_dir: Path,
    *,
    ckpt_name: str = "best.pt",
    bleu_sample_size: int = 256,
) -> dict[str, Any]:
    """加载 ``run_dir/best.pt``（或 ``last.pt``），在 **test** 划分上做 greedy + BLEU/chrF/COMET。"""
    rr = repo_root.resolve()
    if str(rr) not in sys.path:
        sys.path.insert(0, str(rr))

    from train_runtime import PATH_FIELDS_DEFAULT, git_commit_and_dirty, materialize_path_fields

    import mt_eval as mt_eval_mod

    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.is_file():
        alt = run_dir / "last.pt"
        if alt.is_file():
            ckpt_path = alt
            ckpt_name = "last.pt"
        else:
            raise FileNotFoundError(f"无 checkpoint: {run_dir / 'best.pt'}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg") or {}
    if not isinstance(cfg_dict, dict):
        raise ValueError("checkpoint 缺少 cfg 字典")

    Config, TabParallelDataset, collate_batch, load_tokenizers, tokenizer_special_ids, Seq2SeqTransformer = (
        import_training_stack(rr, pkg_name)
    )
    cfg = cfg_from_checkpoint_dict(cfg_dict, Config)
    materialize_path_fields(cfg, rr, PATH_FIELDS_DEFAULT)

    # test 终评：完整 chrF/BERTScore/COMET（不因训练阶段 --eval-light 而跳过）
    cfg.eval_use_chrf = True
    cfg.eval_use_bertscore = True
    cfg.eval_use_comet = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(getattr(cfg, "seed", 0)))

    src_tok, tgt_tok = load_tokenizers(cfg)
    pad_idx, bos_id, eos_id = tokenizer_special_ids(tgt_tok)

    test_ds = TabParallelDataset(cfg, src_tok, tgt_tok, "test")
    test_loader = DataLoader(
        test_ds,
        batch_size=min(cfg.batch_size, 256),
        shuffle=False,
        num_workers=min(getattr(cfg, "num_workers", 0), 4),
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_batch(b, pad_idx),
        persistent_workers=False,
    )

    model = Seq2SeqTransformer(cfg, pad_idx=pad_idx).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    opt_step = int(ckpt.get("optimizer_step", 0))
    test_diag: dict[str, Any] = {}
    bleu, extra = mt_eval_mod.evaluate_generation_corpus(
        model,
        test_loader,
        src_tok,
        tgt_tok,
        pad_idx,
        bos_id,
        eos_id,
        device,
        max_samples=bleu_sample_size,
        max_gen_len=cfg.max_gen_len,
        skip_identical_parallel=cfg.bleu_skip_identical_parallel,
        similarity_threshold=cfg.bleu_skip_similarity_threshold,
        optimizer_step=opt_step,
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
        out_diag=test_diag,
        predictions_jsonl_path=run_dir / "predictions.jsonl",
    )

    gc, gd = git_commit_and_dirty(rr)

    def jsonable(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        return str(obj)

    out = {
        "checkpoint_file": ckpt_name,
        "optimizer_step_at_checkpoint": opt_step,
        "bleu_at_checkpoint_selection": ckpt.get("bleu"),
        "test_bleu": bleu,
        "test_extra_metrics": jsonable(extra),
        "test_bleu_evaluation_meta": jsonable(test_diag),
        "git_commit": gc,
        "git_dirty": gd,
    }
    return out


def merge_ablation_metrics_json(
    run_dir: Path,
    test_eval: dict[str, Any],
) -> None:
    """将训练阶段 metrics.json（val 终评）备份，并写入合并后的 metrics.json（含 test 终评）。"""
    run_dir = run_dir.resolve()
    mp = run_dir / "metrics.json"
    train_metrics: dict[str, Any] = {}
    if mp.is_file():
        train_metrics = json.loads(mp.read_text(encoding="utf-8"))
        bak = run_dir / "metrics_after_train_val_split.json"
        bak.write_text(json.dumps(train_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    merged: dict[str, Any] = {
        "ablation_metrics_version": 1,
        "attention_type": train_metrics.get("attention_type"),
        "seed": train_metrics.get("seed"),
        "selection": {
            "checkpoint": test_eval.get("checkpoint_file", "best.pt"),
            "criterion": "max_corpus_bleu_on_val_during_training",
            "bleu_when_selected": test_eval.get("bleu_at_checkpoint_selection"),
            "training_best_bleu_tracker": train_metrics.get("best_bleu_during_training"),
        },
        "training_end_eval_on_val_split": {
            "final_val_loss": train_metrics.get("final_val_loss"),
            "final_bleu": train_metrics.get("final_bleu"),
            "extra_metrics": train_metrics.get("extra_metrics"),
            "bleu_evaluation": train_metrics.get("bleu_evaluation"),
        },
        "final_eval_on_test_split": {
            "final_bleu": test_eval.get("test_bleu"),
            "extra_metrics": test_eval.get("test_extra_metrics"),
            "bleu_evaluation": test_eval.get("test_bleu_evaluation_meta"),
            "predictions_jsonl": str(run_dir / "predictions.jsonl"),
        },
        "git_commit": test_eval.get("git_commit") or train_metrics.get("git_commit"),
        "git_dirty": test_eval.get("git_dirty") if test_eval.get("git_dirty") is not None else train_metrics.get("git_dirty"),
        "config_snapshot": train_metrics.get("config"),
        "resolved_paths_note": "paths in config_snapshot are post-train resolution; see resolved_config.json",
    }
    mp.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
