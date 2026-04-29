#!/usr/bin/env python3
"""在独立 test.tsv 上对 checkpoint 做最终评估（不做 identical/similarity 跳过）。

写出完整 predictions.jsonl（逐句 source/reference/hypothesis），供 `scripts/significance_test.py` 做配对 bootstrap。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mt_eval import (
    _bleu_score_and_signature,
    average_length_ratio,
    collect_predictions_no_skip_full,
    compute_extra_metrics,
    exact_match_rate,
)


def resolve_under_repo(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def count_test_file_anomalies(test_file: Path) -> dict[str, int]:
    """与 TabParallelDataset 相同的 Tab 行规则；记录未进入 DataLoader 的行。"""
    nonempty_two_col = 0
    malformed = 0
    empty_lines = 0
    line_count = 0
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            raw = line.rstrip("\n")
            if not raw.strip():
                empty_lines += 1
                continue
            parts = raw.split("\t")
            if len(parts) != 2:
                malformed += 1
            else:
                nonempty_two_col += 1
    return {
        "lines_total_read": line_count,
        "lines_empty_or_whitespace_only": empty_lines,
        "lines_nonempty_but_not_two_tab_columns": malformed,
        "lines_loaded_as_parallel_pairs": nonempty_two_col,
    }


def write_predictions_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_wandb_test_metrics(
    *,
    project: str,
    run_name: str,
    bleu: float,
    extra: dict,
    num_examples: int,
) -> None:
    """仅在 --wandb-eval 时调用；只上传标量 summary，不上传 predictions。"""
    try:
        import wandb
    except ImportError:
        print("wandb 未安装，跳过 W&B 上传（已传 --wandb-eval）", file=sys.stderr)
        return
    payload: dict[str, float | int] = {
        "test/bleu": float(bleu),
        "test/num_examples": int(num_examples),
    }
    chrf = extra.get("chrf")
    if chrf is not None:
        payload["test/chrf"] = float(chrf)
    comet = extra.get("comet")
    if comet is not None:
        payload["test/comet"] = float(comet)
    bf1 = extra.get("bertscore_f1")
    if bf1 is not None:
        payload["test/bertscore_f1"] = float(bf1)
    try:
        wandb.init(project=project, name=run_name, job_type="test_eval")
        try:
            wandb.log(payload)
        finally:
            wandb.finish()
    except Exception as e:
        print(f"W&B 上传失败（本地 metrics 已写出）: {e}", file=sys.stderr)


def write_examples_md(path: Path, rows: list[dict], *, limit: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Test-set decode samples",
        "",
        "| # | source | reference | hypothesis |",
        "|---|--------|-----------|------------|",
    ]
    for i, row in enumerate(rows[:limit]):
        se = row["source"].replace("|", "\\|").replace("\n", " ")
        re = row["reference"].replace("|", "\\|").replace("\n", " ")
        hy = row["hypothesis"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i + 1} | {se[:200]} | {re[:200]} | {hy[:200]} |")
    lines.append("")
    lines.append(f"_Showing first {min(limit, len(rows))} of {len(rows)} rows._")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Held-out test.tsv 最终评估（不写回训练）")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test-file", type=str, required=True, dest="test_file")
    p.add_argument("--tokenizer-src", type=str, required=True)
    p.add_argument("--tokenizer-tgt", type=str, required=True)
    p.add_argument("--attention-type", type=str, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--pkg",
        type=str,
        default="small_try",
        choices=("small_try", "small_head", "small_swap"),
        help="checkpoint 对应的代码目录（与 train 时一致）",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--wandb-eval",
        action="store_true",
        help="将 metrics_test.json 核心标量以 test/* 前缀上传到 W&B（默认关闭）",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project（默认：checkpoint cfg.project_name，否则 mt-test-eval）",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run 名称（默认：<checkpoint 文件名 stem>_test_eval）",
    )
    args = p.parse_args()

    ckpt_path = resolve_under_repo(args.checkpoint, _REPO_ROOT)
    test_file = resolve_under_repo(args.test_file, _REPO_ROOT)
    out_dir = resolve_under_repo(args.output_dir, _REPO_ROOT)

    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint 不存在: {ckpt_path}")
    if not test_file.is_file():
        raise SystemExit(f"test-file 不存在: {test_file}")

    pkg_root = _REPO_ROOT / args.pkg
    if not pkg_root.is_dir():
        raise SystemExit(f"找不到代码目录: {pkg_root}")
    pkg_insert = str(pkg_root)
    if pkg_insert not in sys.path:
        sys.path.insert(0, pkg_insert)

    from dataset import TabParallelDataset, collate_batch, load_tokenizers, tokenizer_special_ids
    from model import Seq2SeqTransformer

    anomaly_counts = count_test_file_anomalies(test_file)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = {k: v for k, v in ckpt["cfg"].items() if not str(k).startswith("_")}
    cfg = SimpleNamespace(**raw_cfg)

    tok_src = resolve_under_repo(args.tokenizer_src, _REPO_ROOT)
    tok_tgt = resolve_under_repo(args.tokenizer_tgt, _REPO_ROOT)
    cfg.tokenizer_src = str(tok_src)
    cfg.tokenizer_tgt = str(tok_tgt)

    if args.attention_type is not None:
        cfg.attention_type = args.attention_type
    if args.n_heads is not None:
        cfg.n_heads = args.n_heads

    sf = str(test_file)
    cfg.train_path = sf
    cfg.val_path = sf
    cfg.test_path = sf
    if getattr(cfg, "use_split_files", True) is not True:
        cfg.use_split_files = True
    cfg.data_path = sf

    cfg.max_gen_len = args.max_new_tokens

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    src_tok, tgt_tok = load_tokenizers(cfg)
    cfg.src_vocab_size = src_tok.get_vocab_size()
    cfg.tgt_vocab_size = tgt_tok.get_vocab_size()
    pad_idx, bos_id, eos_id = tokenizer_special_ids(tgt_tok)

    ds = TabParallelDataset(cfg, src_tok, tgt_tok, "test")
    loader = DataLoader(
        ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_batch(b, pad_idx),
    )

    model = Seq2SeqTransformer(cfg, pad_idx=pad_idx).to(device)
    sd = ckpt["model"]
    model.load_state_dict(sd, strict=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    hyps, refs, srcs, rows = collect_predictions_no_skip_full(
        model,
        loader,
        src_tok,
        tgt_tok,
        pad_idx,
        bos_id,
        eos_id,
        device,
        args.max_new_tokens,
    )

    bleu, bleu_sig = _bleu_score_and_signature(hyps, refs)
    extra = compute_extra_metrics(
        hyps,
        refs,
        srcs,
        optimizer_step=0,
        use_chrf=True,
        use_bertscore=True,
        use_comet=True,
        heavy_every=None,
        bertscore_lang=getattr(cfg, "eval_bertscore_lang", "fr"),
        bertscore_model_type=getattr(cfg, "eval_bertscore_model_type", None),
        bertscore_device=getattr(cfg, "eval_bertscore_device", None),
        comet_model=getattr(cfg, "eval_comet_model", "Unbabel/wmt22-comet-da"),
        comet_gpus=getattr(cfg, "eval_comet_gpus", None),
        force_heavy=True,
    )

    em = exact_match_rate(hyps, refs)
    alr = average_length_ratio(hyps, refs)

    filtering_note = (
        "Decoder 覆盖 test.tsv 中所有「非空且恰有两列 Tab」的句对；不对 identical_parallel / "
        "high_similarity 做跳过（与训练期 BLEU 子样本逻辑不同）。"
        " 空行或列数≠2 的行不进入评估，计数见 anomaly_counts_in_test_file。"
    )

    metrics: dict = {
        "BLEU": bleu,
        "chrF": extra.get("chrf"),
        "chrF++": extra.get("chrfpp"),
        "BERTScore": extra.get("bertscore_f1"),
        "COMET": extra.get("comet"),
        "exact_match_rate": em,
        "average_length_ratio": alr,
        "number_of_test_examples": len(hyps),
        "sacrebleu_signature": bleu_sig,
        "filtering_rules": filtering_note,
        "anomaly_counts_in_test_file": anomaly_counts,
        "checkpoint": str(ckpt_path),
        "test_file": str(test_file),
        "pkg": args.pkg,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "attention_type": cfg.attention_type,
        "n_heads": cfg.n_heads,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    met_path = out_dir / "metrics_test.json"
    ex_path = out_dir / "examples.md"

    write_predictions_jsonl(pred_path, rows)
    met_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_examples_md(ex_path, rows)

    print(f"Wrote {pred_path}", file=sys.stderr)
    print(f"Wrote {met_path}", file=sys.stderr)
    print(f"Wrote {ex_path}", file=sys.stderr)
    print(
        f"BLEU={bleu} chrF={extra.get('chrf')} examples={len(hyps)}",
        file=sys.stderr,
    )

    if args.wandb_eval:
        wb_project = args.wandb_project or getattr(cfg, "project_name", None) or "mt-test-eval"
        wb_name = args.wandb_run_name or f"{ckpt_path.stem}_test_eval"
        log_wandb_test_metrics(
            project=wb_project,
            run_name=wb_name,
            bleu=bleu,
            extra=extra,
            num_examples=len(hyps),
        )


if __name__ == "__main__":
    main()
