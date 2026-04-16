"""
共享机器翻译评估：BLEU（调用方或本模块）、chrF++、可选 BERTScore / COMET。
与各项目 train.py 中的 greedy 采样逻辑一致，并复用「跳过退化句对」规则。
"""

from __future__ import annotations

import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    pass

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

# Hugging Face：国内默认走镜像（仅当未预先 export 时生效；官方: export HF_ENDPOINT=https://huggingface.co）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Hub 首次拉 BERT/COMET 权重时默认 10s 易在弱网下超时，可适当加大（可被环境变量覆盖）
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
# 大文件经 HF XET（cas-bridge.xethub.hf.co）时弱网/镜像组合易 SSLEOF；默认走传统 Hub 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def should_skip_src_ref(
    src_str: str,
    ref_str: str,
    *,
    skip_identical: bool,
    similarity_threshold: float | None,
) -> tuple[bool, str]:
    s, r = src_str.strip(), ref_str.strip()
    if skip_identical and s == r:
        return True, "exact_strip"
    if similarity_threshold is None or not s or not r:
        return False, ""
    ratio = SequenceMatcher(None, s, r).ratio()
    if ratio >= similarity_threshold:
        return True, f"similarity_{ratio:.3f}"
    return False, ""


@torch.no_grad()
def collect_greedy_predictions(
    model: Any,
    loader: DataLoader,
    src_tok: Any,
    tgt_tok: Any,
    pad_idx: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    max_samples: int,
    max_gen_len: int,
    *,
    skip_identical_parallel: bool = True,
    similarity_threshold: float | None = 0.98,
    out_diag: dict | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """返回 (hyps, refs, srcs) 字符串列表，对齐同一条样本。"""
    model.eval()
    hyps: list[str] = []
    refs: list[str] = []
    srcs: list[str] = []
    n_collected = 0
    skipped_identical = 0
    skipped_similar = 0
    for src, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        for bi in range(src.size(0)):
            if n_collected >= max_samples:
                break
            src_ids = [t for t in src[bi].tolist() if t != pad_idx]
            ref_ids = [t for t in tgt[bi].tolist() if t not in (pad_idx,)]
            ref_ids_clean = [t for t in ref_ids if t not in (bos_id, eos_id)]
            ref_str = tgt_tok.decode(ref_ids_clean)
            src_str = src_tok.decode(src_ids)
            skip, reason = should_skip_src_ref(
                src_str,
                ref_str,
                skip_identical=skip_identical_parallel,
                similarity_threshold=similarity_threshold,
            )
            if skip:
                if reason.startswith("exact"):
                    skipped_identical += 1
                else:
                    skipped_similar += 1
                continue
            gen = model.greedy_decode(
                src[bi : bi + 1], bos_id, eos_id, max_gen_len
            )
            hyp_ids = gen[0].tolist()
            while hyp_ids and hyp_ids[0] == bos_id:
                hyp_ids.pop(0)
            while hyp_ids and hyp_ids[-1] == eos_id:
                hyp_ids.pop(-1)
            refs.append(ref_str)
            hyps.append(tgt_tok.decode(hyp_ids))
            srcs.append(src_str)
            n_collected += 1
        if n_collected >= max_samples:
            break
    model.train()
    if out_diag is not None:
        out_diag["bleu_skipped_identical_parallel"] = skipped_identical
        out_diag["bleu_skipped_high_similarity"] = skipped_similar
        out_diag["bleu_pairs_used"] = len(hyps)
        if hyps:
            exact = sum(1 for h, r in zip(hyps, refs) if h == r)
            out_diag["bleu_exact_string_match_rate"] = exact / max(1, len(hyps))
        else:
            out_diag["bleu_exact_string_match_rate"] = 0.0
    return hyps, refs, srcs


def score_corpus_bleu(hyps: list[str], refs: list[str]) -> float | None:
    if sacrebleu is None or not hyps:
        return None
    return float(sacrebleu.corpus_bleu(hyps, [[r] for r in refs]).score)


def score_corpus_chrf(hyps: list[str], refs: list[str]) -> float | None:
    if sacrebleu is None or not hyps:
        return None
    from sacrebleu.metrics import CHRF

    chrf = CHRF()
    return float(chrf.corpus_score(hyps, [[r] for r in refs]).score)


_comet_model = None
_comet_name_loaded: str | None = None


def _get_comet_predictor(model_name: str):
    """加载 COMET；下载或 checkpoint 不兼容时返回 None，不向调用方抛异常。"""
    global _comet_model, _comet_name_loaded
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print(
            "[mt_eval] COMET: 未安装 unbabel-comet，请执行 pip install unbabel-comet",
            file=sys.stderr,
        )
        return None
    if _comet_model is not None and _comet_name_loaded == model_name:
        return _comet_model
    try:
        path = download_model(model_name)
        model = load_from_checkpoint(path)
        _comet_model = model
        _comet_name_loaded = model_name
        return model
    except Exception as e:
        print(f"[mt_eval] COMET 模型加载失败 ({model_name}): {e}", file=sys.stderr)
        _comet_model = None
        _comet_name_loaded = None
        return None


def score_comet(
    srcs: list[str],
    hyps: list[str],
    refs: list[str],
    *,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int | None = None,
) -> float | None:
    """gpus: 传给 COMET predict（0=CPU）。

    若 gpus 为 None：优先读环境变量 COMET_GPUS；未设置则与旧版一致（CUDA 可用则用 1 张 GPU）。
    训练脚本与 seq2seq 同进程同卡时，应传入 gpus=0 或 export COMET_GPUS=0，避免争显存 OOM/假死。
    """
    if not hyps or len(srcs) != len(hyps) or len(refs) != len(hyps):
        return None
    model = _get_comet_predictor(model_name)
    if model is None:
        return None
    try:
        import torch as _torch

        if gpus is None:
            raw = os.environ.get("COMET_GPUS", "").strip()
            if raw == "":
                gpus = 1 if _torch.cuda.is_available() else 0
            else:
                gpus = int(raw)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = model.predict(
            data,
            batch_size=batch_size,
            gpus=gpus,
        )
        if isinstance(out, dict):
            scores = out.get("scores", out)
        else:
            scores = getattr(out, "scores", out)
        if hasattr(scores, "mean"):
            return float(scores.mean())
        if isinstance(scores, (list, tuple)):
            return float(sum(scores) / max(1, len(scores)))
        return float(scores)
    except Exception as e:
        print(f"[mt_eval] COMET predict 失败: {e}", file=sys.stderr)
        return None


def score_bertscore_f1(
    hyps: list[str],
    refs: list[str],
    *,
    lang: str = "fr",
    device: str | None = None,
    model_type: str | None = None,
) -> float | None:
    """若未传 model_type，则读环境变量 BERTSCORE_MODEL_TYPE；仍为空则用 bert-score 按 lang 的默认大模型（如 en 的 roberta-large ~1.4GB，首下较慢）。"""
    if not hyps:
        return None
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print(
            "[mt_eval] BERTScore: 未安装 bert-score，请执行 pip install bert-score",
            file=sys.stderr,
        )
        return None
    try:
        import torch as _torch

        d = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        mt = model_type or os.environ.get("BERTSCORE_MODEL_TYPE")
        if mt:
            _p, _r, f1 = bert_score_fn(
                hyps,
                refs,
                model_type=mt,
                lang=lang,
                verbose=False,
                device=d,
            )
        else:
            _p, _r, f1 = bert_score_fn(
                hyps,
                refs,
                lang=lang,
                verbose=False,
                device=d,
            )
        return float(f1.mean().item())
    except Exception as e:
        print(f"[mt_eval] BERTScore 失败: {e}", file=sys.stderr)
        return None


def compute_extra_metrics(
    hyps: list[str],
    refs: list[str],
    srcs: list[str],
    *,
    optimizer_step: int,
    use_chrf: bool = True,
    use_bertscore: bool = False,
    use_comet: bool = False,
    heavy_every: int | None = 2000,
    bertscore_lang: str = "fr",
    bertscore_model_type: str | None = None,
    bertscore_device: str | None = None,
    comet_model: str = "Unbabel/wmt22-comet-da",
    comet_gpus: int | None = None,
    force_heavy: bool = False,
) -> dict[str, float | None]:
    """
    heavy_every: 仅当 optimizer_step % heavy_every == 0 时计算 BERTScore/COMET；
    None 表示每次与 BLEU 同跑（慎用）。
    force_heavy: 为 True 时忽略步数间隔（用于训练结束时的最终评估）。
    """
    out: dict[str, float | None] = {}
    if use_chrf:
        out["chrf"] = score_corpus_chrf(hyps, refs)
    else:
        out["chrf"] = None

    run_heavy = force_heavy or heavy_every is None or (optimizer_step % heavy_every == 0)
    if use_bertscore and run_heavy:
        out["bertscore_f1"] = score_bertscore_f1(
            hyps,
            refs,
            lang=bertscore_lang,
            model_type=bertscore_model_type,
            device=bertscore_device,
        )
    else:
        out["bertscore_f1"] = None

    if use_comet and run_heavy and len(srcs) == len(hyps):
        out["comet"] = score_comet(
            srcs, hyps, refs, model_name=comet_model, gpus=comet_gpus
        )
    else:
        out["comet"] = None

    return out


def eval_sys_path_for_mt_eval(train_file: Path) -> None:
    """train.py 位于 */small_try/train.py、*/base_1/train.py、*/base_improve/train.py 等子目录时注入 autodl-tmp 根目录。"""
    root = train_file.resolve().parent.parent
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


@torch.no_grad()
def evaluate_generation_corpus(
    model: nn.Module,
    loader: DataLoader,
    src_tok: Any,
    tgt_tok: Any,
    pad_idx: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    *,
    max_samples: int,
    max_gen_len: int,
    skip_identical_parallel: bool,
    similarity_threshold: float | None,
    optimizer_step: int,
    use_chrf: bool,
    use_bertscore: bool,
    use_comet: bool,
    heavy_every: int | None,
    bertscore_lang: str,
    bertscore_model_type: str | None = None,
    bertscore_device: str | None = None,
    comet_model: str = "Unbabel/wmt22-comet-da",
    comet_gpus: int | None = None,
    force_heavy: bool = False,
    out_diag: dict | None = None,
) -> tuple[float | None, dict[str, float | None]]:
    """
    一次 greedy 采样后计算 BLEU + chrF++/BERTScore/COMET。
    返回 (bleu, extra)；extra 含 chrf、bertscore_f1、comet（未算则为 None）。
    """
    hyps, refs, srcs = collect_greedy_predictions(
        model,
        loader,
        src_tok,
        tgt_tok,
        pad_idx,
        bos_id,
        eos_id,
        device,
        max_samples,
        max_gen_len,
        skip_identical_parallel=skip_identical_parallel,
        similarity_threshold=similarity_threshold,
        out_diag=out_diag,
    )
    bleu = score_corpus_bleu(hyps, refs)
    extra = compute_extra_metrics(
        hyps,
        refs,
        srcs,
        optimizer_step=optimizer_step,
        use_chrf=use_chrf,
        use_bertscore=use_bertscore,
        use_comet=use_comet,
        heavy_every=heavy_every,
        bertscore_lang=bertscore_lang,
        bertscore_model_type=bertscore_model_type,
        bertscore_device=bertscore_device,
        comet_model=comet_model,
        comet_gpus=comet_gpus,
        force_heavy=force_heavy,
    )
    return bleu, extra


def flatten_extra_for_log(
    extra: dict[str, float | None], prefix: str = "metrics"
) -> dict[str, float]:
    """供 W&B / 打印：去掉 None。"""
    return {f"{prefix}/{k}": float(v) for k, v in extra.items() if v is not None}
