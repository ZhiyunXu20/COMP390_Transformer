"""
共享机器翻译评估：BLEU（SacreBLEU）、chrF / chrF++、可选 BERTScore / COMET。
与各项目 train.py 中的 greedy 采样逻辑一致，并复用「跳过退化句对」规则。
"""

from __future__ import annotations

import json
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
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    """返回 (hyps, refs, srcs, prediction_records)。

    prediction_records 每项含 id、src、ref、hyp；跳过解码时用 hyp=""，skipped_reason 为非空字符串。
    """
    model.eval()
    hyps: list[str] = []
    refs: list[str] = []
    srcs: list[str] = []
    prediction_records: list[dict[str, Any]] = []
    row_id = -1
    n_collected = 0
    skipped_identical = 0
    skipped_similar = 0
    for src, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        for bi in range(src.size(0)):
            if n_collected >= max_samples:
                break
            row_id += 1
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
                prediction_records.append(
                    {
                        "id": row_id,
                        "src": src_str,
                        "ref": ref_str,
                        "hyp": "",
                        "skipped_reason": reason,
                    }
                )
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
            hyp_str = tgt_tok.decode(hyp_ids)
            hyps.append(hyp_str)
            srcs.append(src_str)
            prediction_records.append(
                {
                    "id": row_id,
                    "src": src_str,
                    "ref": ref_str,
                    "hyp": hyp_str,
                    "skipped_reason": "",
                }
            )
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
    return hyps, refs, srcs, prediction_records


def _signature_to_jsonable(sig: Any) -> dict[str, Any]:
    """将 SacreBLEU Signature 转为可 JSON 序列化的字典（含 format 与各分项）。"""
    fmt = sig.format()
    raw = getattr(sig, "info", {})
    info_out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            info_out[str(k)] = v
        else:
            info_out[str(k)] = str(v)
    return {"format": fmt, "info": info_out}


def _bleu_score_and_signature(
    hyps: list[str], refs: list[str]
) -> tuple[float | None, dict[str, Any] | None]:
    """single-reference corpus BLEU，并附带 SacreBLEU signature。"""
    if sacrebleu is None or not hyps:
        return None, None
    from sacrebleu.metrics import BLEU

    metric = BLEU()
    cs = metric.corpus_score(hyps, [refs])
    sig = _signature_to_jsonable(metric.get_signature())
    return float(cs.score), sig


def score_corpus_bleu(hyps: list[str], refs: list[str]) -> float | None:
    """Corpus BLEU（单参考）：``sacrebleu.corpus_bleu(hyps, [refs])``。"""
    if sacrebleu is None or not hyps:
        return None
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def score_corpus_chrf(hyps: list[str], refs: list[str]) -> float | None:
    """chrF（字符级；word_order=0），非 chrF++。"""
    if sacrebleu is None or not hyps:
        return None
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=0)
    return float(metric.corpus_score(hyps, [refs]).score)


def score_corpus_chrfpp(hyps: list[str], refs: list[str]) -> float | None:
    """chrF++（word_order=2）。"""
    if sacrebleu is None or not hyps:
        return None
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=2)
    return float(metric.corpus_score(hyps, [refs]).score)


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


def _chrf_score_signature(
    word_order: int, hyps: list[str], refs: list[str]
) -> tuple[float, dict[str, Any]]:
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=word_order)
    cs = metric.corpus_score(hyps, [refs])
    return float(cs.score), _signature_to_jsonable(metric.get_signature())


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
) -> dict[str, Any]:
    """
    heavy_every: 仅当 optimizer_step % heavy_every == 0 时计算 BERTScore/COMET；
    None 表示每次与 BLEU 同跑（慎用）。
    force_heavy: 为 True 时忽略步数间隔（用于训练结束时的最终评估）。
    """
    out: dict[str, Any] = {}
    if use_chrf and hyps:
        s0, sig0 = _chrf_score_signature(0, hyps, refs)
        s2, sig2 = _chrf_score_signature(2, hyps, refs)
        out["chrf"] = s0
        out["chrfpp"] = s2
        out["chrf_signature"] = sig0
        out["chrfpp_signature"] = sig2
    elif use_chrf:
        out["chrf"] = None
        out["chrfpp"] = None
        out["chrf_signature"] = None
        out["chrfpp_signature"] = None
    else:
        out["chrf"] = None
        out["chrfpp"] = None
        out["chrf_signature"] = None
        out["chrfpp_signature"] = None

    run_heavy = force_heavy or heavy_every is None or (optimizer_step % heavy_every == 0)
    out["bertscore_status"] = "not_run"
    out["bertscore_error"] = None
    out["bertscore_f1"] = None

    if use_bertscore and run_heavy:
        try:
            bs = score_bertscore_f1(
                hyps,
                refs,
                lang=bertscore_lang,
                model_type=bertscore_model_type,
                device=bertscore_device,
            )
            if bs is not None:
                out["bertscore_f1"] = bs
                out["bertscore_status"] = "ok"
            else:
                out["bertscore_status"] = "failed"
                out["bertscore_error"] = (
                    "BERTScore returned None (missing dependency, bad env, or scoring failure; see stderr)."
                )
        except Exception as e:
            out["bertscore_status"] = "failed"
            out["bertscore_error"] = str(e)
    elif use_bertscore:
        out["bertscore_status"] = "not_run"

    if use_comet and run_heavy and len(srcs) == len(hyps):
        out["comet"] = score_comet(
            srcs, hyps, refs, model_name=comet_model, gpus=comet_gpus
        )
    else:
        out["comet"] = None

    return out


def _write_predictions_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            row = dict(rec)
            sr = row.get("skipped_reason")
            if sr == "":
                row["skipped_reason"] = None
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    predictions_jsonl_path: Path | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """
    一次 greedy 采样后计算 BLEU、chrF、chrF++、可选 BERTScore/COMET。
    返回 (bleu, extra)；extra 含 chrf、chrfpp、签名、bertscore_status 等。

    ``force_heavy=True``（训练结束终评）时写出 ``predictions_jsonl_path``
    （默认当前工作目录 ``predictions.jsonl``）：每行 JSON 含 id、src、ref、hyp、skipped_reason。
    """
    hyps, refs, srcs, prediction_records = collect_greedy_predictions(
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
    bleu, bleu_sig = _bleu_score_and_signature(hyps, refs)
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
    extra["bleu_signature"] = bleu_sig
    if force_heavy:
        out_p = predictions_jsonl_path or Path("predictions.jsonl")
        _write_predictions_jsonl(out_p, prediction_records)
    return bleu, extra


@torch.no_grad()
def collect_predictions_no_skip_full(
    model: nn.Module,
    loader: DataLoader,
    src_tok: Any,
    tgt_tok: Any,
    pad_idx: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    max_gen_len: int,
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    """测试集终评：对 loader 中每个样本做 greedy 解码，**不**应用 identical/similarity 跳过逻辑。"""
    model.eval()
    hyps: list[str] = []
    refs: list[str] = []
    srcs: list[str] = []
    rows: list[dict[str, Any]] = []
    for src, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        for bi in range(src.size(0)):
            src_ids = [t for t in src[bi].tolist() if t != pad_idx]
            ref_ids = [t for t in tgt[bi].tolist() if t not in (pad_idx,)]
            ref_ids_clean = [t for t in ref_ids if t not in (bos_id, eos_id)]
            ref_str = tgt_tok.decode(ref_ids_clean)
            src_str = src_tok.decode(src_ids)
            gen = model.greedy_decode(src[bi : bi + 1], bos_id, eos_id, max_gen_len)
            hyp_ids = gen[0].tolist()
            while hyp_ids and hyp_ids[0] == bos_id:
                hyp_ids.pop(0)
            while hyp_ids and hyp_ids[-1] == eos_id:
                hyp_ids.pop(-1)
            hyp_str = tgt_tok.decode(hyp_ids)
            hyps.append(hyp_str)
            refs.append(ref_str)
            srcs.append(src_str)
            rows.append(
                {
                    "source": src_str,
                    "reference": ref_str,
                    "hypothesis": hyp_str,
                    "source_length": len(src_str),
                    "reference_length": len(ref_str),
                    "hypothesis_length": len(hyp_str),
                }
            )
    model.train()
    return hyps, refs, srcs, rows


def exact_match_rate(hyps: list[str], refs: list[str]) -> float | None:
    if not hyps:
        return None
    return sum(1 for h, r in zip(hyps, refs) if h == r) / max(1, len(hyps))


def average_length_ratio(hyps: list[str], refs: list[str]) -> float | None:
    """mean(|hyp| / max(|ref|, 1))，按 UTF-8 字符长度。"""
    if not hyps:
        return None
    ratios: list[float] = []
    for h, r in zip(hyps, refs):
        lr = max(1, len(r))
        ratios.append(len(h) / lr)
    return sum(ratios) / max(1, len(ratios))


def flatten_extra_for_log(
    extra: dict[str, Any], prefix: str = "metrics"
) -> dict[str, Any]:
    """供 W&B：记录标量指标与 bertscore_status 等字符串；*_signature 嵌套 dict 不展开。"""
    out: dict[str, Any] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, dict) and k.endswith("_signature"):
            continue
        if isinstance(v, (int, float)):
            out[f"{prefix}/{k}"] = float(v)
        elif isinstance(v, str):
            out[f"{prefix}/{k}"] = v
        elif isinstance(v, bool):
            out[f"{prefix}/{k}"] = v
    return out
