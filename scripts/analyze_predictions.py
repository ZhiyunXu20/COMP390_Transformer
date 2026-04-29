#!/usr/bin/env python3
"""分析 predictions.jsonl：分桶指标、精确匹配/复制/空输出率、好坏句样本。

用法：
  python scripts/analyze_predictions.py --predictions runs/foo/predictions.jsonl --output-dir runs/foo/pred_analysis

bad cases 默认按 COMET（逐句）升序取最差 20 条；COMET 不可用或 ``--sort-by bleu`` 时改用 sentence BLEU。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalize_skip(rec: dict[str, Any]) -> str:
    sr = rec.get("skipped_reason")
    if sr is None:
        return ""
    return str(sr).strip()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def bucket_label_words(ref_words: int, edges: list[int]) -> str:
    """edges 递增边界，如 [10,25,50] → '<10','10-24','25-49','50+'。"""
    if ref_words < edges[0]:
        return f"<{edges[0]}"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= ref_words < hi:
            return f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def bucket_label_chars(ref_chars: int, edges: list[int]) -> str:
    if ref_chars < edges[0]:
        return f"<{edges[0]}"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= ref_chars < hi:
            return f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def ref_bucket(rec: dict[str, Any], *, mode: str, word_edges: list[int], char_edges: list[int]) -> str:
    ref = rec.get("ref") or ""
    if mode == "chars":
        return bucket_label_chars(len(ref), char_edges)
    words = ref.split()
    return bucket_label_words(len(words), word_edges)


def corpus_bleu_chrf_comet(
    hyps: list[str],
    refs: list[str],
    srcs: list[str],
    *,
    run_comet: bool,
    comet_model: str,
    comet_gpus: int | None,
) -> dict[str, Any]:
    from mt_eval import score_comet  # noqa: PLC0415

    out: dict[str, Any] = {}
    try:
        from mt_eval import score_corpus_bleu, score_corpus_chrf, score_corpus_chrfpp  # noqa: PLC0415
    except ImportError:
        score_corpus_bleu = score_corpus_chrf = score_corpus_chrfpp = None  # type: ignore[misc, assignment]

    out["bleu"] = score_corpus_bleu(hyps, refs) if score_corpus_bleu else None  # type: ignore[misc]
    out["chrf"] = score_corpus_chrf(hyps, refs) if score_corpus_chrf else None  # type: ignore[misc]
    out["chrfpp"] = score_corpus_chrfpp(hyps, refs) if score_corpus_chrfpp else None  # type: ignore[misc]
    if run_comet and len(srcs) == len(hyps):
        out["comet"] = score_comet(srcs, hyps, refs, model_name=comet_model, gpus=comet_gpus)
    else:
        out["comet"] = None
    return out


def sentence_bleu_score(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import BLEU  # noqa: PLC0415

    return float(BLEU(effective_order=True).sentence_score(hyp, [ref]).score)


def comet_segment_scores(
    srcs: list[str],
    hyps: list[str],
    refs: list[str],
    *,
    model_name: str,
    batch_size: int,
    gpus: int | None,
) -> list[float | None]:
    from mt_eval import _get_comet_predictor  # noqa: PLC0415

    model = _get_comet_predictor(model_name)
    if model is None:
        return [None] * len(hyps)
    import torch as _torch  # noqa: PLC0415

    if gpus is None:
        raw = __import__("os").environ.get("COMET_GPUS", "").strip()
        if raw == "":
            gpus = 1 if _torch.cuda.is_available() else 0
        else:
            gpus = int(raw)
    out_scores: list[float | None] = []
    data_all = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    for i in range(0, len(data_all), batch_size):
        chunk = data_all[i : i + batch_size]
        try:
            pred = model.predict(chunk, batch_size=batch_size, gpus=gpus)
            if isinstance(pred, dict):
                sc = pred.get("scores", [])
            else:
                sc = getattr(pred, "scores", [])
            if hasattr(sc, "tolist"):
                sc = sc.tolist()
            for j, v in enumerate(sc):
                out_scores.append(float(v))
            # padding mismatch guard
            if len(sc) != len(chunk):
                while len(out_scores) < i + len(chunk):
                    out_scores.append(None)
        except Exception as e:
            print(f"[analyze_predictions] COMET segment batch failed: {e}", file=sys.stderr)
            out_scores.extend([None] * len(chunk))
    while len(out_scores) < len(hyps):
        out_scores.append(None)
    return out_scores[: len(hyps)]


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    bucket_mode: str,
    word_edges: list[int],
    char_edges: list[int],
    top_k: int,
    sort_by: str,
    run_comet_buckets: bool,
    run_comet_segments: bool,
    comet_model: str,
    comet_batch_size: int,
    comet_gpus: int | None,
) -> dict[str, Any]:
    evaluated = [r for r in rows if not _normalize_skip(r)]
    skipped_n = len(rows) - len(evaluated)

    empty_hyps = [r for r in evaluated if not str(r.get("hyp", "")).strip()]
    non_empty = [r for r in evaluated if str(r.get("hyp", "")).strip()]

    exact = sum(
        1
        for r in non_empty
        if str(r.get("hyp", "")).strip() == str(r.get("ref", "")).strip()
    )
    copy_n = sum(
        1
        for r in non_empty
        if str(r.get("hyp", "")).strip() == str(r.get("src", "")).strip()
    )

    hyp_lens = [len(str(r.get("hyp", ""))) for r in non_empty]
    ref_lens = [len(str(r.get("ref", ""))) for r in non_empty]
    ratios = [h / max(1, r) for h, r in zip(hyp_lens, ref_lens)]
    mean_ratio = sum(ratios) / max(1, len(ratios))

    buckets: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(evaluated):
        key = ref_bucket(r, mode=bucket_mode, word_edges=word_edges, char_edges=char_edges)
        buckets[key].append(i)

    bucket_metrics: dict[str, Any] = {}
    for bname, idxs in sorted(buckets.items()):
        sub = [evaluated[i] for i in idxs]
        hyps = [str(x.get("hyp", "")) for x in sub]
        refs = [str(x.get("ref", "")) for x in sub]
        srcs = [str(x.get("src", "")) for x in sub]
        usable = [
            (s, h, r)
            for s, h, r in zip(srcs, hyps, refs)
            if h.strip()
        ]
        if not usable:
            bucket_metrics[bname] = {
                "n": len(sub),
                "n_decoded_nonempty": 0,
                "bleu": None,
                "chrf": None,
                "chrfpp": None,
                "comet": None,
            }
            continue
        su, hu, ru = zip(*usable)
        m = corpus_bleu_chrf_comet(
            list(hu),
            list(ru),
            list(su),
            run_comet=run_comet_buckets,
            comet_model=comet_model,
            comet_gpus=comet_gpus,
        )
        bucket_metrics[bname] = {
            "n": len(sub),
            "n_decoded_nonempty": len(usable),
            **m,
        }

    # per-row scores for ranking (non-empty hypotheses only)
    srcs_ne = [str(r.get("src", "")) for r in non_empty]
    hyps_ne = [str(r.get("hyp", "")) for r in non_empty]
    refs_ne = [str(r.get("ref", "")) for r in non_empty]

    sbleu_list = [sentence_bleu_score(h, r) for h, r in zip(hyps_ne, refs_ne)]
    comet_list: list[float | None] = [None] * len(non_empty)
    if run_comet_segments:
        comet_list = comet_segment_scores(
            srcs_ne,
            hyps_ne,
            refs_ne,
            model_name=comet_model,
            batch_size=comet_batch_size,
            gpus=comet_gpus,
        )

    ranked_meta = []
    for k, r in enumerate(non_empty):
        ranked_meta.append(
            {
                "id": r.get("id"),
                "src": r.get("src"),
                "ref": r.get("ref"),
                "hyp": r.get("hyp"),
                "sentence_bleu": sbleu_list[k],
                "comet_segment": comet_list[k],
            }
        )

    ranking_primary = sort_by
    use_comet_rank = (
        sort_by == "comet"
        and run_comet_segments
        and any(x.get("comet_segment") is not None for x in ranked_meta)
    )
    if sort_by == "comet" and not use_comet_rank:
        ranking_primary = "bleu"
        print(
            "[analyze_predictions] COMET 逐句不可用或未启用，bad/good 按 sentence BLEU 排序。",
            file=sys.stderr,
        )

    def bad_key(item: dict[str, Any]) -> float:
        sb = float(item["sentence_bleu"])
        c = item.get("comet_segment")
        if use_comet_rank and c is not None:
            return float(c)
        return sb

    def good_key(item: dict[str, Any]) -> float:
        sb = float(item["sentence_bleu"])
        c = item.get("comet_segment")
        if use_comet_rank and c is not None:
            return float(c)
        return sb

    ranked_bad = sorted(ranked_meta, key=bad_key)[:top_k]
    ranked_good = sorted(ranked_meta, key=good_key, reverse=True)[:top_k]

    summary = {
        "n_lines_total": len(rows),
        "n_evaluated_not_skipped": len(evaluated),
        "n_skipped": skipped_n,
        "n_nonempty_hyp": len(non_empty),
        "empty_hyp_rate_among_evaluated": len(empty_hyps) / max(1, len(evaluated)),
        "exact_match_rate_among_nonempty": exact / max(1, len(non_empty)),
        "copy_rate_hyp_equals_src_among_nonempty": copy_n / max(1, len(non_empty)),
        "mean_hyp_over_ref_length_ratio_chars": mean_ratio,
        "bucket_mode": bucket_mode,
        "bucket_edges_words": word_edges,
        "bucket_edges_chars": char_edges,
        "bucket_metrics": bucket_metrics,
        "bad_cases": ranked_bad,
        "good_cases": ranked_good,
        "ranking_sort_primary": ranking_primary,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="分析 predictions.jsonl")
    p.add_argument("--predictions", type=Path, required=True, help="predictions.jsonl 路径")
    p.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    p.add_argument(
        "--bucket-mode",
        choices=("words", "chars"),
        default="words",
        help="长度分桶依据（词数或字符数）",
    )
    p.add_argument(
        "--word-edges",
        type=int,
        nargs="+",
        default=[10, 25, 50],
        help="词数分桶边界（递增），默认 10 25 50 → <10,10-24,...",
    )
    p.add_argument(
        "--char-edges",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="字符分桶边界（递增）",
    )
    p.add_argument("--top-k", type=int, default=20, help="bad/good 各取多少条")
    p.add_argument(
        "--sort-by",
        choices=("comet", "bleu"),
        default="comet",
        help="bad/good 主排序键（COMET 失败时自动退回 BLEU）",
    )
    p.add_argument(
        "--no-bucket-comet",
        action="store_true",
        help="分桶内不计算 COMET（仅靠 BLEU/chrF，更快）",
    )
    p.add_argument(
        "--no-segment-comet",
        action="store_true",
        help="不为每条样本算 COMET（bad/good 仅用 sentence BLEU）",
    )
    p.add_argument("--comet-model", type=str, default="Unbabel/wmt22-comet-da")
    p.add_argument("--comet-batch-size", type=int, default=8)
    p.add_argument(
        "--comet-gpus",
        type=int,
        default=None,
        help="COMET predict gpus（默认继承 mt_eval/COMET_GPUS）",
    )
    args = p.parse_args()

    rows = load_prediction_rows(args.predictions.resolve())
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_comet_seg = not args.no_segment_comet
    run_comet_bucket = not args.no_bucket_comet

    payload = analyze_rows(
        rows,
        bucket_mode=args.bucket_mode,
        word_edges=sorted(args.word_edges),
        char_edges=sorted(args.char_edges),
        top_k=args.top_k,
        sort_by=args.sort_by,
        run_comet_buckets=run_comet_bucket,
        run_comet_segments=run_comet_seg,
        comet_model=args.comet_model,
        comet_batch_size=args.comet_batch_size,
        comet_gpus=args.comet_gpus,
    )

    json_path = out_dir / "prediction_analysis.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Prediction analysis",
        "",
        f"- Source: `{args.predictions}`",
        f"- Lines total: **{payload['n_lines_total']}**, evaluated: **{payload['n_evaluated_not_skipped']}**, skipped: **{payload['n_skipped']}**",
        f"- Non-empty hyp: **{payload['n_nonempty_hyp']}**",
        f"- Empty hyp rate (among evaluated): **{payload['empty_hyp_rate_among_evaluated']:.4f}**",
        f"- Exact match rate (non-empty): **{payload['exact_match_rate_among_nonempty']:.4f}**",
        f"- Copy rate hyp==src (non-empty): **{payload['copy_rate_hyp_equals_src_among_nonempty']:.4f}**",
        f"- Mean |hyp|/|ref| (chars): **{payload['mean_hyp_over_ref_length_ratio_chars']:.4f}**",
        "",
        "## Bucket metrics",
        "",
        "| bucket | n | nonempty | BLEU | chrF | chrF++ | COMET |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bname in sorted(payload["bucket_metrics"].keys()):
        bm = payload["bucket_metrics"][bname]
        md_lines.append(
            f"| {bname} | {bm.get('n')} | {bm.get('n_decoded_nonempty')} | "
            f"{_fmt(bm.get('bleu'))} | {_fmt(bm.get('chrf'))} | {_fmt(bm.get('chrfpp'))} | {_fmt(bm.get('comet'))} |"
        )
    md_lines.extend(["", "## Bad cases", ""])
    for i, rec in enumerate(payload["bad_cases"], 1):
        md_lines.append(f"### {i}. id={rec.get('id')}")
        md_lines.append(f"- sentence_bleu={rec.get('sentence_bleu'):.4f}, comet={rec.get('comet_segment')}")
        md_lines.append(f"- src: {rec.get('src')}")
        md_lines.append(f"- ref: {rec.get('ref')}")
        md_lines.append(f"- hyp: {rec.get('hyp')}")
        md_lines.append("")
    md_lines.extend(["## Good cases", ""])
    for i, rec in enumerate(payload["good_cases"], 1):
        md_lines.append(f"### {i}. id={rec.get('id')}")
        md_lines.append(f"- sentence_bleu={rec.get('sentence_bleu'):.4f}, comet={rec.get('comet_segment')}")
        md_lines.append(f"- src: {rec.get('src')}")
        md_lines.append(f"- ref: {rec.get('ref')}")
        md_lines.append(f"- hyp: {rec.get('hyp')}")
        md_lines.append("")

    md_path = out_dir / "prediction_analysis.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


if __name__ == "__main__":
    main()
