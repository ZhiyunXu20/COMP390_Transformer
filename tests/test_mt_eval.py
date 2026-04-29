"""mt_eval：SacreBLEU 参考格式、chrF/chrF++ 区分、predictions.jsonl 格式。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt_eval import (  # noqa: E402
    _write_predictions_jsonl,
    score_corpus_bleu,
    score_corpus_chrf,
    score_corpus_chrfpp,
)
from sacrebleu.metrics import BLEU  # noqa: E402


def test_score_corpus_bleu_matches_bleu_corpus_score() -> None:
    hyps = ["a b c", "d e f", "g h"]
    refs = ["a b d", "d e x", "g h"]
    s_bleu_fn = score_corpus_bleu(hyps, refs)
    s_metric = float(BLEU().corpus_score(hyps, [refs]).score)
    assert s_bleu_fn is not None
    assert abs(s_bleu_fn - s_metric) < 1e-9


def test_chrf_and_chrfpp_differ() -> None:
    hyps = ["hello world", "foo bar", "baz qux"]
    refs = ["hello there", "foo baz", "baz foo"]
    f = score_corpus_chrf(hyps, refs)
    pp = score_corpus_chrfpp(hyps, refs)
    assert f is not None and pp is not None
    assert f != pp


def test_predictions_jsonl_schema() -> None:
    records = [
        {"id": 0, "src": "a", "ref": "b", "hyp": "c", "skipped_reason": ""},
        {"id": 1, "src": "x", "ref": "y", "hyp": "", "skipped_reason": "exact_strip"},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "predictions.jsonl"
        _write_predictions_jsonl(p, records)
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        o0 = json.loads(lines[0])
        assert set(o0.keys()) >= {"id", "src", "ref", "hyp", "skipped_reason"}
        assert o0["hyp"] == "c"
        assert o0["skipped_reason"] is None
        o1 = json.loads(lines[1])
        assert o1["skipped_reason"] == "exact_strip"


def test_bleu_signature_present_when_scoring() -> None:
    """BLEU 分数路径应对齐 corpus_bleu 与 signature（由 evaluate 组装）。"""
    from mt_eval import _bleu_score_and_signature

    hyps = ["a", "b"]
    refs = ["a", "c"]
    score, sig = _bleu_score_and_signature(hyps, refs)
    assert score is not None and sig is not None
    assert "format" in sig and "info" in sig
    assert "tok" in sig["info"] or "version" in sig["info"]
