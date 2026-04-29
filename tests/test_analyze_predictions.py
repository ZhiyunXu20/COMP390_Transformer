"""analyze_predictions.py：合成 predictions.jsonl 上的聚合统计（跳过 COMET）。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_analyze_predictions():
    path = ROOT / "scripts" / "analyze_predictions.py"
    spec = importlib.util.spec_from_file_location("analyze_predictions_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ap = _load_analyze_predictions()


def test_bucket_exact_copy_rates_sort_bleu() -> None:
    rows = [
        {"id": 0, "src": "a", "ref": "hello world here", "hyp": "hello world here", "skipped_reason": None},
        {"id": 1, "src": "b", "ref": "x", "hyp": "y", "skipped_reason": None},
        {"id": 2, "src": "y", "ref": "y", "hyp": "", "skipped_reason": "exact_strip"},
        {"id": 3, "src": "same", "ref": "ref text here ok", "hyp": "same", "skipped_reason": None},
    ]
    payload = ap.analyze_rows(
        rows,
        bucket_mode="words",
        word_edges=[2, 4],
        char_edges=[10, 20],
        top_k=2,
        sort_by="bleu",
        run_comet_buckets=False,
        run_comet_segments=False,
        comet_model="Unbabel/wmt22-comet-da",
        comet_batch_size=4,
        comet_gpus=0,
    )
    assert payload["n_lines_total"] == 4
    assert payload["n_skipped"] == 1
    assert payload["n_nonempty_hyp"] == 3
    assert payload["exact_match_rate_among_nonempty"] > 0
    assert payload["copy_rate_hyp_equals_src_among_nonempty"] == 1 / 3
    assert payload["ranking_sort_primary"] == "bleu"
    assert len(payload["bad_cases"]) <= 2


def test_writes_outputs(tmp_path: Path) -> None:
    pred = tmp_path / "p.jsonl"
    lines = [
        {"id": i, "src": "s", "ref": "ref " * (i + 1), "hyp": "hyp " * (i + 1), "skipped_reason": None}
        for i in range(5)
    ]
    pred.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    old = sys.argv[:]
    try:
        sys.argv = [
            "analyze_predictions.py",
            "--predictions",
            str(pred),
            "--output-dir",
            str(out),
            "--sort-by",
            "bleu",
            "--no-bucket-comet",
            "--no-segment-comet",
            "--top-k",
            "3",
        ]
        ap.main()
    finally:
        sys.argv = old
    assert (out / "prediction_analysis.json").is_file()
    assert (out / "prediction_analysis.md").is_file()
