"""V9-A30: multi-seed aggregate engineering metrics from training_meta + test_eval."""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_summary():
    path = _ROOT / "scripts" / "summarize_attention_variants.py"
    spec = importlib.util.spec_from_file_location("summarize_attention_variants_a30", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_ablation_csv(tmp: Path, experiment: str) -> Path:
    p = tmp / "results" / "ablation_per_seed.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("experiment", "checkpoint_kind", "BLEU", "chrF", "chrF++", "COMET")
    rows = []
    for i in (1, 2, 3):
        rows.append(
            {
                "experiment": experiment,
                "checkpoint_kind": "best",
                "BLEU": str(10.0 + i * 0.1),
                "chrF": "30",
                "chrF++": "28",
                "COMET": "0.5",
            }
        )
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return p


def test_aggregate_row_uses_mean_not_representative(tmp_path: Path) -> None:
    mod = _load_summary()
    exp = "acmeexp"
    repo = tmp_path
    walls = [700.0, 730.0, 710.0]
    peaks = [1000.0, 1001.0, 1002.0]
    berts = [0.88, 0.881, 0.882]
    for i in (1, 2, 3):
        rn = f"{exp}_s{i}"
        base = repo / "runs" / rn
        (base / "test_eval").mkdir(parents=True, exist_ok=True)
        idx = i - 1
        (base / "training_meta.json").write_text(
            json.dumps({"wall_time_seconds": walls[idx], "peak_gpu_memory_mib": peaks[idx]}),
            encoding="utf-8",
        )
        (base / "test_eval" / "metrics_test.json").write_text(
            json.dumps({"BERTScore": berts[idx], "BLEU": 1.0, "chrF": 1.0, "chrF++": 1.0, "COMET": 0.5}),
            encoding="utf-8",
        )
    row: dict = {}
    mod.apply_aggregate_engineering_fields(row, repo, experiment=exp, n=3)
    wm, ws = statistics.mean(walls), statistics.stdev(walls)
    assert abs(float(row["train_time_seconds_mean"]) - wm) < 1e-9
    assert abs(float(row["train_time_seconds_std"]) - ws) < 1e-9
    assert row["train_time_seconds"].startswith(f"{wm:.2f}")
    out = mod.csv_export_row({**row, "is_aggregate": "true", "run": exp})
    assert out["train_time_seconds"] == round(wm, 2)
    assert out["train_time_seconds_mean"] == wm
    bm, bs = statistics.mean(berts), statistics.stdev(berts)
    assert abs(float(row["BERTScore_mean"]) - bm) < 1e-9
    assert abs(float(row["BERTScore_std"]) - bs) < 1e-9


def test_singleseed_row_has_no_std(tmp_path: Path) -> None:
    mod = _load_summary()
    row = {
        "train_time_seconds": 123.456,
        "peak_gpu_memory_mib": 9999.0,
        "BERTScore": 0.91,
    }
    mod.apply_single_seed_engineering_fields(row, failed=False)
    assert row["train_time_seconds_mean"] == 123.456
    assert row["train_time_seconds_std"] == ""
    assert str(row["train_time_seconds"]).endswith("(n=1)")


def test_aggregate_reads_from_training_meta_not_ablation_csv(tmp_path: Path) -> None:
    mod = _load_summary()
    exp = "wallonly"
    repo = tmp_path
    _fake_ablation_csv(repo, exp)
    for i, w in enumerate([100.0, 200.0, 300.0], start=1):
        rn = f"{exp}_s{i}"
        b = repo / "runs" / rn
        (b / "test_eval").mkdir(parents=True, exist_ok=True)
        (b / "training_meta.json").write_text(
            json.dumps({"wall_time_seconds": w, "peak_gpu_memory_mib": 1.0}),
            encoding="utf-8",
        )
        (b / "test_eval" / "metrics_test.json").write_text(
            json.dumps({"BERTScore": 0.5, "BLEU": 1.0, "chrF": 1.0, "chrF++": 1.0, "COMET": 0.5}),
            encoding="utf-8",
        )
    row: dict = {}
    mod.apply_aggregate_engineering_fields(row, repo, experiment=exp, n=3)
    assert row["train_time_seconds_mean"] == 200.0
    walls, _, _ = mod.load_per_seed_wall_peak_bert(repo, [f"{exp}_s1", f"{exp}_s2", f"{exp}_s3"])
    assert walls == [100.0, 200.0, 300.0]
