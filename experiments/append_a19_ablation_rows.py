#!/usr/bin/env python3
"""Append fast_add_lr3e3_s{1,2,3} rows to results/ablation_per_seed.csv (idempotent by run_name)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "results" / "ablation_per_seed.csv"
CHRPP = "chrF++"


def main() -> int:
    new_rows: list[dict[str, str]] = []
    for seed in (1, 2, 3):
        run = f"fast_add_lr3e3_s{seed}"
        rd = REPO / "runs" / run
        mt = rd / "test_eval" / "metrics_test.json"
        tm_path = rd / "training_meta.json"
        ck = rd / "best.pt"
        if not mt.is_file():
            print(f"missing {mt}", file=sys.stderr)
            return 1
        m = json.loads(mt.read_text(encoding="utf-8"))
        tm: dict = {}
        if tm_path.is_file():
            tm = json.loads(tm_path.read_text(encoding="utf-8"))
        new_rows.append(
            {
                "run_name": run,
                "experiment": "fast_add_lr3e3",
                "attention_type": "additive",
                "n_heads": "4",
                "seed": str(seed),
                "checkpoint_kind": "best",
                "checkpoint_path": str(ck.resolve()),
                "BLEU": str(m["BLEU"]),
                "chrF": str(m["chrF"]),
                CHRPP: str(m[CHRPP]),
                "COMET": str(m["COMET"]),
                "wall_time_seconds": str(tm.get("wall_time_seconds", "")),
                "peak_gpu_memory_mib": str(tm.get("peak_gpu_memory_mib", "")),
                "num_parameters": str(tm.get("num_parameters", "")),
            }
        )

    existing = CSV_PATH.read_text(encoding="utf-8").splitlines()
    reader = csv.DictReader(existing)
    fieldnames = reader.fieldnames
    if not fieldnames:
        print("bad csv header", file=sys.stderr)
        return 1
    seen = {r["run_name"] for r in reader}
    to_add = [r for r in new_rows if r["run_name"] not in seen]
    if not to_add:
        print("A19 rows already present; skip append", file=sys.stderr)
        return 0

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in to_add:
            w.writerow(r)
    print(f"Appended {len(to_add)} rows to {CSV_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
