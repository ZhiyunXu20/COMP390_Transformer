#!/usr/bin/env python3
"""Create conservative training_meta.json from resolved_config + parameter count only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RUNS = ("head_1h_dot", "swap_fr_dot")


def infer_pkg(run_name: str) -> str:
    if run_name == "head_1h_dot" or run_name.startswith("head"):
        return "small_head"
    if run_name == "swap_fr_dot" or run_name.startswith("swap"):
        return "small_swap"
    return "small_try"


def load_resolved_config(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "resolved_config.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    inner = doc.get("resolved_config")
    return inner if isinstance(inner, dict) else None


def build_config_for_pkg(pkg: str, cfg_dict: dict[str, Any]) -> Any:
    """Instantiate pkg's Config dataclass from resolved_config keys only."""
    pkg_root = REPO_ROOT / pkg
    if not pkg_root.is_dir():
        raise FileNotFoundError(f"Package directory not found: {pkg_root}")

    insert = str(pkg_root.resolve())
    old_path = sys.path[:]
    try:
        if insert not in sys.path:
            sys.path.insert(0, insert)
        from config import Config  # type: ignore

        if not is_dataclass(Config):
            raise TypeError("Config is not a dataclass")
        valid = {f.name for f in fields(Config)}
        filtered = {k: v for k, v in cfg_dict.items() if k in valid}
        return Config(**filtered)  # type: ignore[call-arg]
    finally:
        sys.path[:] = old_path


def count_parameters(pkg: str, cfg: Any) -> int:
    pkg_root = REPO_ROOT / pkg
    insert = str(pkg_root.resolve())
    old_path = sys.path[:]
    try:
        if insert not in sys.path:
            sys.path.insert(0, insert)
        from model import Seq2SeqTransformer  # type: ignore

        model = Seq2SeqTransformer(cfg, pad_idx=0)
        return int(sum(p.numel() for p in model.parameters()))
    finally:
        sys.path[:] = old_path


def repair_payload(run_dir: Path, num_parameters: int) -> dict[str, Any]:
    abs_run = str(run_dir.resolve())
    return {
        "num_parameters": num_parameters,
        "wall_time_seconds": None,
        "peak_gpu_memory_bytes": None,
        "peak_gpu_memory_mib": None,
        "run_directory": abs_run,
        "metadata_repaired": True,
        "repair_reason": (
            "original training_meta.json was not present in archive; reconstructed from "
            "resolved_config.json without re-running training"
        ),
        "repaired_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repaired_from": ["resolved_config.json", "model parameter count"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        metavar="RUN",
        help="Run names under runs/ (default: head_1h_dot swap_fr_dot)",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Runs parent directory",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing training_meta.json",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    args = ap.parse_args()

    runs_root = args.runs_root
    if not runs_root.is_absolute():
        runs_root = (REPO_ROOT / runs_root).resolve()

    for run_name in args.runs:
        run_dir = runs_root / run_name
        out_path = run_dir / "training_meta.json"
        print(f"=== {run_name} ===", file=sys.stderr)

        if out_path.is_file() and not args.overwrite:
            print(f"  skip: {out_path} exists (use --overwrite to replace)", file=sys.stderr)
            continue

        cfg_dict = load_resolved_config(run_dir)
        if not cfg_dict:
            print(f"  error: missing or invalid resolved_config.json", file=sys.stderr)
            return 1

        pkg = infer_pkg(run_name)
        print(f"  pkg={pkg}", file=sys.stderr)
        try:
            cfg = build_config_for_pkg(pkg, cfg_dict)
            n_params = count_parameters(pkg, cfg)
        except Exception as e:
            print(f"  error: could not build model: {e}", file=sys.stderr)
            return 1

        payload = repair_payload(run_dir, n_params)
        print(f"  num_parameters={n_params} -> {out_path}", file=sys.stderr)
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
