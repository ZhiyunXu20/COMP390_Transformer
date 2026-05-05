#!/usr/bin/env python3
"""Write pipeline manifests under small_try / small_head / small_swap using repo-relative POSIX paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm_rel(repo: Path, p: Path | str) -> str:
    """Return a POSIX path relative to repo (no leading slash)."""
    repo = repo.resolve()
    path = Path(p)
    if not path.is_absolute():
        path = (repo / path).resolve()
    else:
        path = path.resolve()
    try:
        return path.relative_to(repo).as_posix()
    except ValueError as e:
        raise ValueError(f"path {path} is not under repo {repo}") from e


def write_small_try(
    repo: Path,
    small_try_dir: Path,
    *,
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer_src: str,
    tokenizer_tgt: str,
    tokenizer_metadata: str,
    split_dir: str = "data/splits/en_fr_50k_seed42",
) -> Path:
    repo = repo.resolve()
    out = small_try_dir / "results" / "manifest.json"
    m: dict = {
        "finished_at": _utc_now(),
        "split_protocol": "train_only_tokenizers",
        "split_dir": split_dir,
        "train_path": _norm_rel(repo, train_path),
        "val_path": _norm_rel(repo, val_path),
        "test_path": _norm_rel(repo, test_path),
        "tokenizer_metadata": _norm_rel(repo, tokenizer_metadata),
        "tokenizer_src": _norm_rel(repo, tokenizer_src),
        "tokenizer_tgt": _norm_rel(repo, tokenizer_tgt),
        "report": _norm_rel(repo, small_try_dir / "report.txt"),
        "bundle": _norm_rel(repo, small_try_dir / "results" / "bundle_metrics.json"),
        "runs": {
            "dot": "runs/fast_dot",
            "add": "runs/fast_add",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def write_small_head(
    repo: Path,
    small_head_dir: Path,
    *,
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer_src: str,
    tokenizer_tgt: str,
    tokenizer_metadata: str,
    baseline_metrics: str = "runs/fast_dot/metrics.json",
    split_dir: str = "data/splits/en_fr_50k_seed42",
) -> Path:
    repo = repo.resolve()
    out = small_head_dir / "results" / "manifest_head.json"
    bm = _norm_rel(repo, baseline_metrics)
    m = {
        "finished_at": _utc_now(),
        "split_protocol": "train_only_tokenizers",
        "split_dir": split_dir,
        "train_path": _norm_rel(repo, train_path),
        "val_path": _norm_rel(repo, val_path),
        "test_path": _norm_rel(repo, test_path),
        "tokenizer_metadata": _norm_rel(repo, tokenizer_metadata),
        "tokenizer_src": _norm_rel(repo, tokenizer_src),
        "tokenizer_tgt": _norm_rel(repo, tokenizer_tgt),
        "baseline_metrics_path": bm,
        "report": _norm_rel(repo, small_head_dir / "report_head.txt"),
        "bundle": _norm_rel(repo, small_head_dir / "results" / "bundle_head_metrics.json"),
        "param_counts": _norm_rel(repo, small_head_dir / "results" / "param_counts.json"),
        "runs": {
            "baseline_mh_dot": "runs/fast_dot",
            "head_1h_dot": "runs/head_1h_dot",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def write_small_swap(
    repo: Path,
    small_swap_dir: Path,
    *,
    train_path: str,
    val_path: str,
    test_path: str,
    split_dir: str = "data/splits/en_fr_50k_seed42",
) -> Path:
    repo = repo.resolve()
    out = small_swap_dir / "results" / "manifest_swap.json"
    m = {
        "finished_at": _utc_now(),
        "direction": "fr_en",
        "split_protocol": "same_as_small_try",
        "split_dir": split_dir,
        "train_path": _norm_rel(repo, train_path),
        "val_path": _norm_rel(repo, val_path),
        "test_path": _norm_rel(repo, test_path),
        "report": _norm_rel(repo, small_swap_dir / "report_swap.txt"),
        "bundle": _norm_rel(repo, small_swap_dir / "results" / "bundle_swap_metrics.json"),
        "runs": {
            "dot": "runs/swap_fr_dot",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    repo_default = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "which",
        choices=("small_try", "small_head", "small_swap"),
        help="Which manifest to write",
    )
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--small-try-dir", type=Path, default=None, help="Path to small_try/")
    ap.add_argument("--small-head-dir", type=Path, default=None)
    ap.add_argument("--small-swap-dir", type=Path, default=None)
    ap.add_argument("--train-path", default="data/splits/en_fr_50k_seed42/train.tsv")
    ap.add_argument("--val-path", default="data/splits/en_fr_50k_seed42/val.tsv")
    ap.add_argument("--test-path", default="data/splits/en_fr_50k_seed42/test.tsv")
    ap.add_argument("--tokenizer-src", default="data/tokenizer_src_train_only.json")
    ap.add_argument("--tokenizer-tgt", default="data/tokenizer_tgt_train_only.json")
    ap.add_argument("--tokenizer-metadata", default="data/tokenizer_metadata.json")
    ap.add_argument("--baseline-metrics", default="runs/fast_dot/metrics.json")
    ap.add_argument(
        "--split-dir",
        default="data/splits/en_fr_50k_seed42",
        help='Recorded as "split_dir" (repo-relative)',
    )
    args = ap.parse_args()
    repo = args.repo_root.resolve()

    if args.which == "small_try":
        pkg = (args.small_try_dir or (repo / "small_try")).resolve()
        out = write_small_try(
            repo,
            pkg,
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            tokenizer_src=args.tokenizer_src,
            tokenizer_tgt=args.tokenizer_tgt,
            tokenizer_metadata=args.tokenizer_metadata,
            split_dir=args.split_dir,
        )
    elif args.which == "small_head":
        pkg = (args.small_head_dir or (repo / "small_head")).resolve()
        out = write_small_head(
            repo,
            pkg,
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            tokenizer_src=args.tokenizer_src,
            tokenizer_tgt=args.tokenizer_tgt,
            tokenizer_metadata=args.tokenizer_metadata,
            baseline_metrics=args.baseline_metrics,
            split_dir=args.split_dir,
        )
    else:
        pkg = (args.small_swap_dir or (repo / "small_swap")).resolve()
        out = write_small_swap(
            repo,
            pkg,
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            split_dir=args.split_dir,
        )

    print(out)


if __name__ == "__main__":
    main()
