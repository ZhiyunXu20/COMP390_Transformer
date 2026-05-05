"""训练共用：仓库相对路径解析、resolved_config.json、git 信息。"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch


def configure_determinism(
    seed: int, deterministic: bool, *, cuda_available: bool
) -> dict[str, Any]:
    """Apply (or record) PyTorch/cuDNN determinism settings; return a JSON-safe summary.

    When ``deterministic`` is False, enables ``cudnn.benchmark`` on CUDA to preserve
    the project's default fast path, then returns the resulting cudnn flags.
    """
    del seed  # reserved for API symmetry; DataLoader uses cfg.seed separately
    if deterministic:
        cublas_ws = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        return {
            "deterministic_requested": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "use_deterministic_algorithms": True,
            "cublas_workspace_config": cublas_ws,
            "torch_version": torch.__version__,
        }

    if cuda_available:
        torch.backends.cudnn.benchmark = True
    return {
        "deterministic_requested": False,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "use_deterministic_algorithms": False,
        "torch_version": torch.__version__,
    }


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Re-seed NumPy and ``random`` inside DataLoader workers (with ``num_workers > 0``)."""

    def _worker_init(worker_id: int) -> None:
        import random

        import numpy as np

        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return _worker_init


def infer_repo_root(train_py_file: Path) -> Path:
    """train.py 位于 `<repo>/<pkg>/train.py` 时指向仓库根；可用 REPO_ROOT / AUTODL_REPO_ROOT 覆盖。"""
    for key in ("REPO_ROOT", "AUTODL_REPO_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return train_py_file.resolve().parent.parent


def resolve_repo_relative(path_str: str, repo_root: Path) -> str:
    if not path_str or not str(path_str).strip():
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p.resolve())
    return str((repo_root / path_str).resolve())


def materialize_path_fields(cfg: Any, repo_root: Path, fields: tuple[str, ...]) -> None:
    for name in fields:
        if hasattr(cfg, name):
            v = getattr(cfg, name)
            if isinstance(v, str):
                setattr(cfg, name, resolve_repo_relative(v, repo_root))


PATH_FIELDS_DEFAULT = (
    "train_path",
    "val_path",
    "test_path",
    "data_path",
    "tokenizer_src",
    "tokenizer_tgt",
    "output_dir",
)


def git_commit_and_dirty(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        commit = cp.stdout.strip() if cp.returncode == 0 else None
        st = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
        return commit, dirty
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def config_to_jsonable(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        d = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        d = dict(vars(cfg))
    else:
        d = dict(cfg)  # type: ignore[arg-type]
    return {k: _jsonable_val(v) for k, v in d.items()}


def _jsonable_val(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v.resolve())
    if isinstance(v, Mapping):
        return {str(k): _jsonable_val(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable_val(x) for x in v]
    return str(v)


def save_resolved_config_json(
    out_dir: Path,
    cfg: Any,
    repo_root: Path,
    *,
    argv: list[str] | None = None,
    determinism: dict[str, Any] | None = None,
) -> None:
    git_commit, git_dirty = git_commit_and_dirty(repo_root)
    payload: dict[str, Any] = {
        "repo_root": str(repo_root),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "argv": list(argv or []),
        "resolved_config": config_to_jsonable(cfg),
    }
    if determinism is not None:
        payload["determinism"] = determinism
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def register_shared_cli_arguments(p: Any) -> None:
    """三组 small_* train.py 共用的可移植 CLI（路径均为仓库相对路径，可被命令行覆盖）。"""
    p.add_argument("--train-path", type=str, default=None)
    p.add_argument("--val-path", type=str, default=None)
    p.add_argument("--test-path", type=str, default=None)
    p.add_argument("--tokenizer-src", type=str, default=None)
    p.add_argument("--tokenizer-tgt", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--attention-type",
        "--attention",
        dest="attention_type",
        type=str,
        default=None,
        choices=(
            "dot_product",
            "additive",
            "bilinear",
            "gated_dot_additive",
            "local_window",
            "global_local",
            "sparsemax",
            "entmax15",
        ),
        help="注意力类型（也可用 --attention）",
    )
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--bleu-sample-size", type=int, default=None)
    p.add_argument(
        "--eval-split",
        type=str,
        default=None,
        choices=("train", "val", "test"),
        help="validation_loss / BLEU 使用的划分；训练仍只用 train",
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="更严格的可复现模式（cuDNN deterministic、确定性算法 warn_only、CUBLAS workspace 等）",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="兼容旧用法：未指定 --train-path 时视为训练语料（等同 train_path）",
    )


def apply_shared_cli_to_config(cfg: Any, args: Any) -> None:
    """将 argparse namespace 中的共享字段写入 cfg（None 表示不覆盖默认值）。"""
    if getattr(args, "train_path", None) is not None:
        cfg.train_path = args.train_path
        cfg.data_path = cfg.train_path
    if getattr(args, "val_path", None) is not None:
        cfg.val_path = args.val_path
    if getattr(args, "test_path", None) is not None:
        cfg.test_path = args.test_path
    if getattr(args, "data_path", None) is not None:
        cfg.data_path = args.data_path
        if getattr(args, "train_path", None) is None:
            cfg.train_path = args.data_path
    if getattr(args, "tokenizer_src", None) is not None:
        cfg.tokenizer_src = args.tokenizer_src
    if getattr(args, "tokenizer_tgt", None) is not None:
        cfg.tokenizer_tgt = args.tokenizer_tgt
    if getattr(args, "output_dir", None) is not None:
        cfg.output_dir = args.output_dir
    if getattr(args, "attention_type", None) is not None:
        cfg.attention_type = args.attention_type  # type: ignore[assignment]
    if getattr(args, "n_heads", None) is not None:
        cfg.n_heads = args.n_heads
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "max_steps", None) is not None:
        cfg.max_steps = args.max_steps
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "bleu_sample_size", None) is not None:
        cfg.bleu_sample_size = args.bleu_sample_size
    if getattr(args, "eval_split", None) is not None:
        cfg.eval_split = args.eval_split  # type: ignore[assignment]
    if getattr(args, "no_wandb", False):
        cfg.use_wandb = False
