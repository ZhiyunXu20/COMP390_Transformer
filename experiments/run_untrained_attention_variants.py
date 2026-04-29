#!/usr/bin/env python3
"""
训练 variant_registry 中的 exploratory attention 变体（runs/var_*），统一流程、不写 shell。

元数据（research_question / safe_claim 等）见同目录 **variant_registry.py**。
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from variant_registry import (  # noqa: E402
    DEFAULT_VARIANT_ORDER,
    VARIANT_TO_RUN_NAME,
)

DEFAULT_VARIANTS: Final[tuple[str, ...]] = DEFAULT_VARIANT_ORDER

DEFAULT_SPLITS_PREFIX = "data/splits/en_fr_50k_seed42"

PROTECTED_BASELINE_RUN_DIRS: Final[frozenset[str]] = frozenset(
    {"fast_dot", "fast_add", "head_1h_dot", "swap_fr_dot"}
)

PROTECTED_RUN_DIRS = PROTECTED_BASELINE_RUN_DIRS

BASELINE_ATTENTION_TYPES_FORBIDDEN: Final[frozenset[str]] = frozenset(
    {"dot_product", "additive"}
)

_EXIT_BASELINE = (
    "This script is for untrained exploratory variants and must not overwrite "
    "completed baseline runs.\n"
    "Train dot_product / additive baselines with `small_try/train.py` (or sibling pkgs) "
    "and explicit `--name` such as fast_dot, fast_add, head_1h_dot, swap_fr_dot."
)


def refuse_protected_run(run_name: str, *, attention_type: str, context: str) -> None:
    if run_name not in PROTECTED_BASELINE_RUN_DIRS:
        return
    raise SystemExit(
        f"{_EXIT_BASELINE}\n"
        f"({context}) Target run directory `{run_name}` (attention `{attention_type}`) "
        "is protected."
    )


def _repo_path(repo: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def rel_under_repo(path: Path, root: Path = REPO_ROOT) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def preflight(
    repo: Path,
    *,
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer_src: str,
    tokenizer_tgt: str,
    train_py: Path,
    eval_py: Path,
) -> None:
    checks: list[tuple[str, Path]] = [
        ("small_try/train.py（或所选 pkg）", train_py),
        ("evaluate_test.py（仓库根）", eval_py),
        ("--train-path", _repo_path(repo, train_path)),
        ("--val-path", _repo_path(repo, val_path)),
        ("--test-path", _repo_path(repo, test_path)),
        ("--tokenizer-src", _repo_path(repo, tokenizer_src)),
        ("--tokenizer-tgt", _repo_path(repo, tokenizer_tgt)),
    ]
    bad = [(lbl, path) for lbl, path in checks if not path.is_file()]
    if bad:
        detail = "\n".join(f"  - {lbl}: {path}" for lbl, path in bad)
        raise SystemExit(f"preflight 失败：以下路径不存在或不可读\n{detail}")


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    quoted = shlex.join(cmd)
    print(f"+ cd {cwd} && {quoted}", flush=True)
    if dry_run:
        return 0
    cp = subprocess.run(cmd, cwd=str(cwd))
    return int(cp.returncode)


def parse_variants(arg_list: list[str] | None) -> list[str]:
    if not arg_list:
        return list(DEFAULT_VARIANTS)
    baseline = [v for v in arg_list if v in BASELINE_ATTENTION_TYPES_FORBIDDEN]
    if baseline:
        raise SystemExit(
            f"{_EXIT_BASELINE}\n"
            f"Do not pass baseline attention types as --variants (got {baseline}). "
            "This script only schedules exploratory `var_*` runs from variant_registry."
        )
    unknown = [v for v in arg_list if v not in VARIANT_TO_RUN_NAME]
    if unknown:
        raise SystemExit(
            f"未知 variant: {unknown}；允许: {sorted(VARIANT_TO_RUN_NAME)}",
        )
    return arg_list


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "统一调用 python <pkg>/train.py 与 evaluate_test.py，训练未跑的 attention 变体（runs/var_*）。"
            "不调用任何 .sh；禁止触碰 fast_dot / fast_add / head_1h_dot / swap_fr_dot。"
        ),
    )
    p.add_argument(
        "--train-path",
        type=str,
        default=f"{DEFAULT_SPLITS_PREFIX}/train.tsv",
        help="默认与 fast_dot 管线一致的划分",
    )
    p.add_argument("--val-path", type=str, default=f"{DEFAULT_SPLITS_PREFIX}/val.tsv")
    p.add_argument("--test-path", type=str, default=f"{DEFAULT_SPLITS_PREFIX}/test.tsv")
    p.add_argument(
        "--tokenizer-src",
        type=str,
        default="data/tokenizer_src_train_only.json",
    )
    p.add_argument(
        "--tokenizer-tgt",
        type=str,
        default="data/tokenizer_tgt_train_only.json",
    )
    p.add_argument("--pkg", type=str, default="small_try")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        metavar="ATTENTION_TYPE",
        help=f"子集；默认全部 {list(DEFAULT_VARIANT_ORDER)}",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="若 runs/var_* 已存在则先删除再训练（默认跳过已存在目录）；绝不删除受保护的 baseline 目录",
    )
    wb = p.add_mutually_exclusive_group()
    wb.add_argument("--wandb", action="store_true", help="启用 W&B（默认关闭）")
    wb.add_argument("--no-wandb", action="store_true", help="关闭 W&B（默认）")
    p.add_argument("--no-eval", action="store_true", help="仅训练，不调用 evaluate_test.py")
    args = p.parse_args()

    use_wandb = bool(args.wandb)

    variants = parse_variants(args.variants)
    py = sys.executable
    train_py = REPO_ROOT / args.pkg / "train.py"
    eval_py = REPO_ROOT / "evaluate_test.py"

    preflight(
        REPO_ROOT,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        tokenizer_src=args.tokenizer_src,
        tokenizer_tgt=args.tokenizer_tgt,
        train_py=train_py,
        eval_py=eval_py,
    )

    manifest_common: dict[str, Any] = {
        "train_path": args.train_path,
        "val_path": args.val_path,
        "test_path": args.test_path,
        "tokenizer_src": args.tokenizer_src,
        "tokenizer_tgt": args.tokenizer_tgt,
        "pkg": args.pkg,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "no_wandb": not use_wandb,
        "eval_split": "val",
        "train_script": rel_under_repo(train_py),
        "evaluate_test_script": rel_under_repo(eval_py),
    }
    if args.batch_size is not None:
        manifest_common["batch_size"] = args.batch_size

    print(
        "\n[shared pipeline] 以下字段对所有 variants 相同（公平对比）：",
        file=sys.stderr,
        flush=True,
    )
    for k, v in manifest_common.items():
        print(f"  {k}: {v}", file=sys.stderr, flush=True)
    print(f"[preflight] OK | variants={variants} | dry_run={args.dry_run}\n", file=sys.stderr, flush=True)

    manifest_runs: list[dict[str, Any]] = []

    for att in variants:
        run_name = VARIANT_TO_RUN_NAME[att]
        refuse_protected_run(run_name, attention_type=att, context="scheduled train/eval")

        run_dir = REPO_ROOT / "runs" / run_name
        row: dict[str, Any] = {
            "variant_attention_type": att,
            "run_name": run_name,
            "run_dir": rel_under_repo(run_dir),
            "skipped": False,
            "skip_reason": None,
            "train_returncode": None,
            "eval_returncode": None,
            "train_command": None,
            "eval_command": None,
        }

        if run_name in PROTECTED_BASELINE_RUN_DIRS:
            raise SystemExit(f"内部错误：run_name={run_name} 在受保护列表中，拒绝执行。")

        if run_dir.is_dir() and not args.overwrite:
            msg = "directory_exists_no_overwrite"
            print(
                f"[跳过] 已存在 {run_dir.relative_to(REPO_ROOT)} （使用 --overwrite 强制重跑）",
                file=sys.stderr,
                flush=True,
            )
            row["skipped"] = True
            row["skip_reason"] = msg
            manifest_runs.append(row)
            continue

        if run_dir.is_dir() and args.overwrite:
            if run_name in PROTECTED_BASELINE_RUN_DIRS:
                raise SystemExit(
                    f"拒绝删除受保护 baseline 目录：{run_name}（即使你使用了 --overwrite）。"
                )
            print(f"[overwrite] 删除 {run_dir}", file=sys.stderr, flush=True)
            if not args.dry_run:
                shutil.rmtree(run_dir)

        train_cmd = [
            py,
            str(train_py),
            "--train-path",
            args.train_path,
            "--val-path",
            args.val_path,
            "--test-path",
            args.test_path,
            "--tokenizer-src",
            args.tokenizer_src,
            "--tokenizer-tgt",
            args.tokenizer_tgt,
            "--output-dir",
            "runs",
            "--name",
            run_name,
            "--attention-type",
            att,
            "--seed",
            str(args.seed),
            "--max-steps",
            str(args.max_steps),
            "--eval-split",
            "val",
        ]
        if use_wandb:
            pass
        else:
            train_cmd.append("--no-wandb")
        if args.batch_size is not None:
            train_cmd.extend(["--batch-size", str(args.batch_size)])

        best_pt = run_dir / "best.pt"
        ev_out = run_dir / "test_eval"
        eval_cmd: list[str] | None = None
        if not args.no_eval:
            eval_cmd = [
                py,
                str(eval_py),
                "--checkpoint",
                rel_under_repo(best_pt),
                "--test-file",
                args.test_path,
                "--tokenizer-src",
                args.tokenizer_src,
                "--tokenizer-tgt",
                args.tokenizer_tgt,
                "--output-dir",
                rel_under_repo(ev_out),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--batch-size",
                str(args.eval_batch_size),
                "--pkg",
                args.pkg,
            ]

        row["train_command"] = train_cmd
        row["eval_command"] = eval_cmd

        print("", flush=True)
        print(f"========== {run_name} (attention_type={att}) ==========", flush=True)
        print("TRAIN（即将执行）:", flush=True)
        print(f"  cd {REPO_ROOT} && {shlex.join(train_cmd)}", flush=True)
        if eval_cmd is not None:
            print("EVAL（训练成功后执行）:", flush=True)
            print(f"  cd {REPO_ROOT} && {shlex.join(eval_cmd)}", flush=True)
        print("", flush=True)

        rc = run_command(train_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
        row["train_returncode"] = rc
        if rc != 0:
            print(f"[错误] 训练失败 attention={att} run={run_name} rc={rc}", file=sys.stderr)
            manifest_runs.append(row)
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repo_root": str(REPO_ROOT),
                "dry_run": args.dry_run,
                "overwrite": args.overwrite,
                "no_eval": args.no_eval,
                "common": manifest_common,
                "runs": manifest_runs,
                "aborted_after_run": run_name,
            }
            write_manifest(REPO_ROOT / "results" / "untrained_variants_run_manifest.json", payload)
            print(
                f"Wrote {REPO_ROOT / 'results' / 'untrained_variants_run_manifest.json'} (aborted)",
                file=sys.stderr,
            )
            sys.exit(rc)

        if args.no_eval:
            manifest_runs.append(row)
            continue

        assert eval_cmd is not None
        if args.dry_run:
            run_command(eval_cmd, cwd=REPO_ROOT, dry_run=True)
            row["eval_returncode"] = 0
            manifest_runs.append(row)
            continue

        if not best_pt.is_file():
            print(
                f"[警告] 无 best.pt，跳过 evaluate_test：{best_pt}",
                file=sys.stderr,
            )
            row["eval_skipped_reason"] = "missing_best_pt"
            row["eval_returncode"] = None
            manifest_runs.append(row)
            continue

        rc_e = run_command(eval_cmd, cwd=REPO_ROOT, dry_run=False)
        row["eval_returncode"] = rc_e
        if rc_e != 0:
            print(
                f"[警告] evaluate_test 失败 attention={att} rc={rc_e}",
                file=sys.stderr,
            )
        manifest_runs.append(row)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "no_eval": args.no_eval,
        "common": manifest_common,
        "runs": manifest_runs,
    }
    out_manifest = REPO_ROOT / "results" / "untrained_variants_run_manifest.json"
    write_manifest(out_manifest, payload)
    print(f"Wrote {out_manifest}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
