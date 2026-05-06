#!/usr/bin/env python3
"""
检查仓库 runs/<run_name>/ 与流水线 manifest 的产物是否齐全。

口径说明（脚本输出会再次提示）：
  - metrics.json：训练目录内指标，默认对应 eval_split（通常为 **validation**，非 held-out test）。
  - test_eval/metrics_test.json：由仓库根 evaluate_test.py 在 **test.tsv** 上计算（test 终评）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_UNDER_RUN = (
    "resolved_config.json",
    "metrics.json",
    "training_meta.json",
    "test_eval/metrics_test.json",
    "test_eval/predictions.jsonl",
    "test_eval/examples.md",
)

DEFAULT_MANIFESTS = (
    "small_try/results/manifest.json",
    "small_head/results/manifest_head.json",
    "small_swap/results/manifest_swap.json",
)

# 缺失列表展示用：明确 metrics.json（validation）vs metrics_test.json（test）
REL_HINT: dict[str, str] = {
    "resolved_config.json": "merged config",
    "metrics.json": "validation（训练 eval_split）",
    "training_meta.json": "training run stats",
    "test_eval/metrics_test.json": "test（evaluate_test.py）",
    "test_eval/predictions.jsonl": "test predictions",
    "test_eval/examples.md": "test examples",
}


def _format_missing(rel: str) -> str:
    tip = REL_HINT.get(rel)
    return f"{rel} [{tip}]" if tip else rel


def _should_skip_runs_child(name: str) -> bool:
    """Scratch dirs (leading _) and forensic partial-run backups may omit test_eval."""
    if name.startswith("."):
        return True
    if name.startswith("_"):
        return True
    if ".partial." in name:
        return True
    return False


def check_run_dir(run_dir: Path) -> list[str]:
    missing: list[str] = []
    for rel in REQUIRED_UNDER_RUN:
        p = run_dir / rel
        if not p.is_file():
            missing.append(rel)
    return missing


def training_meta_kind(run_dir: Path) -> str:
    """Return 'missing' | 'repaired' | 'original'."""
    p = run_dir / "training_meta.json"
    if not p.is_file():
        return "missing"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "original"
    if obj.get("metadata_repaired") is True:
        return "repaired"
    return "original"


def validate_metrics_semantics(repo: Path, run_dir: Path) -> list[str]:
    """若 metrics 文件存在，校验字段语义未被混淆（validation vs test）。"""
    errs: list[str] = []
    rel_run = run_dir.relative_to(repo)

    train_m = run_dir / "metrics.json"
    if train_m.is_file():
        try:
            obj = json.loads(train_m.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errs.append(f"{rel_run}/metrics.json: JSON 无效: {e}")
            obj = None
        if isinstance(obj, dict):
            if "eval_split" not in obj:
                errs.append(
                    f"{rel_run}/metrics.json: 缺少 eval_split 字段（应为 validation 口径的训练 metrics）"
                )
            if obj.get("number_of_test_examples") is not None and "eval_split" not in obj:
                errs.append(
                    f"{rel_run}/metrics.json: 疑似误放了 test 终评内容（含 number_of_test_examples 却无 eval_split）"
                )

    test_m = run_dir / "test_eval/metrics_test.json"
    if test_m.is_file():
        try:
            t_obj = json.loads(test_m.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errs.append(f"{rel_run}/test_eval/metrics_test.json: JSON 无效: {e}")
            t_obj = None
        if isinstance(t_obj, dict):
            if "eval_split" in t_obj and "number_of_test_examples" not in t_obj:
                errs.append(
                    f"{rel_run}/test_eval/metrics_test.json: 含 eval_split 却无 number_of_test_examples（疑似 validation metrics 误放）"
                )
            if "number_of_test_examples" not in t_obj:
                errs.append(
                    f"{rel_run}/test_eval/metrics_test.json: 缺少 number_of_test_examples（应为 test 终评）"
                )

    return errs


def _resolve_manifest_path(repo: Path, s: str) -> Path:
    """Interpret manifest path strings relative to repo when not absolute."""
    p = Path(str(s).replace("\\", "/"))
    if p.is_absolute():
        return p.resolve()
    return (repo / p).resolve()


def collect_manifest_errors(repo: Path, manifest_paths: list[Path]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for mf in manifest_paths:
        mf = mf.resolve()
        if not mf.is_file():
            warnings.append(f"（跳过）不存在 manifest 文件: {mf.relative_to(repo)}")
            continue
        try:
            raw = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"{mf.relative_to(repo)}: JSON 解析失败: {e}")
            continue

        # baseline_metrics_path（small_head）：应为文件
        bmp = raw.get("baseline_metrics_path")
        if bmp:
            bp = _resolve_manifest_path(repo, str(bmp))
            if not bp.is_file():
                errors.append(
                    f"{mf.relative_to(repo)}: baseline_metrics_path 不是可读文件: {bmp}"
                )

        runs_block = raw.get("runs")
        if isinstance(runs_block, dict):
            for key, val in runs_block.items():
                if not isinstance(val, str):
                    errors.append(f"{mf.relative_to(repo)}: runs.{key} 不是字符串路径")
                    continue
                norm = val.replace("\\", "/")
                # 旧口径：run 应在仓库 runs/，不应落在 small_*/runs/
                if "/small_swap/runs/" in norm or "/small_try/runs/" in norm or "/small_head/runs/" in norm:
                    errors.append(
                        f"{mf.relative_to(repo)}: runs.{key} 口径错误（应为仓库 runs/…）：{val}"
                    )
                    continue
                rd = _resolve_manifest_path(repo, val)
                if not rd.is_dir():
                    errors.append(
                        f"{mf.relative_to(repo)}: runs.{key} -> 目录不存在: {val}"
                    )

    return errors, warnings


def main() -> None:
    ap = argparse.ArgumentParser(
        description="检查 runs/* 产物与 manifest 中 run 路径是否一致、文件是否齐全。"
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="仓库根目录（默认：本脚本上级目录的上一级）",
    )
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="仅校验 manifest，不扫描 runs/*",
    )
    ap.add_argument(
        "--extra-manifest",
        type=Path,
        action="append",
        default=[],
        help="额外 manifest 路径（相对仓库根），可重复",
    )
    args = ap.parse_args()

    repo = (args.repo_root or Path(__file__).resolve().parent.parent).resolve()
    runs_root = repo / "runs"

    print(
        "口径：metrics.json = 训练脚本 eval_split（多为 **val**）；"
        "test_eval/metrics_test.json = **test** 终评（evaluate_test.py）。\n",
        file=sys.stderr,
    )

    exit_errors: list[str] = []

    manifest_paths = [repo / p for p in DEFAULT_MANIFESTS]
    manifest_paths.extend(repo / p for p in args.extra_manifest)

    m_err, m_warn = collect_manifest_errors(repo, manifest_paths)
    exit_errors.extend(m_err)
    for w in m_warn:
        print(f"[WARN] {w}", file=sys.stderr)

    if not args.manifest_only:
        if not runs_root.is_dir():
            exit_errors.append(f"缺少 runs 目录: {runs_root}")
        else:
            meta_warns: list[str] = []
            for child in sorted(runs_root.iterdir()):
                if not child.is_dir():
                    continue
                if _should_skip_runs_child(child.name):
                    continue
                miss = check_run_dir(child)
                sem = validate_metrics_semantics(repo, child)
                if miss:
                    rel = child.relative_to(repo)
                    exit_errors.append(
                        f"{rel}: 缺少文件 → "
                        + ", ".join(_format_missing(m) for m in miss)
                    )
                exit_errors.extend(sem)
                kind = training_meta_kind(child)
                if kind == "repaired":
                    rel = child.relative_to(repo)
                    meta_warns.append(
                        f"{rel}: training_meta.json present but **repaired** "
                        "(metadata_repaired=true; wall_time / GPU mem not from original training)"
                    )
            for w in meta_warns:
                print(f"[WARN] {w}", file=sys.stderr)

    if exit_errors:
        print("检查失败：", file=sys.stderr)
        for e in exit_errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    print(
        "检查通过：runs/* 必备文件齐全；manifest runs 路径存在。"
        "（若上方有 [WARN] training_meta repaired，表示该 run 的 training_meta 为事后修补，"
        "非训练脚本原始记录。）",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
