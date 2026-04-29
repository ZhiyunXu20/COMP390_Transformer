#!/usr/bin/env python3
"""
多 seed attention ablation：批量训练 small_try（或其它 pkg）、在 test 上评估 best.pt，并汇总 CSV/Markdown。

默认：fast_dot / fast_add × seeds 1,2,3 → runs/fast_dot_s1 … runs/fast_add_s3。
`--legacy-layout` 恢复旧版 runs/<experiment>/seed_<n>/ 与多 heads 组合。
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SPLITS = "data/splits/en_fr_50k_seed42"

# Legacy nested layout: runs/<exp_name>/seed_<seed>/
ABLATION_SPECS_LEGACY: list[tuple[str, str, int]] = [
    ("dot_h4", "dot_product", 4),
    ("add_h4", "additive", 4),
    ("dot_h1", "dot_product", 1),
    ("add_h1", "additive", 1),
]

# Flat layout: experiment label → attention_type, n_heads；run 名为 {label}_s{seed}
DEFAULT_FLAT_SPECS: list[tuple[str, str, int]] = [
    ("fast_dot", "dot_product", 4),
    ("fast_add", "additive", 4),
]

CHRPP_KEY = "chrF++"


def rel_under_repo(path: Path, root: Path = REPO_ROOT) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def fmt_mean_std(mean: float | None, std: float | None, *, nd: int = 4) -> str:
    if mean is None:
        return "n/a"
    if std is None or abs(float(std)) < 1e-12:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def mean_std_tuple(xs: list[float | None]) -> tuple[float | None, float | None]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, std


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_meta(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "training_meta.json"
    if not p.is_file():
        return {}
    try:
        return load_json(p)
    except OSError:
        return {}


def load_metrics_test(eval_dir: Path) -> dict[str, Any]:
    p = eval_dir / "metrics_test.json"
    if not p.is_file():
        return {}
    try:
        return load_json(p)
    except OSError:
        return {}


def _repo_path(repo: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def preflight_ablation(
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
        ("train.py（pkg）", train_py),
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


def flat_run_name(experiment: str, seed: int) -> str:
    return f"{experiment}_s{seed}"


def main() -> None:
    p = argparse.ArgumentParser(description="Attention ablation（多 seed）")
    p.add_argument(
        "--train-path",
        type=str,
        default=f"{DEFAULT_SPLITS}/train.tsv",
    )
    p.add_argument("--val-path", type=str, default=f"{DEFAULT_SPLITS}/val.tsv")
    p.add_argument("--test-path", type=str, default=f"{DEFAULT_SPLITS}/test.tsv")
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
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help="随机种子（flat 默认 1 2 3；legacy 默认 42 43 44）",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="优化步数上限（flat 默认 3000；legacy 默认不设则由 Config 决定）",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument(
        "--legacy-layout",
        action="store_true",
        help="旧版 runs/<experiment>/seed_<seed>/、best+last 双评估、多 heads 组合",
    )
    p.add_argument("--skip-add-h1", action="store_true", help="（仅 legacy）跳过 additive n_heads=1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="仅训练，不调用 evaluate_test.py")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="不训练：仅从现有 runs 目录读取 test_eval/metrics_test.json 重新写出汇总 CSV/Markdown",
    )
    p.add_argument(
        "--experiments",
        nargs="+",
        choices=("fast_dot", "fast_add"),
        default=None,
        metavar="NAME",
        help="（仅 flat）只调度 fast_dot / fast_add 子集；默认二者都做",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="启用 W&B（默认关闭）",
    )
    args = p.parse_args()

    if args.aggregate_only and args.dry_run:
        raise SystemExit("--aggregate-only 不能与 --dry-run 同时使用")

    legacy = args.legacy_layout
    specs_all = list(ABLATION_SPECS_LEGACY if legacy else DEFAULT_FLAT_SPECS)
    if legacy and args.skip_add_h1:
        specs_all = [s for s in specs_all if not (s[1] == "additive" and s[2] == 1)]

    if args.experiments is not None:
        if legacy:
            raise SystemExit("--experiments 仅适用于默认 flat 布局（勿同时传 --legacy-layout）")
        allow = frozenset(args.experiments)
        specs_run = [s for s in specs_all if s[0] in allow]
        if not specs_run:
            raise SystemExit(f"--experiments {args.experiments} 无匹配项")
    else:
        specs_run = specs_all

    if args.seeds is not None:
        seeds: tuple[int, ...] = tuple(args.seeds)
    else:
        seeds = (42, 43, 44) if legacy else (1, 2, 3)

    max_steps = args.max_steps
    if max_steps is None:
        max_steps = None if legacy else 3000

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []

    if args.aggregate_only:
        print(
            f"[aggregate-only] seeds={list(seeds)} experiments(all)={len(specs_all)} layout={'legacy' if legacy else 'flat'}",
            file=sys.stderr,
            flush=True,
        )
        for exp_name, att, n_heads in specs_all:
            for seed in seeds:
                if legacy:
                    run_dir = REPO_ROOT / "runs" / exp_name / f"seed_{seed}"
                    run_name_display = f"{exp_name}/seed_{seed}"
                    best_pt = run_dir / "best.pt"
                    last_pt = run_dir / "last.pt"
                    eval_specs = [
                        ("best_val", best_pt, run_dir / "test_eval_best_val"),
                        ("final", last_pt, run_dir / "test_eval_final"),
                    ]
                else:
                    run_name_display = flat_run_name(exp_name, seed)
                    run_dir = REPO_ROOT / "runs" / run_name_display
                    best_pt = run_dir / "best.pt"
                    eval_specs = [
                        ("best", best_pt, run_dir / "test_eval"),
                    ]

                tm = load_training_meta(run_dir)
                wall = tm.get("wall_time_seconds")
                peak_mib = tm.get("peak_gpu_memory_mib")
                n_params = tm.get("num_parameters")

                for ck_kind, ck_path, ev_out in eval_specs:
                    m = load_metrics_test(ev_out)
                    bleu = chrf = chrfpp = comet = None
                    if m:
                        bleu = m.get("BLEU")
                        chrf = m.get("chrF")
                        chrfpp = m.get(CHRPP_KEY)
                        comet = m.get("COMET")
                    per_seed_rows.append(
                        {
                            "run_name": run_name_display,
                            "experiment": exp_name,
                            "attention_type": att,
                            "n_heads": n_heads,
                            "seed": seed,
                            "checkpoint_kind": ck_kind,
                            "checkpoint_path": str(ck_path) if ck_path.is_file() else "",
                            "BLEU": bleu,
                            "chrF": chrf,
                            CHRPP_KEY: chrfpp,
                            "COMET": comet,
                            "wall_time_seconds": wall,
                            "peak_gpu_memory_mib": peak_mib,
                            "num_parameters": n_params,
                        }
                    )
    else:
        py = sys.executable
        train_py = REPO_ROOT / args.pkg / "train.py"
        eval_py = REPO_ROOT / "evaluate_test.py"

        preflight_ablation(
            REPO_ROOT,
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            tokenizer_src=args.tokenizer_src,
            tokenizer_tgt=args.tokenizer_tgt,
            train_py=train_py,
            eval_py=eval_py,
        )
        layout_note = "legacy nested" if legacy else "flat runs/<name>_s<seed>"
        print(
            f"[preflight] OK | pkg={args.pkg} | layout={layout_note} | seeds={list(seeds)} | train_specs={len(specs_run)}",
            file=sys.stderr,
            flush=True,
        )

        for exp_name, att, n_heads in specs_run:
            for seed in seeds:
                if legacy:
                    run_dir = REPO_ROOT / "runs" / exp_name / f"seed_{seed}"
                    wandb_name = f"seed_{seed}"
                    out_sub = Path("runs") / exp_name
                else:
                    wandb_name = flat_run_name(exp_name, seed)
                    run_dir = REPO_ROOT / "runs" / wandb_name
                    out_sub = Path("runs")

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
                    str(out_sub),
                    "--name",
                    wandb_name,
                    "--attention-type",
                    att,
                    "--n-heads",
                    str(n_heads),
                    "--seed",
                    str(seed),
                    "--eval-split",
                    "val",
                ]
                if not args.wandb:
                    train_cmd.append("--no-wandb")
                if max_steps is not None:
                    train_cmd.extend(["--max-steps", str(max_steps)])
                if args.batch_size is not None:
                    train_cmd.extend(["--batch-size", str(args.batch_size)])

                rc = run_command(train_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
                if rc != 0:
                    print(f"[错误] 训练失败 exp={exp_name} seed={seed} rc={rc}", file=sys.stderr)
                    sys.exit(rc)

                tm = load_training_meta(run_dir)
                wall = tm.get("wall_time_seconds")
                peak_mib = tm.get("peak_gpu_memory_mib")
                n_params = tm.get("num_parameters")

                best_pt = run_dir / "best.pt"
                last_pt = run_dir / "last.pt"

                if legacy:
                    eval_specs = [
                        ("best_val", best_pt, run_dir / "test_eval_best_val"),
                        ("final", last_pt, run_dir / "test_eval_final"),
                    ]
                else:
                    eval_specs = [
                        ("best", best_pt, run_dir / "test_eval"),
                    ]

                for ck_kind, ck_path, ev_out in eval_specs:
                    bleu = chrf = chrfpp = comet = None
                    eval_cmd = [
                        py,
                        str(eval_py),
                        "--checkpoint",
                        rel_under_repo(ck_path),
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
                    if args.no_eval:
                        pass
                    elif args.dry_run:
                        run_command(eval_cmd, cwd=REPO_ROOT, dry_run=True)
                    elif not ck_path.is_file():
                        print(
                            f"[跳过] 无 checkpoint: {ck_path} ({ck_kind})",
                            file=sys.stderr,
                        )
                    else:
                        rc_e = run_command(eval_cmd, cwd=REPO_ROOT, dry_run=False)
                        if rc_e != 0:
                            print(
                                f"[警告] evaluate_test 失败 exp={exp_name} seed={seed} kind={ck_kind}",
                                file=sys.stderr,
                            )
                        else:
                            m = load_metrics_test(ev_out)
                            bleu = m.get("BLEU")
                            chrf = m.get("chrF")
                            chrfpp = m.get(CHRPP_KEY)
                            comet = m.get("COMET")

                    per_seed_rows.append(
                        {
                            "run_name": wandb_name if not legacy else f"{exp_name}/seed_{seed}",
                            "experiment": exp_name,
                            "attention_type": att,
                            "n_heads": n_heads,
                            "seed": seed,
                            "checkpoint_kind": ck_kind,
                            "checkpoint_path": str(ck_path) if ck_path.is_file() else "",
                            "BLEU": bleu,
                            "chrF": chrf,
                            CHRPP_KEY: chrfpp,
                            "COMET": comet,
                            "wall_time_seconds": wall,
                            "peak_gpu_memory_mib": peak_mib,
                            "num_parameters": n_params,
                        }
                    )

    # --- per-seed CSV ---
    per_seed_path = results_dir / "ablation_per_seed.csv"
    fields = [
        "run_name",
        "experiment",
        "attention_type",
        "n_heads",
        "seed",
        "checkpoint_kind",
        "checkpoint_path",
        "BLEU",
        "chrF",
        CHRPP_KEY,
        "COMET",
        "wall_time_seconds",
        "peak_gpu_memory_mib",
        "num_parameters",
    ]
    if not args.dry_run:
        with per_seed_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in per_seed_rows:
                w.writerow(row)

    # --- aggregate by experiment × checkpoint_kind ---
    summary_rows: list[dict[str, Any]] = []
    keys_exp_ck = {(r["experiment"], r["checkpoint_kind"]) for r in per_seed_rows}
    for exp_name, ck_kind in sorted(keys_exp_ck):
        subset = [r for r in per_seed_rows if r["experiment"] == exp_name and r["checkpoint_kind"] == ck_kind]
        bleus = [r.get("BLEU") for r in subset]
        chrfs = [r.get("chrF") for r in subset]
        chrfpps = [r.get(CHRPP_KEY) for r in subset]
        comets = [r.get("COMET") for r in subset]
        walls = [r.get("wall_time_seconds") for r in subset if r.get("wall_time_seconds") is not None]
        peaks = [r.get("peak_gpu_memory_mib") for r in subset if r.get("peak_gpu_memory_mib") is not None]
        np0 = next((r.get("num_parameters") for r in subset if r.get("num_parameters")), None)

        bm, bs = mean_std_tuple([float(x) if x is not None else None for x in bleus])
        cm, cs = mean_std_tuple([float(x) if x is not None else None for x in chrfs])
        cpm, cps = mean_std_tuple([float(x) if x is not None else None for x in chrfpps])
        om, os_ = mean_std_tuple([float(x) if x is not None else None for x in comets])
        wm, ws = mean_std_tuple([float(x) for x in walls])
        pm, ps = mean_std_tuple([float(x) for x in peaks])

        summary_rows.append(
            {
                "experiment": exp_name,
                "checkpoint_kind": ck_kind,
                "bleu_mean": bm,
                "bleu_std": bs,
                "chrf_mean": cm,
                "chrf_std": cs,
                "chrfpp_mean": cpm,
                "chrfpp_std": cps,
                "comet_mean": om,
                "comet_std": os_,
                "train_time_mean_sec": wm,
                "train_time_std_sec": ws,
                "peak_gpu_mib_mean": pm,
                "peak_gpu_mib_std": ps,
                "num_parameters": np0,
                "n_seeds": len(subset),
            }
        )

    summary_csv = results_dir / "ablation_summary.csv"
    if not args.dry_run:
        sfields = list(summary_rows[0].keys()) if summary_rows else []
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            if summary_rows:
                w = csv.DictWriter(f, fieldnames=sfields)
                w.writeheader()
                for row in summary_rows:
                    w.writerow(row)

    # --- Markdown ---
    md_path = results_dir / "ablation_summary.md"
    seeds_note = ", ".join(str(s) for s in seeds)
    run_pat = (
        "`runs/<experiment>/seed_<seed>/`（legacy）"
        if legacy
        else "`runs/fast_dot_s1` … `runs/fast_add_s3`（flat）"
    )
    lines = [
        "# Attention ablation summary",
        "",
        f"Seeds: `{seeds_note}`；训练目录：{run_pat}。"
        + (" **`--aggregate-only`**：从磁盘汇总。" if args.aggregate_only else ""),
        f"同一 `train.tsv` / `val.tsv` / `test.tsv`；`max_steps={max_steps}`（legacy 未指定时为 Config 默认）。",
        "Test 指标在独立 `test.tsv` 上由 `evaluate_test.py` 计算。",
        "",
        "## Aggregate (mean ± std over seeds)",
        "",
        "| experiment | checkpoint | mean BLEU | std BLEU | mean chrF++ | std chrF++ | mean COMET | std COMET | chrF (mean±std) | train time (s) | peak GPU (MiB) | num params |",
        "|------------|------------|-----------|----------|-------------|------------|------------|-----------|-----------------|----------------|----------------|------------|",
    ]
    for row in summary_rows:
        exp = row["experiment"]
        ck = row["checkpoint_kind"]
        bmean = row["bleu_mean"]
        bstd = row["bleu_std"]
        mean_bleu_s = f"{bmean:.4f}" if bmean is not None else "n/a"
        std_bleu_s = f"{bstd:.4f}" if bstd is not None else "n/a"

        cpm = row["chrfpp_mean"]
        cps = row["chrfpp_std"]
        mean_chrfpp_s = f"{cpm:.4f}" if cpm is not None else "n/a"
        std_chrfpp_s = f"{cps:.4f}" if cps is not None else "n/a"

        om = row["comet_mean"]
        os_ = row["comet_std"]
        mean_comet_s = f"{om:.4f}" if om is not None else "n/a"
        std_comet_s = f"{os_:.4f}" if os_ is not None else "n/a"

        chrf_combo = fmt_mean_std(row["chrf_mean"], row["chrf_std"])
        tw = fmt_mean_std(row["train_time_mean_sec"], row["train_time_std_sec"], nd=1)
        pg = fmt_mean_std(row["peak_gpu_mib_mean"], row["peak_gpu_mib_std"], nd=1)
        np_s = str(row["num_parameters"]) if row["num_parameters"] is not None else "n/a"
        lines.append(
            f"| {exp} | {ck} | {mean_bleu_s} | {std_bleu_s} | {mean_chrfpp_s} | {std_chrfpp_s} | "
            f"{mean_comet_s} | {std_comet_s} | {chrf_combo} | {tw} | {pg} | {np_s} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed detail",
            "",
            f"完整行级结果见 `{per_seed_path.relative_to(REPO_ROOT)}`。",
            "",
        ]
    )

    if not args.dry_run:
        md_path.write_text("\n".join(lines), encoding="utf-8")

    if args.dry_run:
        print(
            "[dry-run] 未写入 CSV/Markdown；以上为可复制执行的完整命令。",
            file=sys.stderr,
        )
    else:
        print(f"Wrote {per_seed_path}", file=sys.stderr)
        print(f"Wrote {summary_csv}", file=sys.stderr)
        print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
