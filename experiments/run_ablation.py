#!/usr/bin/env python3
"""
多 seed attention ablation：批量训练 small_try（或其它 pkg）、test 上评估 best.pt 与 last.pt，并汇总 CSV/Markdown。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SPLITS = "data/splits/en_fr_50k_seed42"

ABLATION_SPECS: list[tuple[str, str, int]] = [
    ("dot_h4", "dot_product", 4),
    ("add_h4", "additive", 4),
    ("dot_h1", "dot_product", 1),
    ("add_h1", "additive", 1),
]


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


def run_cmd(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    cp = subprocess.run(cmd, cwd=str(cwd))
    return int(cp.returncode)


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
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--skip-add-h1", action="store_true", help="跳过 additive n_heads=1 组合")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="仅训练，不调用 evaluate_test.py")
    p.add_argument(
        "--wandb",
        action="store_true",
        help="启用 W&B（默认关闭，批量实验建议保持关闭）",
    )
    args = p.parse_args()

    specs = list(ABLATION_SPECS)
    if args.skip_add_h1:
        specs = [s for s in specs if not (s[1] == "additive" and s[2] == 1)]

    seeds = (42, 43, 44)
    py = sys.executable
    train_py = REPO_ROOT / args.pkg / "train.py"
    eval_py = REPO_ROOT / "evaluate_test.py"
    if not train_py.is_file():
        raise SystemExit(f"找不到训练脚本: {train_py}")
    if not eval_py.is_file():
        raise SystemExit(f"找不到 evaluate_test.py: {eval_py}")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []

    for exp_name, att, n_heads in specs:
        for seed in seeds:
            run_dir = REPO_ROOT / "runs" / exp_name / f"seed_{seed}"

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
                str(Path("runs") / exp_name),
                "--name",
                f"seed_{seed}",
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
            if args.max_steps is not None:
                train_cmd.extend(["--max-steps", str(args.max_steps)])
            if args.batch_size is not None:
                train_cmd.extend(["--batch-size", str(args.batch_size)])

            rc = run_cmd(train_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
            if rc != 0:
                print(f"[错误] 训练失败 exp={exp_name} seed={seed} rc={rc}", file=sys.stderr)
                sys.exit(rc)

            tm = load_training_meta(run_dir)
            wall = tm.get("wall_time_seconds")
            peak_mib = tm.get("peak_gpu_memory_mib")
            n_params = tm.get("num_parameters")

            best_pt = run_dir / "best.pt"
            last_pt = run_dir / "last.pt"

            eval_specs = [
                ("best_val", best_pt, run_dir / "test_eval_best_val"),
                ("final", last_pt, run_dir / "test_eval_final"),
            ]

            for ck_kind, ck_path, ev_out in eval_specs:
                bleu = chrf = comet = None
                if args.no_eval or args.dry_run:
                    pass
                elif not ck_path.is_file():
                    print(
                        f"[跳过] 无 checkpoint: {ck_path} ({ck_kind})",
                        file=sys.stderr,
                    )
                else:
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
                    rc_e = run_cmd(eval_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
                    if rc_e != 0:
                        print(
                            f"[警告] evaluate_test 失败 exp={exp_name} seed={seed} kind={ck_kind}",
                            file=sys.stderr,
                        )
                    else:
                        m = load_metrics_test(ev_out)
                        bleu = m.get("BLEU")
                        chrf = m.get("chrF")
                        comet = m.get("COMET")

                per_seed_rows.append(
                    {
                        "experiment": exp_name,
                        "attention_type": att,
                        "n_heads": n_heads,
                        "seed": seed,
                        "checkpoint_kind": ck_kind,
                        "checkpoint_path": str(ck_path) if ck_path.is_file() else "",
                        "BLEU": bleu,
                        "chrF": chrf,
                        "COMET": comet,
                        "wall_time_seconds": wall,
                        "peak_gpu_memory_mib": peak_mib,
                        "num_parameters": n_params,
                    }
                )

    # --- per-seed CSV ---
    per_seed_path = results_dir / "ablation_per_seed.csv"
    fields = [
        "experiment",
        "attention_type",
        "n_heads",
        "seed",
        "checkpoint_kind",
        "checkpoint_path",
        "BLEU",
        "chrF",
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
        comets = [r.get("COMET") for r in subset]
        walls = [r.get("wall_time_seconds") for r in subset if r.get("wall_time_seconds") is not None]
        peaks = [r.get("peak_gpu_memory_mib") for r in subset if r.get("peak_gpu_memory_mib") is not None]
        np0 = next((r.get("num_parameters") for r in subset if r.get("num_parameters")), None)

        bm, bs = mean_std_tuple([float(x) if x is not None else None for x in bleus])
        cm, cs = mean_std_tuple([float(x) if x is not None else None for x in chrfs])
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
    lines = [
        "# Attention ablation summary",
        "",
        "同一 `train.tsv` / `val.tsv` / `test.tsv`；每个组合 3 个 seed（42–44）。",
        "训练目录：`runs/<experiment>/seed_<seed>/`。",
        "**Test 指标**分别来自验证集最优 `best.pt`（`test_eval_best_val`）与训练结束 `last.pt`（`test_eval_final`），二者均在独立 test.tsv 上由 `evaluate_test.py` 计算。",
        "",
        "## Aggregate (mean ± std over seeds)",
        "",
        "| experiment | checkpoint | BLEU | chrF | COMET | train time (s) | peak GPU (MiB) | num params |",
        "|------------|------------|------|------|-------|------------------|----------------|------------|",
    ]
    for row in summary_rows:
        exp = row["experiment"]
        ck = row["checkpoint_kind"]
        bstr = fmt_mean_std(row["bleu_mean"], row["bleu_std"])
        cstr = fmt_mean_std(row["chrf_mean"], row["chrf_std"])
        ostr = fmt_mean_std(row["comet_mean"], row["comet_std"])
        tw = fmt_mean_std(row["train_time_mean_sec"], row["train_time_std_sec"], nd=1)
        pg = fmt_mean_std(row["peak_gpu_mib_mean"], row["peak_gpu_mib_std"], nd=1)
        np_s = str(row["num_parameters"]) if row["num_parameters"] is not None else "n/a"
        lines.append(
            f"| {exp} | {ck} | {bstr} | {cstr} | {ostr} | {tw} | {pg} | {np_s} |"
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

    print(f"Wrote {per_seed_path}", file=sys.stderr)
    print(f"Wrote {summary_csv}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
