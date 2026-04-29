#!/usr/bin/env python3
"""多 seed × 注意力类型消融：训练（val 早停挑 best）→ test 终评 → predictions.jsonl + metrics.json。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
for path in (ROOT, _SCRIPTS):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from ablation_lib import evaluate_checkpoint_on_test, merge_ablation_metrics_json


def main() -> None:
    p = argparse.ArgumentParser(description="消融：多 seed 训练 + test 终评")
    p.add_argument(
        "--attention-types",
        nargs="+",
        required=True,
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
        metavar="TYPE",
    )
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument(
        "--config",
        type=str,
        default="small_try",
        choices=("small_try", "small_head", "small_swap"),
    )
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument(
        "--test-path",
        type=str,
        default="data/splits/en_fr_50k_seed42/test.tsv",
        help="划分 test.tsv（相对仓库根）",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="runs/ablation",
        help="相对仓库根的输出根目录",
    )
    p.add_argument("--bleu-sample-size", type=int, default=256)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将执行的命令，不运行",
    )
    args = p.parse_args()

    repo = ROOT
    train_py = repo / args.config / "train.py"
    if not train_py.is_file():
        raise SystemExit(f"找不到 {train_py}")

    out_root = (repo / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    extra_train = []
    if args.config in ("small_head", "small_swap"):
        extra_train.append("--allow-additive-attention")

    for attn in args.attention_types:
        for seed in args.seeds:
            run_name = f"abl_{attn}_seed{seed}"
            run_dir = out_root / run_name
            cmd = [
                sys.executable,
                str(train_py),
                "--attention-type",
                attn,
                "--seed",
                str(seed),
                "--max-steps",
                str(args.max_steps),
                "--test-path",
                args.test_path,
                "--eval-split",
                "val",
                "--output-dir",
                str(out_root),
                "--name",
                run_name,
                "--no-wandb",
                "--no-attention-plots",
                "--eval-light",
                "--bleu-sample-size",
                str(args.bleu_sample_size),
            ]
            cmd.extend(extra_train)

            print("[run_ablation]", " ".join(cmd), flush=True)
            if args.dry_run:
                continue

            env = dict(**__import__("os").environ)
            pr = subprocess.run(cmd, cwd=str(repo))
            if pr.returncode != 0:
                raise SystemExit(pr.returncode)

            try:
                test_eval = evaluate_checkpoint_on_test(
                    repo,
                    args.config,
                    run_dir,
                    bleu_sample_size=args.bleu_sample_size,
                )
                merge_ablation_metrics_json(run_dir, test_eval)
            except Exception as e:
                print(f"[run_ablation] test 终评失败 ({run_dir}): {e}", file=sys.stderr)
                raise


if __name__ == "__main__":
    main()
