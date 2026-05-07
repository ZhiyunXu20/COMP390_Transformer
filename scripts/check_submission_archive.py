#!/usr/bin/env python3
"""Verify repository contents against docs/CODE_ARCHIVE_CHECKLIST.md (submission readiness)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# When run as `python scripts/check_submission_archive.py`, `scripts/` is on sys.path[0].
from prepare_code_archive import infer_repo_root, validate

Status = Literal["PASS", "WARN", "FAIL"]


@dataclass
class Category:
    name: str
    status: Status = "PASS"
    lines: list[str] = field(default_factory=list)

    def add(self, level: Status, msg: str) -> None:
        self.lines.append(f"- [{level}] {msg}")
        if level == "FAIL":
            self.status = "FAIL"
        elif level == "WARN" and self.status != "FAIL":
            self.status = "WARN"


REPORTED_RUNS = (
    "fast_dot",
    "fast_add",
    "head_1h_dot",
    "swap_fr_dot",
    "var_bilinear",
    "var_gated_dot_additive",
    "var_sparsemax",
    "var_entmax15",
    "var_local_window",
    "var_global_local",
)

CODE_DIRS = (
    "small_try",
    "small_head",
    "small_swap",
    "base_1",
    "base_improve",
    "scripts",
    "experiments",
    "tests",
    "translate_cli",
)

RESULTS_FILES = (
    "attention_variants_test_summary.md",
    "attention_variants_test_summary.csv",
    "cross_seed_significance.md",
    "cross_seed_significance.json",
    "variant_fairness_audit.md",
    "variant_fairness_audit.json",
    "runs_sanity_report.md",
    "runs_sanity_report.json",
    "ablation_summary.md",
    "ablation_summary.csv",
    "ablation_per_seed.csv",
    "significance_fast_dot_vs_fast_add.md",
    "significance_fast_dot_vs_fast_add.json",
)

DOCS_REQUIRED = (
    "RESEARCH_AUDIT.md",
    "ATTENTION_VARIANTS_STATUS.md",
    "THESIS_RESULT_BOUNDARIES.md",
    "REPRODUCIBILITY.md",
    "CODE_ARCHIVE_CHECKLIST.md",
    "ENVIRONMENT.md",
)

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")


def _git_check_ignore(repo: Path, rel_posix: str) -> bool:
    cp = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-q", "--", rel_posix],
        capture_output=True,
        text=True,
    )
    return cp.returncode == 0


def _archive_eligible_bytes(repo: Path) -> int:
    total = 0
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == ".git":
            continue
        rel_posix = rel.as_posix()
        if _git_check_ignore(repo, rel_posix):
            continue
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def run_checks(
    repo: Path, *, max_archive_mib: int | None
) -> tuple[list[Category], list[str]]:
    cats: list[Category] = []
    fail_msgs: list[str] = []

    c_dirs = Category("Python package / script directories")
    for d in CODE_DIRS:
        p = repo / d
        if not p.is_dir():
            c_dirs.add("FAIL", f"missing directory `{d}/`")
            continue
        n_py = sum(1 for _ in p.rglob("*.py"))
        if n_py == 0:
            c_dirs.add("WARN", f"no `.py` files under `{d}/`")
        else:
            c_dirs.add("PASS", f"`{d}/`: {n_py} Python files")
    cats.append(c_dirs)

    c_data = Category("data/splits and tokenizers")
    split_dir = repo / "data" / "splits" / "en_fr_50k_seed42"
    for name in ("train.tsv", "val.tsv", "test.tsv", "split_metadata.json", "manifest.json"):
        fp = split_dir / name
        if fp.is_file():
            c_data.add("PASS", f"`{fp.relative_to(repo)}`")
        else:
            c_data.add("FAIL", f"missing `{fp.relative_to(repo)}`")
    tok_meta = repo / "data" / "tokenizer_metadata.json"
    if tok_meta.is_file():
        c_data.add("PASS", "`data/tokenizer_metadata.json`")
    else:
        c_data.add("WARN", "missing `data/tokenizer_metadata.json` (optional if unused)")
    tok_json = list((repo / "data").glob("tokenizer*.json"))
    if tok_json:
        c_data.add("PASS", f"`data/tokenizer*.json` count={len(tok_json)}")
    else:
        c_data.add("FAIL", "no `data/tokenizer*.json`")
    cats.append(c_data)

    c_di = Category("Data integrity (split TSV SHA-256)")
    script = repo / "scripts" / "check_data_integrity.py"
    if not script.is_file():
        c_di.add("FAIL", "missing `scripts/check_data_integrity.py`")
    else:
        cp = subprocess.run(
            [sys.executable, str(script), "--repo-root", str(repo)],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        out = ((cp.stderr or "").strip() + "\n" + (cp.stdout or "").strip()).strip()
        if cp.returncode == 0:
            c_di.add("PASS", "`scripts/check_data_integrity.py` OK → `results/data_integrity_report.md`")
        else:
            c_di.add(
                "FAIL",
                "`scripts/check_data_integrity.py` FAILED (see `results/data_integrity_report.md`)",
            )
            if out:
                c_di.add("FAIL", out[:1200])
    cats.append(c_di)

    c_ap = Category("absolute path caveats (V9-A31, tracked files via git grep)")
    ap_script = repo / "scripts" / "check_absolute_path_caveats.py"
    if not ap_script.is_file():
        c_ap.add("FAIL", "missing `scripts/check_absolute_path_caveats.py`")
    elif not (repo / ".git").is_dir():
        c_ap.add("WARN", "not a git checkout; skipped `check_absolute_path_caveats.py`")
    else:
        cp = subprocess.run(
            [sys.executable, str(ap_script), "--repo-root", str(repo)],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            c_ap.add(
                "FAIL",
                "`check_absolute_path_caveats.py` failed (see `results/absolute_path_caveats_audit.md`)",
            )
            err = (cp.stderr or "").strip()
            if err:
                c_ap.add("FAIL", err[:1200])
        else:
            c_ap.add(
                "PASS",
                "`check_absolute_path_caveats.py` OK → `results/absolute_path_caveats_audit.md`",
            )
            for legacy in (repo / "base_1" / "config.py", repo / "base_improve" / "config.py"):
                if not legacy.is_file():
                    continue
                try:
                    if "/root/autodl-tmp" in legacy.read_text(encoding="utf-8", errors="replace"):
                        c_ap.add(
                            "WARN",
                            f"Legacy `{legacy.relative_to(repo).as_posix()}` still embeds `/root/autodl-tmp`; "
                            "see `docs/PROVENANCE_CAVEATS.md` (main `small_*` stacks use relative paths).",
                        )
                        break
                except OSError:
                    pass
    cats.append(c_ap)

    c_runs = Category("10 reported runs — test_eval + core JSON")
    for run in REPORTED_RUNS:
        rd = repo / "runs" / run
        if not rd.is_dir():
            c_runs.add("FAIL", f"missing `runs/{run}/`")
            continue
        for rel in (
            "metrics.json",
            "resolved_config.json",
            "training_meta.json",
            "test_eval/metrics_test.json",
            "test_eval/predictions.jsonl",
            "test_eval/examples.md",
        ):
            fp = rd / rel
            if fp.is_file():
                c_runs.add("PASS", f"`runs/{run}/{rel}`")
            else:
                c_runs.add("FAIL", f"missing `runs/{run}/{rel}`")
    cats.append(c_runs)

    c_pc = Category("Checkpoints — must be gitignored (not shipped)")
    leaked: list[str] = []
    for suf in CHECKPOINT_SUFFIXES:
        for path in repo.rglob(f"*{suf}"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(repo)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == ".git":
                continue
            rel_posix = rel.as_posix()
            if not _git_check_ignore(repo, rel_posix):
                leaked.append(rel_posix)
    if leaked:
        for x in sorted(leaked)[:20]:
            c_pc.add("FAIL", f"checkpoint not ignored by git: `{x}` (would enter naive zip)")
        if len(leaked) > 20:
            c_pc.add("FAIL", f"... and {len(leaked) - 20} more")
    else:
        c_pc.add(
            "PASS",
            "no unchecked checkpoint paths (or no checkpoints; `.gitignore` covers `*.pt` etc.)",
        )
    cats.append(c_pc)

    c_res = Category("results/* summaries")
    rdir = repo / "results"
    if not rdir.is_dir():
        c_res.add("FAIL", "missing `results/`")
    else:
        for name in RESULTS_FILES:
            fp = rdir / name
            if fp.is_file():
                c_res.add("PASS", f"`results/{name}`")
            else:
                c_res.add("FAIL", f"missing `results/{name}`")
    cats.append(c_res)

    c_docs = Category("required docs/*.md")
    ddir = repo / "docs"
    if not ddir.is_dir():
        c_docs.add("FAIL", "missing `docs/`")
    else:
        for name in DOCS_REQUIRED:
            fp = ddir / name
            if fp.is_file():
                c_docs.add("PASS", f"`docs/{name}`")
            else:
                c_docs.add("FAIL", f"missing `docs/{name}`")
    cats.append(c_docs)

    c_root = Category("repository root files")
    for name in ("README.md", "requirements.txt", ".gitignore"):
        fp = repo / name
        if fp.is_file():
            c_root.add("PASS", f"`{name}`")
        else:
            c_root.add("FAIL", f"missing `{name}`")
    cats.append(c_root)

    cef = repo / "data" / "EN-FR.txt"
    c_forbid = Category("forbidden / noisy paths (policy)")
    if cef.is_file():
        mib = cef.stat().st_size / (1024 * 1024)
        if mib > 200:
            c_forbid.add(
                "WARN",
                f"large `data/EN-FR.txt` (~{mib:.0f} MiB); exclude from submission zip",
            )
        else:
            c_forbid.add(
                "WARN",
                f"`data/EN-FR.txt` present (~{mib:.1f} MiB); confirm checklist before shipping",
            )
    wd = repo / "wandb"
    if wd.is_dir() and any(wd.iterdir()):
        c_forbid.add("WARN", "`wandb/` non-empty — should be gitignored / omitted from archive")
    n_log = 0
    for log in repo.rglob("*.log"):
        try:
            rel = log.relative_to(repo)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == ".git":
            continue
        c_forbid.add("WARN", f"`.log` (consider omitting): `{rel.as_posix()}`")
        n_log += 1
        if n_log >= 12:
            c_forbid.add("WARN", "(suppressing further `.log` listings)")
            break
    if not c_forbid.lines:
        c_forbid.add("PASS", "no EN-FR / wandb / `.log` policy warnings")
    cats.append(c_forbid)

    c_pcval = Category("prepare_code_archive.validate (test_eval pairs + CSV)")
    v = validate(repo, allow_missing_predictions=False)
    if v == 0:
        c_pcval.add("PASS", "`prepare_code_archive.validate` OK")
    else:
        c_pcval.add("FAIL", "`prepare_code_archive.validate` reported missing artifacts")
    cats.append(c_pcval)

    if max_archive_mib is not None:
        c_sz = Category("estimated archive size (non-ignored files)")
        b = _archive_eligible_bytes(repo)
        mib = b / (1024 * 1024)
        if mib > max_archive_mib:
            c_sz.add(
                "WARN",
                f"estimated non-ignored tree ~{mib:.1f} MiB exceeds --max-archive-size-mib={max_archive_mib}",
            )
        else:
            c_sz.add(
                "PASS",
                f"estimated non-ignored tree ~{mib:.1f} MiB (limit {max_archive_mib} MiB)",
            )
        cats.append(c_sz)
    else:
        c_sz = Category("estimated archive size (non-ignored files)")
        b = _archive_eligible_bytes(repo)
        mib = b / (1024 * 1024)
        c_sz.add(
            "PASS",
            f"~{mib:.1f} MiB for files not matched by `git check-ignore` "
            "(local `du -sh runs/` may be much larger if checkpoints exist but are ignored)",
        )
        cats.append(c_sz)

    for c in cats:
        if c.status == "FAIL":
            fail_msgs.append(c.name)

    return cats, fail_msgs


def write_report(repo: Path, cats: list[Category], *, strict: bool) -> Path:
    out = repo / "results" / "submission_archive_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    overall = "FAIL" if any(c.status == "FAIL" for c in cats) else (
        "WARN" if any(c.status == "WARN" for c in cats) else "PASS"
    )
    lines = [
        "# Submission archive check",
        "",
        f"- **repository**: `{repo}`",
        f"- **--strict**: `{strict}`",
        f"- **overall**: **{overall}**",
        "",
    ]
    for c in cats:
        lines.append(f"## {c.name}")
        lines.append("")
        lines.append(f"- **status**: **{c.status}**")
        lines.extend(c.lines or ["- (no messages)"])
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit 1 if any category is FAIL (default: true)",
    )
    ap.add_argument(
        "--max-archive-size-mib",
        type=int,
        default=None,
        help="Warn if estimated non-gitignored file size exceeds this (MiB)",
    )
    args = ap.parse_args()
    repo = infer_repo_root()

    cats, _ = run_checks(repo, max_archive_mib=args.max_archive_size_mib)
    report_path = write_report(repo, cats, strict=args.strict)

    print(str(report_path), file=sys.stderr)
    text = report_path.read_text(encoding="utf-8")
    print(text)

    if args.strict and any(c.status == "FAIL" for c in cats):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
