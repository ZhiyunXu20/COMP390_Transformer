#!/usr/bin/env python3
"""Audit `/root/autodl-tmp` string occurrences; ensure they are only documented provenance.

With **`.git`**: uses **`git grep`** so only **tracked** files are considered (local `logs/`,
backups, wandb caches, etc. are ignored if untracked or gitignored).

**Without `.git`** (e.g. zip extract): recursive **filesystem scan** with directory excludes;
allowlist rules are unchanged.

Exit 0: all hits are allowlisted; `small_try` / `small_head` / `small_swap`
`config.py` contain no needle.
Exit 1: unexpected file contains the needle, or main `config.py` embeds it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

NEEDLE = "/root/autodl-tmp"

# Scripts allowed to mention the training host path (comments / docstrings only expected).
SCRIPTS_ALLOW_EXACT: frozenset[str] = frozenset(
    {
        "scripts/check_absolute_path_caveats.py",
        "scripts/check_submission_archive.py",
        "scripts/test_mt_eval_metrics.py",
        "scripts/test_mt_eval_metrics_dummy.py",
    }
)

MAIN_CONFIG_RELPATHS: tuple[str, ...] = (
    "small_try/config.py",
    "small_head/config.py",
    "small_swap/config.py",
)

_FS_EXCLUDES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        "venv",
        ".venv",
        "node_modules",
        "wandb",
        "logs",
        "runs",
        ".git",
    }
)

_FS_SKIP_SUFFIXES: tuple[str, ...] = (
    ".pt",
    ".bin",
    ".safetensors",
    ".npy",
    ".npz",
    ".gz",
    ".zip",
    ".tar",
    ".jpg",
    ".png",
)


def _is_probably_binary(blob: bytes) -> bool:
    return b"\0" in blob[:8192]


def file_contains_needle(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if _is_probably_binary(data):
        return False
    if len(data) > 8 * 1024 * 1024:
        return NEEDLE.encode() in data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return NEEDLE in text


def _git_grep_paths(repo: Path, pattern: str) -> list[str]:
    """Use git grep when `.git` is present (tracked files only)."""
    result = subprocess.run(
        ["git", "grep", "-l", "--fixed-strings", pattern],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git grep failed ({result.returncode}): {msg}")
    if not result.stdout.strip():
        return []
    return sorted({ln.strip() for ln in result.stdout.splitlines() if ln.strip()})


def _fs_grep_paths(
    repo: Path,
    pattern: str,
    *,
    excludes: frozenset[str] = _FS_EXCLUDES,
) -> list[str]:
    """Recursive file scan when `.git` is absent."""
    hits: list[str] = []
    repo = repo.resolve()
    for root, dirs, files in os.walk(repo, topdown=True):
        dirs[:] = [d for d in dirs if d not in excludes]
        for f in files:
            if f.endswith(_FS_SKIP_SUFFIXES):
                continue
            p = Path(root) / f
            try:
                if p.stat().st_size > 10_000_000:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern in text:
                try:
                    rel = p.relative_to(repo).as_posix()
                except ValueError:
                    continue
                hits.append(rel)
    return sorted(set(hits))


def is_allowlisted_tracked(rel_posix: str) -> bool:
    if rel_posix.startswith("runs/"):
        return True
    if rel_posix.startswith("results/"):
        return True
    if rel_posix.startswith("docs/") and rel_posix.endswith(".md"):
        return True
    if rel_posix.startswith(("base_1/", "base_improve/")):
        return True
    if rel_posix in (
        "data/splits/en_fr_50k_seed42/manifest.json",
        "data/splits/en_fr_50k_seed42/split_metadata.json",
        "data/tokenizer_metadata.json",
    ):
        return True
    if rel_posix.startswith(("small_try/", "small_head/", "small_swap/")) and "/results/" in rel_posix:
        return True
    p = Path(rel_posix)
    if len(p.parts) == 1 and p.suffix == ".sh":
        return True
    if rel_posix.startswith("scripts/"):
        return rel_posix in SCRIPTS_ALLOW_EXACT
    return False


def check_main_configs_clean(repo: Path) -> list[str]:
    """Critical check: active small_* config defaults must not embed the training-host path."""
    bad: list[str] = []
    for rel_s in MAIN_CONFIG_RELPATHS:
        p = repo / rel_s
        if not p.is_file():
            continue
        if file_contains_needle(p):
            bad.append(rel_s)
    return bad


def partition_hits(hits: list[str]) -> tuple[list[str], list[str]]:
    allow: list[str] = []
    bad: list[str] = []
    for h in hits:
        if is_allowlisted_tracked(h):
            allow.append(h)
        else:
            bad.append(h)
    return sorted(allow), sorted(bad)


def write_report(
    repo: Path,
    *,
    audit_mode: str,
    allowed: list[str],
    unexpected: list[str],
    main_bad: list[str],
) -> Path:
    out = repo / "results" / "absolute_path_caveats_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if "git" in audit_mode.lower():
        scope = "Tracked files only (`git grep`). See **`docs/PROVENANCE_CAVEATS.md`**."
    else:
        scope = (
            "Filesystem scan (no `.git`); directory excludes match the audit script "
            "(e.g. `runs/`, `logs/`, `wandb/`). Allowlist rules are unchanged. "
            "See **`docs/PROVENANCE_CAVEATS.md`**."
        )
    lines = [
        "# Absolute path caveats audit (`/root/autodl-tmp`)",
        "",
        f"**Audit mode**: {audit_mode}",
        "",
        scope,
        "",
        "## Main pipeline configs (`small_*`)",
        "",
    ]
    if main_bad:
        lines.append("**FAIL**: the following embed absolute training-host paths:")
        for m in main_bad:
            lines.append(f"- `{m}`")
    else:
        lines.append(
            "**PASS**: `small_try` / `small_head` / `small_swap` `config.py` contain no `/root/autodl-tmp`."
        )
    lines.extend(["", "## Allowlisted files (historical provenance)", ""])
    for a in allowed:
        lines.append(f"- `{a}`")
    lines.extend(["", "## Unexpected files", ""])
    if unexpected:
        for u in unexpected:
            lines.append(f"- `{u}` **← needs review or allowlist update**")
    else:
        lines.append("(none)")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    repo = args.repo_root.resolve()

    main_bad = check_main_configs_clean(repo)

    if (repo / ".git").is_dir():
        try:
            hits = _git_grep_paths(repo, NEEDLE)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        audit_mode = "git-grep (tracked files)"
    else:
        hits = _fs_grep_paths(repo, NEEDLE)
        audit_mode = "filesystem scan (no .git available)"
        print("INFO: .git not found; using filesystem scan fallback.", file=sys.stderr)

    allowed, unexpected = partition_hits(hits)
    write_report(repo, audit_mode=audit_mode, allowed=allowed, unexpected=unexpected, main_bad=main_bad)

    if main_bad:
        print("FAIL: active config files contain absolute paths: " + ", ".join(main_bad), file=sys.stderr)
        return 1
    if unexpected:
        print("FAIL: file contains /root/autodl-tmp outside documented provenance", file=sys.stderr)
        for u in unexpected:
            print(f"  - {u}", file=sys.stderr)
        return 1
    print("All /root/autodl-tmp occurrences are documented historical provenance.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
