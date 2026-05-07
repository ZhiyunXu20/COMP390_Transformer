#!/usr/bin/env python3
"""Audit `/root/autodl-tmp` string occurrences; ensure they are only documented provenance.

Uses **`git grep`** so only **tracked** files are considered (local `logs/`, backups,
wandb caches, etc. are ignored if untracked or gitignored).

Exit 0: all tracked hits are allowlisted; `small_try` / `small_head` / `small_swap`
`config.py` contain no needle.
Exit 1: unexpected tracked file contains the needle, or main `config.py` embeds it.
"""

from __future__ import annotations

import argparse
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


def tracked_files_with_needle(repo: Path) -> list[str]:
    cp = subprocess.run(
        ["git", "-C", str(repo), "grep", "-l", "--fixed-strings", NEEDLE],
        capture_output=True,
        text=True,
    )
    if cp.returncode not in (0, 1):
        print(cp.stderr or cp.stdout, file=sys.stderr)
        raise SystemExit(2)
    return sorted({ln.strip() for ln in cp.stdout.splitlines() if ln.strip()})


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
    bad: list[str] = []
    for rel_s in MAIN_CONFIG_RELPATHS:
        p = repo / rel_s
        if not p.is_file():
            continue
        if file_contains_needle(p):
            bad.append(rel_s)
    return bad


def audit_tracked(repo: Path) -> tuple[list[str], list[str]]:
    hits = tracked_files_with_needle(repo)
    allow: list[str] = []
    bad: list[str] = []
    for h in hits:
        if is_allowlisted_tracked(h):
            allow.append(h)
        else:
            bad.append(h)
    return allow, bad


def write_report(
    repo: Path,
    *,
    allowed: list[str],
    unexpected: list[str],
    main_bad: list[str],
) -> Path:
    out = repo / "results" / "absolute_path_caveats_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Absolute path caveats audit (`/root/autodl-tmp`)",
        "",
        "Tracked files only (`git grep`). See **`docs/PROVENANCE_CAVEATS.md`**.",
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
    lines.extend(["", "## Allowlisted tracked files (historical provenance)", ""])
    for a in allowed:
        lines.append(f"- `{a}`")
    lines.extend(["", "## Unexpected tracked files", ""])
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
    if not (repo / ".git").is_dir():
        print("FAIL: not a git repository; cannot run git-grep audit", file=sys.stderr)
        return 1

    main_bad = check_main_configs_clean(repo)
    allowed, unexpected = audit_tracked(repo)
    write_report(repo, allowed=allowed, unexpected=unexpected, main_bad=main_bad)

    if main_bad:
        print("FAIL: main small_* config embeds /root/autodl-tmp", file=sys.stderr)
        return 1
    if unexpected:
        print("FAIL: tracked file contains /root/autodl-tmp outside documented provenance", file=sys.stderr)
        for u in unexpected:
            print(f"  - {u}", file=sys.stderr)
        return 1
    print("All /root/autodl-tmp occurrences are documented historical provenance.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
