#!/usr/bin/env bash
# WARNING: This script cleans subpackage runs/ directories (small_try/runs, base_1/runs, etc.),
# NOT the repository-root runs/. The repository-root runs/ contains the dissertation experiment
# outputs and must never be deleted by this script.
#
# 删除各实验包下的训练产物：包内 wandb_cache、包内 runs、日志与部分报告（不删仓库根 runs/）。
# 用法：请不要在仓库根目录的 cwd 下执行；例如：cd /tmp && bash /path/to/repo/cleanup_training_artifacts.sh [--dry-run]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) ;;
  esac
done

# Safety: refuse to run if invoked from repository root (would risk deleting root runs/)
# (Spec had `-d "$ROOT/.git"` alone; that would refuse every invocation when the script lives
#  at repo root. We require cwd == ROOT for that branch.)
if [[ -d "$ROOT/.git" && "$(pwd -P)" == "$(cd "$ROOT" && pwd -P)" ]] ||
  [[ "$ROOT" == "$(pwd -P)" && -f "$ROOT/README.md" && -d "$ROOT/runs" ]]; then
  echo "ERROR: This script must not delete the repository-root runs/." >&2
  echo "       It only cleans subpackage runs/ (small_try/runs, base_1/runs, etc)." >&2
  exit 2
fi

resolve() {
  realpath -m "$1" 2>/dev/null || readlink -f "$1" 2>/dev/null || echo "$1"
}

verify_not_repo_root_runs() {
  local target="$1"
  local target_real root_real
  target_real="$(resolve "$target")"
  root_real="$(resolve "$ROOT")"
  if [[ "$target_real" == "$root_real/runs" ]]; then
    echo "REFUSED: cleanup target equals repository-root runs/" >&2
    exit 3
  fi
}

remove_tree() {
  local target="$1"
  [[ -n "$target" ]] || return 0
  [[ -e "$target" ]] || return 0
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would remove: $target"
  else
    rm -rf "$target"
  fi
}

remove_files() {
  local f
  for f in "$@"; do
    [[ -e "$f" ]] || continue
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] would remove: $f"
    else
      rm -f "$f"
    fi
  done
}

for extra in "$ROOT/wandb_cache_attention_small" "$ROOT/wandb_cache_attention_small_smoke"; do
  remove_tree "$extra"
done

for name in small_try small_head small_swap base_1 base_improve; do
  d="$ROOT/$name"
  [[ -d "$d" ]] || continue
  if [[ "$d" == "$ROOT" ]]; then
    echo "SKIP: package dir equals repo root (unexpected): $d" >&2
    continue
  fi
  remove_tree "$d/wandb_cache"
  _runs="$d/runs"
  verify_not_repo_root_runs "$_runs"
  remove_tree "$_runs"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would mkdir -p $d/runs $d/results $d/logs"
  else
    mkdir -p "$d/runs" "$d/results" "$d/logs"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    for f in "$d/report.txt" "$d/report_head.txt" "$d/report_swap.txt" "$d/report_improve.txt"; do
      [[ -e "$f" ]] && echo "[dry-run] would remove: $f"
    done
    shopt -s nullglob
    for f in "$d/results"/*; do
      echo "[dry-run] would remove: $f"
    done
    shopt -u nullglob
  else
    rm -f "$d/report.txt" "$d/report_head.txt" "$d/report_swap.txt" "$d/report_improve.txt" 2>/dev/null || true
    rm -f "$d/results"/* 2>/dev/null || true
  fi
  echo "cleaned $d${DRY_RUN:+(dry-run)}"
done

echo "done${DRY_RUN:+(dry-run)}."
