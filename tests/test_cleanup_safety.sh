#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO/cleanup_training_artifacts.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

echo "== test 1: from repository root with --dry-run -> exit 2"
cd "$REPO"
set +e
out="$("$SCRIPT" --dry-run 2>&1)"
ec=$?
set -e
[[ "$ec" -eq 2 ]] || fail "expected exit 2 from repo root, got $ec; output:\n$out"
[[ "$out" == *ERROR* ]] || fail "expected ERROR in stderr/stdout; got:\n$out"

echo "== test 2: from temp dir, fake repo layout -> exit 0 and [dry-run] would remove"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/fake_repo/small_try/runs"
echo x > "$tmp/fake_repo/small_try/runs/x"
cp "$SCRIPT" "$tmp/fake_repo/cleanup_training_artifacts.sh"
mkdir -p "$tmp/workdir"
cd "$tmp/workdir"
set +e
out="$(bash "$tmp/fake_repo/cleanup_training_artifacts.sh" --dry-run 2>&1)"
ec=$?
set -e
[[ "$ec" -eq 0 ]] || fail "expected exit 0 from non-repo cwd, got $ec; output:\n$out"
[[ "$out" == *"[dry-run] would remove:"* ]] || fail "expected [dry-run] would remove: in output; got:\n$out"

echo "OK: cleanup safety tests passed"
