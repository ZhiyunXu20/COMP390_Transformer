#!/usr/bin/env bash
# 将本地 main 推送到 origin（你的 fork）。不会修改远端已有分支，仅新增/更新 refs/heads/main。
# 使用前：在 GitHub 生成有 repo 权限的 PAT，写入仓库根目录 token.txt（勿提交，已在 .gitignore）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck source=/dev/null
  source /etc/network_turbo
fi

if [[ ! -f "$ROOT/token.txt" ]]; then
  echo "缺少 $ROOT/token.txt（GitHub PAT）" >&2
  exit 1
fi
TOKEN=$(tr -d '\n\r ' < "$ROOT/token.txt")
printf 'https://x-access-token:%s@github.com\n' "$TOKEN" > "$ROOT/.git-credentials.local"
chmod 600 "$ROOT/.git-credentials.local"

git config --local credential.helper "store --file=$ROOT/.git-credentials.local" 2>/dev/null || true
git push -u origin main "$@"
