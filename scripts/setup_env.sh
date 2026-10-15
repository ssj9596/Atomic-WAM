#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=all
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --server-only) TARGET=server ;;
    --client-only) TARGET=client ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) echo 'Usage: bash scripts/setup_env.sh [--server-only|--client-only] [--dry-run]'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
run() {
  if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if [[ "$DRY_RUN" != 1 ]]; then
  [[ "$(uname -s)" == Linux ]] || { echo 'Run this installer on Linux with an NVIDIA GPU.' >&2; exit 1; }
  command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install "uv>=0.11.3"' >&2; exit 1; }
fi
if [[ "$TARGET" != client ]]; then
  # The released policy imports shared modules from the train dependency set.
  run uv sync --project "$ROOT/cosmos-framework" --python 3.12 --frozen --no-dev --extra train --group cu128-train --group policy-server
  if [[ "$DRY_RUN" != 1 ]]; then
    (cd "$ROOT/cosmos-framework"; COSMOS_TRAINING=0 .venv/bin/python -m cosmos_framework.scripts.action_policy_server_robolab --help >/dev/null)
  fi
fi
if [[ "$TARGET" != server ]]; then
  run uv sync --project "$ROOT/robolab" --python 3.11 --frozen --no-dev
fi
