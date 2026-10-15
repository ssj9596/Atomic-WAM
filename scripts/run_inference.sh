#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SERVER_PY="${SERVER_PY:-$ROOT/cosmos-framework/.venv/bin/python}"
export ROBO_PYTHON="${ROBO_PYTHON:-$ROOT/robolab/.venv/bin/python}"
: "${ROBOLAB_ASSETS:?Set ROBOLAB_ASSETS to the official RoboLab assets directory}"
for python in "$SERVER_PY" "$ROBO_PYTHON"; do
  [[ -x "$python" ]] || { echo "Python environment not found: $python. Run scripts/setup_env.sh first." >&2; exit 1; }
done
[[ -d "$ROOT/cosmos-framework/checkpoints/wam" ]] || { echo 'Download weights with scripts/download_weights.py first.' >&2; exit 1; }
exec bash "$ROOT/robolab/scripts/run_atomic_wam_eval.sh" "$@"
