#!/usr/bin/env bash
set -euo pipefail

# Portable inference entry point for the copied Atomic-WAM evaluation bundle.
# The source checkout is never consulted; all code and model paths resolve
# relative to this bundle unless the caller explicitly overrides them.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTRUCTION_TYPE_VALUE="${INSTRUCTION_TYPE:-default}"
EVAL_LABEL_VALUE="${EVAL_LABEL:-atomic_wam_lora_reasoner_wam_previous_frame_relative_${INSTRUCTION_TYPE_VALUE}}"
EVAL_BASE_OUTDIR_VALUE="${EVAL_BASE_OUTDIR:-atomic_wam_reasoner_wam_previous_frame_relative_${INSTRUCTION_TYPE_VALUE}}"

exec env \
  ROOT="${ROOT:-${REPRO_ROOT}}" \
  ROBOLAB="${ROBOLAB:-${REPRO_ROOT}/robolab}" \
  ROBOLAB_ASSETS="${ROBOLAB_ASSETS:-}" \
  SERVER_REPO="${SERVER_REPO:-${REPRO_ROOT}/cosmos-framework}" \
  LAUNCHER="${LAUNCHER:-${REPRO_ROOT}/robolab/scripts/repro_run_server.sh}" \
  REASONER_ARTIFACTS="${REASONER_ARTIFACTS:-${REPRO_ROOT}/cosmos-framework/checkpoints/reasoner_lora}" \
  WAM_CHECKPOINT_PATH="${WAM_CHECKPOINT_PATH:-${REPRO_ROOT}/cosmos-framework/checkpoints/wam}" \
  EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${REPRO_ROOT}/cosmos-framework/checkpoints/wam}" \
  GPUS="${GPUS:-0}" \
  EVAL_REPEATS="${EVAL_REPEATS:-1}" \
  EVAL_LABEL="${EVAL_LABEL_VALUE}" \
  EVAL_BASE_OUTDIR="${EVAL_BASE_OUTDIR_VALUE}" \
  DROID_VIDEO_SAMPLE_STRIDE=1 WRIST_ROTATE_180=1 \
  SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:---resolution 544x640}" \
  REASONER_WAM=1 REASONER_WAM_MODE=adapter \
  FORMAT_PROMPT_AS_JSON=1 INCLUDE_EXECUTION_SUCCESS=1 OVERLAY_SUBTASK_ON_VIDEO=1 \
  ROBOLAB_VIDEO_QUALITY=4 \
  ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS=1 \
  ROBOLAB_RELATIVE_GRIPPER_COMMANDS=0 \
  ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS=1 \
  ROBOLAB_GRIPPER_RELATIVE_THRESHOLD="${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD:-0.10}" \
  ROBOLAB_REASONER_HISTORY_IMAGES=1 \
  INSTRUCTION_TYPE="${INSTRUCTION_TYPE_VALUE}" \
  MAX_EPISODE_SECONDS="${MAX_EPISODE_SECONDS:-180}" \
  ROBOLAB_DRIVERFIX="${ROBOLAB_DRIVERFIX:-0}" \
  bash "${REPRO_ROOT}/robolab/scripts/repro_eval_loop.sh" "$@"
