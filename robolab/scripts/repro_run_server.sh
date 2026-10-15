#!/usr/bin/env bash
# Local N-GPU RoboLab evaluation using the new code_human Cosmos server and
# the original RoboLab Cosmos3 client.
#
# One policy server per GPU, with an optional pool of independent simulation
# workers sharing that server:
#   GPU k: code_human/cosmos-framework action_policy_server_robolab on
#          BASE_PORT+k + SIM_WORKERS_PER_GPU RoboLab policies/cosmos3/run.py
#          processes connected to that port.
#
# A simulation worker is a separate Isaac process.  This is intentional: while
# one worker is inside its (blocking) Isaac ``env.step``/render path, another
# worker can have a request in flight to the same policy server.  The default
# A800 80GB default: two independent simulator workers provide useful
# inference/step overlap while leaving headroom for Isaac/GL allocations.
# Set SIM_WORKERS_PER_GPU=1 for the historical one-server/one-client layout;
# set 3 only after checking peak VRAM and GL/CUDA stability.
#
# Typical use:
#   GPUS=0,1 CKPT=/path/to/exported_model TASKS_FILE=/path/to/tasks.txt \
#     bash robolab/scripts/repro_run_server.sh
# ViT soft-plan checkpoints additionally set VIT_SOFTPLAN=1. Soft-plan model
# settings themselves are restored from the exported checkpoint config.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CODE_HUMAN="${CODE_HUMAN:-${REPRO_ROOT}}"
USER_ROOT="${USER_ROOT:-${REPRO_ROOT}}"
SERVER_REPO="${SERVER_REPO:-${REPRO_ROOT}/cosmos-framework}"
if [[ -z "${SERVER_PY:-}" ]]; then
  if [[ -n "${COSMOS_SERVER_PYTHON:-}" ]]; then
    SERVER_PY="${COSMOS_SERVER_PYTHON}"
  elif [[ -x "${SERVER_REPO}/.venv/bin/python" ]]; then
    SERVER_PY="${SERVER_REPO}/.venv/bin/python"
  else
    SERVER_PY="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "${SS_BIN:-}" ]]; then
  if [[ -x /usr/bin/ss ]]; then
    SS_BIN=/usr/bin/ss
  else
    SS_BIN="$(command -v ss || true)"
  fi
fi
if [[ -z "${SS_BIN}" || ! -x "${SS_BIN}" ]]; then
  echo "[new-server-old-robolab] ERROR: system ss executable not found; set SS_BIN explicitly" >&2
  exit 1
fi

ROBOLAB="${ROBOLAB:-${REPRO_ROOT}/robolab}"
if [[ -z "${ROBOLAB_PYTHON:-}" ]]; then
  if [[ -n "${ROBO_PYTHON:-}" ]]; then
    ROBOLAB_PYTHON="${ROBO_PYTHON}"
  elif [[ -x "${ROBOLAB}/.venv/bin/python" ]]; then
    ROBOLAB_PYTHON="${ROBOLAB}/.venv/bin/python"
  else
    ROBOLAB_PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi
ROBOLAB_OUTPUT="${ROBOLAB_OUTPUT:-${ROBOLAB}/output}"
CLIENT_SCRIPT="${CLIENT_SCRIPT:-${ROBOLAB}/scripts/repro_run_client.sh}"
LOG_DIR="${LOG_DIR:-${REPRO_ROOT}/logs}"

# Multi-node evaluation is data-parallel rather than model-parallel: each
# node starts its own local policy-server/client workers and receives a
# disjoint shard of the global task list.  These values are also normalized by
# temp.sh, but keeping the detection here makes direct launcher calls work.
if [[ -n "${MLP_WORKER_NUM:-}" && -n "${MLP_WORKER_0_HOST:-}" ]]; then
  NNODES="${MLP_WORKER_NUM}"
  NODE_RANK="${MLP_ROLE_INDEX:?MLP_ROLE_INDEX is required when MLP_WORKER_NUM is set}"
  MASTER_ADDR="${MLP_WORKER_0_HOST}"
  MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"
elif [[ -n "${SENSECORE_PYTORCH_NNODES:-}" && "${SENSECORE_PYTORCH_NNODES}" -gt 1 ]]; then
  NNODES="${SENSECORE_PYTORCH_NNODES}"
  NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-0}}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29500}"
else
  NNODES="${NNODES:-1}"
  NODE_RANK="${NODE_RANK:-0}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29500}"
fi
if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "[new-server-old-robolab] ERROR: invalid NNODES/NODE_RANK: NNODES=${NNODES} NODE_RANK=${NODE_RANK}" >&2
  exit 1
fi

# Configure offline model access once at launcher scope so both policy-server
# and RoboLab-client subprocesses inherit the same behavior.  Explicit caller
# values still win, including HF_HUB_OFFLINE=0 for intentional online runs.
HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export COSMOS_TRAINING=0

: "${CKPT:=${REPRO_ROOT}/cosmos-framework/checkpoints/wam}"
ALLOW_DCP_CHECKPOINT="${ALLOW_DCP_CHECKPOINT:-0}"
EXPERIMENT="${EXPERIMENT:-}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
GUARDRAILS="${GUARDRAILS:-0}"
GUIDANCE="${GUIDANCE:-3.0}"
CAMERA="${CAMERA:-0}"
CAMERA_GUIDANCE="${CAMERA_GUIDANCE:-1.0}"
CAMERA_CFG_MODE="${CAMERA_CFG_MODE:-separate}"
WRIST_ROTATE_180="${WRIST_ROTATE_180:-0}"
VIEWPOINT_ROTATE_180="${VIEWPOINT_ROTATE_180:-0}"
DROID_VIDEO_SAMPLE_STRIDE="${DROID_VIDEO_SAMPLE_STRIDE:-1}"
ACTION_FPS="${ACTION_FPS:-15}"
VIT_SOFTPLAN="${VIT_SOFTPLAN:-0}"
DECODE_SUBTASK="${DECODE_SUBTASK:-0}"
SUBTASK_MAX_NEW_TOKENS="${SUBTASK_MAX_NEW_TOKENS:-128}"
# Opt-in planner modes.  They are deliberately disabled by default and are
# only translated to server CLI flags here; the server owns their semantics.
# Keeping the defaults disabled preserves every legacy action, mid-soft, and
# auxiliary-decoder invocation.
REASONER_WAM="${REASONER_WAM:-0}"
REASONER_THEN_WAM="${REASONER_THEN_WAM:-0}"
REASONER_WAM_MODE="${REASONER_WAM_MODE:-shared}"
# Mode2 keeps the action/WAM checkpoint and the Reasoner checkpoint explicit:
# CKPT remains the primary (Reasoner) checkpoint for this launcher, while
# WAM_CHECKPOINT_PATH points at the no-LoRA standalone WAM DCP/HF directory.
REASONER_CHECKPOINT_PATH="${REASONER_CHECKPOINT_PATH:-}"
REASONER_LORA_PATH="${REASONER_LORA_PATH:-}"
REASONER_PROJECTOR_PATH="${REASONER_PROJECTOR_PATH:-}"
WAM_CHECKPOINT_PATH="${WAM_CHECKPOINT_PATH:-}"
REASONER_MAX_NEW_TOKENS="${REASONER_MAX_NEW_TOKENS:-128}"
REASONER_DO_SAMPLE="${REASONER_DO_SAMPLE:-0}"
REASONER_INCLUDE_PREVIEW_ACTION="${REASONER_INCLUDE_PREVIEW_ACTION:-0}"
REASONER_OPEN_GRIPPER_MOVE_AUGMENT="${REASONER_OPEN_GRIPPER_MOVE_AUGMENT:-0}"
REASONER_REQUIRE_THINK="${REASONER_REQUIRE_THINK:-0}"
REASONER_SEPARATE_VIEWS="${REASONER_SEPARATE_VIEWS:-0}"
REASONER_THREE_GROUP_HISTORY="${REASONER_THREE_GROUP_HISTORY:-0}"
REASONER_IMAGE_LONG_EDGE="${REASONER_IMAGE_LONG_EDGE:-448}"
FORMAT_PROMPT_AS_JSON="${FORMAT_PROMPT_AS_JSON:-0}"
INCLUDE_EXECUTION_SUCCESS="${INCLUDE_EXECUTION_SUCCESS:-0}"
ROBOLAB_VIDEO_QUALITY="${ROBOLAB_VIDEO_QUALITY:-4}"
ROBOLAB_DRIVERFIX="${ROBOLAB_DRIVERFIX:-0}"
ROBOLAB_DRIVERFIX_ASSETS="${ROBOLAB_DRIVERFIX_ASSETS:-}"
ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS="${ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS:-0}"
ROBOLAB_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_RELATIVE_GRIPPER_COMMANDS:-0}"
ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS:-0}"
ROBOLAB_GRIPPER_RELATIVE_THRESHOLD="${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD:-0.10}"
ROBOLAB_REASONER_HISTORY_IMAGES="${ROBOLAB_REASONER_HISTORY_IMAGES:-0}"
# Preserve full evaluation artifacts, including HDF5, by default.
ROBO_INTERACTIVE_COMPACT_ARTIFACTS="${ROBO_INTERACTIVE_COMPACT_ARTIFACTS:-0}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-default}"
ROBO_INTERACTIVE_SUPERVISOR_PER_WORKER="${ROBO_INTERACTIVE_SUPERVISOR_PER_WORKER:-0}"
ROBOLAB_OFFLINE_ASSET_PREFLIGHT="${ROBOLAB_OFFLINE_ASSET_PREFLIGHT:-1}"
# Keep legacy runs subtitle-free.  A cascade run enables the existing subtitle
# renderer by default, but an explicit caller value (including 0) wins.
if [[ -z "${OVERLAY_SUBTASK_ON_VIDEO+x}" ]]; then
  case "${REASONER_WAM}:${REASONER_THEN_WAM}" in
    1:*|true:*|TRUE:*|yes:*|YES:*|on:*|ON:*|*:1|*:true|*:TRUE|*:yes|*:YES|*:on|*:ON)
      OVERLAY_SUBTASK_ON_VIDEO=1
      ;;
    *)
      OVERLAY_SUBTASK_ON_VIDEO=0
      ;;
  esac
fi
VIT_PROCESSOR_MODEL_NAME="${VIT_PROCESSOR_MODEL_NAME:-${REPRO_ROOT}/cosmos-framework/checkpoints/wam}"
VIT_PROCESSOR_CONFIG_VARIANT="${VIT_PROCESSOR_CONFIG_VARIANT:-hf}"
VIT_PROCESSOR_CACHE_DIR="${VIT_PROCESSOR_CACHE_DIR:-}"
VIT_FIRST_FRAME_MAX_PIXELS="${VIT_FIRST_FRAME_MAX_PIXELS:-348160}"
VIT_FIRST_FRAME_MIN_PIXELS="${VIT_FIRST_FRAME_MIN_PIXELS:-}"
# Used by the mode2 WAM child to avoid downloading the VAE in offline eval.
WAN_VAE_PATH="${WAN_VAE_PATH:-${REPRO_ROOT}/cosmos-framework/checkpoints/wam/wan2.2_vae.pth}"
COSMOS3_NANO_ROOT="${COSMOS3_NANO_ROOT:-${VIT_PROCESSOR_MODEL_NAME}}"
ROBOLAB_ASSETS="${ROBOLAB_ASSETS:-}"
export WAN_VAE_PATH COSMOS3_NANO_ROOT ROBOLAB_ASSETS
ROBOLAB_WRIST_FOCAL_LENGTH="${ROBOLAB_WRIST_FOCAL_LENGTH:-2.8}"

validate_robolab_wrist_focal_length() {
  "${ROBOLAB_PYTHON}" - "${ROBOLAB_WRIST_FOCAL_LENGTH}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"ROBOLAB_WRIST_FOCAL_LENGTH must be numeric, got {sys.argv[1]!r}") from exc
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(f"ROBOLAB_WRIST_FOCAL_LENGTH must be finite and positive, got {value!r}")
PY
}

validate_guidance_args() {
  "${ROBOLAB_PYTHON}" - "${GUIDANCE}" "${CAMERA_GUIDANCE}" <<'PY'
import math
import sys

for name, raw in zip(("GUIDANCE", "CAMERA_GUIDANCE"), sys.argv[1:], strict=True):
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"{name} must be finite and non-negative, got {value!r}")
PY
  case "${CAMERA}" in
    0|1|false|true|FALSE|TRUE) ;;
    *) echo "CAMERA must be 0/1/false/true, got '${CAMERA}'" >&2; return 1 ;;
  esac
  case "${CAMERA_CFG_MODE}" in
    camera|both|separate) ;;
    *)
      echo "CAMERA_CFG_MODE must be camera/both/separate, got '${CAMERA_CFG_MODE}'" >&2
      return 1
      ;;
  esac
}

validate_reasoner_mode_flags() {
  local name
  local value
  for name in REASONER_WAM REASONER_THEN_WAM REASONER_DO_SAMPLE REASONER_INCLUDE_PREVIEW_ACTION REASONER_OPEN_GRIPPER_MOVE_AUGMENT REASONER_REQUIRE_THINK REASONER_SEPARATE_VIEWS REASONER_THREE_GROUP_HISTORY OVERLAY_SUBTASK_ON_VIDEO; do
    value="${!name}"
    case "${value}" in
      0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
      *)
        echo "${name} must be 0/1 or false/true, got '${value}'" >&2
        return 1
        ;;
    esac
  done
  if ! [[ "${REASONER_IMAGE_LONG_EDGE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REASONER_IMAGE_LONG_EDGE must be a positive integer, got '${REASONER_IMAGE_LONG_EDGE}'" >&2
    return 1
  fi
  case "${VIEWPOINT_ROTATE_180}" in
    0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
    *)
      echo "VIEWPOINT_ROTATE_180 must be 0/1 or false/true, got '${VIEWPOINT_ROTATE_180}'" >&2
      return 1
      ;;
  esac
  if ! [[ "${REASONER_MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REASONER_MAX_NEW_TOKENS must be a positive integer, got '${REASONER_MAX_NEW_TOKENS}'" >&2
    return 1
  fi
  case "${REASONER_WAM_MODE}" in
    shared|mode2|adapter) ;;
    *)
      echo "REASONER_WAM_MODE must be shared, mode2, or adapter, got '${REASONER_WAM_MODE}'" >&2
      return 1
      ;;
  esac
  if [[ "${REASONER_WAM_MODE}" == "mode2" && "${REASONER_WAM}" != "1" && "${REASONER_WAM}" != "true" && "${REASONER_WAM}" != "TRUE" && "${REASONER_WAM}" != "yes" && "${REASONER_WAM}" != "on" && "${REASONER_THEN_WAM}" != "1" && "${REASONER_THEN_WAM}" != "true" && "${REASONER_THEN_WAM}" != "TRUE" && "${REASONER_THEN_WAM}" != "yes" && "${REASONER_THEN_WAM}" != "on" ]]; then
    echo "REASONER_WAM_MODE=mode2 requires REASONER_WAM=1 (or REASONER_THEN_WAM=1)" >&2
    return 1
  fi
  if [[ "${REASONER_WAM_MODE}" == "mode2" && -z "${WAM_CHECKPOINT_PATH}" ]]; then
    echo "REASONER_WAM_MODE=mode2 requires WAM_CHECKPOINT_PATH (standalone no-LoRA WAM run/checkpoint)" >&2
    return 1
  fi
  case "${REASONER_INCLUDE_PREVIEW_ACTION}" in
    1|true|TRUE|yes|YES|on|ON)
      case "${REASONER_WAM}:${REASONER_THEN_WAM}" in
        1:*|true:*|TRUE:*|yes:*|YES:*|on:*|ON:*|*:1|*:true|*:TRUE|*:yes|*:YES|*:on|*:ON) ;;
        *)
          echo "REASONER_INCLUDE_PREVIEW_ACTION requires REASONER_WAM=1 (or REASONER_THEN_WAM=1)" >&2
          return 1
          ;;
      esac
      ;;
  esac
  if [[ "${REASONER_WAM_MODE}" == "adapter" && ! -f "${REASONER_LORA_PATH}" ]]; then
    echo "REASONER_WAM_MODE=adapter requires a REASONER_LORA_PATH file" >&2
    return 1
  fi
  if [[ "${REASONER_WAM_MODE}" == "adapter" && -n "${REASONER_PROJECTOR_PATH}" && ! -f "${REASONER_PROJECTOR_PATH}" ]]; then
    echo "REASONER_PROJECTOR_PATH was provided but is not a file: ${REASONER_PROJECTOR_PATH}" >&2
    return 1
  fi
  case "${FORMAT_PROMPT_AS_JSON}" in
    0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
    *) echo "FORMAT_PROMPT_AS_JSON must be 0/1 or false/true, got '${FORMAT_PROMPT_AS_JSON}'" >&2; return 1 ;;
  esac
  case "${INCLUDE_EXECUTION_SUCCESS}" in
    0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
    *) echo "INCLUDE_EXECUTION_SUCCESS must be 0/1 or false/true, got '${INCLUDE_EXECUTION_SUCCESS}'" >&2; return 1 ;;
  esac
  if ! [[ "${ROBOLAB_VIDEO_QUALITY}" =~ ^([0-9]|10)$ ]]; then
    echo "ROBOLAB_VIDEO_QUALITY must be an integer in [0,10], got '${ROBOLAB_VIDEO_QUALITY}'" >&2
    return 1
  fi
}

validate_sim_workers() {
  if ! [[ "${SIM_WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SIM_WORKERS_PER_GPU must be a positive integer, got '${SIM_WORKERS_PER_GPU}'" >&2
    return 1
  fi
  if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got '${NUM_ENVS}'" >&2
    return 1
  fi
}

validate_video_mode() {
  case "${VIDEO_MODE}" in
    all|sensor|viewport) ;;
    none)
      echo "VIDEO_MODE=none is disabled for evaluation; keep video recording enabled (all, sensor, or viewport)" >&2
      return 1
      ;;
    *)
      echo "VIDEO_MODE must be all, sensor, or viewport (none is disabled), got '${VIDEO_MODE}'" >&2
      return 1
      ;;
  esac

  # EXTRA is ultimately word-split by the RoboLab client.  Inspect every
  # explicit video-mode spelling here so a custom EXTRA cannot silently turn
  # recording off, including the --video-mode=none form.
  local -a extra_args=()
  local i token mode
  read -r -a extra_args <<< "${EXTRA}"
  for ((i = 0; i < ${#extra_args[@]}; i++)); do
    token="${extra_args[i]}"
    mode=""
    case "${token}" in
      --video-mode|--video_mode)
        if (( i + 1 >= ${#extra_args[@]} )); then
          echo "EXTRA contains '${token}' without a mode value" >&2
          return 1
        fi
        mode="${extra_args[i + 1]}"
        ;;
      --video-mode=*|--video_mode=*)
        mode="${token#*=}"
        ;;
      *)
        continue
        ;;
    esac
    case "${mode}" in
      all|sensor|viewport) ;;
      none)
        echo "EXTRA requests video mode 'none'; video recording cannot be disabled for evaluation" >&2
        return 1
        ;;
      *)
        echo "EXTRA requests unsupported video mode '${mode}'; use all, sensor, or viewport" >&2
        return 1
        ;;
    esac
  done
}

conditioning_fps_for_stride() {
  "${ROBOLAB_PYTHON}" - <<PY
action_fps = float("${ACTION_FPS}")
stride = float("${DROID_VIDEO_SAMPLE_STRIDE}")
print(f"{action_fps / stride:g}")
PY
}

server_alignment_args() {
  local args=()
  args+=(--guidance "${GUIDANCE}")
  if [[ "${CAMERA}" == "1" || "${CAMERA}" == "true" || "${CAMERA}" == "TRUE" ]]; then
    args+=(--camera-cfg-mode "${CAMERA_CFG_MODE}")
    if [[ "${CAMERA_CFG_MODE}" == "separate" ]]; then
      args+=(--camera-guidance "${CAMERA_GUIDANCE}")
    fi
  fi
  if [[ "${WRIST_ROTATE_180}" == "1" || "${WRIST_ROTATE_180}" == "true" || "${WRIST_ROTATE_180}" == "TRUE" ]]; then
    args+=(--wrist-rotate-180)
  fi
  args+=(--video-sample-stride "${DROID_VIDEO_SAMPLE_STRIDE}")
  if [[ "${DROID_VIDEO_SAMPLE_STRIDE}" != "1" ]]; then
    args+=(--conditioning-fps "$(conditioning_fps_for_stride)")
  fi
  args+=(--action-fps "${ACTION_FPS}")
  if [[ "${VIT_SOFTPLAN}" == "1" || "${VIT_SOFTPLAN}" == "true" || "${VIT_SOFTPLAN}" == "TRUE" || "${REASONER_WAM}" == "1" || "${REASONER_WAM}" == "true" || "${REASONER_WAM}" == "TRUE" || "${REASONER_WAM}" == "yes" || "${REASONER_WAM}" == "on" || "${REASONER_THEN_WAM}" == "1" || "${REASONER_THEN_WAM}" == "true" || "${REASONER_THEN_WAM}" == "TRUE" || "${REASONER_THEN_WAM}" == "yes" || "${REASONER_THEN_WAM}" == "on" ]]; then
    args+=(--include-vit-first-frame)
    args+=(--vit-processor-model-name "${VIT_PROCESSOR_MODEL_NAME}")
    args+=(--vit-processor-config-variant "${VIT_PROCESSOR_CONFIG_VARIANT}")
    if [[ -n "${VIT_PROCESSOR_CACHE_DIR}" ]]; then
      args+=(--vit-processor-cache-dir "${VIT_PROCESSOR_CACHE_DIR}")
    fi
    if [[ -n "${VIT_FIRST_FRAME_MAX_PIXELS}" ]]; then
      args+=(--vit-first-frame-max-pixels "${VIT_FIRST_FRAME_MAX_PIXELS}")
    fi
    if [[ -n "${VIT_FIRST_FRAME_MIN_PIXELS}" ]]; then
      args+=(--vit-first-frame-min-pixels "${VIT_FIRST_FRAME_MIN_PIXELS}")
    fi
  fi
  if [[ "${DECODE_SUBTASK}" == "1" || "${DECODE_SUBTASK}" == "true" || "${DECODE_SUBTASK}" == "TRUE" ]]; then
    args+=(--decode-subtask)
    args+=(--subtask-max-new-tokens "${SUBTASK_MAX_NEW_TOKENS}")
  fi
  if [[ "${REASONER_WAM}" == "1" || "${REASONER_WAM}" == "true" || "${REASONER_WAM}" == "TRUE" || "${REASONER_WAM}" == "yes" || "${REASONER_WAM}" == "on" ]]; then
    args+=(--reasoner-wam)
  fi
  if [[ "${REASONER_THEN_WAM}" == "1" || "${REASONER_THEN_WAM}" == "true" || "${REASONER_THEN_WAM}" == "TRUE" || "${REASONER_THEN_WAM}" == "yes" || "${REASONER_THEN_WAM}" == "on" ]]; then
    args+=(--reasoner-then-wam)
  fi
  if [[ "${REASONER_WAM}" == "1" || "${REASONER_WAM}" == "true" || "${REASONER_WAM}" == "TRUE" || "${REASONER_WAM}" == "yes" || "${REASONER_WAM}" == "on" || "${REASONER_THEN_WAM}" == "1" || "${REASONER_THEN_WAM}" == "true" || "${REASONER_THEN_WAM}" == "TRUE" || "${REASONER_THEN_WAM}" == "yes" || "${REASONER_THEN_WAM}" == "on" ]]; then
    args+=(--reasoner-wam-mode "${REASONER_WAM_MODE}")
    if [[ -n "${REASONER_CHECKPOINT_PATH}" ]]; then
      args+=(--reasoner-checkpoint-path "${REASONER_CHECKPOINT_PATH}")
    fi
    if [[ -n "${REASONER_LORA_PATH}" ]]; then
      args+=(--reasoner-lora-path "${REASONER_LORA_PATH}")
    fi
    if [[ -n "${REASONER_PROJECTOR_PATH}" ]]; then
      args+=(--reasoner-projector-path "${REASONER_PROJECTOR_PATH}")
    fi
    if [[ -n "${WAM_CHECKPOINT_PATH}" ]]; then
      args+=(--wam-checkpoint-path "${WAM_CHECKPOINT_PATH}")
    fi
    args+=(--reasoner-max-new-tokens "${REASONER_MAX_NEW_TOKENS}")
    if [[ "${REASONER_DO_SAMPLE}" == "1" || "${REASONER_DO_SAMPLE}" == "true" || "${REASONER_DO_SAMPLE}" == "TRUE" || "${REASONER_DO_SAMPLE}" == "yes" || "${REASONER_DO_SAMPLE}" == "on" ]]; then
      args+=(--reasoner-do-sample)
    fi
    if [[ "${REASONER_INCLUDE_PREVIEW_ACTION}" == "1" || "${REASONER_INCLUDE_PREVIEW_ACTION}" == "true" || "${REASONER_INCLUDE_PREVIEW_ACTION}" == "TRUE" || "${REASONER_INCLUDE_PREVIEW_ACTION}" == "yes" || "${REASONER_INCLUDE_PREVIEW_ACTION}" == "on" ]]; then
      args+=(--reasoner-include-preview-action)
    fi
    if [[ "${REASONER_OPEN_GRIPPER_MOVE_AUGMENT}" == "1" || "${REASONER_OPEN_GRIPPER_MOVE_AUGMENT}" == "true" || "${REASONER_OPEN_GRIPPER_MOVE_AUGMENT}" == "TRUE" || "${REASONER_OPEN_GRIPPER_MOVE_AUGMENT}" == "yes" || "${REASONER_OPEN_GRIPPER_MOVE_AUGMENT}" == "on" ]]; then
      args+=(--reasoner-open-gripper-move-augment)
    fi
    if [[ "${REASONER_REQUIRE_THINK}" == "1" || "${REASONER_REQUIRE_THINK}" == "true" || "${REASONER_REQUIRE_THINK}" == "TRUE" || "${REASONER_REQUIRE_THINK}" == "yes" || "${REASONER_REQUIRE_THINK}" == "on" ]]; then
      args+=(--reasoner-require-think)
    fi
    if [[ "${REASONER_SEPARATE_VIEWS}" == "1" || "${REASONER_SEPARATE_VIEWS}" == "true" || "${REASONER_SEPARATE_VIEWS}" == "TRUE" || "${REASONER_SEPARATE_VIEWS}" == "yes" || "${REASONER_SEPARATE_VIEWS}" == "on" ]]; then
      args+=(--reasoner-separate-views --reasoner-image-long-edge "${REASONER_IMAGE_LONG_EDGE}")
    fi
    if [[ "${REASONER_THREE_GROUP_HISTORY}" == "1" || "${REASONER_THREE_GROUP_HISTORY}" == "true" || "${REASONER_THREE_GROUP_HISTORY}" == "TRUE" || "${REASONER_THREE_GROUP_HISTORY}" == "yes" || "${REASONER_THREE_GROUP_HISTORY}" == "on" ]]; then
      args+=(--reasoner-three-group-history)
    fi
  fi
  if [[ "${FORMAT_PROMPT_AS_JSON}" == "1" || "${FORMAT_PROMPT_AS_JSON}" == "true" || "${FORMAT_PROMPT_AS_JSON}" == "TRUE" || "${FORMAT_PROMPT_AS_JSON}" == "yes" || "${FORMAT_PROMPT_AS_JSON}" == "on" ]]; then
    # This field is Optional[bool] so Tyro requires an explicit value; a bare
    # flag consumes the following option (for example --resolution) as its
    # value and aborts before model loading.
    args+=(--format-prompt-as-json True)
  fi
  if [[ "${INCLUDE_EXECUTION_SUCCESS}" == "1" || "${INCLUDE_EXECUTION_SUCCESS}" == "true" || "${INCLUDE_EXECUTION_SUCCESS}" == "TRUE" || "${INCLUDE_EXECUTION_SUCCESS}" == "yes" || "${INCLUDE_EXECUTION_SUCCESS}" == "on" ]]; then
    # Same Optional[bool] Tyro shape as --format-prompt-as-json above.
    args+=(--include-execution-success True)
  fi
  printf '%s\n' "${args[*]}"
}

detect_gpus() {
  if [[ -n "${GPUS:-}" ]]; then
    echo "${GPUS}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local detected
    detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
    if [[ -n "${detected}" ]]; then
      echo "${detected}"
      return
    fi
  fi
  echo "0,1"
}

GPUS_CSV="$(detect_gpus)"

validate_gpu_list() {
  local item
  local -A seen=()
  if [[ -z "${GPUS_CSV}" ]]; then
    echo "[new-server-old-robolab] ERROR: no GPUs selected. Set GPUS=0,1" >&2
    return 1
  fi
  if ! [[ "${GPUS_CSV}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[new-server-old-robolab] ERROR: GPUS must be a comma-separated list of non-negative integers, got '${GPUS_CSV}'" >&2
    return 1
  fi
  IFS=',' read -r -a GPU_LIST <<<"${GPUS_CSV}"
  if (( ${#GPU_LIST[@]} == 0 )); then
    echo "[new-server-old-robolab] ERROR: no GPUs selected. Set GPUS=0,1" >&2
    return 1
  fi
  for item in "${GPU_LIST[@]}"; do
    if ! [[ "${item}" =~ ^[0-9]+$ ]]; then
      echo "[new-server-old-robolab] ERROR: GPUS must be a comma-separated list of non-negative integers, got '${GPUS_CSV}'" >&2
      return 1
    fi
    if [[ -n "${seen[${item}]+present}" ]]; then
      echo "[new-server-old-robolab] ERROR: duplicate GPU '${item}' in GPUS='${GPUS_CSV}'" >&2
      return 1
    fi
    seen["${item}"]=1
  done
  NGPU="${#GPU_LIST[@]}"
}

validate_gpu_list

BASE_PORT="${BASE_PORT:-8000}"
DIST_BASE_PORT="${DIST_BASE_PORT:-28000}"
HOST="${HOST:-localhost}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-18000}"
NUM_ENVS="${NUM_ENVS:-1}"
SIM_WORKERS_PER_GPU="${SIM_WORKERS_PER_GPU:-1}"
VIDEO_MODE="${VIDEO_MODE:-all}"
EXTRA="${EXTRA:---headless --video-mode ${VIDEO_MODE}}"
GRASP_HORIZON="${GRASP_HORIZON:-8}"
GRASP_CONFIRM="${GRASP_CONFIRM:-10}"
GRIPPER_OPEN="${GRIPPER_OPEN:-0.3}"
GRIPPER_CLOSE="${GRIPPER_CLOSE:-0.7}"
MAX_EPISODE_SECONDS="${MAX_EPISODE_SECONDS:-}"
MAX_EPISODE_SECONDS_EXCEPT_TASKS="${MAX_EPISODE_SECONDS_EXCEPT_TASKS:-}"
DRY_RUN="${DRY_RUN:-0}"
MERGE_OUTPUTS="${MERGE_OUTPUTS:-1}"
KEEP_WORKER_OUTPUTS="${KEEP_WORKER_OUTPUTS:-0}"
OVERWRITE_FINAL="${OVERWRITE_FINAL:-0}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-2}"
CLIENT_RETRY_DELAY_SECONDS="${CLIENT_RETRY_DELAY_SECONDS:-30}"

if ! [[ "${CLIENT_MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "[new-server-old-robolab] ERROR: CLIENT_MAX_RETRIES must be a non-negative integer, got '${CLIENT_MAX_RETRIES}'" >&2
  exit 2
fi
if ! [[ "${CLIENT_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "[new-server-old-robolab] ERROR: CLIENT_RETRY_DELAY_SECONDS must be a non-negative integer, got '${CLIENT_RETRY_DELAY_SECONDS}'" >&2
  exit 2
fi

validate_base_port() {
  if ! [[ "${BASE_PORT}" =~ ^[0-9]+$ ]] || (( BASE_PORT < 1 || BASE_PORT > 65535 )); then
    echo "[new-server-old-robolab] ERROR: BASE_PORT must be an integer in [1,65535], got '${BASE_PORT}'" >&2
    return 1
  fi
  if (( BASE_PORT + NGPU - 1 > 65535 )); then
    echo "[new-server-old-robolab] ERROR: GPU port range exceeds 65535: BASE_PORT=${BASE_PORT}, GPUs=${NGPU}" >&2
    return 1
  fi
  if ! [[ "${DIST_BASE_PORT}" =~ ^[0-9]+$ ]] || (( DIST_BASE_PORT < 1 || DIST_BASE_PORT > 65535 )); then
    echo "[new-server-old-robolab] ERROR: DIST_BASE_PORT must be an integer in [1,65535], got '${DIST_BASE_PORT}'" >&2
    return 1
  fi
  if (( DIST_BASE_PORT + NGPU - 1 > 65535 )); then
    echo "[new-server-old-robolab] ERROR: distributed port range exceeds 65535: DIST_BASE_PORT=${DIST_BASE_PORT}, GPUs=${NGPU}" >&2
    return 1
  fi
  if (( BASE_PORT <= DIST_BASE_PORT + NGPU - 1 && DIST_BASE_PORT <= BASE_PORT + NGPU - 1 )); then
    echo "[new-server-old-robolab] ERROR: HTTP and distributed port ranges overlap" >&2
    return 1
  fi
}

validate_base_port

TASK_SET="${TASK_SET:-simple}"
TASKS_FILE="${TASKS_FILE:-}"
TASKS_SOURCE="${TASKS_SOURCE:-${ROBOLAB}/output/2026-06-09_06-31-43_cosmos3/episode_results.jsonl}"
TASK_METADATA="${TASK_METADATA:-${ROBOLAB}/robolab/tasks/_metadata/task_metadata.json}"
MAX_TASKS="${MAX_TASKS:-}"

BASE_OUTDIR="${BASE_OUTDIR:-$(date -u +%Y-%m-%d_%H-%M-%S)_cosmos3_code_human_server_original_${TASK_SET}_ngpu}"
FINAL_OUTDIR="${FINAL_OUTDIR:-${BASE_OUTDIR}}"

# Shared coordination for multi-node runs.  FINAL_OUTDIR is the run key, so
# both nodes can use the same command without an extra run-id argument.
# Keep coordination paths independent of pod hostnames, master ports, and
# session identifiers.  These values are runtime metadata, not output names.
COORD_KEY="${FINAL_OUTDIR//[^a-zA-Z0-9_.-]/_}"
COORD_ROOT="${LOG_DIR}/new_server_old_robolab_coord_${COORD_KEY}"
mkdir -p "${COORD_ROOT}"
COORD_SESSION_ID="${EVAL_SESSION_ID:-default}"
# In multi-node mode node 0 owns the merge, but every node must observe that
# merge before its outer repeat loop can start the next evaluation.  The
# marker is scoped by FINAL_OUTDIR (which includes the repeat suffix), so
# different repeats cannot accidentally release one another.
GLOBAL_MERGE_DONE="${COORD_ROOT}/merge.done"

marker_matches_session() {
  local marker="$1"
  [[ -s "${marker}" ]] && grep -Fqx "session=${COORD_SESSION_ID}" "${marker}"
}

mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
if (( NNODES > 1 )); then
  TASKS_DIR="${LOG_DIR}/new_server_old_robolab_tasks_${COORD_KEY}_node${NODE_RANK}"
else
  TASKS_DIR="${LOG_DIR}/new_server_old_robolab_tasks_${STAMP}"
fi
TASKS_SUMMARY="${TASKS_DIR}/summary.tsv"
mkdir -p "${TASKS_DIR}"

server_pids=()
client_pids=()
server_indices=()
worker_indices=()
worker_gpu_indices=()
worker_sim_indices=()
worker_outdirs=()
server_logs=()
client_logs=()
task_files=()

cleanup() {
  local status=$?
  local pid
  local alive
  local pids=()

  trap - EXIT INT TERM

  for pid in "${client_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[new-server-old-robolab] stopping client pid=${pid}"
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${server_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[new-server-old-robolab] stopping server pid=${pid}"
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  pids+=("${client_pids[@]}")
  pids+=("${server_pids[@]}")
  for _ in $(seq 1 20); do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    (( alive == 0 )) && break
    sleep 1
  done

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[new-server-old-robolab] forcing remaining child pid=${pid} to exit"
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT INT TERM

check_prereqs() {
  test -x "${SERVER_PY}"
  test -f "${SERVER_REPO}/cosmos_framework/scripts/action_policy_server_robolab.py"
  test -x "${ROBOLAB_PYTHON}"
  test -f "${CLIENT_SCRIPT}"
  if [[ -z "${ROBOLAB_ASSETS}" || ! -d "${ROBOLAB_ASSETS}" ]]; then
    echo "[new-server-old-robolab] ERROR: set ROBOLAB_ASSETS to the external RoboLab assets directory" >&2
    return 1
  fi
  for required_asset_dir in backgrounds fixtures objects robots scenes; do
    if [[ ! -d "${ROBOLAB_ASSETS}/${required_asset_dir}" ]]; then
      echo "[new-server-old-robolab] ERROR: missing RoboLab asset directory: ${ROBOLAB_ASSETS}/${required_asset_dir}" >&2
      return 1
    fi
  done
  test -e "${CKPT}"

  case "${VIT_SOFTPLAN}:${REASONER_WAM}:${REASONER_THEN_WAM}" in
    1:*|true:*|TRUE:*|yes:*|YES:*|on:*|ON:*|\
    *:1:*|*:true:*|*:TRUE:*|*:yes:*|*:YES:*|*:on:*|*:ON:*|\
    *:*:1|*:*:true|*:*:TRUE|*:*:yes|*:*:YES|*:*:on|*:*:ON)
      if [[ ! -d "${VIT_PROCESSOR_MODEL_NAME}" ]]; then
        echo "[new-server-old-robolab] ERROR: local ViT processor model not found: ${VIT_PROCESSOR_MODEL_NAME}" >&2
        return 1
      fi
      ;;
  esac
}

preflight_robolab_graphics() {
  echo "[new-server-old-robolab] checking RoboLab NVIDIA graphics before loading policy servers"
  env \
    ROBOLAB="${ROBOLAB}" \
    ROBOLAB_PY="${ROBOLAB_PYTHON}" \
    ROBOLAB_DRIVERFIX="${ROBOLAB_DRIVERFIX}" \
    ROBOLAB_DRIVERFIX_ASSETS="${ROBOLAB_DRIVERFIX_ASSETS}" \
    ROBOLAB_DRIVERFIX_CHECK_ONLY=1 \
    TASK=DriverfixPreflight \
    bash "${CLIENT_SCRIPT}"
}

preflight_robolab_assets() {
  case "${ROBOLAB_OFFLINE_ASSET_PREFLIGHT}" in
    0|false|FALSE|no|NO|off|OFF)
      echo "[new-server-old-robolab] WARNING: offline asset preflight disabled (ROBOLAB_OFFLINE_ASSET_PREFLIGHT=${ROBOLAB_OFFLINE_ASSET_PREFLIGHT})" >&2
      return 0
      ;;
    1|true|TRUE|yes|YES|on|ON) ;;
    *)
      echo "[new-server-old-robolab] ERROR: ROBOLAB_OFFLINE_ASSET_PREFLIGHT must be 0/1 or false/true, got '${ROBOLAB_OFFLINE_ASSET_PREFLIGHT}'" >&2
      return 1
      ;;
  esac

  local checker="${ROBOLAB}/scripts/check_offline_assets.py"
  if [[ ! -f "${checker}" ]]; then
    echo "[new-server-old-robolab] ERROR: offline asset checker not found: ${checker}" >&2
    return 1
  fi

  local -a args=(
    --robolab "${ROBOLAB}"
    --assets "${ROBOLAB_ASSETS}"
    --task-set "${TASK_SET}"
    --task-metadata "${TASK_METADATA}"
    --strict
  )
  if [[ -n "${TASKS_FILE}" ]]; then
    args+=(--tasks-file "${TASKS_FILE}")
  fi
  if [[ -n "${TASKS_SOURCE}" ]]; then
    args+=(--tasks-source "${TASKS_SOURCE}")
  fi

  echo "[new-server-old-robolab] checking RoboLab USD/MDL assets before loading policy servers"
  "${ROBOLAB_PYTHON}" "${checker}" "${args[@]}"
}

check_port_free() {
  local port="$1"
  if "${SS_BIN}" -ltnp 2>/dev/null | grep -qE ":${port}[[:space:]]"; then
    echo "[new-server-old-robolab] ERROR: port ${port} is already in use." >&2
    "${SS_BIN}" -ltnp | grep -E ":${port}[[:space:]]" || true
    exit 1
  fi
}

build_task_splits() {
  if [[ -n "${TASKS_FILE}" && ! -f "${TASKS_FILE}" ]]; then
    echo "[new-server-old-robolab] ERROR: TASKS_FILE not found: ${TASKS_FILE}" >&2
    exit 1
  fi
  if [[ -z "${TASKS_FILE}" && ( "${TASK_SET}" == "simple" || "${TASK_SET}" == "all" ) && ! -f "${TASK_METADATA}" ]]; then
    echo "[new-server-old-robolab] ERROR: TASK_METADATA not found: ${TASK_METADATA}" >&2
    exit 1
  fi
  if [[ -z "${TASKS_FILE}" && "${TASK_SET}" != "simple" && "${TASK_SET}" != "all" && ! -f "${TASKS_SOURCE}" ]]; then
    echo "[new-server-old-robolab] ERROR: TASKS_SOURCE not found: ${TASKS_SOURCE}" >&2
    exit 1
  fi

  "${ROBOLAB_PYTHON}" - \
    "${NGPU}" "${SIM_WORKERS_PER_GPU}" "${GPUS_CSV}" "${NNODES}" "${NODE_RANK}" "${TASKS_DIR}" \
    "${TASKS_SUMMARY}" "${TASKS_FILE}" "${TASKS_SOURCE}" "${TASK_METADATA}" \
    "${TASK_SET}" "${MAX_TASKS}" <<'PY'
import json
import sys
from collections import OrderedDict
from pathlib import Path

ngpu = int(sys.argv[1])
sim_workers_per_gpu = int(sys.argv[2])
gpu_ids = sys.argv[3].split(",")
nnodes = int(sys.argv[4])
node_rank = int(sys.argv[5])
tasks_dir = Path(sys.argv[6])
summary_path = Path(sys.argv[7])
tasks_file = sys.argv[8]
tasks_source = sys.argv[9]
task_metadata = sys.argv[10]
task_set = sys.argv[11]
max_tasks_s = sys.argv[12]
max_tasks = int(max_tasks_s) if max_tasks_s else None
if len(gpu_ids) != ngpu:
    raise SystemExit(f"GPU list/count mismatch: ngpu={ngpu}, gpu_ids={gpu_ids}")
if nnodes < 1 or node_rank < 0 or node_rank >= nnodes:
    raise SystemExit(f"invalid node topology: nnodes={nnodes}, node_rank={node_rank}")

def source_weights(path_s: str) -> OrderedDict[str, float]:
    weights: OrderedDict[str, float] = OrderedDict()
    path = Path(path_s)
    if not path.is_file():
        return weights
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row.get("env_name") or row.get("task_name")
        if not name or name in weights:
            continue
        timing = row.get("timing") or {}
        weight = timing.get("wall_total_s", row.get("episode_step", 1.0))
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        weights[name] = max(weight, 1.0)
    return weights

weights = source_weights(tasks_source)
names: list[str] = []
if tasks_file:
    for line in Path(tasks_file).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
elif task_set in {"simple", "all"}:
    rows = json.loads(Path(task_metadata).read_text())
    names = [
        row["task_name"]
        for row in rows
        if (task_set == "all" or row.get("difficulty_label") == "simple") and row.get("task_name")
    ]
else:
    names = list(weights)

seen = set()
deduped = []
for name in names:
    if name not in seen:
        seen.add(name)
        deduped.append(name)
names = deduped
if max_tasks is not None:
    names = names[:max_tasks]
if not names:
    raise SystemExit("[new-server-old-robolab] no tasks found")

items = [(name, weights.get(name, 1.0)) for name in names]
items.sort(key=lambda x: x[1], reverse=True)

# Keep the hierarchy explicit: first balance work across GPUs, then balance
# each GPU's assignment across its simulator processes.  This avoids a later
# change in SIM_WORKERS_PER_GPU silently changing which GPU owns a task.
global_ngpu = nnodes * ngpu
gpu_buckets: list[list[tuple[str, float]]] = [[] for _ in range(global_ngpu)]
gpu_totals = [0.0 for _ in range(global_ngpu)]
for name, weight in items:
    gpu_idx = min(range(global_ngpu), key=lambda i: gpu_totals[i])
    gpu_buckets[gpu_idx].append((name, weight))
    gpu_totals[gpu_idx] += weight

worker_buckets: list[list[tuple[str, float]]] = []
for gpu_bucket in gpu_buckets:
    local_buckets: list[list[tuple[str, float]]] = [[] for _ in range(sim_workers_per_gpu)]
    local_totals = [0.0 for _ in range(sim_workers_per_gpu)]
    for name, weight in gpu_bucket:
        sim_idx = min(range(sim_workers_per_gpu), key=lambda i: local_totals[i])
        local_buckets[sim_idx].append((name, weight))
        local_totals[sim_idx] += weight
    worker_buckets.extend(local_buckets)

global_worker_start = node_rank * ngpu * sim_workers_per_gpu
global_worker_end = global_worker_start + ngpu * sim_workers_per_gpu
local_worker_buckets = worker_buckets[global_worker_start:global_worker_end]

tasks_dir.mkdir(parents=True, exist_ok=True)
for idx, bucket in enumerate(local_worker_buckets):
    path = tasks_dir / f"worker_{idx}.tasks"
    with path.open("w") as f:
        for name, _ in bucket:
            f.write(name + "\n")

with summary_path.open("w") as f:
    f.write("worker\tgpu_slot\tgpu\tsim_worker\tnum_tasks\tweight_sum\ttasks_file\n")
    for local_idx, bucket in enumerate(local_worker_buckets):
        global_idx = global_worker_start + local_idx
        gpu_idx, sim_idx = divmod(global_idx, sim_workers_per_gpu)
        total = sum(weight for _, weight in bucket)
        f.write(
            f"{global_idx}\t{gpu_idx}\t{gpu_ids[gpu_idx % ngpu]}\t{sim_idx}\t"
            f"{len(bucket)}\t{total:.1f}\t"
            f"{tasks_dir / f'worker_{local_idx}.tasks'}\n"
        )

print(
    f"[new-server-old-robolab] task_set={task_set} total_tasks={len(names)} "
    f"gpus={global_ngpu} (local={ngpu}) nodes={nnodes} node_rank={node_rank} "
    f"sim_workers_per_gpu={sim_workers_per_gpu} total_sim_workers={len(worker_buckets)}"
)
for local_idx, bucket in enumerate(local_worker_buckets):
    global_idx = global_worker_start + local_idx
    gpu_idx, sim_idx = divmod(global_idx, sim_workers_per_gpu)
    total = sum(weight for _, weight in bucket)
    print(
        f"[new-server-old-robolab] node{node_rank} gpu{gpu_idx}/sim{sim_idx} worker{global_idx}: "
        f"tasks={len(bucket)} weight={total:.1f} "
        f"file={tasks_dir / f'worker_{local_idx}.tasks'}"
    )
print(f"[new-server-old-robolab] summary={summary_path}")
PY
}

wait_ready() {
  local pid="$1"
  local log="$2"
  local label="$3"
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[new-server-old-robolab] ${label} exited before ready; last log lines:"
      tail -n 120 "${log}" || true
      exit 1
    fi
    if grep -Eq "Server accessible at:|ready domain=" "${log}"; then
      echo "[new-server-old-robolab] ${label} ready"
      return 0
    fi
    sleep 5
  done
  echo "[new-server-old-robolab] timeout waiting for ${label}; last log lines:"
  tail -n 160 "${log}" || true
  exit 1
}

task_string() {
  paste -sd' ' "$1"
}

client_extra_args() {
  local extra="$1"
  if [[ -n "${MAX_EPISODE_SECONDS}" ]]; then
    extra="${extra} --max-episode-seconds ${MAX_EPISODE_SECONDS}"
  fi
  if [[ -n "${MAX_EPISODE_SECONDS_EXCEPT_TASKS}" ]]; then
    extra="${extra} --max-episode-seconds-except-task"
    local except_task
    IFS=',' read -r -a _except_tasks <<<"${MAX_EPISODE_SECONDS_EXCEPT_TASKS}"
    for except_task in "${_except_tasks[@]}"; do
      if [[ -n "${except_task}" ]]; then
        extra="${extra} ${except_task}"
      fi
    done
  fi
  printf '%s\n' "${extra}"
}

run_client_with_retries() {
  local task="$1"
  local worker_outdir="$2"
  local gpu="$3"
  local port="$4"
  local client_extra="$5"
  local supervisor_dir="${ROBO_INTERACTIVE_SUPERVISOR_DIR}"
  if [[ "${ROBO_INTERACTIVE_SUPERVISOR_PER_WORKER}" == "1" || "${ROBO_INTERACTIVE_SUPERVISOR_PER_WORKER}" == "true" || "${ROBO_INTERACTIVE_SUPERVISOR_PER_WORKER}" == "TRUE" || "${ROBO_INTERACTIVE_SUPERVISOR_PER_WORKER}" == "yes" || "${ROBO_INTERACTIVE_SUPERVISOR_PER_WORKER}" == "on" ]]; then
    supervisor_dir="${supervisor_dir%/}/${worker_outdir}"
  fi
  local attempt=0
  local child_pid=""
  local rc=0

  stop_client_child() {
    trap - INT TERM
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill -TERM "${child_pid}" 2>/dev/null || true
      wait "${child_pid}" 2>/dev/null || true
    fi
    exit 143
  }
  trap stop_client_child INT TERM

  while true; do
    echo "[new-server-old-robolab] client gpu${gpu} attempt=$((attempt + 1))/$((CLIENT_MAX_RETRIES + 1))"
    (
      export ALL=0 TASK="${task}" OUTDIR="${worker_outdir}" GPU="${gpu}" HOST PORT="${port}" NUM_ENVS
      export ROBOLAB ROBOLAB_PYTHON
      export ROBOLAB_DRIVERFIX ROBOLAB_DRIVERFIX_ASSETS
      export ROBOLAB_WRIST_FOCAL_LENGTH OVERLAY_SUBTASK_ON_VIDEO VIEWPOINT_ROTATE_180 ROBOLAB_VIDEO_QUALITY
      export ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS ROBOLAB_RELATIVE_GRIPPER_COMMANDS ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS ROBOLAB_GRIPPER_RELATIVE_THRESHOLD
      export ROBOLAB_REASONER_HISTORY_IMAGES REASONER_INCLUDE_PREVIEW_ACTION REASONER_OPEN_GRIPPER_MOVE_AUGMENT REASONER_REQUIRE_THINK REASONER_SEPARATE_VIEWS REASONER_THREE_GROUP_HISTORY REASONER_IMAGE_LONG_EDGE
      export INSTRUCTION_TYPE
      export ROBO_INTERACTIVE_REASONER_FIRST ROBO_INTERACTIVE_FLAT_LAYOUT ROBO_INTERACTIVE_SUPERVISOR_PER_TASK ROBO_INTERACTIVE_COMPACT_ARTIFACTS
      export ROBO_INTERACTIVE_SUPERVISOR_DIR="${supervisor_dir}"
      export EXTRA="${client_extra}"
      export GRASP_HORIZON GRASP_CONFIRM GRIPPER_OPEN GRIPPER_CLOSE MAX_EPISODE_SECONDS MAX_EPISODE_SECONDS_EXCEPT_TASKS
      exec bash "${CLIENT_SCRIPT}"
    ) &
    child_pid=$!
    if wait "${child_pid}"; then
      child_pid=""
      trap - INT TERM
      return 0
    else
      rc=$?
    fi
    child_pid=""

    if (( attempt >= CLIENT_MAX_RETRIES )); then
      echo "[new-server-old-robolab] client gpu${gpu} failed rc=${rc}; retries exhausted" >&2
      trap - INT TERM
      return "${rc}"
    fi

    attempt=$((attempt + 1))
    echo "[new-server-old-robolab] client gpu${gpu} failed rc=${rc}; retrying in ${CLIENT_RETRY_DELAY_SECONDS}s (completed episodes will be skipped)" >&2
    sleep "${CLIENT_RETRY_DELAY_SECONDS}"
  done
}

start_server() {
  local gpu="$1"
  local port="$2"
  local dist_port="$3"
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES="${gpu}"

  # Each GPU runs an independent single-process HTTP policy server.  Cluster
  # shells commonly inherit torchrun variables (for example WORLD_SIZE=8 and
  # MASTER_PORT=23456) from the parent job.  Normalize this replica to a
  # one-rank group and give every server its own deterministic rendezvous port.
  # init_script() correctly skips multi-rank init for WORLD_SIZE=1; the later
  # FSDP-aware standalone initializer then creates the required one-rank group
  # using this replica-specific env:// endpoint.
  export RANK=0
  export WORLD_SIZE=1
  export LOCAL_RANK=0
  export LOCAL_WORLD_SIZE=1
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="${dist_port}"
  unset GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
  unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
  export PATH="${SERVER_REPO}/.venv/bin:${PATH}"
  export PYTHONPATH="${SERVER_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

  export HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${USER_ROOT}/.cache}"
  export TMPDIR="${TMPDIR:-${USER_ROOT}/.tmp}"
  mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TMPDIR}" "${XDG_CACHE_HOME}"

  # TorchInductor/Triton kernels are sensitive to the exact torch, CUDA,
  # model-code, and dynamic-shape combination.  A shared historical cache can
  # otherwise reuse a stale bmm kernel and fail on the first real request with
  # CUBLAS_STATUS_INVALID_VALUE.  Keep compile enabled, but give each server
  # process a fresh per-run cache; this also prevents different GPU replicas
  # from racing on the same compiled artifact.
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR}/torchinductor_robolab_${STAMP}_gpu${gpu}_port${port}}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton_robolab_${STAMP}_gpu${gpu}_port${port}}"
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    # Torch 2.10+cu128 must resolve CUDA math libraries from its own
    # environment.  Remove every system CUDA path, not only the usual
    # /usr/local/cuda/lib64 spelling; otherwise a CUDA 12.9 libcublasLt can
    # be loaded beside Torch's CUDA 12.8 libcublas and fail at runtime.
    LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -vE '^/usr/local/cuda' | paste -sd ':' - || true)"
    export LD_LIBRARY_PATH
  fi
  FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-${USER_ROOT}/miniconda3/envs/cosmos3/lib}"
  if [[ -d "${FFMPEG_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    # Do not inherit a caller's arbitrary preload list.  Keep only the C++
    # runtime needed by torchcodec; CUDA libraries must still come from Torch.
    if [[ -f "${FFMPEG_LIB_DIR}/libstdc++.so.6" ]]; then
      export LD_PRELOAD="${FFMPEG_LIB_DIR}/libstdc++.so.6"
    else
      unset LD_PRELOAD
    fi
  fi

  cd "${SERVER_REPO}"
  SERVER_ARGS=(
    --checkpoint-path "${CKPT}"
    --port "${port}"
    --host 0.0.0.0
  )
  if [[ "${ALLOW_DCP_CHECKPOINT}" == "1" ]]; then
    SERVER_ARGS+=(--allow-dcp-checkpoint)
  fi
  if [[ -n "${EXPERIMENT}" ]]; then
    SERVER_ARGS+=(--experiment "${EXPERIMENT}")
  fi
  if [[ "${GUARDRAILS}" == "1" ]]; then
    SERVER_ARGS+=(--guardrails)
  else
    SERVER_ARGS+=(--no-guardrails)
  fi
  # shellcheck disable=SC2207
  ALIGNMENT_ARGS=($(server_alignment_args))
  SERVER_ARGS+=("${ALIGNMENT_ARGS[@]}")
  if [[ -n "${SERVER_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${SERVER_EXTRA_ARGS})
    SERVER_ARGS+=("${EXTRA_ARGS[@]}")
  fi

  echo "[new-server] repo=${SERVER_REPO}"
  echo "[new-server] python=${SERVER_PY}"
  echo "[new-server] GPU=${gpu} PORT=${port} standalone_ddp=${MASTER_ADDR}:${MASTER_PORT} ckpt=${CKPT} allow_dcp=${ALLOW_DCP_CHECKPOINT} experiment='${EXPERIMENT}' guardrails=${GUARDRAILS} compile_cache=${TORCHINDUCTOR_CACHE_DIR} triton_cache=${TRITON_CACHE_DIR}"
  exec "${SERVER_PY}" -m cosmos_framework.scripts.action_policy_server_robolab "${SERVER_ARGS[@]}"
}

merge_outputs() {
  if [[ "${MERGE_OUTPUTS}" != "1" ]]; then
    echo "[new-server-old-robolab] MERGE_OUTPUTS=${MERGE_OUTPUTS}; skip merge"
    return
  fi

  local final="${ROBOLAB_OUTPUT}/${FINAL_OUTDIR}"
  if [[ -e "${final}" && "${OVERWRITE_FINAL}" != "1" ]]; then
    echo "[new-server-old-robolab] ERROR: final output exists: ${final}" >&2
    echo "[new-server-old-robolab] Set OVERWRITE_FINAL=1 or choose a new FINAL_OUTDIR." >&2
    exit 1
  fi

  "${ROBOLAB_PYTHON}" - \
    "${final}" "${KEEP_WORKER_OUTPUTS}" "${TASKS_SUMMARY}" "${worker_outdirs[@]}" <<'PY'
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

final = Path(sys.argv[1])
keep_worker_outputs = sys.argv[2] == "1"
tasks_summary = Path(sys.argv[3])
sources = [Path(p) for p in sys.argv[4:]]

if final.exists():
    shutil.rmtree(final)
final.mkdir(parents=True)

def row_key(row: dict) -> tuple:
    return (
        row.get("env_name") or row.get("task_name"),
        row.get("run"),
        row.get("episode"),
        row.get("env_id"),
        row.get("instruction_type"),
    )

rows: list[dict] = []
seen: set[tuple] = set()
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "final_output": str(final),
    "sources": [str(s) for s in sources],
    "tasks_summary": str(tasks_summary),
    "rows_by_source": {},
    "copied_task_dirs": [],
    "duplicate_rows_skipped": [],
    "keep_worker_outputs": keep_worker_outputs,
}

for src in sources:
    if not src.is_dir():
        raise SystemExit(f"missing worker output: {src}")
    for item in src.iterdir():
        if item.name == "episode_results.jsonl":
            continue
        target = final / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
            manifest["copied_task_dirs"].append(item.name)
        elif item.is_file():
            shutil.copy2(item, target)

    result_file = src / "episode_results.jsonl"
    source_rows = 0
    if result_file.exists():
        for line in result_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_rows += 1
            key = row_key(row)
            if key in seen:
                manifest["duplicate_rows_skipped"].append({"source": str(src), "key": list(key)})
                continue
            seen.add(key)
            rows.append(row)
    manifest["rows_by_source"][str(src)] = source_rows

with (final / "episode_results.jsonl").open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

manifest["merged_rows"] = len(rows)
manifest["merged_task_dir_count"] = len({p.name for p in final.iterdir() if p.is_dir()})
manifest["copied_task_dirs"] = sorted(set(manifest["copied_task_dirs"]))
if tasks_summary.is_file():
    shutil.copy2(tasks_summary, final / "task_split_summary.tsv")
(final / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

if not rows:
    raise SystemExit(
        "[new-server-old-robolab] evaluation produced zero episode results; "
        "check the worker/server logs (worker outputs were retained)"
    )

if not keep_worker_outputs:
    for src in sources:
        shutil.rmtree(src)

print(f"[new-server-old-robolab] merged output={final}")
print(f"[new-server-old-robolab] merged_rows={manifest['merged_rows']} task_dirs={manifest['merged_task_dir_count']}")
PY
}

check_prereqs
if [[ "${DRY_RUN}" != "1" ]]; then
  preflight_robolab_assets
  preflight_robolab_graphics
else
  echo "[new-server-old-robolab] DRY_RUN=1; skipping Isaac/USD and graphics preflight"
fi
validate_robolab_wrist_focal_length
validate_guidance_args
validate_reasoner_mode_flags
validate_sim_workers
validate_video_mode

echo "[new-server-old-robolab] server_repo=${SERVER_REPO}"
echo "[new-server-old-robolab] server_py=${SERVER_PY}"
echo "[new-server-old-robolab] client_script=${CLIENT_SCRIPT}"
echo "[new-server-old-robolab] ckpt=${CKPT}"
echo "[new-server-old-robolab] GPUS=${GPUS_CSV} gpu_workers=${NGPU} nodes=${NNODES} node_rank=${NODE_RANK} sim_workers_per_gpu=${SIM_WORKERS_PER_GPU} base_port=${BASE_PORT} dist_base_port=${DIST_BASE_PORT}"
WORKER_COUNT=$((NGPU * SIM_WORKERS_PER_GPU))
echo "[new-server-old-robolab] total_sim_workers=${WORKER_COUNT} local (${NNODES} nodes => $((NNODES * WORKER_COUNT)) global)"
echo "[new-server-old-robolab] GUARDRAILS=${GUARDRAILS}"
echo "[new-server-old-robolab] GUIDANCE=${GUIDANCE} CAMERA=${CAMERA} CAMERA_CFG_MODE=${CAMERA_CFG_MODE} CAMERA_GUIDANCE=${CAMERA_GUIDANCE}"
echo "[new-server-old-robolab] WRIST_ROTATE_180=${WRIST_ROTATE_180} DROID_VIDEO_SAMPLE_STRIDE=${DROID_VIDEO_SAMPLE_STRIDE} ACTION_FPS=${ACTION_FPS}"
echo "[new-server-old-robolab] VIT_SOFTPLAN=${VIT_SOFTPLAN} VIT_PROCESSOR_MODEL_NAME=${VIT_PROCESSOR_MODEL_NAME} VIT_FIRST_FRAME_MAX_PIXELS=${VIT_FIRST_FRAME_MAX_PIXELS}"
echo "[new-server-old-robolab] DECODE_SUBTASK=${DECODE_SUBTASK} SUBTASK_MAX_NEW_TOKENS=${SUBTASK_MAX_NEW_TOKENS} OVERLAY_SUBTASK_ON_VIDEO=${OVERLAY_SUBTASK_ON_VIDEO} VIEWPOINT_ROTATE_180=${VIEWPOINT_ROTATE_180}"
echo "[new-server-old-robolab] FORMAT_PROMPT_AS_JSON=${FORMAT_PROMPT_AS_JSON} INCLUDE_EXECUTION_SUCCESS=${INCLUDE_EXECUTION_SUCCESS} ROBOLAB_VIDEO_QUALITY=${ROBOLAB_VIDEO_QUALITY}"
echo "[new-server-old-robolab] gripper_filter=${ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS} relative=${ROBOLAB_RELATIVE_GRIPPER_COMMANDS} previous_frame_relative=${ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS} relative_threshold=${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD}"
echo "[new-server-old-robolab] reasoner_history_images=${ROBOLAB_REASONER_HISTORY_IMAGES} reasoner_include_preview_action=${REASONER_INCLUDE_PREVIEW_ACTION} reasoner_require_think=${REASONER_REQUIRE_THINK}"
echo "[new-server-old-robolab] instruction_type=${INSTRUCTION_TYPE}"
echo "[new-server-old-robolab] reasoner_separate_views=${REASONER_SEPARATE_VIEWS} reasoner_three_group_history=${REASONER_THREE_GROUP_HISTORY} reasoner_image_long_edge=${REASONER_IMAGE_LONG_EDGE}"
echo "[new-server-old-robolab] HF_HOME=${HF_HOME} HF_HUB_OFFLINE=${HF_HUB_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "[new-server-old-robolab] REASONER_WAM=${REASONER_WAM} REASONER_THEN_WAM=${REASONER_THEN_WAM} REASONER_WAM_MODE=${REASONER_WAM_MODE} (server flags are opt-in passthroughs)"
if [[ "${REASONER_WAM_MODE}" == "mode2" || "${REASONER_WAM_MODE}" == "adapter" ]]; then
  echo "[new-server-old-robolab] ${REASONER_WAM_MODE} reasoner_checkpoint=${REASONER_CHECKPOINT_PATH:-<CKPT>} wam_checkpoint=${WAM_CHECKPOINT_PATH:-<CKPT>} lora=${REASONER_LORA_PATH:-<embedded-or-none>} projector=${REASONER_PROJECTOR_PATH:-<embedded-or-none>}"
fi
echo "[new-server-old-robolab] REASONER_MAX_NEW_TOKENS=${REASONER_MAX_NEW_TOKENS} REASONER_DO_SAMPLE=${REASONER_DO_SAMPLE}"
echo "[new-server-old-robolab] REASONER_OPEN_GRIPPER_MOVE_AUGMENT=${REASONER_OPEN_GRIPPER_MOVE_AUGMENT}"
echo "[new-server-old-robolab] ROBOLAB_WRIST_FOCAL_LENGTH=${ROBOLAB_WRIST_FOCAL_LENGTH}"
echo "[new-server-old-robolab] SERVER_ALIGNMENT_ARGS='$(server_alignment_args)'"
echo "[new-server-old-robolab] TASK_SET=${TASK_SET} TASKS_FILE=${TASKS_FILE:-<auto>}"
echo "[new-server-old-robolab] MAX_EPISODE_SECONDS=${MAX_EPISODE_SECONDS:-<task-default>}"
echo "[new-server-old-robolab] MAX_EPISODE_SECONDS_EXCEPT_TASKS=${MAX_EPISODE_SECONDS_EXCEPT_TASKS:-<none>}"
echo "[new-server-old-robolab] CLIENT_MAX_RETRIES=${CLIENT_MAX_RETRIES} CLIENT_RETRY_DELAY_SECONDS=${CLIENT_RETRY_DELAY_SECONDS}"
echo "[new-server-old-robolab] BASE_OUTDIR=${BASE_OUTDIR} FINAL_OUTDIR=${FINAL_OUTDIR}"
echo "[new-server-old-robolab] EXTRA='${EXTRA}'"
echo "[new-server-old-robolab] CLIENT_EXTRA='$(client_extra_args "${EXTRA}")'"

build_task_splits

# Build two explicitly different inventories:
#   * server_*: one entry per GPU slot with at least one task;
#   * worker_*: one entry per non-empty (GPU slot, simulator slot) task file.
# A logical worker index is mapped with divmod so its server port is always the
# GPU's port, while its output/log names remain unique within that GPU.
for ((idx = 0; idx < WORKER_COUNT; idx++)); do
  gpu_slot=$((idx / SIM_WORKERS_PER_GPU))
  sim_slot=$((idx % SIM_WORKERS_PER_GPU))
  gpu="${GPU_LIST[$gpu_slot]}"
  global_gpu_slot=$((NODE_RANK * NGPU + gpu_slot))
  global_gpu_slot_for_log="${global_gpu_slot}"
  task_file="${TASKS_DIR}/worker_${idx}.tasks"
  if [[ ! -s "${task_file}" ]]; then
    echo "[new-server-old-robolab] gpu${global_gpu_slot_for_log}/sim${sim_slot} worker${idx} has no tasks; skipping"
    continue
  fi

  # Add the GPU server once, even when several simulator workers share it.
  server_seen=0
  for server_slot in "${server_indices[@]:-}"; do
    if [[ "${server_slot}" == "${gpu_slot}" ]]; then
      server_seen=1
      break
    fi
  done
  if (( server_seen == 0 )); then
    port=$((BASE_PORT + gpu_slot))
    dist_port=$((DIST_BASE_PORT + gpu_slot))
    if [[ "${DRY_RUN}" != "1" ]]; then
      check_port_free "${port}"
      check_port_free "${dist_port}"
    fi
    server_indices+=("${gpu_slot}")
    if (( NNODES > 1 )); then
      server_logs+=("${LOG_DIR}/new_server_old_robolab_server_node${NODE_RANK}_gpu${global_gpu_slot}_${STAMP}.log")
    else
      server_logs+=("${LOG_DIR}/new_server_old_robolab_server_gpu${gpu}_${STAMP}.log")
    fi
  fi

  if (( SIM_WORKERS_PER_GPU == 1 )); then
    # Preserve historical output/log names for the single-node topology;
    # multi-node runs use global GPU slots (_gpu0 ... _gpu3).
    if (( NNODES > 1 )); then
      worker_outdir="${BASE_OUTDIR}_gpu${global_gpu_slot}"
      client_log="${LOG_DIR}/new_server_old_robolab_client_node${NODE_RANK}_gpu${global_gpu_slot}_${STAMP}.log"
    else
      worker_outdir="${BASE_OUTDIR}_gpu${gpu}"
      client_log="${LOG_DIR}/new_server_old_robolab_client_gpu${gpu}_${STAMP}.log"
    fi
  else
    if (( NNODES > 1 )); then
      worker_outdir="${BASE_OUTDIR}_gpu${global_gpu_slot}_sim${sim_slot}"
      client_log="${LOG_DIR}/new_server_old_robolab_client_node${NODE_RANK}_gpu${global_gpu_slot}_sim${sim_slot}_${STAMP}.log"
    else
      worker_outdir="${BASE_OUTDIR}_gpu${gpu}_sim${sim_slot}"
      client_log="${LOG_DIR}/new_server_old_robolab_client_gpu${gpu}_sim${sim_slot}_${STAMP}.log"
    fi
  fi
  worker_indices+=("${idx}")
  worker_gpu_indices+=("${gpu_slot}")
  worker_sim_indices+=("${sim_slot}")
  task_files+=("${task_file}")
  worker_outdirs+=("${ROBOLAB_OUTPUT}/${worker_outdir}")
  client_logs+=("${client_log}")
done

if (( ${#worker_indices[@]} == 0 )); then
  echo "[new-server-old-robolab] no non-empty task splits; exiting 0"
  if (( NNODES > 1 && NODE_RANK != 0 )); then
    printf 'session=%s\n' "${COORD_SESSION_ID}" > "${COORD_ROOT}/node_${NODE_RANK}.done.tmp.$$"
    mv -f "${COORD_ROOT}/node_${NODE_RANK}.done.tmp.$$" "${COORD_ROOT}/node_${NODE_RANK}.done"
  fi
  if (( NNODES == 1 || NODE_RANK != 0 )); then
    exit 0
  fi
fi

for local_i in "${!server_indices[@]}"; do
  gpu_slot="${server_indices[$local_i]}"
  gpu="${GPU_LIST[$gpu_slot]}"
  global_gpu_slot=$((NODE_RANK * NGPU + gpu_slot))
  port=$((BASE_PORT + gpu_slot))
  dist_port=$((DIST_BASE_PORT + gpu_slot))
  echo "[new-server-old-robolab] server plan: gpu${global_gpu_slot} (local=${gpu}) slot=${gpu_slot} port=${port} dist_port=${dist_port} log=${server_logs[$local_i]}"
done
for local_i in "${!worker_indices[@]}"; do
  idx="${worker_indices[$local_i]}"
  gpu_slot="${worker_gpu_indices[$local_i]}"
  sim_slot="${worker_sim_indices[$local_i]}"
  gpu="${GPU_LIST[$gpu_slot]}"
  global_gpu_slot=$((NODE_RANK * NGPU + gpu_slot))
  port=$((BASE_PORT + gpu_slot))
  echo "[new-server-old-robolab] client plan: worker${idx} gpu${global_gpu_slot} (local=${gpu})/sim${sim_slot} port=${port} tasks=${task_files[$local_i]} OUTDIR=$(basename "${worker_outdirs[$local_i]}") log=${client_logs[$local_i]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[new-server-old-robolab] DRY_RUN=1; launch plan validated, not starting servers/clients"
  exit 0
fi

for local_i in "${!server_indices[@]}"; do
  gpu_slot="${server_indices[$local_i]}"
  gpu="${GPU_LIST[$gpu_slot]}"
  port=$((BASE_PORT + gpu_slot))
  dist_port=$((DIST_BASE_PORT + gpu_slot))
  echo "[new-server-old-robolab] starting server gpu${gpu} slot=${gpu_slot} port=${port} dist_port=${dist_port} log=${server_logs[$local_i]}"
  start_server "${gpu}" "${port}" "${dist_port}" >"${server_logs[$local_i]}" 2>&1 &
  server_pids+=("$!")
done

for local_i in "${!server_pids[@]}"; do
  gpu_slot="${server_indices[$local_i]}"
  wait_ready "${server_pids[$local_i]}" "${server_logs[$local_i]}" "server_gpu${GPU_LIST[$gpu_slot]}"
done

for local_i in "${!worker_indices[@]}"; do
  idx="${worker_indices[$local_i]}"
  gpu_slot="${worker_gpu_indices[$local_i]}"
  sim_slot="${worker_sim_indices[$local_i]}"
  gpu="${GPU_LIST[$gpu_slot]}"
  port=$((BASE_PORT + gpu_slot))
  task="$(task_string "${task_files[$local_i]}")"
  worker_outdir="$(basename "${worker_outdirs[$local_i]}")"
  client_extra="$(client_extra_args "${EXTRA}")"
  global_gpu_slot=$((NODE_RANK * NGPU + gpu_slot))
  echo "[new-server-old-robolab] starting client worker${idx} gpu${global_gpu_slot} (local=${gpu})/sim${sim_slot} port=${port} OUTDIR=${worker_outdir} log=${client_logs[$local_i]}"
  run_client_with_retries \
    "${task}" "${worker_outdir}" "${gpu}" "${port}" "${client_extra}" \
    >"${client_logs[$local_i]}" 2>&1 &
  client_pids+=("$!")
done

status=0
for pid in "${client_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[new-server-old-robolab] failed; client logs:"
  printf '  %s\n' "${client_logs[@]}"
  exit "${status}"
fi

if (( NNODES > 1 )); then
  # Publish this node's worker directories on the shared filesystem. Node 0
  # waits for all ranks, imports their directories, and performs the existing
  # merge path by default.  Non-zero ranks must stay alive until node 0 has
  # completed that merge; otherwise their outer EVAL_REPEATS loop can start
  # the next repeat while node 0 is still merging the previous one.
  {
    printf 'session=%s\n' "${COORD_SESSION_ID}"
    printf '%s\n' "${worker_outdirs[@]}"
  } > "${COORD_ROOT}/node_${NODE_RANK}.done.tmp.$$"
  mv -f "${COORD_ROOT}/node_${NODE_RANK}.done.tmp.$$" "${COORD_ROOT}/node_${NODE_RANK}.done"
  if (( NODE_RANK != 0 )); then
    echo "[new-server-old-robolab] node${NODE_RANK} complete; waiting for node0 merge: ${GLOBAL_MERGE_DONE}"
    deadline=$((SECONDS + WAIT_TIMEOUT))
    while ! marker_matches_session "${GLOBAL_MERGE_DONE}" && (( SECONDS < deadline )); do
      sleep 5
    done
    if ! marker_matches_session "${GLOBAL_MERGE_DONE}"; then
      echo "[new-server-old-robolab] ERROR: timeout waiting for node0 merge: ${GLOBAL_MERGE_DONE}" >&2
      exit 1
    fi
    if ! grep -qx 'status=ok' "${GLOBAL_MERGE_DONE}"; then
      echo "[new-server-old-robolab] ERROR: node0 did not report a successful merge: ${GLOBAL_MERGE_DONE}" >&2
      cat "${GLOBAL_MERGE_DONE}" >&2 || true
      exit 1
    fi
    echo "[new-server-old-robolab] node${NODE_RANK} released after node0 merge; entering next repeat"
    exit 0
  fi
  deadline=$((SECONDS + WAIT_TIMEOUT))
  for ((other_rank = 1; other_rank < NNODES; other_rank++)); do
    marker="${COORD_ROOT}/node_${other_rank}.done"
    while ! marker_matches_session "${marker}" && (( SECONDS < deadline )); do
      sleep 5
    done
    if ! marker_matches_session "${marker}"; then
      echo "[new-server-old-robolab] ERROR: timeout waiting for node${other_rank}: ${marker}" >&2
      exit 1
    fi
    while IFS= read -r remote_outdir; do
      [[ -n "${remote_outdir}" && "${remote_outdir}" != session=* ]] && worker_outdirs+=("${remote_outdir}")
    done < "${marker}"
  done
fi

merge_outputs

# Release the other nodes only after merge_outputs has returned successfully.
# Write atomically so waiters never observe a partially-written barrier file.
{
  printf 'session=%s\n' "${COORD_SESSION_ID}"
  printf 'status=ok\n'
  printf 'final=%s\n' "${ROBOLAB_OUTPUT}/${FINAL_OUTDIR}"
} > "${GLOBAL_MERGE_DONE}.tmp.$$"
mv -f "${GLOBAL_MERGE_DONE}.tmp.$$" "${GLOBAL_MERGE_DONE}"
echo "[new-server-old-robolab] global merge barrier released: ${GLOBAL_MERGE_DONE}"

echo "[new-server-old-robolab] completed successfully"
echo "[new-server-old-robolab] final=${ROBOLAB_OUTPUT}/${FINAL_OUTDIR}"
if [[ "${KEEP_WORKER_OUTPUTS}" == "1" ]]; then
  echo "[new-server-old-robolab] worker_outputs retained (KEEP_WORKER_OUTPUTS=1):"
  printf '  %s\n' "${worker_outdirs[@]}"
else
  echo "[new-server-old-robolab] worker_outputs removed after merge (KEEP_WORKER_OUTPUTS=0)"
fi
