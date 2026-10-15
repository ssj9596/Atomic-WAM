#!/usr/bin/env bash
set -euo pipefail

# 120-task Reasoner adapter -> JSON WAM RoboLab evaluation. Projector weights
# are optional so a LoRA-only Reasoner uses the frozen base visual projector.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_HUMAN="${CODE_HUMAN:-${SCRIPT_DIR}}"
ROOT="${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
# RoboLab's evaluator resolves ``--output-folder-name`` under the repository's
# own ``output`` directory.  Keep the launcher/merge path identical to that
# location; using a separate runtime mirror here makes workers write to one
# tree while the merge step looks in another.
ROBOLAB="${ROBOLAB:-${ROOT}/robolab}"
ROBOLAB_OUTPUT="${ROBOLAB_OUTPUT:-${ROBOLAB}/output}"
WATCHER="${WATCHER:-}"
LAUNCHER="${LAUNCHER:-${ROBOLAB}/scripts/repro_run_server.sh}"
LAUNCHER_SNAPSHOT=""
TASKS_SOURCE="${TASKS_SOURCE:-}"
TASKS_FILE="${TASKS_FILE:-}"
SERVER_REPO="${SERVER_REPO:-${ROOT}/cosmos-framework}"
REASONER_ARTIFACTS="${REASONER_ARTIFACTS:-${ROOT}/cosmos-framework/checkpoints/reasoner_lora}"
WAM_CHECKPOINT_PATH="${WAM_CHECKPOINT_PATH:-${ROOT}/cosmos-framework/checkpoints/wam}"
if [[ -z "${SERVER_PY:-}" ]]; then
  if [[ -n "${COSMOS_SERVER_PYTHON:-}" ]]; then
    SERVER_PY="${COSMOS_SERVER_PYTHON}"
  elif [[ -x "${SERVER_REPO:-${ROOT}/cosmos-framework}/.venv/bin/python" ]]; then
    SERVER_PY="${SERVER_REPO:-${ROOT}/cosmos-framework}/.venv/bin/python"
  else
    SERVER_PY="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "${ROBOLAB_PYTHON:-}" ]]; then
  ROBOLAB_PYTHON="${ROBO_PYTHON:-}"
fi
if [[ -z "${SS_BIN:-}" ]]; then
  if [[ -x /usr/bin/ss ]]; then
    SS_BIN=/usr/bin/ss
  else
    SS_BIN="$(command -v ss || true)"
  fi
fi
if [[ -z "${SS_BIN}" || ! -x "${SS_BIN}" ]]; then
  printf 'system ss executable not found; set SS_BIN explicitly\n' >&2
  exit 1
fi

detect_gpus() {
  if [[ -n "${GPUS:-}" ]]; then
    printf '%s\n' "${GPUS}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local detected
    detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
    if [[ -n "${detected}" ]]; then
      printf '%s\n' "${detected}"
      return
    fi
  fi
  printf 'Unable to auto-detect GPUs; set GPUS=0,1 explicitly\n' >&2
  return 1
}

GPUS="$(detect_gpus)"
# Multi-node inference is data-parallel: every node owns a local GPU list and
# receives a deterministic shard of the global task inventory.  Reuse the
# same platform variables as the training launchers; single-node defaults are
# unchanged.
if [[ -n "${MLP_WORKER_NUM:-}" && -n "${MLP_WORKER_0_HOST:-}" ]]; then
  NNODES="${MLP_WORKER_NUM}"
  NODE_RANK="${MLP_ROLE_INDEX:?MLP_ROLE_INDEX is required when MLP_WORKER_NUM is set}"
  MASTER_ADDR="${MLP_WORKER_0_HOST}"
  MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"
elif [[ -n "${SENSECORE_PYTORCH_NNODES:-}" && "${SENSECORE_PYTORCH_NNODES}" -gt 1 ]]; then
  NNODES="${SENSECORE_PYTORCH_NNODES}"
  NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-0}}"
  MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR for multi-node inference}"
  MASTER_PORT="${MASTER_PORT:-29500}"
else
  NNODES="${NNODES:-1}"
  NODE_RANK="${NODE_RANK:-0}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29500}"
fi
if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  printf 'NNODES/NODE_RANK invalid: NNODES=%q NODE_RANK=%q\n' "${NNODES}" "${NODE_RANK}" >&2
  exit 1
fi
BASE_PORT="${BASE_PORT:-8000}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-18000}"
# Two independent concurrency axes:
#   NUM_ENVS: vector environments inside one Isaac process.
#   SIM_WORKERS_PER_GPU: independent Isaac/client processes sharing one
#                        policy server on each GPU.
# Keep the historical topology: one independent simulator/client process per
# GPU.  The policy server in the public main branch handles requests
# synchronously, so multiple clients per GPU are not a supported default.
NUM_ENVS="${NUM_ENVS:-1}"
SIM_WORKERS_PER_GPU="${SIM_WORKERS_PER_GPU:-1}"
VIDEO_MODE="${VIDEO_MODE:-all}"
EXPORT_POLL_SECONDS="${EXPORT_POLL_SECONDS:-60}"
GUIDANCE="${GUIDANCE:-3.0}"
CAMERA="${CAMERA:-0}"
CAMERA_GUIDANCE="${CAMERA_GUIDANCE:-1.0}"
CAMERA_CFG_MODE="${CAMERA_CFG_MODE:-separate}"
EVAL_PLAN="${EVAL_PLAN:-single}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}"
EVAL_LABEL="${EVAL_LABEL:-}"
VIT_SOFTPLAN="${VIT_SOFTPLAN:-0}"
MID_SOFT="${MID_SOFT:-0}"
DECODE_SUBTASK="${DECODE_SUBTASK:-0}"
SUBTASK_MAX_NEW_TOKENS="${SUBTASK_MAX_NEW_TOKENS:-128}"
VIEWPOINT_ROTATE_180="${VIEWPOINT_ROTATE_180:-0}"
# Planner-mode flags are opt-in passthroughs to the RoboLab server.  Keep both
# disabled by default so existing direct-WAM and auxiliary-decoder evaluations
# are byte-for-byte unchanged unless the server-side cascade is explicitly enabled.
REASONER_WAM="${REASONER_WAM:-1}"
REASONER_THEN_WAM="${REASONER_THEN_WAM:-0}"
REASONER_WAM_MODE="${REASONER_WAM_MODE:-adapter}"
REASONER_CHECKPOINT_PATH="${REASONER_CHECKPOINT_PATH:-}"
REASONER_LORA_PATH="${REASONER_LORA_PATH:-${REASONER_ARTIFACTS}/reasoner_lora.safetensors}"
REASONER_LORA_METADATA="${REASONER_LORA_METADATA:-${REASONER_LORA_PATH%.safetensors}.json}"
REASONER_PROJECTOR_PATH="${REASONER_PROJECTOR_PATH-${REASONER_ARTIFACTS}/reasoner_projector.safetensors}"
if [[ ! -f "${REASONER_PROJECTOR_PATH}" ]]; then
  REASONER_PROJECTOR_PATH=""
fi
REASONER_MAX_NEW_TOKENS="${REASONER_MAX_NEW_TOKENS:-128}"
REASONER_DO_SAMPLE="${REASONER_DO_SAMPLE:-0}"
REASONER_INCLUDE_PREVIEW_ACTION="${REASONER_INCLUDE_PREVIEW_ACTION:-0}"
REASONER_OPEN_GRIPPER_MOVE_AUGMENT="${REASONER_OPEN_GRIPPER_MOVE_AUGMENT:-0}"
REASONER_REQUIRE_THINK="${REASONER_REQUIRE_THINK:-0}"
REASONER_SEPARATE_VIEWS="${REASONER_SEPARATE_VIEWS:-0}"
REASONER_THREE_GROUP_HISTORY="${REASONER_THREE_GROUP_HISTORY:-0}"
REASONER_IMAGE_LONG_EDGE="${REASONER_IMAGE_LONG_EDGE:-448}"
FORMAT_PROMPT_AS_JSON="${FORMAT_PROMPT_AS_JSON:-1}"
INCLUDE_EXECUTION_SUCCESS="${INCLUDE_EXECUTION_SUCCESS:-0}"
ROBOLAB_VIDEO_QUALITY="${ROBOLAB_VIDEO_QUALITY:-4}"
ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS="${ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS:-0}"
ROBOLAB_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_RELATIVE_GRIPPER_COMMANDS:-0}"
ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS:-0}"
ROBOLAB_GRIPPER_RELATIVE_THRESHOLD="${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD:-0.10}"
ROBOLAB_REASONER_HISTORY_IMAGES="${ROBOLAB_REASONER_HISTORY_IMAGES:-0}"
# Keep full evaluation artifacts (including run_*.hdf5) unless a caller
# explicitly opts into compact mode for a lightweight run.
ROBO_INTERACTIVE_COMPACT_ARTIFACTS="${ROBO_INTERACTIVE_COMPACT_ARTIFACTS:-0}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-default}"
ROBO_INTERACTIVE_REASONER_FIRST="${ROBO_INTERACTIVE_REASONER_FIRST:-0}"
ROBO_INTERACTIVE_SUPERVISOR_DIR="${ROBO_INTERACTIVE_SUPERVISOR_DIR:-}"
ROBOLAB_DRIVERFIX="${ROBOLAB_DRIVERFIX:-0}"
ROBOLAB_DRIVERFIX_ASSETS="${ROBOLAB_DRIVERFIX_ASSETS:-}"

# Respect an explicit subtitle setting. If it is omitted, show the generated
# Reasoner plan for the opt-in cascade and retain the historical off default
# for direct-WAM/mid-soft evaluations.
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
EXPORT_SUBTASK_DECODER="${EXPORT_SUBTASK_DECODER:-1}"
VIT_PROCESSOR_MODEL_NAME="${VIT_PROCESSOR_MODEL_NAME:-${ROOT}/cosmos-framework/checkpoints/wam}"
VIT_PROCESSOR_CONFIG_VARIANT="${VIT_PROCESSOR_CONFIG_VARIANT:-hf}"
VIT_PROCESSOR_CACHE_DIR="${VIT_PROCESSOR_CACHE_DIR:-}"
VIT_FIRST_FRAME_MAX_PIXELS="${VIT_FIRST_FRAME_MAX_PIXELS:-348160}"
VIT_FIRST_FRAME_MIN_PIXELS="${VIT_FIRST_FRAME_MIN_PIXELS:-}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/cosmos-framework/checkpoints/wam/wan2.2_vae.pth}"
COSMOS3_NANO_ROOT="${COSMOS3_NANO_ROOT:-${VIT_PROCESSOR_MODEL_NAME}}"
ROBOLAB_ASSETS="${ROBOLAB_ASSETS:-}"
export WAN_VAE_PATH COSMOS3_NANO_ROOT ROBOLAB_ASSETS
export COSMOS_TRAINING=0

if [[ "${CAMERA}" == "1" || "${CAMERA}" == "true" || "${CAMERA}" == "TRUE" ]]; then
  TRAIN_NAME="${TRAIN_NAME:-droid_joint_cam2base_768_viddown4_plucker}"
  TRAIN_DIR="${TRAIN_DIR:-${ROOT}/outputs/cosmos3-droid_joint_cam2base_768_viddown4_plucker/cosmos3_action/action_sft_droid/action_policy_droid_success_both_ext_cam2base_joint_16_global768_wristrot180_viddown4_plucker}"
else
  TRAIN_NAME="${TRAIN_NAME:-droid_joint_cam2base_768_viddown4}"
  TRAIN_DIR="${TRAIN_DIR:-${ROOT}/outputs/cosmos3-droid_joint_cam2base_768_viddown4/cosmos3_action/action_sft_droid/action_policy_droid_success_both_ext_cam2base_joint_16_global768_wristrot180_viddown4}"
fi
EVAL_ITER="${EVAL_ITER:-12000}"
EVAL_REPEATS="${EVAL_REPEATS:-1}"
EVAL_RETRY_COUNT="${EVAL_RETRY_COUNT:-3}"
EVAL_RETRY_DELAY_SECONDS="${EVAL_RETRY_DELAY_SECONDS:-30}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${WAM_CHECKPOINT_PATH}}"
EVAL_LABEL="${EVAL_LABEL:-reasoner5500_wam12000_json}"
EVAL_BASE_OUTDIR="${EVAL_BASE_OUTDIR:-cosmos3_droid_all120_reasoner5500_wam12000_json_q4}"
DROID_VIDEO_SAMPLE_STRIDE="${DROID_VIDEO_SAMPLE_STRIDE:-1}"
MAX_EPISODE_SECONDS="${MAX_EPISODE_SECONDS:-180}"
DRY_RUN="${DRY_RUN:-0}"
WRIST_ROTATE_180="${WRIST_ROTATE_180:-1}"
if [[ -z "${SERVER_EXTRA_ARGS+x}" ]]; then
  SERVER_EXTRA_ARGS='--resolution 544x640'
fi

case "${CAMERA_CFG_MODE}" in
  camera|both|separate) ;;
  *)
    printf 'CAMERA_CFG_MODE must be camera/both/separate, got %q\n' "${CAMERA_CFG_MODE}" >&2
    exit 1
    ;;
esac

for _reasoner_mode_name in REASONER_WAM REASONER_THEN_WAM REASONER_INCLUDE_PREVIEW_ACTION REASONER_OPEN_GRIPPER_MOVE_AUGMENT REASONER_REQUIRE_THINK REASONER_SEPARATE_VIEWS REASONER_THREE_GROUP_HISTORY; do
  _reasoner_mode_value="${!_reasoner_mode_name}"
  case "${_reasoner_mode_value}" in
    0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
    *)
      printf '%s must be 0/1 or false/true, got %q\n' "${_reasoner_mode_name}" "${_reasoner_mode_value}" >&2
      exit 1
      ;;
  esac
done
unset _reasoner_mode_name _reasoner_mode_value
case "${REASONER_INCLUDE_PREVIEW_ACTION}" in
  1|true|TRUE|yes|YES|on|ON)
    case "${REASONER_WAM}:${REASONER_THEN_WAM}" in
      1:*|true:*|TRUE:*|yes:*|YES:*|on:*|ON:*|*:1|*:true|*:TRUE|*:yes|*:YES|*:on|*:ON) ;;
      *)
        printf 'REASONER_INCLUDE_PREVIEW_ACTION requires REASONER_WAM=1 (or REASONER_THEN_WAM=1)\n' >&2
        exit 1
        ;;
    esac
    ;;
esac
case "${REASONER_DO_SAMPLE}" in
  0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
  *)
    printf 'REASONER_DO_SAMPLE must be 0/1 or false/true, got %q\n' "${REASONER_DO_SAMPLE}" >&2
    exit 1
    ;;
esac
case "${REASONER_WAM_MODE}" in
  shared|mode2|adapter) ;;
  *)
    printf 'REASONER_WAM_MODE must be shared, mode2, or adapter, got %q\n' "${REASONER_WAM_MODE}" >&2
    exit 1
    ;;
esac
if [[ "${REASONER_WAM_MODE}" == "mode2" ]]; then
  case "${REASONER_WAM}:${REASONER_THEN_WAM}" in
    1:*|true:*|TRUE:*|yes:*|YES:*|on:*|ON:*|*:1|*:true|*:TRUE|*:yes|*:YES|*:on|*:ON) ;;
    *)
      printf 'REASONER_WAM_MODE=mode2 requires REASONER_WAM=1 (or REASONER_THEN_WAM=1)\n' >&2
      exit 1
      ;;
  esac
  if [[ -z "${WAM_CHECKPOINT_PATH}" ]]; then
    printf 'REASONER_WAM_MODE=mode2 requires WAM_CHECKPOINT_PATH (standalone no-LoRA WAM run/checkpoint)\n' >&2
    exit 1
  fi
fi
if ! [[ "${REASONER_MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'REASONER_MAX_NEW_TOKENS must be a positive integer, got %q\n' "${REASONER_MAX_NEW_TOKENS}" >&2
  exit 1
fi
if ! [[ "${EVAL_RETRY_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EVAL_RETRY_COUNT must be a positive integer, got %q\n' "${EVAL_RETRY_COUNT}" >&2
  exit 1
fi
if ! [[ "${EVAL_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  printf 'EVAL_RETRY_DELAY_SECONDS must be a non-negative integer, got %q\n' "${EVAL_RETRY_DELAY_SECONDS}" >&2
  exit 1
fi
case "${OVERLAY_SUBTASK_ON_VIDEO}" in
  0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
  *)
    printf 'OVERLAY_SUBTASK_ON_VIDEO must be 0/1 or false/true, got %q\n' "${OVERLAY_SUBTASK_ON_VIDEO}" >&2
    exit 1
    ;;
esac
case "${INCLUDE_EXECUTION_SUCCESS}" in
  0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
  *)
    printf 'INCLUDE_EXECUTION_SUCCESS must be 0/1 or false/true, got %q\n' "${INCLUDE_EXECUTION_SUCCESS}" >&2
    exit 1
    ;;
esac
case "${VIEWPOINT_ROTATE_180}" in
  0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
  *)
    printf 'VIEWPOINT_ROTATE_180 must be 0/1 or false/true, got %q\n' "${VIEWPOINT_ROTATE_180}" >&2
    exit 1
    ;;
esac
case "${DRY_RUN}" in
  0|1) ;;
  *)
    printf 'DRY_RUN must be 0/1 or false/true, got %q\n' "${DRY_RUN}" >&2
    exit 1
    ;;
esac
case "${WRIST_ROTATE_180}" in
  0|1|false|true|FALSE|TRUE|no|yes|NO|YES|off|on|OFF|ON) ;;
  *)
    printf 'WRIST_ROTATE_180 must be 0/1 or false/true, got %q\n' "${WRIST_ROTATE_180}" >&2
    exit 1
    ;;
esac
if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NUM_ENVS must be a positive integer, got %q\n' "${NUM_ENVS}" >&2
  exit 1
fi
if ! [[ "${SIM_WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SIM_WORKERS_PER_GPU must be a positive integer, got %q\n' "${SIM_WORKERS_PER_GPU}" >&2
  exit 1
fi
case "${VIDEO_MODE}" in
  all|sensor|viewport) ;;
  none)
    printf 'VIDEO_MODE=none is disabled for evaluation; keep video recording enabled (all, sensor, or viewport)\n' >&2
    exit 1
    ;;
  *)
    printf 'VIDEO_MODE must be all, sensor, or viewport (none is disabled), got %q\n' "${VIDEO_MODE}" >&2
    exit 1
    ;;
esac
if [[ -n "${EXTRA:-}" ]]; then
  read -r -a _extra_args <<< "${EXTRA}"
  for ((_extra_i = 0; _extra_i < ${#_extra_args[@]}; _extra_i++)); do
    _extra_token="${_extra_args[_extra_i]}"
    _extra_video_mode=""
    case "${_extra_token}" in
      --video-mode|--video_mode)
        if (( _extra_i + 1 >= ${#_extra_args[@]} )); then
          printf 'EXTRA contains %q without a mode value\n' "${_extra_token}" >&2
          exit 1
        fi
        _extra_video_mode="${_extra_args[_extra_i + 1]}"
        ;;
      --video-mode=*|--video_mode=*)
        _extra_video_mode="${_extra_token#*=}"
        ;;
      *)
        continue
        ;;
    esac
    case "${_extra_video_mode}" in
      all|sensor|viewport) ;;
      none)
        printf 'EXTRA requests video mode none; video recording cannot be disabled for evaluation\n' >&2
        exit 1
        ;;
      *)
        printf 'EXTRA requests unsupported video mode %q; use all, sensor, or viewport\n' "${_extra_video_mode}" >&2
        exit 1
        ;;
    esac
  done
  unset _extra_args _extra_i _extra_token _extra_video_mode
fi

IFS=',' read -r -a GPU_LIST <<<"${GPUS}"
declare -A _gpu_seen=()
if ! [[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'GPUS must be a comma-separated list of non-negative integers, got %q\n' "${GPUS}" >&2
  exit 1
fi
for _gpu in "${GPU_LIST[@]}"; do
  if ! [[ "${_gpu}" =~ ^[0-9]+$ ]]; then
    printf 'GPUS must be a comma-separated list of non-negative integers, got %q\n' "${GPUS}" >&2
    exit 1
  fi
  if [[ -n "${_gpu_seen[${_gpu}]+present}" ]]; then
    printf 'duplicate GPU %q in GPUS=%q\n' "${_gpu}" "${GPUS}" >&2
    exit 1
  fi
  _gpu_seen["${_gpu}"]=1
done
unset _gpu _gpu_seen
if ! [[ "${BASE_PORT}" =~ ^[0-9]+$ ]] || (( BASE_PORT < 1 || BASE_PORT > 65535 )); then
  printf 'BASE_PORT must be an integer in [1,65535], got %q\n' "${BASE_PORT}" >&2
  exit 1
fi
PORT_END=$((BASE_PORT + ${#GPU_LIST[@]} - 1))
if (( PORT_END > 65535 )); then
  printf 'GPU port range exceeds 65535: BASE_PORT=%s GPUs=%s\n' "${BASE_PORT}" "${#GPU_LIST[@]}" >&2
  exit 1
fi
RUN_PGID=""

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

iter_name() {
  printf 'iter_%09d' "$1"
}

export_dir() {
  printf '%s/export_%s\n' "${TRAIN_DIR}" "$(iter_name "$1")"
}

export_has_subtask_decoder() {
  python - "$1/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
enabled = config.get("model", {}).get("config", {}).get("subtask_decoder", {}).get("enabled", False)
raise SystemExit(0 if enabled is True else 1)
PY
}

job_outdir() {
  local base_outdir="$1"
  local repeat="$2"
  local outdir

  if (( repeat == 0 )); then
    outdir="${base_outdir}"
  else
    outdir="${base_outdir}_${repeat}"
  fi
  if [[ "${CAMERA}" == "1" || "${CAMERA}" == "true" || "${CAMERA}" == "TRUE" ]]; then
    case "${CAMERA_CFG_MODE}" in
      separate)
        outdir+="_camera_textg${GUIDANCE//./p}_camg${CAMERA_GUIDANCE//./p}"
        ;;
      camera|both)
        outdir+="_camera_cfg${CAMERA_CFG_MODE}_textg${GUIDANCE//./p}"
        ;;
    esac
  fi
  printf '%s\n' "${outdir}"
}

final_output_complete() {
  local result_file="$1"
  local output_dir="${result_file%/episode_results.jsonl}"
  local manifest="${output_dir}/merge_manifest.json"
  # A non-empty JSONL can be left behind when a job is terminated during
  # merge.  Only a manifest produced after the merge is a durable completion
  # marker; otherwise the next invocation must resume this repeat.
  [[ -s "${result_file}" && -s "${manifest}" ]] || return 1
  python - "${result_file}" "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

result_path, manifest_path = map(Path, sys.argv[1:])
try:
    rows = sum(1 for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merged_rows = int(manifest.get("merged_rows", -1))
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if rows > 0 and rows == merged_rows else 1)
PY
}

repeat_sync_key() {
  local label="$1"
  local base_outdir="$2"
  # Keep the coordination path short even when labels contain long checkpoint
  # names and multi-node rendezvous metadata.  The full values are hashed only
  # for the logs/synchronization namespace; result output paths are unchanged.
  printf '%s\n' "${label}|${base_outdir}" \
    | sha256sum | cut -c1-16
}

repeat_sync_init() {
  local label="$1"
  local base_outdir="$2"
  local key
  local sync_dir
  local session
  local nonce
  local ready_file
  local deadline
  local rank
  [[ "${NNODES}" -gt 1 ]] || return 0
  key="$(repeat_sync_key "${label}" "${base_outdir}")"
  sync_dir="${ROOT}/logs/eval_repeat_sync_${key}"
  mkdir -p "${sync_dir}"
  if (( NODE_RANK == 0 )); then
    # Start a fresh handshake.  The nonce is deliberately short; it is only
    # coordination metadata and never appears in an output directory name.
    nonce="${EVAL_SESSION_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}"
    printf '%s\n' "${nonce}" > "${sync_dir}/round.tmp.$$"
    mv -f -- "${sync_dir}/round.tmp.$$" "${sync_dir}/round"
  else
    deadline=$((SECONDS + WAIT_TIMEOUT))
    while [[ ! -s "${sync_dir}/round" ]] && (( SECONDS < deadline )); do
      sleep 2
    done
    if [[ ! -s "${sync_dir}/round" ]]; then
      log "timeout waiting for node0 repeat round: ${sync_dir}/round"
      return 1
    fi
    nonce="$(head -n 1 "${sync_dir}/round")"
  fi
  [[ -n "${nonce}" ]] || return 1

  ready_file="${sync_dir}/node_${NODE_RANK}.ready"
  printf '%s\n' "${nonce}" > "${ready_file}.tmp.$$"
  mv -f -- "${ready_file}.tmp.$$" "${ready_file}"
  if (( NODE_RANK == 0 )); then
    deadline=$((SECONDS + WAIT_TIMEOUT))
    for (( rank = 1; rank < NNODES; rank++ )); do
      ready_file="${sync_dir}/node_${rank}.ready"
      while [[ "$(head -n 1 "${ready_file}" 2>/dev/null || true)" != "${nonce}" ]] && (( SECONDS < deadline )); do
        sleep 2
      done
      if [[ "$(head -n 1 "${ready_file}" 2>/dev/null || true)" != "${nonce}" ]]; then
        log "timeout waiting for node${rank} repeat handshake"
        return 1
      fi
    done
    rm -f -- "${sync_dir}"/repeat_*.decision "${sync_dir}/session"
    session="${nonce}"
    printf '%s\n' "${session}" > "${sync_dir}/session.tmp.$$"
    mv -f -- "${sync_dir}/session.tmp.$$" "${sync_dir}/session"
  else
    deadline=$((SECONDS + WAIT_TIMEOUT))
    while [[ "$(head -n 1 "${sync_dir}/session" 2>/dev/null || true)" != "${nonce}" ]] && (( SECONDS < deadline )); do
      sleep 2
    done
    if [[ "$(head -n 1 "${sync_dir}/session" 2>/dev/null || true)" != "${nonce}" ]]; then
      log "timeout waiting for node0 repeat session"
      return 1
    fi
  fi
  REPEAT_SYNC_DIR="${sync_dir}"
  REPEAT_SYNC_SESSION="${nonce}"
  [[ -n "${REPEAT_SYNC_SESSION}" ]] || return 1
}

sync_repeat_decision() {
  local label="$1"
  local base_outdir="$2"
  local repeat="$3"
  local decision_file
  local tmp_file
  local deadline
  local decision
  local final_dir
  local session

  [[ "${NNODES}" -gt 1 ]] || { REPEAT_SYNC_ACTION=run; return 0; }
  [[ -n "${REPEAT_SYNC_DIR:-}" ]] || repeat_sync_init "${label}" "${base_outdir}"
  decision_file="${REPEAT_SYNC_DIR}/repeat_${repeat}.decision"
  if (( NODE_RANK == 0 )); then
    final_dir="$(job_outdir "${base_outdir}" "${repeat}")"
    decision=run
    if final_output_complete "${ROBOLAB_OUTPUT}/${final_dir}/episode_results.jsonl"; then
      decision=skip
    fi
    tmp_file="${decision_file}.tmp.$$"
    printf '%s\n%s\n' "${REPEAT_SYNC_SESSION}" "${decision}" > "${tmp_file}"
    mv -f -- "${tmp_file}" "${decision_file}"
  else
    deadline=$((SECONDS + WAIT_TIMEOUT))
    while [[ ! -s "${decision_file}" ]] && (( SECONDS < deadline )); do
      sleep 2
    done
    if [[ ! -s "${decision_file}" ]]; then
      log "timeout waiting for node0 repeat ${repeat} decision: ${decision_file}"
      return 1
    fi
    session="$(sed -n '1p' "${decision_file}")"
    if [[ "${session}" != "${REPEAT_SYNC_SESSION}" ]]; then
      log "stale repeat ${repeat} decision ignored: expected session ${REPEAT_SYNC_SESSION}, got ${session}"
      return 1
    fi
    decision="$(sed -n '2p' "${decision_file}")"
  fi
  case "${decision}" in
    run|skip) REPEAT_SYNC_ACTION="${decision}" ;;
    *) log "invalid node0 repeat ${repeat} decision: ${decision}"; return 1 ;;
  esac
}

JOB_LABELS=()
JOB_CKPTS=()
JOB_STRIDES=()
JOB_OUTDIRS=()
JOB_REPEATS=()

add_job() {
  JOB_LABELS+=("$1")
  JOB_CKPTS+=("$2")
  JOB_STRIDES+=("$3")
  JOB_OUTDIRS+=("$4")
  JOB_REPEATS+=("$5")
}

build_jobs() {
  local checkpoint
  local label
  case "${EVAL_PLAN}" in
    single)
      checkpoint="${EVAL_CHECKPOINT:-$(export_dir "${EVAL_ITER}")}"
      label="${EVAL_LABEL:-iter${EVAL_ITER}}"
      add_job \
        "${label}" \
        "${checkpoint}" \
        "${DROID_VIDEO_SAMPLE_STRIDE}" \
        "${EVAL_BASE_OUTDIR}" \
        "${EVAL_REPEATS}"
      ;;
    requested_four)
      add_job \
        "wristrot180_iter13000" \
        "${ROOT}/outputs/cosmos3-droid_joint_cam2base_768_wristrot180/cosmos3_action/action_sft_droid/action_policy_droid_success_both_ext_cam2base_joint_16_global768_wristrot180/export_iter_000013000" \
        "1" \
        "cosmos3_droid_all120_officialtime_8gpu_544x640_wristrot180_iter13000" \
        "4"
      add_job \
        "viddown4_iter12000" \
        "${ROOT}/outputs/cosmos3-droid_joint_cam2base_768_viddown4/cosmos3_action/action_sft_droid/action_policy_droid_success_both_ext_cam2base_joint_16_global768_wristrot180_viddown4/export_iter_000012000" \
        "4" \
        "cosmos3_droid_all120_officialtime_8gpu_544x640_wristrot180_viddown4_iter12000" \
        "4"
      add_job \
        "viddown4_iter13000" \
        "${ROOT}/outputs/cosmos3-droid_joint_cam2base_768_viddown4/cosmos3_action/action_sft_droid/action_policy_droid_success_both_ext_cam2base_joint_16_global768_wristrot180_viddown4/export_iter_000013000" \
        "4" \
        "cosmos3_droid_all120_officialtime_8gpu_544x640_wristrot180_viddown4_iter13000" \
        "4"
      add_job \
        "viddown2_iter15500" \
        "${ROOT}/outputs/cosmos3-droid_joint_cam2base_768_viddown2/cosmos3_action/action_sft_droid/action_policy_droid_success_both_ext_cam2base_joint_16_global768_wristrot180_viddown2/export_iter_000015500" \
        "2" \
        "cosmos3_droid_all120_officialtime_8gpu_544x640_wristrot180_viddown2_iter15500" \
        "4"
      ;;
    *)
      log "unsupported EVAL_PLAN=${EVAL_PLAN}; use single or requested_four"
      return 1
      ;;
  esac
}

export_ready() {
  local dir
  dir="$(export_dir "$1")"
  [[ -f "${dir}/checkpoint.json" && -f "${dir}/model.safetensors.index.json" ]] || return 1
  if [[ "${DECODE_SUBTASK}" == "1" || "${DECODE_SUBTASK}" == "true" || "${DECODE_SUBTASK}" == "TRUE" ]]; then
    export_has_subtask_decoder "${dir}"
  fi
}

validate_mid_soft_export() {
  local iter="$1"
  local dir
  local train_mid_count
  dir="$(export_dir "${iter}")"
  case "${MID_SOFT}" in
    1|true|TRUE|yes|on)
      if ! grep -Eq '"mid_soft"[[:space:]]*:[[:space:]]*true' "${dir}/config.json"; then
        log "${dir}: exported model config is not mid_soft=true"
        return 1
      fi
      train_mid_count="$(grep -Ec '^[[:space:]]*mid_soft:[[:space:]]*true[[:space:]]*$' "${TRAIN_DIR}/config.yaml" || true)"
      if (( train_mid_count < 2 )); then
        log "${TRAIN_DIR}/config.yaml: expected model and action-transform mid_soft=true, found ${train_mid_count}"
        return 1
      fi
      log "${dir}: verified model + training transform are mid_soft=true"
      ;;
    0|false|FALSE|no|off)
      ;;
    *)
      log "MID_SOFT must be 0/1 or false/true, got ${MID_SOFT}"
      return 1
      ;;
  esac

  case "${DECODE_SUBTASK}" in
    1|true|TRUE|yes|on)
      if ! export_has_subtask_decoder "${dir}"; then
        log "${dir}: decoder evaluation requested, but exported subtask_decoder.enabled is not true"
        return 1
      fi
      log "${dir}: verified subtask decoder is included"
      ;;
    0|false|FALSE|no|off)
      ;;
    *)
      log "DECODE_SUBTASK must be 0/1 or false/true, got ${DECODE_SUBTASK}"
      return 1
      ;;
  esac
}

dcp_started() {
  local name
  name="$(iter_name "$1")"
  [[ -f "${TRAIN_DIR}/checkpoints/${name}/model/.metadata" ]]
}

wait_for_export() {
  local iter="$1"
  local dir
  local -a watcher_env
  dir="$(export_dir "${iter}")"

  while ! export_ready "${iter}"; do
    log "waiting for exported checkpoint: ${dir}"
    if dcp_started "${iter}"; then
      log "DCP $(iter_name "${iter}") exists; asking watcher for a one-pass export"
      watcher_env=(
        "TRAIN_FILTER=${TRAIN_NAME}"
        "TARGET_ITERS=${iter}"
        "RUN_ONCE=1"
        "EXPORT_SUBTASK_DECODER=${EXPORT_SUBTASK_DECODER}"
      )
      # The default watcher list intentionally excludes the opt-in
      # Reasoner-WAM run.  Register this run for the one-pass export so a
      # Reasoner evaluation does not wait forever after its DCP appears.
      case "${REASONER_WAM}:${REASONER_THEN_WAM}" in
        1:*|true:*|TRUE:*|yes:*|YES:*|on:*|ON:*|*:1|*:true|*:TRUE|*:yes|*:YES|*:on|*:ON)
          watcher_env+=(
            "REASONER_WAM_TRAIN_NAME=${TRAIN_NAME}"
            "REASONER_WAM_TRAIN_DIR=${TRAIN_DIR}"
            "REASONER_WAM_TARGET_ITERS=${iter}"
            "REASONER_WAM_EXPORT_SUBTASK_DECODER=${EXPORT_SUBTASK_DECODER}"
          )
          ;;
      esac
      if ! env "${watcher_env[@]}" bash "${WATCHER}"; then
        log "one-pass export attempt failed; the background watcher may still complete it"
      fi
    fi
    sleep "${EXPORT_POLL_SECONDS}"
  done

  log "export ready: ${dir}"
  validate_mid_soft_export "${iter}"
}

ports_in_use() {
  "${SS_BIN}" -H -ltn "( sport >= :${BASE_PORT} and sport <= :${PORT_END} )" | grep -q .
}

show_ports() {
  "${SS_BIN}" -ltnp "( sport >= :${BASE_PORT} and sport <= :${PORT_END} )" || true
}

require_ports_free() {
  if ports_in_use; then
    log "ports ${BASE_PORT}-${PORT_END} are already in use; refusing to kill unrelated processes"
    show_ports
    return 1
  fi
}

cleanup_run_group() {
  local pgid="${RUN_PGID}"
  [[ -n "${pgid}" ]] || return 0

  if kill -0 -- "-${pgid}" 2>/dev/null; then
    log "stopping remaining evaluation processes in process group ${pgid}"
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-${pgid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-${pgid}" 2>/dev/null; then
      log "forcing remaining evaluation processes in process group ${pgid} to exit"
      kill -KILL -- "-${pgid}" 2>/dev/null || true
      for _ in $(seq 1 10); do
        kill -0 -- "-${pgid}" 2>/dev/null || break
        sleep 1
      done
    fi
  fi
}

wait_for_ports_free() {
  for _ in $(seq 1 60); do
    ports_in_use || return 0
    sleep 1
  done

  log "ports ${BASE_PORT}-${PORT_END} did not clear after evaluation cleanup"
  show_ports
  return 1
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT INT TERM
  cleanup_run_group
  if [[ -n "${LAUNCHER_SNAPSHOT}" ]]; then
    rm -f -- "${LAUNCHER_SNAPSHOT}"
  fi
  exit "${status}"
}
trap cleanup_on_exit EXIT INT TERM

freeze_launcher() {
  local snapshot
  # Keep the snapshot directly beside the launcher so its BASH_SOURCE-derived
  # SCRIPT_DIR/USER_ROOT paths remain identical to the canonical file.
  snapshot="$(mktemp "${CODE_HUMAN}/.cosmos3_launcher.XXXXXX.sh")"
  cp -- "${LAUNCHER}" "${snapshot}"
  LAUNCHER_SNAPSHOT="${snapshot}"
  LAUNCHER="${snapshot}"
  log "using immutable launcher snapshot: ${LAUNCHER_SNAPSHOT}"
}

run_eval() {
  local label="$1"
  local ckpt="$2"
  local stride="$3"
  local base_outdir="$4"
  local repeat="$5"
  local outdir
  local pgid_file
  local wait_pid
  local status

  outdir="$(job_outdir "${base_outdir}" "${repeat}")"
  if (( NNODES > 1 )); then
    pgid_file="${ROOT}/.tmp/cosmos3_eval_${label}_${repeat}_node${NODE_RANK}_pgid"
  else
    pgid_file="${ROOT}/.tmp/cosmos3_eval_${label}_${repeat}_pgid"
  fi
  mkdir -p "${ROOT}/.tmp"
  rm -f "${pgid_file}"

  if final_output_complete "${ROBOLAB_OUTPUT}/${outdir}/episode_results.jsonl"; then
    log "final output already exists; skipping repeat ${repeat}: ${ROBOLAB_OUTPUT}/${outdir}"
    return 0
  fi

  if ! require_ports_free; then
    log "repeat ${repeat} cannot start: evaluation ports are not free"
    return 1
  fi
  log "starting ${label} repeat ${repeat} stride=${stride} on GPUs ${GPUS}, ports ${BASE_PORT}-${PORT_END} num_envs=${NUM_ENVS} sim_workers_per_gpu=${SIM_WORKERS_PER_GPU} video_mode=${VIDEO_MODE}"
  log "planner flags: REASONER_WAM=${REASONER_WAM} REASONER_THEN_WAM=${REASONER_THEN_WAM}"

  env \
    SERVER_REPO="${SERVER_REPO:-${ROOT}/cosmos-framework}" \
    SERVER_PY="${SERVER_PY}" \
    ROBOLAB="${ROBOLAB}" \
    ROBOLAB_PYTHON="${ROBOLAB_PYTHON:-}" \
    COSMOS3_NANO_ROOT="${COSMOS3_NANO_ROOT}" \
    WAN_VAE_PATH="${WAN_VAE_PATH}" \
    ROBOLAB_ASSETS="${ROBOLAB_ASSETS}" \
    GPUS="${GPUS}" \
    NNODES="${NNODES}" \
    NODE_RANK="${NODE_RANK}" \
    MASTER_ADDR="${MASTER_ADDR}" \
    MASTER_PORT="${MASTER_PORT}" \
    BASE_PORT="${BASE_PORT}" \
    NUM_ENVS="${NUM_ENVS}" \
    SIM_WORKERS_PER_GPU="${SIM_WORKERS_PER_GPU}" \
    VIDEO_MODE="${VIDEO_MODE}" \
    MAX_EPISODE_SECONDS="${MAX_EPISODE_SECONDS}" \
    WAIT_TIMEOUT="${WAIT_TIMEOUT}" \
    TASK_SET=all \
    TASKS_FILE="${TASKS_FILE}" \
    TASKS_SOURCE="${TASKS_SOURCE}" \
    CKPT="${ckpt}" \
    BASE_OUTDIR="${outdir}" \
    FINAL_OUTDIR="${outdir}" \
    WRIST_ROTATE_180="${WRIST_ROTATE_180}" \
    DROID_VIDEO_SAMPLE_STRIDE="${stride}" \
    ACTION_FPS=15 \
    GUIDANCE="${GUIDANCE}" \
    CAMERA="${CAMERA}" \
    CAMERA_GUIDANCE="${CAMERA_GUIDANCE}" \
    CAMERA_CFG_MODE="${CAMERA_CFG_MODE}" \
    VIT_SOFTPLAN="${VIT_SOFTPLAN}" \
    MID_SOFT="${MID_SOFT}" \
    DECODE_SUBTASK="${DECODE_SUBTASK}" \
    SUBTASK_MAX_NEW_TOKENS="${SUBTASK_MAX_NEW_TOKENS}" \
    VIEWPOINT_ROTATE_180="${VIEWPOINT_ROTATE_180}" \
    OVERLAY_SUBTASK_ON_VIDEO="${OVERLAY_SUBTASK_ON_VIDEO}" \
    REASONER_WAM="${REASONER_WAM}" \
    REASONER_THEN_WAM="${REASONER_THEN_WAM}" \
    REASONER_WAM_MODE="${REASONER_WAM_MODE}" \
    REASONER_CHECKPOINT_PATH="${REASONER_CHECKPOINT_PATH}" \
    REASONER_LORA_PATH="${REASONER_LORA_PATH}" \
    REASONER_PROJECTOR_PATH="${REASONER_PROJECTOR_PATH}" \
    WAM_CHECKPOINT_PATH="${WAM_CHECKPOINT_PATH}" \
    REASONER_MAX_NEW_TOKENS="${REASONER_MAX_NEW_TOKENS}" \
    REASONER_DO_SAMPLE="${REASONER_DO_SAMPLE}" \
    REASONER_INCLUDE_PREVIEW_ACTION="${REASONER_INCLUDE_PREVIEW_ACTION}" \
    REASONER_OPEN_GRIPPER_MOVE_AUGMENT="${REASONER_OPEN_GRIPPER_MOVE_AUGMENT}" \
    REASONER_REQUIRE_THINK="${REASONER_REQUIRE_THINK}" \
    REASONER_SEPARATE_VIEWS="${REASONER_SEPARATE_VIEWS}" \
    REASONER_THREE_GROUP_HISTORY="${REASONER_THREE_GROUP_HISTORY}" \
    REASONER_IMAGE_LONG_EDGE="${REASONER_IMAGE_LONG_EDGE}" \
    FORMAT_PROMPT_AS_JSON="${FORMAT_PROMPT_AS_JSON}" \
    INCLUDE_EXECUTION_SUCCESS="${INCLUDE_EXECUTION_SUCCESS}" \
    ROBOLAB_OUTPUT="${ROBOLAB_OUTPUT}" \
    ROBOLAB_VIDEO_QUALITY="${ROBOLAB_VIDEO_QUALITY}" \
    ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS="${ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS}" \
    ROBOLAB_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_RELATIVE_GRIPPER_COMMANDS}" \
    ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS}" \
    ROBOLAB_GRIPPER_RELATIVE_THRESHOLD="${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD}" \
    ROBOLAB_REASONER_HISTORY_IMAGES="${ROBOLAB_REASONER_HISTORY_IMAGES}" \
    ROBO_INTERACTIVE_COMPACT_ARTIFACTS="${ROBO_INTERACTIVE_COMPACT_ARTIFACTS}" \
    INSTRUCTION_TYPE="${INSTRUCTION_TYPE}" \
    REASONER_SEPARATE_VIEWS="${REASONER_SEPARATE_VIEWS}" \
    REASONER_THREE_GROUP_HISTORY="${REASONER_THREE_GROUP_HISTORY}" \
    REASONER_IMAGE_LONG_EDGE="${REASONER_IMAGE_LONG_EDGE}" \
    ROBO_INTERACTIVE_REASONER_FIRST="${ROBO_INTERACTIVE_REASONER_FIRST}" \
    ROBO_INTERACTIVE_SUPERVISOR_DIR="${ROBO_INTERACTIVE_SUPERVISOR_DIR}" \
    ROBOLAB_DRIVERFIX="${ROBOLAB_DRIVERFIX}" \
    ROBOLAB_DRIVERFIX_ASSETS="${ROBOLAB_DRIVERFIX_ASSETS}" \
    VIT_PROCESSOR_MODEL_NAME="${VIT_PROCESSOR_MODEL_NAME}" \
    VIT_PROCESSOR_CONFIG_VARIANT="${VIT_PROCESSOR_CONFIG_VARIANT}" \
    VIT_PROCESSOR_CACHE_DIR="${VIT_PROCESSOR_CACHE_DIR}" \
    VIT_FIRST_FRAME_MAX_PIXELS="${VIT_FIRST_FRAME_MAX_PIXELS}" \
    VIT_FIRST_FRAME_MIN_PIXELS="${VIT_FIRST_FRAME_MIN_PIXELS}" \
    MAX_EPISODE_SECONDS_EXCEPT_TASKS="${MAX_EPISODE_SECONDS_EXCEPT_TASKS:-}" \
    EVAL_SESSION_ID="${REPEAT_SYNC_SESSION:-}" \
    DRY_RUN="${DRY_RUN}" \
    SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS}" \
    EVAL_PGID_FILE="${pgid_file}" \
    EVAL_LAUNCHER="${LAUNCHER}" \
    setsid --wait bash -c '
      ps -o pgid= -p "$$" | tr -d "[:space:]" > "${EVAL_PGID_FILE}"
      printf "\n" >> "${EVAL_PGID_FILE}"
      exec bash "${EVAL_LAUNCHER}"
    ' &
  wait_pid=$!

  for _ in $(seq 1 100); do
    [[ -s "${pgid_file}" ]] && break
    kill -0 "${wait_pid}" 2>/dev/null || break
    sleep 0.1
  done
  if [[ ! -s "${pgid_file}" ]]; then
    log "evaluation process exited before publishing its process group"
    wait "${wait_pid}" || true
    return 1
  fi
  RUN_PGID="$(tr -d '[:space:]' < "${pgid_file}")"

  status=0
  wait "${wait_pid}" || status=$?
  cleanup_run_group
  if ! wait_for_ports_free; then
    log "repeat ${repeat} cleanup failed: evaluation ports did not become free"
    RUN_PGID=""
    rm -f "${pgid_file}"
    return 1
  fi
  RUN_PGID=""
  rm -f "${pgid_file}"

  if (( status != 0 )); then
    log "${label} repeat ${repeat} evaluation failed with status ${status}; rerun this script to resume"
    return "${status}"
  fi
  log "repeat ${repeat} evaluation complete; ports ${BASE_PORT}-${PORT_END} are free"
}

main() {
  freeze_launcher
  bash -n "${LAUNCHER}"
  test -x "${SERVER_PY}"
  test -d "${REASONER_ARTIFACTS}"
  test -f "${REASONER_LORA_PATH}"
  test -f "${REASONER_LORA_METADATA}"
  if [[ -n "${REASONER_PROJECTOR_PATH}" ]]; then
    test -f "${REASONER_PROJECTOR_PATH}"
    test -f "${REASONER_PROJECTOR_PATH%.safetensors}.json"
    log "Reasoner adapter artifacts: LoRA + projector"
  else
    log "Reasoner adapter artifacts: LoRA only; using the base visual projector"
  fi
  test -d "${WAM_CHECKPOINT_PATH}"

  build_jobs
  for job_idx in "${!JOB_LABELS[@]}"; do
    label="${JOB_LABELS[$job_idx]}"
    ckpt="${JOB_CKPTS[$job_idx]}"
    stride="${JOB_STRIDES[$job_idx]}"
    base_outdir="${JOB_OUTDIRS[$job_idx]}"
    repeats="${JOB_REPEATS[$job_idx]}"

    repeat_sync_init "${label}" "${base_outdir}"
    if (( NNODES > 1 )); then
      log "repeat coordinator: node${NODE_RANK}/${NNODES} session=${REPEAT_SYNC_SESSION} dir=${REPEAT_SYNC_DIR}"
    fi

    if [[ "${EVAL_PLAN}" == "single" ]]; then
      if ! [[ "${repeats}" =~ ^[1-9][0-9]*$ ]]; then
        log "EVAL_REPEATS must be a positive integer, got: ${repeats}"
        return 1
      fi
      if [[ -n "${EVAL_CHECKPOINT}" ]]; then
        if [[ ! -d "${ckpt}" ]]; then
          log "checkpoint directory does not exist for ${label}: ${ckpt}"
          return 1
        fi
        log "using existing exported checkpoint: ${ckpt}"
      else
        test -d "${TRAIN_DIR}"
        wait_for_export "${EVAL_ITER}"
      fi
    else
      if [[ ! -d "${ckpt}" ]]; then
        log "checkpoint directory does not exist for ${label}: ${ckpt}"
        return 1
      fi
    fi

    for (( repeat = 0; repeat < repeats; repeat++ )); do
      if ! sync_repeat_decision "${label}" "${base_outdir}" "${repeat}"; then
        return 1
      fi
      if [[ "${REPEAT_SYNC_ACTION}" == "skip" ]]; then
        log "node${NODE_RANK}: repeat ${repeat} globally complete; skipping on all nodes"
        continue
      fi
      local attempt=0
      while true; do
        if run_eval "${label}" "${ckpt}" "${stride}" "${base_outdir}" "${repeat}"; then
          break
        fi
        attempt=$((attempt + 1))
        if (( attempt >= EVAL_RETRY_COUNT )); then
          log "aborting job after repeat ${repeat} failed ${attempt} time(s); completed repeats remain resumable"
          return 1
        fi
        log "repeat ${repeat} failed (attempt ${attempt}/${EVAL_RETRY_COUNT}); retrying after ${EVAL_RETRY_DELAY_SECONDS}s"
        sleep "${EVAL_RETRY_DELAY_SECONDS}"
      done
    done
  done

  log "all queued evaluations are complete"
}

main "$@"
