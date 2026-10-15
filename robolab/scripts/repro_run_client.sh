#!/usr/bin/env bash
set -euo pipefail

# Local RoboLab client for the Cosmos3 websocket policy server.  The N-GPU
# launcher fills TASK, OUTDIR, GPU, HOST, PORT, and NUM_ENVS.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
USER_ROOT="${USER_ROOT:-${REPRO_ROOT}}"
ROBOLAB="${ROBOLAB:-${REPRO_ROOT}/robolab}"
ROBOLAB_PY="${ROBOLAB_PY:-${ROBOLAB_PYTHON:-${ROBOLAB}/.venv/bin/python}}"

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
ALL="${ALL:-0}"
TASK="${TASK:-}"
OUTDIR="${OUTDIR:-}"
EXTRA="${EXTRA:---headless --video-mode all}"
ISAACSIM_NO_DLSS="${ISAACSIM_NO_DLSS:-1}"
# AppLauncher receives --device cuda:N below, but Isaac Sim 5 can still honor
# the persistent/default renderer GPU (usually physical GPU 0) during startup.
# Append explicit Kit settings last so rendering, PhysX, and CUDA all use the
# worker's assigned physical GPU.  Disabling multi-GPU also prevents every
# simulator process from reserving memory on every GPU.
# A800/A100 headless Vulkan runs need the optional NGX/DLSS post-processing
# paths disabled.  ISAACSIM_NO_DLSS alone is not sufficient: Kit can still
# load the RTX post-processing plugin and crash while creating Hydra.
ROBOLAB_KIT_ARGS="${ROBOLAB_KIT_ARGS:+${ROBOLAB_KIT_ARGS} }--/renderer/activeGpu=${GPU} --/physics/cudaDevice=${GPU} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/ngx/enabled=false --/rtx-transient/dldenoiser/enabled=false --/rtx-transient/dlssg/enabled=false"
ROBOLAB_VIDEO_QUALITY="${ROBOLAB_VIDEO_QUALITY:-4}"
# Gripper handling is opt-in.  RoboLab's simulator action is binary, so an
# experiment can first use the current observed state plus a dead-band, then
# apply the existing one-frame pulse filter without changing legacy launches.
ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS="${ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS:-0}"
ROBOLAB_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_RELATIVE_GRIPPER_COMMANDS:-0}"
ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS="${ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS:-0}"
ROBOLAB_GRIPPER_RELATIVE_THRESHOLD="${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD:-0.10}"
ROBOLAB_REASONER_HISTORY_IMAGES="${ROBOLAB_REASONER_HISTORY_IMAGES:-0}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-default}"
REASONER_SEPARATE_VIEWS="${REASONER_SEPARATE_VIEWS:-0}"
REASONER_THREE_GROUP_HISTORY="${REASONER_THREE_GROUP_HISTORY:-0}"
REASONER_IMAGE_LONG_EDGE="${REASONER_IMAGE_LONG_EDGE:-448}"
REASONER_INCLUDE_PREVIEW_ACTION="${REASONER_INCLUDE_PREVIEW_ACTION:-0}"
ROBO_INTERACTIVE_REASONER_FIRST="${ROBO_INTERACTIVE_REASONER_FIRST:-0}"
ROBO_INTERACTIVE_SUPERVISOR_DIR="${ROBO_INTERACTIVE_SUPERVISOR_DIR:-}"
# Most evaluation containers expose the kernel driver but not a matching set
# of NVIDIA EGL/GLX/Vulkan userspace libraries.  Prefer RoboLab's versioned,
# read-only graphics store.  Set ROBOLAB_DRIVERFIX=0 only on an image whose
# system graphics stack is already known to match its kernel driver.
ROBOLAB_DRIVERFIX="${ROBOLAB_DRIVERFIX:-0}"
ROBOLAB_DRIVERFIX_ASSETS="${ROBOLAB_DRIVERFIX_ASSETS:-}"
ROBOLAB_DRIVERFIX_CHECK_ONLY="${ROBOLAB_DRIVERFIX_CHECK_ONLY:-0}"
ROBOLAB_DRIVERFIX_VULKAN_TIMEOUT_SECONDS="${ROBOLAB_DRIVERFIX_VULKAN_TIMEOUT_SECONDS:-30}"
ROBOLAB_DRIVERFIX_SCRIPT="${ROBOLAB_DRIVERFIX_SCRIPT:-}"
DRIVERFIX_DIR="${DRIVERFIX_DIR:-${TMPDIR:-/tmp}/robolab-driverfix}"

# Keep the simulator client offline as well as the policy server.  The client
# uses the external RoboLab assets directory supplied by ROBOLAB_ASSETS, while
# libraries imported during Kit startup may still consult Hugging Face unless
# the offline flags are inherited.
HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
ROBOLAB_ASSETS="${ROBOLAB_ASSETS:-}"
export ROBOLAB_ASSETS

[[ -x "${ROBOLAB_PY}" ]] || { echo "[cosmos3-client] missing RoboLab Python: ${ROBOLAB_PY}" >&2; exit 1; }
[[ -f "${ROBOLAB}/policies/cosmos3/run.py" ]] || {
  echo "[cosmos3-client] missing ${ROBOLAB}/policies/cosmos3/run.py" >&2
  exit 1
}
[[ -n "${ROBOLAB_ASSETS}" && -d "${ROBOLAB_ASSETS}" ]] || {
  echo "[cosmos3-client] set ROBOLAB_ASSETS to the external RoboLab assets directory" >&2
  exit 1
}
for required_asset_dir in backgrounds fixtures objects robots scenes; do
  [[ -d "${ROBOLAB_ASSETS}/${required_asset_dir}" ]] || {
    echo "[cosmos3-client] missing RoboLab asset directory: ${ROBOLAB_ASSETS}/${required_asset_dir}" >&2
    exit 1
  }
done
if [[ "${ALL}" != "1" && -z "${TASK}" ]]; then
  echo "[cosmos3-client] set TASK='TaskName ...' or ALL=1" >&2
  exit 2
fi
if ! [[ "${ROBOLAB_VIDEO_QUALITY}" =~ ^([0-9]|10)$ ]]; then
  echo "[cosmos3-client] ROBOLAB_VIDEO_QUALITY must be an integer in [0,10]" >&2
  exit 2
fi

case "${ROBOLAB_DRIVERFIX}" in
  1|true|TRUE|yes|YES|on|ON)
    [[ -n "${ROBOLAB_DRIVERFIX_SCRIPT}" && -f "${ROBOLAB_DRIVERFIX_SCRIPT}" ]] || {
      echo "[cosmos3-client] set ROBOLAB_DRIVERFIX_SCRIPT to a host driverfix script, or use ROBOLAB_DRIVERFIX=0" >&2
      exit 1
    }
    export DRIVERFIX_DIR
    export DRIVERFIX_REQUIRE_GRAPHICS=1
    export DRIVERFIX_GRAPHICS_STORE="${ROBOLAB_DRIVERFIX_ASSETS}/graphics-$({
      awk '/NVRM version:/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+/) {print $i; exit}}' \
        /proc/driver/nvidia/version
    })"
    # The canonical script is written for ordinary bash. Temporarily disable
    # nounset while sourcing, then restore this launcher's strict mode.
    set +u
    source "${ROBOLAB_DRIVERFIX_SCRIPT}"
    set -u
    ;;
  0|false|FALSE|no|NO|off|OFF)
    echo "[cosmos3-client] driverfix disabled by ROBOLAB_DRIVERFIX=${ROBOLAB_DRIVERFIX}"
    ;;
  *)
    echo "[cosmos3-client] ROBOLAB_DRIVERFIX must be 0/1 or false/true, got '${ROBOLAB_DRIVERFIX}'" >&2
    exit 1
    ;;
esac

if [[ "${ROBOLAB_DRIVERFIX_CHECK_ONLY}" == "1" ]]; then
  if command -v vulkaninfo >/dev/null 2>&1; then
    if ! vulkan_summary="$(timeout "${ROBOLAB_DRIVERFIX_VULKAN_TIMEOUT_SECONDS}" vulkaninfo --summary 2>&1)"; then
      echo "[cosmos3-client] Vulkan preflight failed:" >&2
      printf '%s\n' "${vulkan_summary}" >&2
      exit 1
    fi
    echo "[cosmos3-client] Vulkan instance preflight passed"
  else
    echo "[cosmos3-client] WARNING: vulkaninfo is unavailable; skipping Vulkan instance preflight" >&2
  fi
  echo "[cosmos3-client] driverfix preflight passed"
  exit 0
fi

# Isaac Sim must see the same physical device numbering in CUDA and Vulkan.
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="${ROBOLAB}/robolab_sitecustomize:${ROBOLAB}${PYTHONPATH:+:${PYTHONPATH}}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export ISAACSIM_NO_DLSS ROBOLAB_VIDEO_QUALITY
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,${HOST}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,${HOST}"

args=(
  "${ROBOLAB_PY}"
  "${ROBOLAB}/policies/cosmos3/run.py"
  --remote-host "${HOST}"
  --remote-port "${PORT}"
  --num-envs "${NUM_ENVS}"
  --num-runs "${NUM_RUNS}"
  --device "cuda:${GPU}"
)
if [[ -n "${ROBO_INTERACTIVE_SUPERVISOR_DIR}" ]]; then
  args[1]="${ROBOLAB}/policies/cosmos3/run_interactive.py"
  args+=(--supervisor-mode file --supervisor-dir "${ROBO_INTERACTIVE_SUPERVISOR_DIR}" --supervisor-timeout-seconds 0)
  if [[ "${ROBO_INTERACTIVE_REASONER_FIRST}" == "1" || "${ROBO_INTERACTIVE_REASONER_FIRST}" == "true" || "${ROBO_INTERACTIVE_REASONER_FIRST}" == "TRUE" || "${ROBO_INTERACTIVE_REASONER_FIRST}" == "yes" || "${ROBO_INTERACTIVE_REASONER_FIRST}" == "on" ]]; then
    args+=(--reasoner-first)
  fi
fi
if [[ "${ALL}" != "1" ]]; then
  read -r -a tasks <<<"${TASK}"
  args+=(--task "${tasks[@]}")
fi
[[ -z "${OUTDIR}" ]] || args+=(--output-folder-name "${OUTDIR}")
if [[ "${INSTRUCTION_TYPE}" != "default" ]]; then
  args+=(--instruction-type "${INSTRUCTION_TYPE}")
fi
if [[ -n "${EXTRA}" ]]; then
  read -r -a extra_args <<<"${EXTRA}"
  args+=("${extra_args[@]}")
fi
case "${ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--filter-isolated-gripper-commands)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "[cosmos3-client] ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS must be 0/1 or false/true, got '${ROBOLAB_FILTER_ISOLATED_GRIPPER_COMMANDS}'" >&2
    exit 2
    ;;
esac
case "${ROBOLAB_RELATIVE_GRIPPER_COMMANDS}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--relative-gripper-commands --gripper-relative-threshold "${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD}")
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "[cosmos3-client] ROBOLAB_RELATIVE_GRIPPER_COMMANDS must be 0/1 or false/true, got '${ROBOLAB_RELATIVE_GRIPPER_COMMANDS}'" >&2
    exit 2
    ;;
esac
case "${ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--previous-frame-relative-gripper-commands --gripper-relative-threshold "${ROBOLAB_GRIPPER_RELATIVE_THRESHOLD}")
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "[cosmos3-client] ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS must be 0/1 or false/true, got '${ROBOLAB_PREVIOUS_FRAME_RELATIVE_GRIPPER_COMMANDS}'" >&2
    exit 2
    ;;
esac
case "${ROBOLAB_REASONER_HISTORY_IMAGES}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--reasoner-history-images)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "[cosmos3-client] ROBOLAB_REASONER_HISTORY_IMAGES must be 0/1 or false/true, got '${ROBOLAB_REASONER_HISTORY_IMAGES}'" >&2
    exit 2
    ;;
esac
case "${REASONER_SEPARATE_VIEWS}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--reasoner-separate-views --reasoner-image-long-edge "${REASONER_IMAGE_LONG_EDGE}")
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "[cosmos3-client] REASONER_SEPARATE_VIEWS must be 0/1 or false/true, got '${REASONER_SEPARATE_VIEWS}'" >&2
    exit 2
    ;;
esac
case "${REASONER_THREE_GROUP_HISTORY}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--reasoner-three-group-history)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "[cosmos3-client] REASONER_THREE_GROUP_HISTORY must be 0/1 or false/true, got '${REASONER_THREE_GROUP_HISTORY}'" >&2
    exit 2
    ;;
esac
case "${REASONER_INCLUDE_PREVIEW_ACTION}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--reasoner-include-preview-action)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "[cosmos3-client] REASONER_INCLUDE_PREVIEW_ACTION must be 0/1 or false/true, got '${REASONER_INCLUDE_PREVIEW_ACTION}'" >&2
    exit 2
    ;;
esac
[[ -z "${ROBOLAB_KIT_ARGS}" ]] || args+=(--kit_args "${ROBOLAB_KIT_ARGS}")

cd "${ROBOLAB}"
printf '[cosmos3-client] GPU=%s server=%s:%s video_quality=%s command:' \
  "${GPU}" "${HOST}" "${PORT}" "${ROBOLAB_VIDEO_QUALITY}"
printf ' %q' "${args[@]}"
printf '\n'
exec "${args[@]}"
