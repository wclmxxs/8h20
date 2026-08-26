#!/usr/bin/env bash
set -euo pipefail

API_PORT=${PORT:-${TCE_SERVICE_PORT:-30010}}
SGLANG_PORT=${SGLANG_PORT:-30020}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-1800}
DATA_ROOT=${DATA_ROOT:-/opt/tiger/minimax-h3/data}
CSDE_MODEL_ROOT=${CSDE_MODEL_ROOT:-/opt/tiger/csde/MiniMax-H3}
HDFS_BIN=${HDFS_BIN:-/opt/tiger/hdfs_client/bin/hdfs}

required_model_entries=(
  modular_model_index.json
  FL2VA
  audio_vae
  processor
  scheduler
  text_encoder
  tokenizer
  transformer
  vae
)

model_is_complete() {
  local root=$1
  local entry
  [[ -d ${root} ]] || return 1
  for entry in "${required_model_entries[@]}"; do
    [[ -e ${root}/${entry} ]] || return 1
  done
}

model_path_identifies_minimax_h3() {
  local identity=${1,,}
  identity=${identity//-/}
  identity=${identity//_/}
  [[ ${identity} == *minimaxh3* ]]
}

prepare_model() {
  local partial_root

  if [[ -n ${MODEL:-} && ${MODEL} != /* ]]; then
    return
  fi
  if [[ -n ${MODEL:-} && ${MODEL} == /* ]]; then
    model_is_complete "${MODEL}" || {
      echo "Explicit MODEL is incomplete: ${MODEL}" >&2
      exit 1
    }
    return
  fi
  if [[ -n ${MODEL_PATH:-} && -d ${MODEL_PATH} ]]; then
    model_is_complete "${MODEL_PATH}" || {
      echo "Local MODEL_PATH is incomplete: ${MODEL_PATH}" >&2
      exit 1
    }
    export MODEL=${MODEL_PATH}
    return
  fi
  if model_is_complete "${CSDE_MODEL_ROOT}"; then
    export MODEL=${CSDE_MODEL_ROOT}
    echo "Reusing localized MiniMax H3 model: ${MODEL}"
    return
  fi
  if [[ ${MODEL_PATH:-} != hdfs://* ]]; then
    echo "No complete local model and MODEL_PATH is not an HDFS URI" >&2
    exit 1
  fi
  if ! model_path_identifies_minimax_h3 "${CSDE_MODEL_ROOT}"; then
    echo "CSDE_MODEL_ROOT must retain the MiniMax-H3 model identity: ${CSDE_MODEL_ROOT}" >&2
    exit 1
  fi
  if [[ ! -x ${HDFS_BIN} ]]; then
    echo "HDFS client is required at ${HDFS_BIN} to fetch ${MODEL_PATH}" >&2
    exit 1
  fi
  case ${CSDE_MODEL_ROOT} in
    /opt/tiger/* | /dev/shm/*) ;;
    *)
      echo "Refusing to manage unsafe CSDE_MODEL_ROOT: ${CSDE_MODEL_ROOT}" >&2
      exit 1
      ;;
  esac

  partial_root=${CSDE_MODEL_ROOT}.partial
  mkdir -p "$(dirname "${CSDE_MODEL_ROOT}")"
  if [[ -e ${partial_root} ]]; then
    rm -rf -- "${partial_root}"
  fi
  echo "Downloading MiniMax H3 model from ${MODEL_PATH} to ${CSDE_MODEL_ROOT}"
  "${HDFS_BIN}" get -s -c 128 --ct 32 -t 8 "${MODEL_PATH}" "${partial_root}"
  model_is_complete "${partial_root}" || {
    echo "Downloaded MiniMax H3 model is incomplete: ${partial_root}" >&2
    exit 1
  }
  if [[ -e ${CSDE_MODEL_ROOT} ]]; then
    rm -rf -- "${CSDE_MODEL_ROOT}"
  fi
  mv "${partial_root}" "${CSDE_MODEL_ROOT}"
  export MODEL=${CSDE_MODEL_ROOT}
}

if [[ ${API_PORT} == "${SGLANG_PORT}" ]]; then
  echo "API_PORT and SGLANG_PORT must differ; both resolved to ${API_PORT}" >&2
  exit 1
fi

mapfile -t gpu_lines < <(nvidia-smi -L)
if (( ${#gpu_lines[@]} != 8 )); then
  echo "Bernard deployment must expose exactly 8 GPUs; got ${#gpu_lines[@]}" >&2
  exit 1
fi
for line in "${gpu_lines[@]}"; do
  if [[ ! ${line} =~ (^|[[:space:]])H20([[:space:]]|$) ]]; then
    echo "Bernard deployment must expose only NVIDIA H20 GPUs; got: ${line}" >&2
    exit 1
  fi
done

export NUM_GPUS=8
export TP=1
export ULYSSES=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SGLANG_HOST=127.0.0.1
export SGLANG_PORT
export SGLANG_URL="http://127.0.0.1:${SGLANG_PORT}"
export OUTPUT_PATH="${DATA_ROOT}/videos"
export DATA_ROOT="${DATA_ROOT}/api"
export GPU_GROUP_INDEX=0
export GPU_INDEXES=0,1,2,3,4,5,6,7
export RELEASE_ID=${RELEASE_ID:-h3-8h20-20260826-v1}
export PYTHONPATH=/opt/minimax-h3/api${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${OUTPUT_PATH}" "${DATA_ROOT}"
prepare_model

sglang_pid=
api_pid=
shutdown() {
  local status=$?
  trap - EXIT INT TERM
  [[ -z ${api_pid} ]] || kill "${api_pid}" 2>/dev/null || true
  [[ -z ${sglang_pid} ]] || kill "${sglang_pid}" 2>/dev/null || true
  [[ -z ${api_pid} ]] || wait "${api_pid}" 2>/dev/null || true
  [[ -z ${sglang_pid} ]] || wait "${sglang_pid}" 2>/dev/null || true
  exit "${status}"
}
trap shutdown EXIT INT TERM

/opt/minimax-h3/bin/launch_sglang.sh &
sglang_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if ! kill -0 "${sglang_pid}" 2>/dev/null; then
    wait "${sglang_pid}"
    exit $?
  fi
  if curl -fsS "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null; then
    break
  fi
  sleep 5
done
if ! curl -fsS "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null; then
  echo "SGLang did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s" >&2
  exit 1
fi

/opt/minimax-h3/api-venv/bin/python /opt/minimax-h3/api/run_dual_stack.py &
api_pid=$!

echo "READY: Bernard API dual-stack port=${API_PORT}; SGLang port=${SGLANG_PORT}; topology=TP1xUlysses8 on 8xH20"
wait -n "${sglang_pid}" "${api_pid}"
