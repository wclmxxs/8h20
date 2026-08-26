#!/usr/bin/env bash
set -euo pipefail

API_PORT=${PORT:-${TCE_SERVICE_PORT:-30010}}
SGLANG_PORT=${SGLANG_PORT:-30020}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-1800}
DATA_ROOT=${DATA_ROOT:-/opt/tiger/minimax-h3/data}

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

/opt/minimax-h3/api-venv/bin/uvicorn app.server:app \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --workers 1 \
  --timeout-keep-alive 120 &
api_pid=$!

echo "READY: Bernard API port=${API_PORT}; SGLang port=${SGLANG_PORT}; topology=TP1xUlysses8 on 8xH20"
wait -n "${sglang_pid}" "${api_pid}"
