#!/usr/bin/env bash
set -Eeuo pipefail

# Manage manually launched SGLang/API processes in a BERNARD_DEBUG_HOLD=1 Pod.
# The permanent PID 1 is debug_hold.py, so restarting these children never
# terminates the container or triggers another model download.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=/sgl-workspace/sglang
model_root=${CSDE_MODEL_ROOT:-/opt/tiger/csde/MiniMax-H3}
sglang_port=${SGLANG_PORT:-30020}
api_port=${PORT:-${TCE_SERVICE_PORT:-30010}}
sglang_log=/tmp/minimax-h3-debug-sglang.log
api_log=/tmp/minimax-h3-debug-api.log
pid_file=/tmp/minimax-h3-debug-services.pids
action=${1:-restart}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case ${action} in
  start|stop|restart|status) ;;
  *) die "usage: $0 [start|stop|restart|status]" ;;
esac
if (( EUID != 0 )); then
  die "run this script as root inside the Bernard Pod"
fi

pid1_command=$(tr '\0' ' ' </proc/1/cmdline)
[[ ${pid1_command} == *debug_hold.py* ]] || die \
  "this command requires a BERNARD_DEBUG_HOLD=1 image; PID 1 is: ${pid1_command}"

process_is_live() {
  local pid=$1
  local stat rest state
  [[ -r /proc/${pid}/stat ]] || return 1
  stat=$(</proc/${pid}/stat)
  rest=${stat#*) }
  state=${rest%% *}
  [[ ${state} != Z && ${state} != X ]]
}

find_service_roots() {
  local pid command
  while read -r pid command; do
    if [[ ${command} == *"/usr/local/bin/sglang serve"* ]] \
      || [[ ${command} == *"/run_dual_stack.py"* ]]; then
      process_is_live "${pid}" && printf '%s\n' "${pid}"
    fi
  done < <(ps -eo pid=,args=)
}

collect_descendants() {
  local parent=$1
  local child
  while read -r child; do
    [[ -n ${child} ]] || continue
    collect_descendants "${child}"
  done < <(pgrep -P "${parent}" || true)
  printf '%s\n' "${parent}"
}

stop_services() {
  local root_pid child_pid pid
  local -a roots=()
  local -a service_pids=()
  local -a survivors=()

  mapfile -t roots < <(find_service_roots | sort -nu)
  if (( ${#roots[@]} == 0 )); then
    echo "No live debug SGLang/API processes found."
    return
  fi
  for root_pid in "${roots[@]}"; do
    while read -r child_pid; do
      [[ -n ${child_pid} ]] && service_pids+=("${child_pid}")
    done < <(collect_descendants "${root_pid}")
  done
  mapfile -t service_pids < <(printf '%s\n' "${service_pids[@]}" | sort -nu)

  echo "Stopping debug service PIDs: ${service_pids[*]}"
  kill -TERM "${service_pids[@]}" 2>/dev/null || true
  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    survivors=()
    for pid in "${service_pids[@]}"; do
      process_is_live "${pid}" && survivors+=("${pid}")
    done
    (( ${#survivors[@]} == 0 )) && break
    sleep 1
  done
  if (( ${#survivors[@]} > 0 )); then
    echo "Force-stopping remaining PIDs: ${survivors[*]}"
    kill -KILL "${survivors[@]}" 2>/dev/null || true
  fi
}

apply_runtime_patch() {
  local runtime_source backup runtime_patch static_lora_patch
  local -a runtime_patches=(
    "${repo_root}/patches/minimax-h3-cache-dit-residual-preservation.patch"
    "${repo_root}/patches/minimax-h3-sol-attn-path-observability.patch"
  )

  [[ -d ${runtime_root} ]] || die "missing SGLang workspace: ${runtime_root}"
  cd "${runtime_root}"
  static_lora_patch=${repo_root}/patches/minimax-h3-static-lora-before-fp8.patch
  [[ -f ${static_lora_patch} ]] || die "missing static LoRA rollback patch: ${static_lora_patch}"
  if git apply -p1 --reverse --check "${static_lora_patch}"; then
    git apply -p1 --reverse "${static_lora_patch}"
    echo "Removed static-LoRA-before-FP8 runtime patch."
  elif git apply -p1 --check "${static_lora_patch}"; then
    echo "Static-LoRA-before-FP8 runtime patch is already absent."
  else
    die "cannot prove the static-LoRA-before-FP8 patch is absent"
  fi
  for runtime_patch in "${runtime_patches[@]}"; do
    [[ -f ${runtime_patch} ]] || die "missing runtime patch: ${runtime_patch}"
    if git apply -p1 --check "${runtime_patch}"; then
      git apply -p1 "${runtime_patch}"
      echo "Applied runtime patch: $(basename "${runtime_patch}")"
    elif git apply -p1 --reverse --check "${runtime_patch}"; then
      echo "Runtime patch already present: $(basename "${runtime_patch}")"
    else
      die "runtime patch does not match the active SGLang tree: ${runtime_patch}"
    fi
  done
  runtime_source=$(
    python3 - <<'PY'
import inspect
import sglang.multimodal_gen.runtime.models.dits.minimax_h3 as module

print(inspect.getsourcefile(module))
PY
  )
  [[ -f ${runtime_source} ]] || die "cannot locate imported MiniMax H3 source"
  case ${runtime_source} in
    "${runtime_root}"/* | /usr/local/lib/python*/dist-packages/sglang/*) ;;
    *) die "refusing to patch unexpected runtime source: ${runtime_source}" ;;
  esac

  backup=${runtime_source}.before-final-gather-bcg
  if [[ ! -e ${backup} ]]; then
    cp -p -- "${runtime_source}" "${backup}"
  fi
  python3 - "${runtime_source}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = """_minimax_h3_sp_all_gather_eager = torch.compiler.disable(
    _minimax_h3_sp_all_gather_impl
)
"""
new = """_minimax_h3_sp_all_gather_compiler_eager = torch.compiler.disable(
    _minimax_h3_sp_all_gather_impl
)
_minimax_h3_sp_all_gather_eager = eager_on_graph(True)(
    _minimax_h3_sp_all_gather_compiler_eager
)
"""
source = path.read_text()
if new in source:
    print(f"Runtime patch already present: {path}")
elif old in source:
    path.write_text(source.replace(old, new, 1))
    print(f"Patched runtime source: {path}")
else:
    raise SystemExit(f"Unexpected MiniMax H3 runtime source: {path}")
PY
  python3 -m py_compile "${runtime_source}"
}

export_runtime() {
  local data_base=${MINIMAX_H3_DATA_BASE:-/opt/tiger/minimax-h3/data}

  [[ -f ${model_root}/modular_model_index.json ]] \
    || die "localized MiniMax-H3 model is missing: ${model_root}"
  export NUM_GPUS=8
  export TP=1
  export ULYSSES=8
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
  export MODEL=${model_root}
  export SGLANG_HOST=127.0.0.1
  export SGLANG_PORT=${sglang_port}
  export SGLANG_URL=http://127.0.0.1:${sglang_port}
  export OUTPUT_PATH=${data_base}/videos
  export DATA_ROOT=${data_base}/api
  export GPU_GROUP_INDEX=0
  export GPU_INDEXES=0,1,2,3,4,5,6,7
  export RELEASE_ID=${RELEASE_ID:-h3-8h20-debug}
  export PORT=${api_port}
  export PYTHONPATH=${repo_root}/api${PYTHONPATH:+:${PYTHONPATH}}
  mkdir -p "${OUTPUT_PATH}" "${DATA_ROOT}"
}

start_services() {
  local -a existing=()
  mapfile -t existing < <(find_service_roots | sort -nu)
  (( ${#existing[@]} == 0 )) \
    || die "debug services are already running (${existing[*]}); use restart"
  [[ -x ${repo_root}/scripts/launch_sglang.sh ]] \
    || die "missing launcher: ${repo_root}/scripts/launch_sglang.sh"
  [[ -f ${repo_root}/api/run_dual_stack.py ]] \
    || die "missing API runner: ${repo_root}/api/run_dual_stack.py"
  [[ -x /opt/minimax-h3/api-venv/bin/python ]] \
    || die "missing Bernard API virtualenv"

  export_runtime
  : >"${sglang_log}"
  nohup setsid "${repo_root}/scripts/launch_sglang.sh" \
    >"${sglang_log}" 2>&1 </dev/null &
  sglang_pid=$!

  : >"${api_log}"
  nohup setsid /opt/minimax-h3/api-venv/bin/python \
    "${repo_root}/api/run_dual_stack.py" >"${api_log}" 2>&1 </dev/null &
  api_pid=$!
  sleep 3
  process_is_live "${sglang_pid}" || {
    tail -n 80 "${sglang_log}" >&2 || true
    die "SGLang exited immediately"
  }
  process_is_live "${api_pid}" || {
    tail -n 80 "${api_log}" >&2 || true
    die "API exited immediately"
  }
  printf 'SGLANG_PID=%s\nAPI_PID=%s\n' "${sglang_pid}" "${api_pid}" >"${pid_file}"
  echo "STARTED: SGLang PID ${sglang_pid}; API PID ${api_pid}"
  echo "SGLang warmup continues in the background: ${sglang_log}"
  echo "API log: ${api_log}"
  echo "Run '$0 status' until both endpoints report healthy."
}

status_services() {
  local failed=0
  local sglang_healthy=0
  local -a roots=()
  local -a headers=()

  mapfile -t roots < <(find_service_roots | sort -nu)
  echo "Live service root PIDs: ${roots[*]:-none}"
  if grep -Fq "Synthetic server warmup failed" "${sglang_log}" 2>/dev/null \
    || grep -Fq "NCCL Error" "${sglang_log}" 2>/dev/null; then
    echo "SGLang: warmup failed (see ${sglang_log})"
    failed=1
  elif curl -fsS --max-time 5 "http://127.0.0.1:${sglang_port}/health" \
    >/dev/null 2>&1; then
    echo "SGLang: healthy"
    sglang_healthy=1
  else
    echo "SGLang: starting or failed (see ${sglang_log})"
    failed=1
  fi
  if [[ -n ${API_KEY:-} ]]; then
    headers=(-H "Authorization: Bearer ${API_KEY}")
  fi
  if (( sglang_healthy == 1 )) && curl -fsS --max-time 5 "${headers[@]}" \
    "http://127.0.0.1:${api_port}/healthz" 2>/dev/null \
    | python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("ok") is True else 1)' \
      >/dev/null 2>&1; then
    echo "API: healthy"
  else
    echo "API: waiting for SGLang or failed (see ${api_log})"
    failed=1
  fi
  return "${failed}"
}

case ${action} in
  start)
    apply_runtime_patch
    start_services
    ;;
  stop)
    stop_services
    ;;
  restart)
    apply_runtime_patch
    stop_services
    start_services
    ;;
  status)
    status_services
    ;;
esac
