#!/usr/bin/env bash
set -Eeuo pipefail

# Hot-patch the already running Bernard container for short-lived validation.
# This intentionally avoids an image build and reuses the localized model. PID 1
# is left stopped after the old children are replaced because its original
# `wait -n` would otherwise observe an exited child and terminate the container.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=/sgl-workspace/sglang
model_root=${CSDE_MODEL_ROOT:-/opt/tiger/csde/MiniMax-H3}
sglang_port=${SGLANG_PORT:-30020}
api_port=${PORT:-${TCE_SERVICE_PORT:-30010}}
startup_timeout=${HOTPATCH_STARTUP_TIMEOUT_SECONDS:-1800}
sglang_log=/tmp/minimax-h3-hotpatch-sglang.log
api_log=/tmp/minimax-h3-hotpatch-api.log

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if (( EUID != 0 )); then
  die "run this script as root inside the Bernard Pod"
fi
[[ -d ${runtime_root} ]] || die "missing SGLang workspace: ${runtime_root}"
[[ -f ${model_root}/modular_model_index.json ]] \
  || die "localized MiniMax-H3 model is missing: ${model_root}"
[[ -x /opt/minimax-h3/bin/launch_sglang.sh ]] \
  || die "missing SGLang launcher in the current image"
[[ -x /opt/minimax-h3/api-venv/bin/python ]] \
  || die "missing Bernard API virtualenv in the current image"

pid1_command=$(tr '\0' ' ' </proc/1/cmdline)
[[ ${pid1_command} == *start_bernard.sh* ]] \
  || die "PID 1 is not the Bernard supervisor: ${pid1_command}"

cd "${runtime_root}"
runtime_source=$(
  python3 - <<'PY'
import inspect
import sglang.multimodal_gen.runtime.models.dits.minimax_h3 as module

print(inspect.getsourcefile(module))
PY
)
[[ -f ${runtime_source} ]] || die "cannot locate the imported MiniMax H3 source"
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
    print(f"Patch already present: {path}")
elif old in source:
    path.write_text(source.replace(old, new, 1))
    print(f"Patched runtime source: {path}")
else:
    raise SystemExit(
        "Runtime source is neither the expected v1.0.0.13 code nor the patched code: "
        f"{path}"
    )
PY
python3 -m py_compile "${runtime_source}"

if [[ ${1:-} == --patch-only ]]; then
  echo "Patch-only mode complete; processes were not restarted."
  exit 0
fi
if [[ $# -gt 0 ]]; then
  die "unknown argument: $1 (the only supported argument is --patch-only)"
fi

service_roots=()
found_sglang=0
found_api=0
while read -r pid ppid command; do
  if [[ ${ppid} == 1 && ${command} == *"/usr/local/bin/sglang serve"* ]]; then
    service_roots+=("${pid}")
    found_sglang=1
  elif [[ ${ppid} == 1 && ${command} == *"/opt/minimax-h3/api/run_dual_stack.py"* ]]; then
    service_roots+=("${pid}")
    found_api=1
  fi
done < <(ps -eo pid=,ppid=,args=)
(( found_sglang == 1 )) || die "could not find the current SGLang child of PID 1"
(( found_api == 1 )) || die "could not find the current API child of PID 1"

collect_descendants() {
  local parent=$1
  local child
  while read -r child; do
    [[ -n ${child} ]] || continue
    collect_descendants "${child}"
  done < <(pgrep -P "${parent}" || true)
  printf '%s\n' "${parent}"
}

service_pids=()
for root_pid in "${service_roots[@]}"; do
  while read -r child_pid; do
    [[ -n ${child_pid} ]] && service_pids+=("${child_pid}")
  done < <(collect_descendants "${root_pid}")
done
mapfile -t service_pids < <(printf '%s\n' "${service_pids[@]}" | sort -nu)

echo "Pausing Bernard PID 1 and replacing only its SGLang/API children."
kill -STOP 1
kill -TERM "${service_pids[@]}" 2>/dev/null || true

deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  survivors=()
  for pid in "${service_pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && survivors+=("${pid}")
  done
  (( ${#survivors[@]} == 0 )) && break
  sleep 1
done
if (( ${#survivors[@]} > 0 )); then
  echo "Force-stopping stale worker PIDs: ${survivors[*]}"
  kill -KILL "${survivors[@]}" 2>/dev/null || true
fi

data_base=${MINIMAX_H3_DATA_BASE:-/opt/tiger/minimax-h3/data}
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
export RELEASE_ID=${RELEASE_ID:-h3-8h20-hotpatch}
export PORT=${api_port}
export PYTHONPATH=/opt/minimax-h3/api${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${OUTPUT_PATH}" "${DATA_ROOT}"

: >"${sglang_log}"
nohup setsid /opt/minimax-h3/bin/launch_sglang.sh \
  >"${sglang_log}" 2>&1 </dev/null &
new_sglang_pid=$!
echo "Started SGLang PID ${new_sglang_pid}; log: ${sglang_log}"

deadline=$((SECONDS + startup_timeout))
next_update=$((SECONDS + 30))
while ! curl -fsS --max-time 5 "http://127.0.0.1:${sglang_port}/health" \
  >/dev/null 2>&1; do
  if ! kill -0 "${new_sglang_pid}" 2>/dev/null; then
    tail -n 120 "${sglang_log}" >&2 || true
    die "hot-patched SGLang exited during startup; restart the Pod to restore supervision"
  fi
  if (( SECONDS >= deadline )); then
    tail -n 120 "${sglang_log}" >&2 || true
    die "SGLang did not become healthy within ${startup_timeout}s"
  fi
  if (( SECONDS >= next_update )); then
    echo "Still waiting for SGLang warmup ($((deadline - SECONDS))s remaining)..."
    next_update=$((SECONDS + 30))
  fi
  sleep 5
done

: >"${api_log}"
nohup setsid /opt/minimax-h3/api-venv/bin/python \
  /opt/minimax-h3/api/run_dual_stack.py >"${api_log}" 2>&1 </dev/null &
new_api_pid=$!
echo "Started API PID ${new_api_pid}; log: ${api_log}"

headers=()
if [[ -n ${API_KEY:-} ]]; then
  headers=(-H "Authorization: Bearer ${API_KEY}")
fi
deadline=$((SECONDS + 120))
while ! curl -fsS --max-time 5 "${headers[@]}" \
  "http://127.0.0.1:${api_port}/healthz" >/dev/null 2>&1; do
  if ! kill -0 "${new_api_pid}" 2>/dev/null; then
    tail -n 120 "${api_log}" >&2 || true
    die "hot-patched API exited during startup; restart the Pod to restore supervision"
  fi
  if (( SECONDS >= deadline )); then
    tail -n 120 "${api_log}" >&2 || true
    die "API did not become healthy within 120s"
  fi
  sleep 2
done

cat >/tmp/minimax-h3-hotpatch.pids <<EOF
SGLANG_PID=${new_sglang_pid}
API_PID=${new_api_pid}
EOF

echo "READY: hot-patched API port=${api_port}; SGLang port=${sglang_port}"
echo "PID 1 remains stopped by design. Do not run 'kill -CONT 1'."
echo "After validation, restart the Pod (or deploy a fixed image) to restore supervision."
echo "Repository used for the hot-patch: ${repo_root}"
