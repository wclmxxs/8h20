#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

if [[ ! -f .env || ! -f .generated/compose.yaml || ! -f .generated/instances.json ]]; then
  echo "not installed: run ./install.sh first" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
service_count=$(jq '.instances | length' .generated/instances.json)

echo "=== containers ==="
"${compose[@]}" ps

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo "=== single 8-H20 worker ==="
for ((slot=0; slot<service_count; slot++)); do
  port=$((API_BASE_PORT + slot))
  printf 'slot=%d port=%d family=ipv4 ' "${slot}" "${port}"
  if ! curl --noproxy '*' -fsS --max-time 10 \
      -H "Authorization: Bearer ${API_KEY}" \
      "http://127.0.0.1:${port}/healthz" |
      jq -c '{ok,healthy_workers,gpu_indexes,gpu_uuids,deployment}'; then
    echo '{"ok":false,"error":"health request failed"}'
  fi
  printf 'slot=%d port=%d family=ipv6 ' "${slot}" "${port}"
  if ! curl --noproxy '*' -g -6 -fsS --max-time 10 \
      -H "Authorization: Bearer ${API_KEY}" \
      "http://[::1]:${port}/healthz" |
      jq -c '{ok,healthy_workers,gpu_indexes,gpu_uuids,deployment}'; then
    echo '{"ok":false,"error":"health request failed"}'
  fi
done

echo "=== registration ==="
if [[ -f ${DATA_ROOT}/reporter/status.json ]]; then
  jq . "${DATA_ROOT}/reporter/status.json"
else
  echo "reporter has not written status yet"
fi

echo "=== cleanup ==="
if [[ -f ${DATA_ROOT}/cleaner/status.json ]]; then
  jq . "${DATA_ROOT}/cleaner/status.json"
else
  echo "cleaner has not written status yet"
fi

echo "=== queue watchdog ==="
if [[ -f ${DATA_ROOT}/watchdog/status.json ]]; then
  jq . "${DATA_ROOT}/watchdog/status.json"
else
  echo "watchdog has not written status yet"
fi
