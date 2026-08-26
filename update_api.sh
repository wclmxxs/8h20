#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

[[ -f .env ]] || { echo ".env is missing; run ./install.sh first" >&2; exit 1; }
[[ -f .generated/compose.yaml ]] || {
  echo ".generated/compose.yaml is missing; run ./install.sh first" >&2
  exit 1
}

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${API_IMAGE:?API_IMAGE is missing from .env}"
: "${API_KEY:?API_KEY is missing from .env}"
: "${API_BASE_PORT:?API_BASE_PORT is missing from .env}"

echo "Building API image ${API_IMAGE}..."
sudo docker build --progress=plain \
  -f docker/Dockerfile.api \
  -t "${API_IMAGE}" \
  .

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
mapfile -t api_services < <("${compose[@]}" config --services | grep '^h3-api-')
(( ${#api_services[@]} > 0 )) || { echo "no API services found" >&2; exit 1; }

echo "Recreating API services only: ${api_services[*]}"
"${compose[@]}" up -d --no-deps --force-recreate "${api_services[@]}"

service_count=$(jq '.instances | length' .generated/instances.json)
for ((slot=0; slot<service_count; slot++)); do
  port=$((API_BASE_PORT + slot))
  deadline=$((SECONDS + 180))
  api_healthy=false
  while (( SECONDS < deadline )); do
    ipv4_healthy=false
    ipv6_healthy=false
    if curl --noproxy '*' -fsS --max-time 10 \
      -H "Authorization: Bearer ${API_KEY}" \
      "http://127.0.0.1:${port}/healthz" 2>/dev/null \
      | jq -e '.ok == true' >/dev/null 2>&1; then
      ipv4_healthy=true
    fi
    if curl --noproxy '*' -g -6 -fsS --max-time 10 \
      -H "Authorization: Bearer ${API_KEY}" \
      "http://[::1]:${port}/healthz" 2>/dev/null \
      | jq -e '.ok == true' >/dev/null 2>&1; then
      ipv6_healthy=true
    fi
    if [[ ${ipv4_healthy} == true && ${ipv6_healthy} == true ]]; then
      echo "API partition ${slot} is healthy on IPv4+IPv6 port ${port}"
      api_healthy=true
      break
    fi
    sleep 3
  done
  if [[ ${api_healthy} != true ]]; then
    "${compose[@]}" logs --tail 200 "h3-api-${slot}"
    echo "API partition ${slot} did not become healthy" >&2
    exit 1
  fi
done

for service in "${api_services[@]}"; do
  container=$("${compose[@]}" ps -q "${service}")
  command=$(sudo docker inspect -f '{{json .Config.Cmd}}' "${container}")
  [[ ${command} == *'--timeout-keep-alive","120"'* ]] || {
    echo "${service} is not running with the 120-second keep-alive" >&2
    exit 1
  }
done

echo "READY: API is dual-stack with 120-second keep-alive; GPU workers were not restarted"
