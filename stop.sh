#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

if [[ ! -f .env || ! -f .generated/compose.yaml ]]; then
  echo "nothing to stop"
  exit 0
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
"${compose[@]}" stop h3-watchdog >/dev/null 2>&1 || true
mapfile -t inference_services < <(
  "${compose[@]}" config --services | sed -n '/^h3-\(api\|sglang\)-/p'
)
if ((${#inference_services[@]})); then
  "${compose[@]}" stop "${inference_services[@]}"
  # Let the reporter publish alive=false before it exits.
  sleep "$((REPORT_INTERVAL_SECONDS + 2))"
fi
"${compose[@]}" stop h3-reporter h3-cleaner
echo "Stopped MiniMax H3 services; model cache and outputs remain under ${DATA_ROOT}."
