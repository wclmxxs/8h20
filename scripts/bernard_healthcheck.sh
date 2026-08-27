#!/usr/bin/env bash
set -euo pipefail

if [[ ${BERNARD_DEBUG_HOLD:-0} == 1 ]]; then
  printf '{"ok":true,"mode":"debug_hold"}\n'
  exit 0
fi

port=${PORT:-${TCE_SERVICE_PORT:-30010}}
headers=()
if [[ -n ${API_KEY:-} ]]; then
  headers=(-H "Authorization: Bearer ${API_KEY}")
fi

curl -fsS --max-time 10 "${headers[@]}" "http://127.0.0.1:${port}/healthz" \
  | python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("ok") is True else 1)'
