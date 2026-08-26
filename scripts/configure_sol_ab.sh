#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

mode=${1:-}
if [[ ${mode} != enable && ${mode} != disable ]]; then
  echo "Usage: $0 enable|disable" >&2
  exit 2
fi
if [[ ! -f .env ]]; then
  echo ".env is missing; run ./install.sh once before changing the optimization stack" >&2
  exit 1
fi

set_env() {
  local key=$1 value=$2
  if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*$|${key}=${value}|" .env
  else
    printf '%s=%s\n' "${key}" "${value}" >> .env
  fi
}

if [[ ${mode} == enable ]]; then
  set_env OPTIMIZATION_STACK_ENABLED 1
  echo "Enabling Sol-Attn + FP8 + Cache-DiT on the single 8-H20 worker..."
else
  set_env OPTIMIZATION_STACK_ENABLED 0
  echo "Disabling the optimization stack on the single 8-H20 worker..."
fi

exec ./install.sh
