#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${ROOT}/scripts/configure_sol_ab.sh" disable
