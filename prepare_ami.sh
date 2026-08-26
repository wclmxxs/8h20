#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

[[ -f .env ]] || {
  echo ".env is missing; complete ./install.sh before preparing an AMI" >&2
  exit 1
}
[[ -f .generated/compose.yaml ]] || {
  echo ".generated/compose.yaml is missing; complete ./install.sh before preparing an AMI" >&2
  exit 1
}

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)

echo "Stopping Watchdog and Reporter before inference services..."
"${compose[@]}" stop h3-watchdog >/dev/null 2>&1 || true
"${compose[@]}" stop h3-reporter >/dev/null 2>&1 || true
echo "Stopping all deployment containers while preserving images and caches..."
"${compose[@]}" stop

mkdir -p .state
cat >.state/ami-ready <<EOF
prepared_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_instance_id=$(sed -n 's/^INSTANCE_ID=//p' .env)
EOF
sync

cat <<'EOF'
AMI_READY: containers are stopped and will remain stopped across a Docker restart.
Create the AMI now. On each cloned instance run:
  git pull --ff-only
  ./install.sh --from-ami

To resume this source machine instead, run:
  ./install.sh --from-ami
EOF
