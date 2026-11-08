#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && . ./.env
NAME=${NAME:-cmp170hx-glm53-flash-exl3}
docker stop -t 120 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "stopped $NAME"
