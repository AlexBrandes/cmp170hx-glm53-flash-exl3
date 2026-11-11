#!/usr/bin/env bash
# Start GLM-5.3-Flash (EXL3 2.60 bpw) pipeline-parallel on two CMP 170HX cards.
#   ./start.sh          set the power limit (sudo), start the container, wait for /health
# Settings: .env (see .env.example) or environment variables. Engine settings
# are in docker/serve.sh, mounted over the image's copy.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env

IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/GLM-5.3-Flash-exl3-sc_2.60bpw}
DRAFT_DIR=${DRAFT_DIR:-$MODELS_DIR/GLM-5.3-Flash-DFlash2}
W8_DIR=${W8_DIR:-$MODELS_DIR/GLM-5.3-Flash-dense-w8}
CACHE_DIR=${CACHE_DIR:-$PWD/cache}
GPUS=${GPUS:-0,1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
NAME=${NAME:-cmp170hx-glm53-flash-exl3}
HOST_TUNE=${HOST_TUNE:-1}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

die() { echo "start.sh: $*" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || die "no model at $MODEL_DIR (run ./download.sh)"
[ -f "$DRAFT_DIR/config.json" ] || die "no drafter at $DRAFT_DIR (run ./download.sh)"
[ -f "$W8_DIR/dense.safetensors" ] || die "no INT8 dense sidecar at $W8_DIR (run ./download.sh)"
[ "$(echo "$GPUS" | tr ',' '\n' | grep -c .)" = 2 ] || die "GPUS must name two cards, e.g. 0,1"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
mkdir -p "$CACHE_DIR"

if [ "$HOST_TUNE" = 1 ]; then
  sudo ./host-tune.sh "$GPUS"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --shm-size 16g \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HUB_OFFLINE=1 \
  -p "$HOST:$PORT:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$DRAFT_DIR:/draft:ro" \
  -v "$W8_DIR:/w8:ro" \
  -v "$CACHE_DIR:/cache" \
  -v "$PWD/config/chat_template.jinja:/model/chat_template.jinja:ro" \
  -v "$PWD/docker/serve.sh:/opt/cmp170hx/serve.sh:ro" \
  "$IMAGE" >/dev/null

echo "started $NAME ($IMAGE) on $HOST:$PORT; loading (the first start compiles kernels)..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -fs "http://$HOST:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps -q -f "name=^$NAME$" | grep -q .; then
    docker logs --tail 50 "$NAME" >&2 || true
    die "container exited during startup"
  fi
  [ "$(date +%s)" -lt "$deadline" ] || die "not ready after ${READY_TIMEOUT}s (docker logs $NAME)"
  sleep 5
done
echo "ready: http://$HOST:$PORT/v1 (model glm-5.3-flash-exl3-2.60bpw)"
