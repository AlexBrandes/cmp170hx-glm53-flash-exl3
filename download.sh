#!/usr/bin/env bash
# Download the checkpoint and the drafter at the pinned revisions.
#   ./download.sh       (MODELS_DIR=./models by default; ~113 GB kept, ~126 GB peak)
# Also builds the INT8 dense sidecar: fetches the dense text weights of the original
# checkpoint (13.2 GB, range requests; the routed experts are not downloaded) and
# converts them in the engine image (tools/convert_dense.py, a few CPU minutes).
# The drafter is CC BY-NC-ND 4.0 (non-commercial); read its model card first.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env
IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/GLM-5.3-Flash-exl3-sc_2.60bpw}
DRAFT_DIR=${DRAFT_DIR:-$MODELS_DIR/GLM-5.3-Flash-DFlash2}
ORIG_DIR=${ORIG_DIR:-$MODELS_DIR/GLM-5.3-Flash-dense-orig}
W8_DIR=${W8_DIR:-$MODELS_DIR/GLM-5.3-Flash-dense-w8}
mkdir -p "$MODEL_DIR" "$DRAFT_DIR" "$ORIG_DIR" "$W8_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$MODEL_DIR:/model" -v "$DRAFT_DIR:/draft" --entrypoint python3 "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model')
snapshot_download('$DRAFT_REPO', revision='$DRAFT_REVISION', local_dir='/draft')
print('model and drafter ready')"

if [ ! -s "$W8_DIR/dense.safetensors" ]; then
  run() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/tools:/tools:ro" \
            -v "$MODEL_DIR:/model:ro" -v "$ORIG_DIR:/orig" -v "$W8_DIR:/w8" --entrypoint python3 "$IMAGE" "$@"; }
  [ -s "$ORIG_DIR/dense.safetensors" ] || run /tools/fetch_dense.py "$ORIG_REPO" "$ORIG_REVISION" /orig/dense.safetensors
  run /tools/convert_dense.py /orig/dense.safetensors /model /w8/dense.safetensors.tmp
  mv "$W8_DIR/dense.safetensors.tmp" "$W8_DIR/dense.safetensors"
  [ "${KEEP_ORIG:-0}" = 1 ] || rm -f "$ORIG_DIR/dense.safetensors"
  echo "INT8 dense sidecar ready: $W8_DIR/dense.safetensors"
fi
