#!/usr/bin/env bash
# Download the checkpoint and the drafter at the pinned revisions, and build the
# 4-bit drafter.
#   ./download.sh       (MODELS_DIR=./models by default; ~118 GB kept, ~120 GB peak)
# The checkpoint repo includes the INT8 dense weights the engine serves
# (int8/dense.safetensors). The drafter is downloaded in BF16 and converted to
# 4 bits in the engine image (tools/convert_draft.py, about a minute on the CPU);
# the BF16 copy is removed afterwards unless KEEP_DRAFT_BF16=1.
# The drafter is CC BY-NC-ND 4.0 (non-commercial); read its model card first.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env
IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/GLM-5.3-Flash-A4NX-2.6bpw}
DRAFT_DIR=${DRAFT_DIR:-$MODELS_DIR/GLM-5.3-Flash-DFlash2-w4a16}
DRAFT_SRC_DIR=${DRAFT_SRC_DIR:-$MODELS_DIR/GLM-5.3-Flash-DFlash2}
mkdir -p "$MODEL_DIR" "$DRAFT_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
run() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN="${HF_TOKEN:-}" \
          --entrypoint python3 "$@"; }

run -v "$MODEL_DIR:/model" "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model')
print('checkpoint ready: $MODEL_DIR')"

if [ ! -s "$DRAFT_DIR/model.safetensors" ]; then
  mkdir -p "$DRAFT_SRC_DIR"
  run -v "$DRAFT_SRC_DIR:/draft-src" "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$DRAFT_REPO', revision='$DRAFT_REVISION', local_dir='/draft-src')"
  run -v "$PWD/tools:/tools:ro" -v "$DRAFT_SRC_DIR:/draft-src:ro" -v "$DRAFT_DIR:/draft" "$IMAGE" \
    /tools/convert_draft.py /draft-src /draft >/dev/null
  [ "${KEEP_DRAFT_BF16:-0}" = 1 ] || rm -rf "$DRAFT_SRC_DIR"
fi
got=$(sha256sum "$DRAFT_DIR/model.safetensors" | cut -d' ' -f1)
if [ "$got" = "$DRAFT_W4_SHA256" ]; then
  echo "drafter ready: $DRAFT_DIR (identical to the measured build)"
else
  echo "drafter ready: $DRAFT_DIR (sha256 $got; the measured build's is $DRAFT_W4_SHA256)"
fi
