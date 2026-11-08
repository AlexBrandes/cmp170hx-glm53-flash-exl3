#!/usr/bin/env bash
# Download the checkpoint and the drafter at the pinned revisions.
#   ./download.sh       (MODELS_DIR=./models by default; ~105 GB)
# The drafter is CC BY-NC-ND 4.0 (non-commercial); read its model card first.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env
IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/GLM-5.3-Flash-exl3-sc_2.60bpw}
DRAFT_DIR=${DRAFT_DIR:-$MODELS_DIR/GLM-5.3-Flash-DFlash2}
mkdir -p "$MODEL_DIR" "$DRAFT_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$MODEL_DIR:/model" -v "$DRAFT_DIR:/draft" --entrypoint python3 "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model')
snapshot_download('$DRAFT_REPO', revision='$DRAFT_REVISION', local_dir='/draft')
print('model and drafter ready')"
