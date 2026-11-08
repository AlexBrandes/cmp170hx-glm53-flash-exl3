#!/usr/bin/env bash
# Host setting the published numbers were measured with (run as root;
# start.sh calls it through sudo unless HOST_TUNE=0).
#   host-tune.sh <gpu list, e.g. 0,1>
# Sets the 170 W power limit on both cards. Nothing else on the host changes.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "host-tune.sh: run as root" >&2; exit 1; }
gpus=${1:?gpu list}
for g in $(echo "$gpus" | tr ',' ' '); do
  nvidia-smi -i "$g" -pl 170 >/dev/null
done
echo "host tuned: 170 W on GPUs $gpus"
