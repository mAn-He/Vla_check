#!/usr/bin/env bash
# Extract manually downloaded nuScenes-mini archives into the layout the devkit expects.
# This script does NOT download anything: nuscenes.org requires a login and accepting the Terms of Use.
#
# 1) Log in on the nuScenes download page (https://www.nuscenes.org/) and download into 3dcheck/data/downloads/:
#      - v1.0-mini.tgz
#      - nuScenes-lidarseg-mini-v1.0.tar.bz2
#    File names are the ones used in nuscenes-devkit python-sdk/tutorials/nuscenes_lidarseg_panoptic_tutorial.ipynb.
#    TODO(verify): exact menu labels on the download page (site was unreachable during step 0).
# 2) Run:  bash outdoor/scripts/place_nuscenes_mini.sh
# 3) Then: python outdoor/scripts/verify_nuscenes_mini.py
#
# Resulting layout (data_root in outdoor/config.yaml):
#   data/nuscenes/
#     maps/  samples/  sweeps/
#     v1.0-mini/       *.json tables (+ lidarseg.json, category.json from the lidarseg archive)
#     lidarseg/v1.0-mini/*_lidarseg.bin

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DL="${ARCHIVES_DIR:-$ROOT/data/downloads}"
DST="${DATA_ROOT:-$ROOT/data/nuscenes}"
MAIN="$DL/v1.0-mini.tgz"
SEG="$DL/nuScenes-lidarseg-mini-v1.0.tar.bz2"

missing=0
for f in "$MAIN" "$SEG"; do
  if [[ ! -f "$f" ]]; then echo "missing: $f"; missing=1; fi
done
if [[ $missing == 1 ]]; then
  echo "Download the files listed at the top of this script into $DL first."
  exit 1
fi

mkdir -p "$DST"
echo "extracting $(basename "$MAIN") -> $DST"
tar -xf "$MAIN" -C "$DST"
# The lidarseg archive adds lidarseg/ and extra tables into v1.0-mini/, so it must go into the same root.
echo "extracting $(basename "$SEG") -> $DST"
tar -xf "$SEG" -C "$DST"

echo "done. top-level entries:"
ls "$DST"
