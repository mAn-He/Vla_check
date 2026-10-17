#!/usr/bin/env bash
# Build mmdetection3d info files for nuScenes-mini, then the eval4 subset + custom devkit splits.
# Run once in the 3dv env after place_nuscenes_mini.sh / verify_nuscenes_mini.py:
#     bash outdoor/scripts/prepare_mmdet3d_infos.sh
#
# Step 1 is the official command (mmdetection3d v1.4.0 docs/en/user_guides/dataset_prepare.md, "NuScenes"):
#     python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
# plus `--version v1.0-mini`, which tools/create_data.py routes to nuscenes_data_prep(version='v1.0-mini')
# (mini_train -> nuscenes_infos_train.pkl, mini_val -> nuscenes_infos_val.pkl, then update_pkl_infos -> v2 format,
#  then create_groundtruth_database for train-time ObjectSample; the latter is unused at test time but official).
# --max-sweeps: default 10 in create_data.py; PointPillars nuScenes uses 10 sweeps, CenterPoint/BEVFusion 9.
#
# Why a scratch working directory: tools/dataset_converters/update_infos_to_v2.py::update_nuscenes_infos opens
# NuScenes(dataroot='./data/nuscenes') (hard-coded, relative to the cwd). We therefore run from a temp dir whose
# ./data/nuscenes is a symlink to DATA_ROOT, so any DATA_ROOT works. Outputs land in DATA_ROOT through the link:
#     nuscenes_infos_train.pkl  nuscenes_infos_val.pkl  nuscenes_dbinfos_train.pkl  nuscenes_gt_database/
#
# Step 2 (make_eval4_infos.py): nuscenes_infos_eval4.pkl + <DATA_ROOT>/v1.0-mini/splits.json (eval4, probe6,
# mini_val_official) for bench3d.metrics.detection.evaluate_detection.
#
# Env overrides: PYTHON (3dv python), THIRD_PARTY (pinned clones), DATA_ROOT, MAX_SWEEPS.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
THIRD_PARTY="${THIRD_PARTY:-$ROOT/third_party}"
MMDET3D="$THIRD_PARTY/mmdetection3d"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/nuscenes}"   # must match dataset.data_root in outdoor/config.yaml
MAX_SWEEPS="${MAX_SWEEPS:-10}"
EXTRA_TAG=nuscenes                               # info file prefix used by every official nuScenes config

if (( MAX_SWEEPS < 10 )); then
  echo "MAX_SWEEPS=$MAX_SWEEPS < 10: PointPillars (nuScenes) loads 10 sweeps per keyframe" >&2
  exit 1
fi
for p in "$MMDET3D/tools/create_data.py" "$DATA_ROOT/v1.0-mini/scene.json" "$DATA_ROOT/samples/LIDAR_TOP" \
         "$DATA_ROOT/sweeps/LIDAR_TOP"; do
  if [[ ! -e "$p" ]]; then echo "missing: $p" >&2; exit 1; fi
done
"$PY" -c "import mmdet3d, nuscenes; assert mmdet3d.__version__ == '1.4.0', mmdet3d.__version__" \
  || { echo "run with the 3dv env python (PYTHON=...)" >&2; exit 1; }

DATA_ABS="$(cd "$DATA_ROOT" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/data"
ln -s "$DATA_ABS" "$WORK/data/nuscenes"

echo "[1/2] tools/create_data.py nuscenes --version v1.0-mini --max-sweeps $MAX_SWEEPS (root: $DATA_ABS)"
# The clone is appended to sys.path (not PYTHONPATH) so `tools.dataset_converters` resolves from the clone while
# the installed mmdet3d 1.4.0 package keeps precedence over the clone's mmdet3d/ source tree.
(cd "$WORK" && "$PY" - "$MMDET3D" "$MAX_SWEEPS" "$EXTRA_TAG" <<'PYEOF'
import runpy
import sys

repo, sweeps, tag = sys.argv[1:4]
sys.path.append(repo)
script = f"{repo}/tools/create_data.py"
sys.argv = [script, "nuscenes", "--root-path", "./data/nuscenes", "--out-dir", "./data/nuscenes",
            "--extra-tag", tag, "--version", "v1.0-mini", "--max-sweeps", sweeps]
runpy.run_path(script, run_name="__main__")
PYEOF
)

echo "[2/2] eval4 info subset + splits.json"
"$PY" "$ROOT/outdoor/scripts/make_eval4_infos.py" --data-root "$DATA_ABS" --extra-tag "$EXTRA_TAG"

echo "done:"
ls -1 "$DATA_ABS"/"$EXTRA_TAG"_infos_*.pkl "$DATA_ABS/v1.0-mini/splits.json"
