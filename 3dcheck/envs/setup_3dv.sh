#!/usr/bin/env bash
# Build the `3dv` env: conda layer (python/nvcc/gcc) + uv layer + pinned third-party repos + kernel.
#
# Usage:  bash envs/setup_3dv.sh
# Options (env vars):
#   MMCV_INDEX   wheel index for mmcv 2.1.0 (default: OpenMMLab cu121/torch2.1).
#                Set to "" to build mmcv from the PyPI sdist instead (slow; CPU ops only without a GPU).
#   BUILD_EXT    1 = also build optional CUDA extensions (BEVFusion ops, DUSt3R curope, tiny-cuda-nn).
#                Needs nvcc from the env; set TORCH_CUDA_ARCH_LIST if not building for sm86.
#   BUILD_TCNN   defaults to BUILD_EXT; set 0 to skip the (slow) tiny-cuda-nn build.
#   plus the variables documented in _lib.sh (CONDA_FRONTEND, ENV_PREFIX, THIRD_PARTY, REGISTER_KERNEL)

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

NAME=3dv
MMCV_INDEX="${MMCV_INDEX-https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html}"
BUILD_EXT="${BUILD_EXT:-0}"
BUILD_TCNN="${BUILD_TCNN:-$BUILD_EXT}"
C="$ENVS_DIR/3dv.constraints.txt"

pick_frontend
create_env "$ENVS_DIR/3dv.environment.yml" "$NAME"
PY="$(env_python "$NAME")"
PREFIX="$(dirname "$(dirname "$PY")")"
log "python: $PY"

# 1) torch first: mmcv (if built from source) and xformers need it present at build time
log "uv layer 1/3: torch stack"
uvi "$PY" torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 -c "$C"
uvi "$PY" "setuptools<81" wheel

# 2) everything else
log "uv layer 2/3: requirements"
MMCV_ARGS=()
if [[ -n "$MMCV_INDEX" ]]; then
  MMCV_ARGS=(-f "$MMCV_INDEX")
else
  log "MMCV_INDEX empty -> building mmcv 2.1.0 from sdist (this takes a while)"
  export MMCV_WITH_OPS=1
  MMCV_ARGS=(--no-build-isolation-package mmcv)
fi
uvi "$PY" -r "$ENVS_DIR/3dv.requirements.txt" -c "$C" "${MMCV_ARGS[@]}"

# 3) pinned source repos
log "uv layer 3/3: third-party repos"
for r in mast3r vggt Depth-Anything-3 Depth-Anything-V2 mmdetection3d; do clone_pinned "$r"; done
uvi "$PY" -e "$THIRD_PARTY/vggt" -c "$C"
uvi "$PY" -e "$THIRD_PARTY/Depth-Anything-3" -c "$C"
uvi "$PY" -e "$ROOT_DIR/common"
# DUSt3R and MASt3R share MASt3R's dust3r submodule so only one `dust3r` package is importable.
# DA V2 metric models live in metric_depth/ (no setup.py upstream).
add_pth "$PY" 3dcheck_3dv_src.pth \
  "$THIRD_PARTY/mast3r" \
  "$THIRD_PARTY/mast3r/dust3r" \
  "$THIRD_PARTY/Depth-Anything-V2/metric_depth"

if [[ "$BUILD_EXT" == "1" ]]; then
  cuda_env "$PREFIX"
  log "optional ext: BEVFusion ops (arch $TORCH_CUDA_ARCH_LIST)"
  # Upstream README says `setup.py develop`; with setuptools>=80 that turns into an isolated
  # `pip install -e .` of the whole repo root (fails: no torch). build_ext --inplace produces the same .so files.
  (cd "$THIRD_PARTY/mmdetection3d" && "$PY" projects/BEVFusion/setup.py build_ext --inplace)
  log "optional ext: DUSt3R curope"
  (cd "$THIRD_PARTY/mast3r/dust3r/croco/models/curope" && "$PY" setup.py build_ext --inplace)
  if [[ "$BUILD_TCNN" == "1" ]]; then
    log "optional ext: tiny-cuda-nn (Instant-NGP speed-up; long build)"
    TCNN_CUDA_ARCHITECTURES="${TORCH_CUDA_ARCH_LIST//./}" \
      uvi "$PY" --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" -c "$C"
  fi
fi

register_kernel "$PY" "$NAME" "3dv (torch 2.1.2, cu121)"
log "smoke test"
"$PY" "$ENVS_DIR/smoke_test.py" "$NAME"
