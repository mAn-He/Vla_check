#!/usr/bin/env bash
# Build the `3dv-point` env (Pointcept / Sonata / Concerto): conda layer + uv layer + pinned repos + kernel.
#
# Usage:  bash envs/setup_3dv-point.sh
# Options (env vars):
#   PYG_INDEX    wheel index for torch-scatter / torch-cluster (default: PyG torch-2.5.0+cu124).
#                Set to "" to build them from the PyPI sdist instead (slow).
#   FLASH_ATTN   1 = install flash-attn (Ampere+ only). Without it, set enable_flash=False in configs.
#   POINTOPS     1 = build Pointcept libs/pointops (only needed for PTv1/PTv2 or precise eval).
#   plus the variables documented in _lib.sh

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

NAME=3dv-point
PYG_INDEX="${PYG_INDEX-https://data.pyg.org/whl/torch-2.5.0+cu124.html}"
FLASH_ATTN="${FLASH_ATTN:-0}"
POINTOPS="${POINTOPS:-0}"
C="$ENVS_DIR/3dv-point.constraints.txt"

pick_frontend
create_env "$ENVS_DIR/3dv-point.environment.yml" "$NAME"
PY="$(env_python "$NAME")"
PREFIX="$(dirname "$(dirname "$PY")")"
log "python: $PY"

log "uv layer 1/3: torch stack"
uvi "$PY" torch==2.5.0 torchvision==0.20.0 -c "$C"
uvi "$PY" "setuptools<81" wheel

log "uv layer 2/3: requirements"
PYG_ARGS=()
if [[ -n "$PYG_INDEX" ]]; then
  PYG_ARGS=(-f "$PYG_INDEX")
else
  log "PYG_INDEX empty -> building torch-scatter / torch-cluster from sdist"
  PYG_ARGS=(--no-build-isolation-package torch-scatter --no-build-isolation-package torch-cluster)
fi
uvi "$PY" -r "$ENVS_DIR/3dv-point.requirements.txt" -c "$C" "${PYG_ARGS[@]}"

log "uv layer 3/3: third-party repos"
for r in Pointcept Pointcept-v1.5.1 Concerto sonata; do clone_pinned "$r"; done
# upstream setup.py files import pkg_resources -> build without isolation against setuptools<81
uvi "$PY" --no-build-isolation -e "$THIRD_PARTY/Concerto"
uvi "$PY" --no-build-isolation -e "$THIRD_PARTY/sonata"
uvi "$PY" -e "$ROOT_DIR/common"
# Pointcept is used from its repo root (no setup.py). Notebooks that need v1.5.1 put
# $THIRD_PARTY/Pointcept-v1.5.1 on PYTHONPATH, which takes precedence over this .pth entry.
add_pth "$PY" 3dcheck_point_src.pth "$THIRD_PARTY/Pointcept"

if [[ "$FLASH_ATTN" == "1" ]]; then
  cuda_env "$PREFIX"
  # TODO(verify): flash-attn version with a prebuilt wheel for torch 2.5 / cu12 / cp310
  log "optional: flash-attn"
  uvi "$PY" --no-build-isolation flash-attn -c "$C"
fi
if [[ "$POINTOPS" == "1" ]]; then
  cuda_env "$PREFIX"
  log "optional: pointops (arch $TORCH_CUDA_ARCH_LIST)"
  (cd "$THIRD_PARTY/Pointcept/libs/pointops" && "$PY" setup.py install)
fi

register_kernel "$PY" "$NAME" "3dv-point (torch 2.5.0, cu124)"
log "smoke test"
"$PY" "$ENVS_DIR/smoke_test.py" "$NAME"
