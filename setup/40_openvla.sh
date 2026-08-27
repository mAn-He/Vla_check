#!/usr/bin/env bash
# OpenVLA-OFT server environment. Mirrors openvla-oft SETUP.md + LIBERO.md:
# python 3.10, pip install -e ., flash-attn 2.5.5 (--no-build-isolation),
# the custom transformers v4.40.1 fork, and LIBERO installed into the SAME env
# (the OFT eval helpers import libero at module load). Idempotent.
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-$(pwd)/.venvs}"
export THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$(pwd)/third_party}"
VENV="${VENV_ROOT}/openvla"
OFT_DIR="${OFT_DIR:-${THIRD_PARTY_DIR}/openvla-oft}"
LIBERO_DIR="${THIRD_PARTY_DIR}/LIBERO"

if [ ! -d "${OFT_DIR}" ]; then
  git clone https://github.com/moojink/openvla-oft.git "${OFT_DIR}"
fi
if [ ! -d "${LIBERO_DIR}" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi

if [ ! -d "${VENV}" ]; then
  uv venv --python 3.10 "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Paper-pinned versions: PyTorch 2.2.0, custom transformers 4.40.1 fork,
# flash-attn 2.5.5 ("Please stick to these package versions", LIBERO.md).
uv pip install torch==2.2.0 torchvision==0.17.0
uv pip install -e "${OFT_DIR}"
uv pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
uv pip install packaging ninja
uv pip install "flash-attn==2.5.5" --no-build-isolation

# LIBERO into the same env (per openvla-oft LIBERO.md).
uv pip install -e "${LIBERO_DIR}"
uv pip install -r "${OFT_DIR}/experiments/robot/libero/libero_requirements.txt"

# Deps for our websocket server (servers/serve_openvla.py).
uv pip install websockets msgpack

python -c "import prismatic, libero; print('openvla-oft + libero importable')"
echo "[OK] openvla server env ready (${VENV}, OFT_DIR=${OFT_DIR})"
