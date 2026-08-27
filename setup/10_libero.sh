#!/usr/bin/env bash
# Eval-client environment: Python 3.8 venv with LIBERO (source install) and the
# openpi client package. Mirrors the official installs:
#   - LIBERO README: python 3.8.13, pip install -r requirements.txt,
#     torch 1.11.0+cu113 from the cu113 extra index, pip install -e .
#   - openpi examples/libero: a separate python 3.8 venv with openpi-client
# Idempotent; safe to re-run.
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-$(pwd)/.venvs}"
export THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$(pwd)/third_party}"
VENV="${VENV_ROOT}/libero"
LIBERO_DIR="${THIRD_PARTY_DIR}/LIBERO"
# CPU wheels exist for torch 1.11 too; cu113 is what LIBERO's README pins.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu113}"

if [ ! -d "${LIBERO_DIR}" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi

if [ ! -d "${VENV}" ]; then
  uv venv --python 3.8 "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

uv pip install -r "${LIBERO_DIR}/requirements.txt"
uv pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url "${TORCH_INDEX}"
uv pip install -e "${LIBERO_DIR}"

# Official protocol client (+ its websockets/msgpack deps) straight from openpi.
uv pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"

# Client-side extras: config parsing + heatmaps.
uv pip install pyyaml matplotlib seaborn pandas

python -c "import libero; print(libero.__file__)"
python -c "import openpi_client; print(openpi_client.__file__)"
echo "[OK] libero eval-client env ready (${VENV})"
