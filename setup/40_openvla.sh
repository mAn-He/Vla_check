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

# --- RunPod 실행 중 발견된 누락 항목 (2026-08-28) ---

# openvla venv 도 LIBERO 를 import 한다.
# serve_openvla.py -> experiments.robot.libero.run_libero_eval 이
# 최상단에서 `from libero.libero import benchmark` 를 호출하기 때문.
cd "$REPO_ROOT/third_party/LIBERO"
uv pip install -e .
cd "$REPO_ROOT"

# editable 설치가 MAPPING 을 비운 채 끝나므로 .pth 를 직접 만든다
echo "$REPO_ROOT/third_party/LIBERO" \
  > "$VENV/lib/python3.10/site-packages/libero_path.pth"

# 첫 import 시 데이터셋 경로를 묻는 대화형 프롬프트가 스크립트를 멈춘다
echo "N" | python -c "from libero.libero import benchmark" 2>/dev/null || true

# protobuf 3자 충돌 회피:
#   tensorflow_datasets -> tensorflow_metadata 는 protobuf runtime_version(5.27+) 요구
#   tensorflow 본체는 MessageFactory.GetPrototype(6.x 에서 제거) 요구
#   -> tensorflow-metadata 를 낮춰 runtime_version 요구 자체를 없앤다
uv pip install "tensorflow-metadata==1.13.1"

# RunPod 이미지가 HF_HUB_ENABLE_HF_TRANSFER=1 을 켜두지만 패키지는 없다
uv pip install hf_transfer || echo "[!] hf_transfer 설치 실패 -> HF_HUB_ENABLE_HF_TRANSFER=0 필요"

python -c "import tensorflow_datasets; from libero.libero import benchmark; print('[OK] openvla deps 확인')"
