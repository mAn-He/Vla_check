#!/usr/bin/env bash
# GR00T N1.7 server environment. Isaac-GR00T (main branch = N1.7) is
# uv-managed: Python >=3.12,<3.13, torch 2.9.0, flash-attn 2.8.3 pinned to
# prebuilt wheels via [tool.uv.sources] — `uv sync` installs them; no
# source build. (README FAQ: the "Installing flash-attn..." message on every
# `uv run` is uv re-validating the cached URL-pinned wheel, ~2-3s, not a
# rebuild.) Idempotent; safe to re-run.
set -euo pipefail

export THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$(pwd)/third_party}"
GROOT_DIR="${GROOT_DIR:-${THIRD_PARTY_DIR}/Isaac-GR00T}"

if [ ! -d "${GROOT_DIR}" ]; then
  git clone https://github.com/NVIDIA/Isaac-GR00T.git "${GROOT_DIR}"
fi

cd "${GROOT_DIR}"
uv sync

# Deps for our websocket adapter (servers/serve_groot.py) inside this env.
uv pip install websockets msgpack

# NOTE (required, from the GR00T README): the VLM backbone
# nvidia/Cosmos-Reason2-2B is GATED on Hugging Face and is loaded on first use
# by every GR00T checkpoint. Request access on its model page, then:
#   huggingface-cli login   (or export HF_TOKEN=...)
#
# Server start (documented for reference; do NOT start it here):
#   cd ${GROOT_DIR} && uv run python gr00t/eval/run_gr00t_server.py \
#     --model-path nvidia/GR00T-N1.7-LIBERO --embodiment-tag LIBERO_PANDA \
#     --use-sim-policy-wrapper --port 5555
# TODO(verify): the GR00T-N1.7-LIBERO HF repo stores the model in a nested
# folder (see Isaac-GR00T scripts/deployment/README.md); if loading by repo id
# fails, download and pass the local subfolder path as --model-path.

uv run python -c "import gr00t; print('GR00T installed successfully')"
echo "[OK] groot server env ready (${GROOT_DIR})"
