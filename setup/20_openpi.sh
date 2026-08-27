#!/usr/bin/env bash
# π0.5 server environment. openpi is uv-managed: `uv sync` creates the repo's
# own .venv (official install: GIT_LFS_SKIP_SMUDGE=1 uv sync, then run scripts
# with `uv run`). Idempotent; safe to re-run.
set -euo pipefail

export THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$(pwd)/third_party}"
OPENPI_DIR="${OPENPI_DIR:-${THIRD_PARTY_DIR}/openpi}"

if [ ! -d "${OPENPI_DIR}" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}"
fi

cd "${OPENPI_DIR}"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Server start (documented for reference; do NOT start it here):
#   cd ${OPENPI_DIR} && uv run scripts/serve_policy.py --env LIBERO --port 8000
# The pi05_libero checkpoint (gs://openpi-assets/checkpoints/pi05_libero) is
# downloaded automatically on first use.

uv run python -c "import openpi; print(openpi.__file__)"
echo "[OK] openpi server env ready (${OPENPI_DIR})"
