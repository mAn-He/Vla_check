#!/usr/bin/env bash
# Common prerequisites: apt packages, uv, shared env vars, cache directories.
# Idempotent; safe to re-run. Works on Colab / RunPod / EC2 (Debian/Ubuntu).
set -euo pipefail

# --- overridable locations (point these at Drive / volume / EBS mounts) ------
export VENV_ROOT="${VENV_ROOT:-$(pwd)/.venvs}"
export CKPT_CACHE_DIR="${CKPT_CACHE_DIR:-$(pwd)/checkpoints}"
export HF_HOME="${HF_HOME:-${CKPT_CACHE_DIR}/hf}"
export THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$(pwd)/third_party}"
mkdir -p "${VENV_ROOT}" "${CKPT_CACHE_DIR}" "${HF_HOME}" "${THIRD_PARTY_DIR}" results

# --- apt packages ------------------------------------------------------------
# Headless MuJoCo rendering needs EGL and/or OSMesa; ffmpeg for rollout videos.
# TODO(verify): trim to the minimal set on a clean image; this list is the
# union of what LIBERO/robosuite headless setups commonly need.
APT_PKGS=(git git-lfs curl build-essential cmake ffmpeg
          libegl1 libgles2 libgl1 libosmesa6-dev libglew-dev libglfw3 patchelf)
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  ${SUDO} apt-get update -y
  DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get install -y "${APT_PKGS[@]}"
else
  echo "[warn] apt-get not found; install equivalents of: ${APT_PKGS[*]}" >&2
fi

# --- uv (used to create per-model venvs with pinned Python versions) --------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

git lfs install --skip-repo

# --- verification ------------------------------------------------------------
uv --version
echo "[OK] common ready (VENV_ROOT=${VENV_ROOT}, CKPT_CACHE_DIR=${CKPT_CACHE_DIR}, HF_HOME=${HF_HOME})"
