#!/usr/bin/env bash
# Shared helpers for setup_*.sh. Source this file; do not execute it.
#
# Environment variables (all optional):
#   CONDA_FRONTEND   mamba | micromamba | conda   (default: first one found in that order)
#   ENV_PREFIX       create the env at this path instead of by name
#   THIRD_PARTY      where pinned repos are cloned (default: <repo>/third_party)
#   REGISTER_KERNEL  1 (default) registers the env as a Jupyter kernel, 0 skips

set -euo pipefail

ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$ENVS_DIR/.." && pwd)"
export THIRD_PARTY="${THIRD_PARTY:-$ROOT_DIR/third_party}"
REGISTER_KERNEL="${REGISTER_KERNEL:-1}"

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

pick_frontend() {
  if [[ -n "${CONDA_FRONTEND:-}" ]]; then FE="$CONDA_FRONTEND"; return; fi
  for c in mamba micromamba conda; do
    if command -v "$c" >/dev/null 2>&1; then FE="$c"; return; fi
  done
  echo "No conda frontend found (mamba / micromamba / conda)." >&2; exit 1
}

# create_env <yml> <name> : skip if the env already exists
create_env() {
  local yml="$1" name="$2"
  local target
  if [[ -n "${ENV_PREFIX:-}" ]]; then target=(-p "$ENV_PREFIX"); else target=(-n "$name"); fi
  if [[ -x "$(env_python "$name" 2>/dev/null || true)" ]]; then
    log "env '$name' already exists, skipping conda layer"; return
  fi
  log "creating conda layer for '$name' with $FE"
  if [[ "$(basename "$FE")" == "micromamba" ]]; then
    "$FE" create -y -f "$yml" "${target[@]}"
  else
    "$FE" env create -y -f "$yml" "${target[@]}"
  fi
}

# env_python <name> : print the env's python path
env_python() {
  local name="$1"
  if [[ -n "${ENV_PREFIX:-}" ]]; then echo "$ENV_PREFIX/bin/python"; return; fi
  "$FE" run -n "$name" python -c 'import sys; print(sys.executable)'
}

# uvi <python> <args...> : uv pip install into the given interpreter
uvi() {
  local py="$1"; shift
  "$(dirname "$py")/uv" pip install --python "$py" "$@"
}

# clone_pinned <name> : clone a repo listed in third_party.lock and check out its pinned commit/tag
clone_pinned() {
  local name="$1" line url ref dir
  line="$(grep -E "^$name[[:space:]]" "$ENVS_DIR/third_party.lock")" || { echo "unknown repo $name" >&2; exit 1; }
  ref="$(awk '{print $3}' <<<"$line")"; url="$(awk '{print $4}' <<<"$line")"
  dir="$THIRD_PARTY/$name"
  mkdir -p "$THIRD_PARTY"
  if [[ ! -d "$dir/.git" ]]; then
    log "clone $name @ $ref"
    git clone --filter=blob:none --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet --tags origin || true
  git -C "$dir" checkout --quiet "$ref"
  git -C "$dir" submodule update --init --recursive --quiet
}

# add_pth <python> <file-name> <dir...> : put plain source dirs on sys.path via a .pth file
add_pth() {
  local py="$1" fname="$2"; shift 2
  local site
  site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf '%s\n' "$@" > "$site/$fname"
  log "wrote $site/$fname"
}

# cuda_env <env prefix> : point torch extension builds at the conda-forge CUDA toolkit
cuda_env() {
  local prefix="$1"
  export CUDA_HOME="$prefix"
  # conda-forge CUDA 12 puts headers under targets/; torch.utils.cpp_extension only looks in $CUDA_HOME/include
  export CPATH="$prefix/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # RTX 3060 = sm86
  # mmcv-style setup.py files fall back to CPU-only builds when no GPU is visible (e.g. docker build)
  export FORCE_CUDA=1
}

register_kernel() {
  local py="$1" name="$2" display="$3"
  if [[ "$REGISTER_KERNEL" == "1" ]]; then
    "$py" -m ipykernel install --user --name "$name" --display-name "$display"
  fi
}
