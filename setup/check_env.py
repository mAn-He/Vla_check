"""Pre-flight environment check. Prints a table; exits 1 if anything fails.

Checks: GPU presence/name/VRAM, free disk (checkpoints need tens of GB),
MuJoCo rendering backend (egl -> osmesa -> glx), per-model venv/checkout
presence. Runs on the SYSTEM python (no venv needed).
"""

import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENV_ROOT = pathlib.Path(os.environ.get("VENV_ROOT", REPO_ROOT / ".venvs"))
THIRD_PARTY = pathlib.Path(os.environ.get("THIRD_PARTY_DIR", REPO_ROOT / "third_party"))

MIN_FREE_GB = 60  # three checkpoints + datasets comfortably


def check_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "nvidia-smi not found (no GPU?)"
    if out.returncode != 0:
        return False, f"nvidia-smi failed: {out.stderr.strip()[:100]}"
    lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    return bool(lines), "; ".join(lines) if lines else "no GPU listed"


def check_disk():
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / 1e9
    return free_gb >= MIN_FREE_GB, f"{free_gb:.0f} GB free (need >= {MIN_FREE_GB} GB)"


def check_mujoco_gl():
    """Probe egl -> osmesa -> glx in a libero-venv subprocess if available,
    else with the system python (needs mujoco importable)."""
    python = VENV_ROOT / "libero" / "bin" / "python"
    if not python.exists():
        python = pathlib.Path(sys.executable)
    probe = (
        "import mujoco\n"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>')\n"
        "r = mujoco.Renderer(m, 16, 16); r.update_scene(mujoco.MjData(m)); r.render()\n"
    )
    for backend in ("egl", "osmesa", "glx"):
        env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        try:
            res = subprocess.run([str(python), "-c", probe], env=env, capture_output=True, timeout=120)
        except (subprocess.TimeoutExpired, OSError):
            continue
        if res.returncode == 0:
            return True, f"backend '{backend}' works (probed with {python})"
    return False, "no backend of egl/osmesa/glx worked (or mujoco not installed in the probe python)"


def check_path(path: pathlib.Path, what: str):
    return path.exists(), f"{what}: {path} {'exists' if path.exists() else 'MISSING'}"


def main() -> None:
    checks = [
        ("gpu", *check_gpu()),
        ("disk", *check_disk()),
        ("mujoco_gl", *check_mujoco_gl()),
        ("venv:libero", *check_path(VENV_ROOT / "libero", "eval-client venv")),
        ("venv:openvla", *check_path(VENV_ROOT / "openvla", "openvla-oft venv")),
        ("checkout:openpi", *check_path(THIRD_PARTY / "openpi", "openpi repo (own .venv via uv)")),
        ("checkout:Isaac-GR00T", *check_path(THIRD_PARTY / "Isaac-GR00T", "GR00T repo (own .venv via uv)")),
        ("checkout:LIBERO", *check_path(THIRD_PARTY / "LIBERO", "LIBERO repo")),
    ]
    width = max(len(name) for name, _, _ in checks)
    failed = 0
    print(f"{'check':<{width}}  status  detail")
    print("-" * (width + 50))
    for name, ok, detail in checks:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL':<6}  {detail}")
        failed += 0 if ok else 1
    if failed:
        print(f"\n{failed} check(s) failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
