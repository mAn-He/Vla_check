"""LIBERO environment wrapper for the eval client.

Every LIBERO call in this file is verified against the LIBERO repository
(master branch) and/or the official eval harnesses of the three models; see
DOC_TRACE.md for the URL of each. In particular:

  - benchmark.get_benchmark_dict() -> {suite_name: benchmark class}
  - suite = benchmark_dict[name]();  suite.n_tasks;  suite.get_task(i)
  - task fields: name, language, problem_folder, bddl_file
  - suite.get_task_init_states(i) -> array of init states (torch.load'ed)
  - OffScreenRenderEnv(bddl_file_name=..., camera_heights=..., camera_widths=...)
  - env.seed(s); env.reset(); obs = env.set_init_state(init_states[k])
  - obs keys: agentview_image, robot0_eye_in_hand_image, robot0_eef_pos,
    robot0_eef_quat, robot0_gripper_qpos
  - env.step(a7) -> (obs, reward, done, info); done becoming True within the
    step budget is the success signal used by openpi's examples/libero/main.py,
    openvla-oft's run_libero_eval.py, and LIBERO's own libero/lifelong/metric.py
    (env.check_success() exposes the same underlying signal).

max_steps: LIBERO itself has no per-suite step budget (its own eval config uses
a single global max_steps=600). The per-suite values below are the ones used by
BOTH openpi's and openvla-oft's official LIBERO harnesses (identical numbers,
including the comments) — using them keeps our results comparable to published
numbers. Do not change them.
"""

import math
import os
import subprocess
import sys

import numpy as np

# Same values + provenance comments as openpi examples/libero/main.py and
# openvla-oft experiments/robot/libero/run_libero_eval.py.
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}

# openpi/OFT convention: LIBERO renders at 256; models resize themselves (or
# the client resizes for openpi, see run_eval.py).
ENV_RESOLUTION = 256

# openpi/OFT convention: no-op action while objects settle after reset.
DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10

# Task counts per suite, from LIBERO's libero_suite_task_map.py. Used only by
# --dry-run when LIBERO is not importable (real runs always ask LIBERO).
STATIC_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

_GL_BACKENDS = ("egl", "osmesa", "glx")


def pick_mujoco_gl(verbose: bool = True) -> str:
    """Choose a working MUJOCO_GL backend (egl -> osmesa -> glx fallback).

    MUJOCO_GL is read when mujoco initializes, so this must run BEFORE any
    libero/robosuite import. Each candidate is probed in a subprocess; the
    first one that can create a headless GL context wins and is exported into
    os.environ for this process. (openpi's docker setup uses egl with a
    documented glx fallback; osmesa is this harness's extra middle step.)
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    probe = (
        "import mujoco, numpy as np\n"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>')\n"
        "r = mujoco.Renderer(m, 16, 16); r.update_scene(mujoco.MjData(m)); r.render()\n"
    )
    for backend in _GL_BACKENDS:
        env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        try:
            res = subprocess.run(
                [sys.executable, "-c", probe], env=env, capture_output=True, timeout=120
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if res.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            os.environ["PYOPENGL_PLATFORM"] = backend
            if verbose:
                print(f"[libero_env] MUJOCO_GL={backend}", flush=True)
            return backend
    raise RuntimeError(
        "No working MuJoCo GL backend (tried egl, osmesa, glx). "
        "Install libegl1 / libosmesa6 or run under X."
    )


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion -> axis-angle. Copied (via openpi examples/libero/main.py)
    from robosuite transform_utils; both openpi and OFT build the 8-dim state
    with exactly this conversion."""
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def extract(obs: dict) -> dict:
    """LIBERO raw obs -> harness observation pieces.

    Images are rotated 180° — the shared convention of all three official
    harnesses ("rotate 180 degrees to match train preprocessing" in both
    openpi's and OFT's LIBERO utils; GR00T's LIBERO env does the same flip).
    State layout [eef_pos(3), eef axis-angle(3), gripper_qpos(2)] is likewise
    identical across all three.
    """
    return {
        "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


class LiberoSuite:
    """Thin wrapper over LIBERO's benchmark API. Import happens lazily so that
    --dry-run works on machines without LIBERO installed."""

    def __init__(self, suite_name: str):
        pick_mujoco_gl()
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite_name not in benchmark_dict:
            raise KeyError(f"unknown task suite {suite_name!r}; available: {sorted(benchmark_dict)}")
        self.name = suite_name
        self.suite = benchmark_dict[suite_name]()
        self.n_tasks = self.suite.n_tasks

    def task(self, task_id: int):
        return self.suite.get_task(task_id)

    def init_states(self, task_id: int):
        return self.suite.get_task_init_states(task_id)

    def make_env(self, task_id: int, seed: int, resolution: int = ENV_RESOLUTION):
        """Create the env exactly like openpi/OFT do (both call env.seed right
        after construction: 'seed seems to affect object positions even when
        using fixed initial state')."""
        import pathlib

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task = self.task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
        env.seed(seed)
        return env, str(task.language)
