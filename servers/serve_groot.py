"""GR00T N1.7 policy server: websocket front-end + adapter to NVIDIA's official
GR00T inference server (ZeroMQ). Runs inside the GR00T venv (needs `gr00t`).

We do not reimplement GR00T loading. Start NVIDIA's own server first
(command verified against Isaac-GR00T gr00t/eval/run_gr00t_server.py):

    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.7-LIBERO \
        --embodiment-tag LIBERO_PANDA \
        --use-sim-policy-wrapper \
        --port 5555

`--use-sim-policy-wrapper` makes the server accept the flat "video.*"/"state.*"
observation format used by GR00T's own sim environments (Gr00tSimPolicyWrapper).

This adapter then translates one harness observation into the exact flat format
GR00T's own LIBERO env produces (verified against
gr00t/eval/sim/LIBERO/libero_env.py::LiberoEnv._process_observation and
Gr00tSimPolicyWrapper.check_observation):

    video.image        uint8   (B=1, T, 256, 256, 3)   agentview, rotated 180°
    video.wrist_image  uint8   (B=1, T, 256, 256, 3)   eye-in-hand, rotated 180°
    state.x/y/z        float32 (1, T, 1)               eef position
    state.roll/pitch/yaw float32 (1, T, 1)             eef quat -> axis-angle
    state.gripper      float32 (1, T, 2)               gripper qpos
    annotation.human.action.task_description  [prompt] (batch of 1 string)

and converts the returned flat action dict (action.x ... action.gripper, each
(B, T, 1)) into an env-ready (T, 7) chunk, applying the same two gripper
transforms GR00T's LIBERO env applies before env.step:
normalize_gripper_action(binarize=True) then invert_gripper_action.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from base_server import PolicyAdapter, run

ACTION_KEYS = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]
STATE_SLICES = {"x": (0, 1), "y": (1, 2), "z": (2, 3), "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6), "gripper": (6, 8)}


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """[0,1] -> [-1,+1] on the last dim. Copied from GR00T's LIBERO env
    (gr00t/eval/sim/LIBERO/libero_env.py); identical helper exists in
    openvla-oft experiments/robot/robot_utils.py."""
    out = action.copy()
    out[..., -1] = 2.0 * (out[..., -1] - 0.0) / (1.0 - 0.0) - 1.0
    if binarize:
        out[..., -1] = np.sign(out[..., -1])
    return out


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip gripper sign (RLDS 0=close/1=open -> env -1=open/+1=close). Same
    provenance as normalize_gripper_action."""
    out = action.copy()
    out[..., -1] *= -1.0
    return out


class GrootAdapter(PolicyAdapter):
    def __init__(self, backend_host: str, backend_port: int, obs_horizon: int) -> None:
        self._backend_host = backend_host
        self._backend_port = backend_port
        # Number of observation timesteps T the checkpoint's modality config
        # expects (len(delta_indices)).
        # TODO(verify): read the actual value from the nvidia/GR00T-N1.7-LIBERO
        # checkpoint's processor/modality config once it can be downloaded; 1 is
        # the plain "current frame only" assumption. If the server asserts a
        # different horizon, the assertion message from Gr00tSimPolicyWrapper
        # will name the expected value — set --obs-horizon accordingly.
        self._obs_horizon = obs_horizon
        self._client = None

    def load_model(self) -> None:
        # Official client class, verified against gr00t/policy/server_client.py:
        # PolicyClient(host="localhost", port=5555, timeout_ms=15000, ...) with
        # the same get_action(observation, options) interface as a local policy.
        from gr00t.policy.server_client import PolicyClient

        self._client = PolicyClient(host=self._backend_host, port=self._backend_port)

    def predict(self, obs: dict) -> dict:
        img = np.asarray(obs["observation/image"], dtype=np.uint8)
        wrist = np.asarray(obs["observation/wrist_image"], dtype=np.uint8)
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        prompt = str(obs["prompt"])
        if state.shape != (8,):
            raise ValueError(f"expected 8-dim state, got {state.shape}")

        T = self._obs_horizon

        def tile_video(x):  # (H, W, 3) -> (1, T, H, W, 3)
            return np.repeat(x[None, None], T, axis=1)

        def tile_state(lo, hi):  # -> (1, T, hi-lo) float32
            return np.repeat(state[None, None, lo:hi], T, axis=1)

        flat_obs = {
            "video.image": tile_video(img),
            "video.wrist_image": tile_video(wrist),
            # The wrapper reads the language key named by the checkpoint's
            # modality config; GR00T's LIBERO env emits
            # "annotation.human.action.task_description". "task" is the other
            # name Gr00tSimPolicyWrapper knows; sending both is harmless.
            "annotation.human.action.task_description": [prompt],
            "task": [prompt],
        }
        for name, (lo, hi) in STATE_SLICES.items():
            flat_obs[f"state.{name}"] = tile_state(lo, hi)

        action_dict, _info = self._client.get_action(flat_obs)

        # (B, T, 1) per key -> (T, 7), same concat order as GR00T's LIBERO env.
        parts = [np.asarray(action_dict[k], dtype=np.float32)[0] for k in ACTION_KEYS]
        chunk = np.concatenate(parts, axis=-1)
        chunk = invert_gripper_action(normalize_gripper_action(chunk, binarize=True))
        return {"actions": chunk}

    def metadata(self) -> dict:
        return {"model": "groot_n17_libero", "ready": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--backend-host", default="localhost", help="host of the official GR00T ZMQ server")
    parser.add_argument("--backend-port", type=int, default=5555, help="port of the official GR00T ZMQ server")
    parser.add_argument("--obs-horizon", type=int, default=1, help="observation timesteps T expected by the checkpoint")
    args = parser.parse_args()
    run(GrootAdapter(args.backend_host, args.backend_port, args.obs_horizon), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
