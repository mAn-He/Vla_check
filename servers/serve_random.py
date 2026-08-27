"""Dummy policy server: random action chunks. Validates the whole pipeline
(client <-> websocket protocol <-> JSONL results <-> heatmap) without any model,
GPU, or model venv. Needs only: numpy, msgpack, websockets.

Usage:
    python servers/serve_random.py --port 8000
    curl localhost:8000/healthz   # -> 200 OK
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from base_server import PolicyAdapter, run


class RandomPolicy(PolicyAdapter):
    def __init__(self, action_horizon: int = 10, seed: int = 0) -> None:
        # action_horizon=10 matches the pi05_libero chunk length so the client
        # exercises the same replan path it will use with real models.
        self._horizon = action_horizon
        self._rng = np.random.default_rng(seed)

    def load_model(self) -> None:
        pass  # nothing to load

    def predict(self, obs: dict) -> dict:
        for key in ("observation/image", "observation/wrist_image", "observation/state", "prompt"):
            if key not in obs:
                raise KeyError(f"observation missing required key {key!r}")
        chunk = self._rng.uniform(-0.2, 0.2, size=(self._horizon, 7))
        chunk[:, 6] = -1.0  # keep gripper open (LIBERO no-op gripper value)
        return {"actions": chunk}

    def metadata(self) -> dict:
        return {"model": "random", "ready": True, "action_horizon": self._horizon}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=10)
    args = parser.parse_args()
    run(RandomPolicy(action_horizon=args.action_horizon), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
