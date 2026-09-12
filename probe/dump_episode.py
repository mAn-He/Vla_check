#!/usr/bin/env python3
"""에피소드 하나를 굴리면서 매 스텝 관측을 통째로 저장한다.

이후 compare.py 가 이 관측을 여러 모델에 **똑같이** 먹여서 반응을 비교한다.
한 번만 뽑아두면 모델을 몇 개 추가하든 동일 입력으로 비교할 수 있다.

사용 (서버가 떠 있는 상태, libero venv):
    python probe/dump_episode.py --model groot_n17_libero --suite libero_spatial --task-id 0

출력: probe/episodes/<suite>_t<id>_by-<model>.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "client"))
OUT = REPO / "probe" / "episodes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="롤아웃을 굴릴 정책")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--config", default=str(REPO / "configs" / "eval.yaml"))
    a = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(a.config).read_text())["models"][a.model]
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host=cfg["host"], port=cfg["port"])
    print(f"[dump] rollout policy = {a.model} @ {cfg['host']}:{cfg['port']}")

    from libero_env import LiberoSuite, extract
    suite = LiberoSuite(a.suite)
    env, instruction = suite.make_env(a.task_id, seed=a.seed)
    print(f"[dump] instruction: {instruction}")

    frames = []
    done = False
    try:
        env.reset()
        init = suite.init_states(a.task_id)
        obs = env.set_init_state(init[a.episode_idx % len(init)])

        step = 0
        while step < a.max_steps and not done:
            o = extract(obs)
            result = client.infer({
                "observation/image": o["image"],
                "observation/wrist_image": o["wrist_image"],
                "observation/state": o["state"],
                "prompt": instruction,
            })
            chunk = np.asarray(result["actions"], dtype=np.float32)

            frames.append({
                "step": step,
                "image": o["image"], "wrist_image": o["wrist_image"],
                "state": np.asarray(o["state"], dtype=np.float32),
                "chunk": chunk,
            })

            for t in range(max(1, len(chunk)//4)):
                obs, _, done, _ = env.step(chunk[t].tolist())
                step += 1
                if done or step >= a.max_steps:
                    break
            print(f"  step {step}/{a.max_steps}  frames={len(frames)}", end="\r")
    finally:
        env.close()

    print(f"\n[dump] {len(frames)} frames, success={done}, steps={frames[-1]['step']}")

    OUT.mkdir(parents=True, exist_ok=True)
    L = min(len(f["chunk"]) for f in frames)
    tag = f"{a.suite}_t{a.task_id}_by-{a.model}"
    np.savez_compressed(
        OUT / f"{tag}.npz",
        suite=a.suite, task_id=a.task_id, episode_idx=a.episode_idx, seed=a.seed,
        instruction=instruction, rollout_policy=a.model,
        n_frames=len(frames), success=bool(done),
        images=np.stack([f["image"] for f in frames]),
        wrist_images=np.stack([f["wrist_image"] for f in frames]),
        states=np.stack([f["state"] for f in frames]),
        chunks=np.stack([f["chunk"][:L] for f in frames]),
        steps=np.array([f["step"] for f in frames]),
    )
    p = OUT / f"{tag}.npz"
    print(f"[saved] {p}  ({p.stat().st_size/1e6:.1f} MB)")

    if a.save_png:
        from PIL import Image
        d = OUT / f"{tag}_frames"; d.mkdir(exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f["image"]).save(d / f"{i:03d}_agentview.png")
            Image.fromarray(f["wrist_image"]).save(d / f"{i:03d}_wrist.png")
        print(f"[saved] PNG -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
