#!/usr/bin/env python3
"""인지/계획/제어 3층 프로브 — 데이터 수집.

얼린 관측(frozen observation) 하나 위에서 전부 측정한다.
롤아웃이 필요 없으므로 저렴하고, 모델 간 조건이 완전히 동일하다.

사용:
    # 서버가 떠 있는 상태에서 (libero venv)
    python probe/collect.py --model groot_n17_libero --suite libero_spatial --task-id 0
    python probe/collect.py --model openvla_oft_libero --suite libero_spatial --task-id 0

출력: probe/data/<model>_<suite>_t<id>.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "client"))

import libero_env                      # noqa: E402
from policy_client import WebsocketClientPolicy  # noqa: E402

OUT = REPO / "probe" / "data"

# ---- 언어 프로브 변형 -------------------------------------------------------
def language_variants(instr: str) -> dict[str, str]:
    syn = (instr.replace("pick up", "grab")
                .replace("place", "put")
                .replace("on the", "onto the"))
    # 대상 교체: 첫 명사구를 다른 물체로 (LIBERO spatial 기준 휴리스틱)
    swapped = instr
    for a, b in (("black bowl", "plate"), ("plate", "black bowl")):
        if a in swapped:
            swapped = swapped.replace(a, b, 1)
            break
    return {
        "original": instr,
        "synonym": syn,
        "target_swap": swapped,
        "empty": "",
        "irrelevant": "the weather is nice today",
    }


# ---- 관측 조작 --------------------------------------------------------------
def mask_patch(img: np.ndarray, i: int, j: int, grid: int) -> np.ndarray:
    """격자 (i,j) 칸을 회색으로 가린 복사본."""
    h, w = img.shape[:2]
    ph, pw = h // grid, w // grid
    out = img.copy()
    out[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = 128
    return out


def add_noise(img: np.ndarray, sigma: float, rng) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0, sigma * 255, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---- 메인 -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="관측을 얼리기 전에 진행할 스텝 수. 0=리셋 직후")
    ap.add_argument("--grid", type=int, default=8, help="가림 격자 크기")
    ap.add_argument("--config", default=str(REPO / "configs" / "eval.yaml"))
    a = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(a.config).read_text())["models"][a.model]
    client = WebsocketClientPolicy(host=cfg["host"], port=cfg["port"])
    print(f"[probe] {a.model} @ {cfg['host']}:{cfg['port']}")

    # ---- 관측 얼리기 ----
    runner = libero_env.LiberoRunner(a.suite) if hasattr(libero_env, "LiberoRunner") \
        else None
    if runner is None:
        raise SystemExit("libero_env.LiberoRunner 를 찾을 수 없다. "
                         "client/libero_env.py 의 클래스명을 확인할 것.")

    env = runner._make_env(a.task_id)
    try:
        obs = env.reset()
        init_states = runner._bench.get_task_init_states(a.task_id)
        obs = env.set_init_state(init_states[0])
        for _ in range(a.warmup_steps):
            obs, *_ = env.step([0.0] * 6 + [-1.0])

        images, state = runner._extract(obs)
        instr = runner.task_spec(a.task_id).name
    finally:
        env.close()

    print(f"[probe] 관측 확보: 카메라 {list(images)} / state dim {state.shape}")
    print(f"[probe] instruction: {instr}")

    rec: dict[str, object] = {
        "model": a.model, "suite": a.suite, "task_id": a.task_id,
        "instruction": instr, "state": state,
        "cameras": list(images), "grid": a.grid,
    }
    for k, v in images.items():
        rec[f"image_{k}"] = v

    def infer(imgs, st, ins) -> np.ndarray:
        return np.asarray(client.infer(imgs, st, ins), dtype=np.float32)

    # ---- 기준 액션 ----
    t0 = time.time()
    base = infer(images, state, instr)
    rec["action_base"] = base
    print(f"[probe] base chunk shape={base.shape} ({time.time()-t0:.2f}s)")

    # ---- P1 가림 민감도 ----
    for cam in images:
        sal = np.zeros((a.grid, a.grid), dtype=np.float32)
        for i in range(a.grid):
            for j in range(a.grid):
                imgs = dict(images)
                imgs[cam] = mask_patch(images[cam], i, j, a.grid)
                sal[i, j] = np.linalg.norm(infer(imgs, state, instr) - base)
            print(f"  [P1:{cam}] row {i+1}/{a.grid}", end="\r")
        rec[f"saliency_{cam}"] = sal
        print(f"  [P1:{cam}] done. max={sal.max():.4f} min={sal.min():.4f}")

    # ---- P2 카메라 기여도 ----
    cam_rel = {}
    for cam in images:
        imgs = dict(images)
        imgs[cam] = np.full_like(images[cam], 128)
        cam_rel[cam] = float(np.linalg.norm(infer(imgs, state, instr) - base))
    rec["camera_reliance"] = json.dumps(cam_rel)
    print(f"  [P2] {cam_rel}")

    # ---- P3 / L2 언어 프로브 ----
    variants = language_variants(instr)
    lang_actions, lang_dist = {}, {}
    for name, text in variants.items():
        act = infer(images, state, text)
        lang_actions[name] = act
        lang_dist[name] = float(np.linalg.norm(act - base))
        rec[f"action_lang_{name}"] = act
    rec["language_variants"] = json.dumps(variants, ensure_ascii=False)
    rec["language_dist"] = json.dumps(lang_dist)
    print(f"  [P3] {json.dumps(lang_dist, indent=None)}")

    # ---- C3 노이즈 민감도 ----
    rng = np.random.default_rng(0)
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.10]
    sens = []
    for s in sigmas:
        ds = []
        for _ in range(3 if s > 0 else 1):
            imgs = {k: add_noise(v, s, rng) for k, v in images.items()}
            ds.append(float(np.linalg.norm(infer(imgs, state, instr) - base)))
        sens.append(float(np.mean(ds)))
    rec["noise_sigmas"] = np.array(sigmas, dtype=np.float32)
    rec["noise_sensitivity"] = np.array(sens, dtype=np.float32)
    print(f"  [C3] sigma={sigmas} -> dist={[round(x,4) for x in sens]}")
    if sens[0] > 1e-6:
        print(f"  [C3] 주의: sigma=0 에서 거리 {sens[0]:.6f} -> 비결정적 출력")

    # ---- L3 재계획 안정성 ----
    env = runner._make_env(a.task_id)
    try:
        obs = env.reset()
        obs = env.set_init_state(init_states[0])
        chunks, steps_done = [], []
        for k in range(4):
            imgs_k, st_k = runner._extract(obs)
            ch = infer(imgs_k, st_k, instr)
            chunks.append(ch)
            steps_done.append(k)
            n_exec = max(1, len(ch) // 2)
            for t in range(n_exec):
                obs, *_ = env.step(ch[t].tolist())
        rec["replan_chunks"] = np.stack(
            [c[:min(len(c) for c in chunks)] for c in chunks])
        rec["replan_offsets"] = np.array(
            [i * max(1, len(chunks[0]) // 2) for i in range(len(chunks))])
        print(f"  [L3] {len(chunks)} chunks 수집")
    finally:
        env.close()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.model}_{a.suite}_t{a.task_id}.npz"
    np.savez_compressed(path, **rec)
    print(f"\n[saved] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
