#!/usr/bin/env python3
"""저장된 에피소드의 관측을 여러 모델에 똑같이 먹여 반응을 비교한다.

환경을 굴리지 않고 저장된 관측을 재생하므로 **모든 모델이 완전히 동일한 입력**을
받는다. 롤아웃 비교의 고질적 문제(모델마다 다른 상태를 지나가 입력이 달라짐)를 피한다.

사용:
    # 서버를 하나씩 띄우면서
    python probe/compare.py --episode probe/episodes/libero_spatial_t0_by-groot_n17_libero.npz --model groot_n17_libero
    python probe/compare.py --episode <같은 파일> --model openvla_oft_libero
    # 다 모이면
    python probe/compare.py --render
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "client"))
OUT = REPO / "probe" / "compare"
FIG = OUT / "figures"

LANG_ORDER = ["original", "synonym", "target_swap", "empty", "irrelevant"]
COLORS = {"groot": "#2E86AB", "openvla": "#E8743B", "pi05": "#6A4C93"}
ACT_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def color_for(m):
    return next((v for k, v in COLORS.items() if k in m.lower()), "#555")


def short(m):
    return (m.replace("_libero", "").replace("_oft", "-OFT")
             .replace("groot_n17", "GR00T-N1.7").replace("openvla", "OpenVLA")
             .replace("pi05", "pi0.5"))


def lang_variants(instr: str) -> dict[str, str]:
    syn = (instr.replace("pick up", "grab").replace("place", "put")
                .replace(" on the ", " onto the "))
    swap = instr
    for x, y in (("black bowl", "plate"), ("plate", "black bowl")):
        if x in swap:
            swap = swap.replace(x, y, 1)
            break
    return {"original": instr, "synonym": syn, "target_swap": swap,
            "empty": "", "irrelevant": "the weather is nice today"}


def mask_patch(img, i, j, grid):
    h, w = img.shape[:2]
    ph, pw = h // grid, w // grid
    out = img.copy()
    out[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = 128
    return out


# =============================================================== collect
def collect(a) -> int:
    ep = np.load(a.episode, allow_pickle=True)
    instr = str(ep["instruction"])
    n = int(ep["n_frames"])
    print(f"[cmp] episode : {pathlib.Path(a.episode).name}")
    print(f"[cmp] frames  : {n}")
    print(f"[cmp] prompt  : {instr}")

    cfg = yaml.safe_load(pathlib.Path(a.config).read_text())["models"][a.model]
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host=cfg["host"], port=cfg["port"])
    print(f"[cmp] model   : {a.model} @ {cfg['host']}:{cfg['port']}")

    def infer(k, image=None, wrist=None, prompt=None) -> np.ndarray:
        r = client.infer({
            "observation/image": ep["images"][k] if image is None else image,
            "observation/wrist_image": ep["wrist_images"][k] if wrist is None else wrist,
            "observation/state": ep["states"][k],
            "prompt": instr if prompt is None else prompt,
        })
        return np.asarray(r["actions"], dtype=np.float32)

    idx = np.unique(np.linspace(0, n - 1, min(a.n_frames, n)).astype(int))
    print(f"[cmp] 비교 프레임 {len(idx)}개: {idx.tolist()}")

    rec = {"model": a.model, "episode": pathlib.Path(a.episode).stem,
           "instruction": instr, "frame_idx": idx,
           "states": ep["states"][idx],
           "images": ep["images"][idx], "wrist_images": ep["wrist_images"][idx]}

    # 1) 프레임별 액션
    acts = []
    for c, k in enumerate(idx):
        acts.append(infer(int(k)))
        print(f"  [action] {c+1}/{len(idx)}", end="\r")
    L = min(len(x) for x in acts)
    rec["actions"] = np.stack([x[:L] for x in acts])
    print(f"\n[cmp] actions {rec['actions'].shape}")

    # 2) 언어 프로브 (대표 프레임)
    kk = int(idx[len(idx) // 3])
    rec["probe_frame"] = kk
    base = infer(kk)
    rec["lang_base"] = base
    variants = lang_variants(instr)
    dist = {}
    for name, text in variants.items():
        x = infer(kk, prompt=text)[:len(base)]
        rec[f"lang_{name}"] = x
        dist[name] = float(np.linalg.norm(x - base))
    rec["lang_variants"] = json.dumps(variants, ensure_ascii=False)
    rec["lang_dist"] = json.dumps(dist)
    print(f"[cmp] language: {json.dumps(dist)}")

    # 3) 카메라 기여도
    blank = np.full_like(ep["images"][kk], 128)
    rec["cam_reliance"] = json.dumps({
        "agentview": float(np.linalg.norm(infer(kk, image=blank)[:len(base)] - base)),
        "wrist": float(np.linalg.norm(infer(kk, wrist=blank)[:len(base)] - base)),
    })
    print(f"[cmp] camera  : {rec['cam_reliance']}")

    # 4) 가림 민감도
    if a.grid > 0:
        for cam, arr in (("agentview", ep["images"]), ("wrist", ep["wrist_images"])):
            img = arr[kk]
            sal = np.zeros((a.grid, a.grid), np.float32)
            for i in range(a.grid):
                for j in range(a.grid):
                    kw = {"image" if cam == "agentview" else "wrist":
                          mask_patch(img, i, j, a.grid)}
                    sal[i, j] = np.linalg.norm(infer(kk, **kw)[:len(base)] - base)
                print(f"  [saliency:{cam}] {i+1}/{a.grid}", end="\r")
            rec[f"saliency_{cam}"] = sal
            print(f"\n[cmp] saliency {cam}: max={sal.max():.4f} min={sal.min():.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{pathlib.Path(a.episode).stem}__{a.model}.npz"
    np.savez_compressed(p, **rec)
    print(f"[saved] {p}")
    return 0


# =============================================================== render
def render() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for p in sorted(OUT.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        runs[str(d["model"])] = {k: d[k] for k in d.files}
    if not runs:
        raise SystemExit(f"비교 데이터 없음: {OUT}/*.npz")
    models = list(runs)
    print(f"[viz] {len(models)} models: {models}")
    FIG.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] {FIG/name}.png")

    # 1. 액션 불일치 (핵심)
    if len(models) >= 2:
        a, b = models[0], models[1]
        A, B = runs[a]["actions"], runs[b]["actions"]
        T, L = min(len(A), len(B)), min(A.shape[1], B.shape[1])
        diff = np.linalg.norm(A[:T, :L] - B[:T, :L], axis=2)

        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        im = ax[0].imshow(diff.T, aspect="auto", cmap="magma")
        ax[0].set_xlabel("frame (episode progress →)")
        ax[0].set_ylabel("step within chunk")
        ax[0].set_title(f"||Δaction||   {short(a)}  vs  {short(b)}")
        fig.colorbar(im, ax=ax[0])
        ax[1].plot(diff.mean(axis=1), "-o", ms=4, color="#333")
        ax[1].set_xlabel("frame"); ax[1].set_ylabel("mean ||Δaction||")
        ax[1].set_title("where they disagree most"); ax[1].grid(alpha=.3)
        save(fig, "01_action_disagreement")
        w = int(diff.mean(axis=1).argmax())
        print(f"[viz] 최대 불일치 프레임: {w} "
              f"(episode step {int(runs[a]['frame_idx'][w])})")

    # 2. 액션 시계열 + jerk
    fig, axes = plt.subplots(2, 4, figsize=(17, 6)); axes = axes.ravel()
    for d in range(7):
        for m in models:
            axes[d].plot(runs[m]["actions"][:, 0, d], "-o", ms=3,
                         color=color_for(m), label=short(m))
        axes[d].set_title(ACT_NAMES[d], fontsize=10); axes[d].grid(alpha=.3)
        if d == 0:
            axes[d].legend(fontsize=8)
    for m in models:
        A = runs[m]["actions"][:, 0, :]
        if len(A) >= 3:
            axes[7].plot(np.linalg.norm(np.diff(A, n=2, axis=0), axis=1),
                         "-o", ms=3, color=color_for(m), label=short(m))
    axes[7].set_title("jerk", fontsize=10); axes[7].grid(alpha=.3)
    axes[7].legend(fontsize=8)
    fig.suptitle("Control — first action of each chunk, same observations")
    plt.tight_layout(); save(fig, "02_action_timeseries")

    # 3. 의도 궤적 (첫 프레임)
    def integ(ch, s0):
        p = np.zeros((len(ch)+1, 3), np.float32); p[0] = s0[:3]
        for t, x in enumerate(ch):
            p[t+1] = p[t] + x[:3]
        return p

    fig = plt.figure(figsize=(11, 5))
    a3 = fig.add_subplot(121, projection="3d"); a2 = fig.add_subplot(122)
    allp = [integ(runs[m]["actions"][0], runs[m]["states"][0]) for m in models]
    lo = np.min(np.concatenate(allp), 0); hi = np.max(np.concatenate(allp), 0)
    pad = np.maximum((hi-lo)*.15, 1e-3); lo, hi = lo-pad, hi+pad
    for m, p in zip(models, allp):
        a3.plot(p[:, 0], p[:, 1], p[:, 2], "-o", ms=3,
                color=color_for(m), label=short(m))
        a2.plot(runs[m]["actions"][0][:, 6], "-o", ms=3,
                color=color_for(m), label=short(m))
    a3.set_xlim(lo[0], hi[0]); a3.set_ylim(lo[1], hi[1]); a3.set_zlim(lo[2], hi[2])
    a3.set_title("Planning — intent from the SAME first frame")
    a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z"); a3.legend(fontsize=8)
    a2.set_title("gripper intent"); a2.set_xlabel("step in chunk")
    a2.grid(alpha=.3); a2.legend(fontsize=8)
    plt.tight_layout(); save(fig, "03_intent_trajectory")

    # 4. 언어 프로브
    mat = [json.loads(str(runs[m]["lang_dist"])) for m in models
           if "lang_dist" in runs[m]]
    used = [m for m in models if "lang_dist" in runs[m]]
    if mat:
        M = np.array([[d.get(k, np.nan) for k in LANG_ORDER] for d in mat])
        fig, ax = plt.subplots(figsize=(8, 1.1*len(used)+2.6))
        im = ax.imshow(M, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(LANG_ORDER)))
        ax.set_xticklabels(LANG_ORDER, rotation=20, ha="right")
        ax.set_yticks(range(len(used)))
        ax.set_yticklabels([short(m) for m in used])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="w", fontsize=9)
        fig.colorbar(im, ax=ax, label="||Δaction|| vs original prompt")
        ax.set_title("Perception — language probe\n"
                     "empty/irrelevant ≈ 0 ⇒ prompt is being ignored", fontsize=12)
        plt.tight_layout(); save(fig, "04_language_probe")

    # 5. 카메라 기여도
    rel = {m: json.loads(str(runs[m]["cam_reliance"])) for m in models
           if "cam_reliance" in runs[m]}
    if rel:
        cams = ["agentview", "wrist"]; x = np.arange(2)
        w = .8/len(rel)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for i, (m, v) in enumerate(rel.items()):
            ax.bar(x + i*w - .4 + w/2, [v[c] for c in cams], w,
                   color=color_for(m), label=short(m))
        ax.set_xticks(x); ax.set_xticklabels(cams)
        ax.set_ylabel("||Δaction|| when camera blanked")
        ax.set_title("Perception — camera reliance")
        ax.legend(); ax.grid(axis="y", alpha=.3)
        plt.tight_layout(); save(fig, "05_camera_reliance")

    # 6. saliency
    for m in models:
        cams = [c for c in ("agentview", "wrist") if f"saliency_{c}" in runs[m]]
        if not cams:
            continue
        fidx = runs[m]["frame_idx"].tolist()
        kk = int(runs[m]["probe_frame"])
        ki = fidx.index(kk) if kk in fidx else 0
        fig, ax = plt.subplots(2, len(cams), figsize=(5*len(cams), 9), squeeze=False)
        for c, cam in enumerate(cams):
            img = runs[m]["images" if cam == "agentview" else "wrist_images"][ki]
            sal = runs[m][f"saliency_{cam}"]
            ax[0, c].imshow(img); ax[0, c].set_title(cam); ax[0, c].axis("off")
            ax[1, c].imshow(img)
            up = np.kron(sal, np.ones((img.shape[0]//sal.shape[0],
                                       img.shape[1]//sal.shape[1])))
            h = ax[1, c].imshow(up, cmap="inferno", alpha=.55,
                                extent=(0, img.shape[1], img.shape[0], 0))
            ax[1, c].set_title("occlusion sensitivity"); ax[1, c].axis("off")
            fig.colorbar(h, ax=ax[1, c], fraction=.046)
        fig.suptitle(f"Perception — what {short(m)} looks at", fontsize=14)
        plt.tight_layout(); save(fig, f"06_saliency_{m}")

    print(f"\n[done] {FIG}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode")
    ap.add_argument("--model")
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--grid", type=int, default=8, help="0 이면 saliency 생략")
    ap.add_argument("--config", default=str(REPO / "configs" / "eval.yaml"))
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    sys.exit(render() if a.render else collect(a))
