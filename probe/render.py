#!/usr/bin/env python3
"""인지/계획/제어 3층 프로브 — 시각화.

probe/data/*.npz 를 읽어 probe/figures/ 에 PNG+SVG 를 만든다.
모델이 2개 이상이면 자동으로 비교 그림을 그린다.

사용:
    python probe/render.py
    python probe/render.py --demo      # 가짜 데이터로 파이프라인만 검증
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "probe" / "data"
FIG = REPO / "probe" / "figures"
COLORS = {"groot": "#2E86AB", "openvla": "#E8743B", "pi05": "#6A4C93"}
LANG_ORDER = ["original", "synonym", "target_swap", "empty", "irrelevant"]


def color_for(model: str) -> str:
    for k, v in COLORS.items():
        if k in model.lower():
            return v
    return "#555555"


def short(model: str) -> str:
    return (model.replace("_libero", "").replace("_oft", "-OFT")
                 .replace("groot_n17", "GR00T-N1.7").replace("openvla", "OpenVLA")
                 .replace("pi05", "pi0.5"))


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {FIG/name}.png")


def load_all() -> dict[str, dict]:
    out = {}
    for p in sorted(DATA.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        out[str(d["model"])] = {k: d[k] for k in d.files}
    return out


# ---------- 층 1: 인지 ----------
def fig_saliency(runs: dict) -> None:
    """P1 — 원본 씬 위에 겹친 가림 민감도."""
    for model, r in runs.items():
        cams = [c for c in r["cameras"].tolist()] if hasattr(r["cameras"], "tolist") \
            else list(r["cameras"])
        cams = [c for c in cams if f"saliency_{c}" in r]
        if not cams:
            continue
        fig, axes = plt.subplots(2, len(cams), figsize=(5 * len(cams), 9))
        axes = np.atleast_2d(axes)
        if len(cams) == 1:
            axes = axes.reshape(2, 1)
        for k, cam in enumerate(cams):
            img, sal = r[f"image_{cam}"], r[f"saliency_{cam}"]
            axes[0, k].imshow(img)
            axes[0, k].set_title(f"{cam}", fontsize=11)
            axes[0, k].axis("off")

            axes[1, k].imshow(img)
            up = np.kron(sal, np.ones((img.shape[0] // sal.shape[0],
                                       img.shape[1] // sal.shape[1])))
            im = axes[1, k].imshow(up, cmap="inferno", alpha=0.55,
                                   extent=(0, img.shape[1], img.shape[0], 0))
            axes[1, k].set_title("occlusion sensitivity", fontsize=11)
            axes[1, k].axis("off")
            fig.colorbar(im, ax=axes[1, k], fraction=0.046,
                         label="||Δaction||")
        fig.suptitle(f"Perception — what {short(model)} looks at", fontsize=14)
        plt.tight_layout()
        save(fig, f"perception_saliency_{model}")


def fig_camera_reliance(runs: dict) -> None:
    """P2 — 카메라별 의존도."""
    models = list(runs)
    rel = {m: json.loads(str(runs[m]["camera_reliance"])) for m in models
           if "camera_reliance" in runs[m]}
    if not rel:
        return
    cams = sorted({c for v in rel.values() for c in v})
    x = np.arange(len(cams))
    w = 0.8 / max(1, len(rel))

    fig, ax = plt.subplots(figsize=(1.8 * len(cams) + 4, 4.2))
    for i, (m, v) in enumerate(rel.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [v.get(c, 0) for c in cams], w,
               label=short(m), color=color_for(m))
    ax.set_xticks(x)
    ax.set_xticklabels(cams)
    ax.set_ylabel("||Δaction|| when camera is blanked")
    ax.set_title("Perception — camera reliance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "perception_camera_reliance")


def fig_language(runs: dict) -> None:
    """P3 — 지시문 변형에 대한 반응."""
    models = list(runs)
    mat, used = [], []
    for m in models:
        if "language_dist" not in runs[m]:
            continue
        d = json.loads(str(runs[m]["language_dist"]))
        mat.append([d.get(k, np.nan) for k in LANG_ORDER])
        used.append(m)
    if not mat:
        return
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(8, 1.1 * len(used) + 2.6))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_xticklabels(LANG_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(used)))
    ax.set_yticklabels([short(m) for m in used])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="||Δaction|| vs original instruction")
    ax.set_title("Perception — language probe\n"
                 "empty/irrelevant ≈ 0 → instruction is being ignored",
                 fontsize=12)
    plt.tight_layout()
    save(fig, "perception_language_probe")


# ---------- 층 2: 계획 ----------
def integrate(chunk: np.ndarray, start: np.ndarray) -> np.ndarray:
    """delta action chunk -> EEF 경로."""
    p = np.zeros((len(chunk) + 1, 3), dtype=np.float32)
    p[0] = start[:3]
    for t, a in enumerate(chunk):
        p[t + 1] = p[t] + a[:3]
    return p


def fig_intent_3d(runs: dict) -> None:
    """L1 — 얼린 관측에서 각 모델이 그리는 의도 궤적."""
    fig = plt.figure(figsize=(11, 5))
    ax3 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    for m, r in runs.items():
        if "action_base" not in r:
            continue
        p = integrate(r["action_base"], r["state"])
        c = color_for(m)
        ax3.plot(p[:, 0], p[:, 1], p[:, 2], "-o", ms=3, color=c, label=short(m))
        ax3.scatter(*p[0], color=c, s=70, marker="^")
        ax2.plot(r["action_base"][:, 6], "-o", ms=3, color=c, label=short(m))

    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title("Planning — intent trajectory\n(same frozen observation)")
    ax3.legend(fontsize=8)
    ax2.set_xlabel("step in chunk"); ax2.set_ylabel("gripper command")
    ax2.set_title("Planning — gripper intent")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    plt.tight_layout()
    save(fig, "planning_intent_3d")


def fig_language_conditioning(runs: dict) -> None:
    """L2 — 지시문을 바꾸면 계획이 실제로 바뀌는가. 핵심 그림."""
    models = [m for m in runs if any(f"action_lang_{k}" in runs[m]
                                     for k in LANG_ORDER)]
    if not models:
        return
    # 모든 모델의 궤적을 모아 공통 축 범위를 구한다 (안 그러면 눈으로 비교 불가)
    allp = []
    for m in models:
        for k in LANG_ORDER:
            key = f"action_lang_{k}"
            if key in runs[m]:
                allp.append(integrate(runs[m][key], runs[m]["state"]))
    allp = np.concatenate(allp, axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    pad = np.maximum((hi - lo) * 0.1, 1e-3)
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(5.2 * len(models), 5))
    for i, m in enumerate(models):
        ax = fig.add_subplot(1, len(models), i + 1, projection="3d")
        for k, style in zip(LANG_ORDER, ["-", "--", "-.", ":", ":"]):
            key = f"action_lang_{k}"
            if key not in runs[m]:
                continue
            pth = integrate(runs[m][key], runs[m]["state"])
            ax.plot(pth[:, 0], pth[:, 1], pth[:, 2], style,
                    lw=2.4 if k == "original" else 1.4, label=k)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_title(short(m), fontsize=12)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Planning — is the plan actually conditioned on language?\n"
                 "overlapping lines ⇒ the instruction is not changing the plan",
                 fontsize=13)
    plt.tight_layout()
    save(fig, "planning_language_conditioning")


def fig_replan(runs: dict) -> None:
    """L3 — 재계획 시 궤적이 튀는가."""
    models = [m for m in runs if "replan_chunks" in runs[m]]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4),
                             squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0, i]
        chunks = runs[m]["replan_chunks"]
        offs = runs[m]["replan_offsets"]
        for k, (ch, off) in enumerate(zip(chunks, offs)):
            p = integrate(ch, runs[m]["state"])
            ax.plot(np.arange(len(p)) + off, p[:, 2], "-o", ms=3,
                    alpha=0.85, label=f"replan @ t={off}")
        ax.set_xlabel("timestep"); ax.set_ylabel("planned z")
        ax.set_title(short(m)); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Planning — replanning stability (overlap = consistent plan)")
    plt.tight_layout()
    save(fig, "planning_replan_stability")


# ---------- 층 3: 제어 ----------
def fig_control_timeseries(runs: dict) -> None:
    """C1/C2 — 액션 7차원과 jerk."""
    models = [m for m in runs if "action_base" in runs[m]]
    if not models:
        return
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.ravel()
    for d in range(7):
        for m in models:
            a = runs[m]["action_base"]
            axes[d].plot(a[:, d], "-o", ms=3, color=color_for(m), label=short(m))
        axes[d].set_title(names[d], fontsize=10)
        axes[d].grid(alpha=0.3)
        if d == 0:
            axes[d].legend(fontsize=8)
    # jerk
    for m in models:
        a = runs[m]["action_base"]
        if len(a) >= 3:
            jerk = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1)
            axes[7].plot(jerk, "-o", ms=3, color=color_for(m), label=short(m))
    axes[7].set_title("jerk  ||a$_{t+1}$-2a$_t$+a$_{t-1}$||", fontsize=10)
    axes[7].grid(alpha=0.3); axes[7].legend(fontsize=8)
    fig.suptitle("Control — action chunk from the same frozen observation")
    plt.tight_layout()
    save(fig, "control_action_timeseries")


def fig_noise(runs: dict) -> None:
    """C3 — 입력 노이즈 민감도."""
    models = [m for m in runs if "noise_sensitivity" in runs[m]]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in models:
        ax.plot(runs[m]["noise_sigmas"], runs[m]["noise_sensitivity"],
                "-o", color=color_for(m), label=short(m))
    ax.set_xlabel("image noise σ (fraction of 255)")
    ax.set_ylabel("||Δaction||")
    ax.set_title("Control — sensitivity to input noise\n"
                 "steeper ⇒ less robust to real camera noise", fontsize=11)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    save(fig, "control_noise_sensitivity")


def fig_efficiency(runs: dict) -> None:
    """C4 — results/*.jsonl 의 스텝 수. 추가 실행 불필요."""
    rows = {}
    for p in sorted((REPO / "results").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("success"):
                rows.setdefault(r["model"], {}).setdefault(
                    f"{r['task_suite']}/t{r['task_id']}", []).append(r["num_steps"])
    if len(rows) < 1:
        return
    models = list(rows)
    tasks = sorted({t for v in rows.values() for t in v})
    mat = np.full((len(models), len(tasks)), np.nan)
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            if t in rows[m]:
                mat[i, j] = np.mean(rows[m][t])

    fig, ax = plt.subplots(figsize=(max(9, 0.42 * len(tasks)),
                                    0.6 * len(models) + 3))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short(m) for m in models])
    if len(tasks) <= 45:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                            color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean steps (successful episodes)")
    ax.set_title("Control — efficiency: fewer steps is better\n"
                 "(success rate saturates at ~98%; step count does not)",
                 fontsize=12)
    plt.tight_layout()
    save(fig, "control_efficiency")


# ---------- 데모 ----------
def make_demo() -> None:
    """GPU/서버 없이 렌더 파이프라인만 검증."""
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for model, bias in (("groot_n17_libero", 0.0), ("openvla_oft_libero", 0.6)):
        g, H = 8, 256
        img = (rng.integers(40, 210, (H, H, 3))).astype(np.uint8)
        wrist = (rng.integers(40, 210, (H, H, 3))).astype(np.uint8)
        yy, xx = np.mgrid[0:g, 0:g]
        sal = np.exp(-(((xx - 3 - bias * 2) ** 2 + (yy - 4) ** 2) / 4.0)) * 0.4
        base = np.cumsum(rng.normal(0, 0.02, (8, 7)), axis=0)
        base[:, 6] = np.tanh(np.linspace(-2, 2, 8))
        rec = dict(
            model=model, suite="libero_spatial", task_id=0,
            instruction="pick up the black bowl and place it on the plate",
            state=np.array([0.1, 0.0, 1.0, 0, 0, 0, 0.04, 0.04], dtype=np.float32),
            cameras=np.array(["agentview_image", "wrist_image"]),
            grid=g, image_agentview_image=img, image_wrist_image=wrist,
            saliency_agentview_image=sal.astype(np.float32),
            saliency_wrist_image=(sal.T * 0.6).astype(np.float32),
            action_base=base.astype(np.float32),
            camera_reliance=json.dumps({"agentview_image": 0.41 + bias * 0.2,
                                        "wrist_image": 0.33 - bias * 0.1}),
            language_dist=json.dumps({"original": 0.0, "synonym": 0.05 + bias * 0.1,
                                      "target_swap": 0.42 - bias * 0.3,
                                      "empty": 0.31 - bias * 0.25,
                                      "irrelevant": 0.29 - bias * 0.24}),
            noise_sigmas=np.array([0, .01, .02, .05, .1], dtype=np.float32),
            noise_sensitivity=np.array([0, .04, .09, .25, .55],
                                       dtype=np.float32) * (1 + bias),
            replan_chunks=np.stack([base + rng.normal(0, .01 + bias * .02, base.shape)
                                    for _ in range(4)]).astype(np.float32),
            replan_offsets=np.array([0, 4, 8, 12]),
        )
        for k in LANG_ORDER:
            scale = {"original": 0, "synonym": .05, "target_swap": .45,
                     "empty": .3, "irrelevant": .28}[k] * (1 - bias * 0.7)
            rec[f"action_lang_{k}"] = (base + rng.normal(0, scale, base.shape)
                                       ).astype(np.float32)
        np.savez_compressed(DATA / f"{model}_libero_spatial_t0.npz", **rec)
    print(f"[demo] 가짜 데이터 생성 -> {DATA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        make_demo()

    runs = load_all()
    if not runs:
        raise SystemExit(f"데이터 없음: {DATA}/*.npz  (--demo 로 테스트 가능)")
    print(f"[load] {len(runs)} models: {list(runs)}")

    fig_saliency(runs)
    fig_camera_reliance(runs)
    fig_language(runs)
    fig_intent_3d(runs)
    fig_language_conditioning(runs)
    fig_replan(runs)
    fig_control_timeseries(runs)
    fig_noise(runs)
    fig_efficiency(runs)
    print(f"\n[done] {FIG}")
