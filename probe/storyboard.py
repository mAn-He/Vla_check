#!/usr/bin/env python3
"""probe/compare/*.npz 를 사람이 읽을 수 있는 스토리보드로 만든다.

한 장에 전부 담는다:
    어떤 장면을 봤는가 (실제 이미지)
    지시문은 무엇이었는가 (원문 그대로)
    어디를 보고 판단했는가 (가림 민감도 오버레이)
    무슨 행동을 냈는가 (방향 화살표 + 7차원 막대 + 말로 푼 해석)

사용:
    python probe/storyboard.py                 # 전체
    python probe/storyboard.py --frame 2       # 특정 프레임만 상세
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.patches import FancyArrow # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
CMP = REPO / "probe" / "compare"
FIG = CMP / "figures"
ACT = ["dx", "dy", "dz", "dRoll", "dPitch", "dYaw", "grip"]
COLORS = {"groot": "#2E86AB", "openvla": "#E8743B", "pi05": "#6A4C93"}
LANG_ORDER = ["original", "synonym", "target_swap", "empty", "irrelevant"]


def color_for(m):
    return next((v for k, v in COLORS.items() if k in m.lower()), "#555")


def short(m):
    return (m.replace("_libero", "").replace("_oft", "-OFT")
             .replace("groot_n17", "GR00T-N1.7").replace("openvla", "OpenVLA")
             .replace("pi05", "pi0.5"))


def load():
    runs = {}
    for p in sorted(CMP.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        runs[str(d["model"])] = {k: d[k] for k in d.files}
    if not runs:
        raise SystemExit(f"no data in {CMP}/*.npz")
    return runs


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"[sb] {FIG/name}.png")


# ---------------------------------------------------------------- 해석
def describe(a: np.ndarray) -> str:
    """7차원 액션을 사람 말로 푼다."""
    dx, dy, dz = a[:3]
    parts = []
    thr = 0.05
    if abs(dz) > thr:
        parts.append("DOWN" if dz < 0 else "UP")
    if abs(dx) > thr:
        parts.append("FWD" if dx > 0 else "BACK")
    if abs(dy) > thr:
        parts.append("LEFT" if dy > 0 else "RIGHT")
    rot = np.abs(a[3:6]).max()
    if rot > thr:
        parts.append("ROTATE")
    if not parts:
        parts.append("hold")
    grip = "CLOSE" if a[6] > 0 else "OPEN"
    return " + ".join(parts) + f"  |  gripper {grip}"


def path_of(chunk):
    p = np.zeros((len(chunk) + 1, 3), np.float32)
    for t, a in enumerate(chunk):
        p[t + 1] = p[t] + a[:3]
    return p


def common_lims(chunks):
    """여러 청크의 경로를 같은 축에 놓기 위한 공통 범위."""
    P = np.concatenate([path_of(c) for c in chunks], axis=0)
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    pad = np.maximum((hi - lo) * .18, 1e-3)
    return (lo[0]-pad[0], hi[0]+pad[0]), (lo[1]-pad[1], hi[1]+pad[1])


def arrow_panel(ax, chunk, col, title, lims=None, cbar=False, fig=None):
    """청크를 위에서 본 경로 + 높이 변화. 3D보다 읽기 쉽다."""
    p = path_of(chunk)
    ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2)
    ax.scatter(p[0, 0], p[0, 1], s=90, marker="o", color=col,
               edgecolor="k", zorder=5, label="start")
    ax.scatter(p[-1, 0], p[-1, 1], s=130, marker="*", color=col,
               edgecolor="k", zorder=5, label="end")
    # 진행에 따라 높이를 색으로
    sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="coolwarm_r",
                    s=28, zorder=4)
    ax.set_xlabel("Δx (forward +)"); ax.set_ylabel("Δy (left +)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.3)
    if lims is not None:
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    if cbar and fig is not None:
        fig.colorbar(sc, ax=ax, fraction=.046, pad=.02, label="Δz (height)")
    return sc


def bars_panel(ax, ax_g, chunks: dict[str, np.ndarray]):
    """첫 액션. 이동/회전 6축과 그리퍼는 스케일이 달라 따로 그린다."""
    x = np.arange(6); w = .8 / max(1, len(chunks))
    for i, (m, ch) in enumerate(chunks.items()):
        ax.bar(x + i * w - .4 + w / 2, ch[0][:6], w,
               color=color_for(m), label=short(m))
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(ACT[:6], fontsize=8)
    ax.set_ylabel("first action")
    ax.set_title("motion / rotation", fontsize=9)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)

    xs = np.arange(1)
    for i, (m, ch) in enumerate(chunks.items()):
        ax_g.bar(xs + i * w - .4 + w / 2, [ch[0][6]], w, color=color_for(m))
    ax_g.axhline(0, color="k", lw=.8)
    ax_g.set_xticks([0]); ax_g.set_xticklabels(["grip"], fontsize=8)
    ax_g.set_ylim(-1.15, 1.15)
    ax_g.set_yticks([-1, 0, 1]); ax_g.set_yticklabels(["OPEN", "", "CLOSE"], fontsize=8)
    ax_g.set_title("gripper", fontsize=9)
    ax_g.grid(axis="y", alpha=.3)


# ---------------------------------------------------------------- 프레임 카드
def frame_card(runs, ki: int):
    models = list(runs)
    m0 = models[0]
    step = int(runs[m0]["frame_idx"][ki])
    instr = str(runs[m0]["instruction"])
    has_sal = any(f"saliency_agentview" in runs[m] for m in models)

    ncol = 2 + len(models) * (2 if has_sal else 1)
    fig = plt.figure(figsize=(3.6 * ncol, 9.0))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.25, 1],
                          hspace=.34, wspace=.45)

    # --- 위: 무엇을 봤는가 ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(runs[m0]["images"][ki]); ax.axis("off")
    ax.set_title("SCENE  (agentview)\nwhat the robot sees", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(runs[m0]["wrist_images"][ki]); ax.axis("off")
    ax.set_title("WRIST  (close-up)", fontsize=10)

    c = 2
    if has_sal:
        for m in models:
            for cam, key in (("agentview", "images"), ("wrist", "wrist_images")):
                if f"saliency_{cam}" not in runs[m]:
                    continue
                ax = fig.add_subplot(gs[0, c]); c += 1
                img = runs[m][key][ki]
                sal = runs[m][f"saliency_{cam}"]
                ax.imshow(img)
                up = np.kron(sal, np.ones((img.shape[0] // sal.shape[0],
                                           img.shape[1] // sal.shape[1])))
                ax.imshow(up, cmap="inferno", alpha=.55,
                          extent=(0, img.shape[1], img.shape[0], 0))
                ax.axis("off")
                ax.set_title(f"{short(m)}\nwhere it looked ({cam})", fontsize=9)
    else:
        for m in models:
            ax = fig.add_subplot(gs[0, c]); c += 1
            ax.axis("off")
            ax.text(.5, .5, f"{short(m)}\n(saliency not collected\nrun with --grid 8)",
                    ha="center", va="center", fontsize=10, color="#888")

    # --- 아래: 무엇을 하려는가 ---
    chunks = {m: runs[m]["actions"][ki] for m in models}
    lims = common_lims(list(chunks.values()))
    for i, m in enumerate(models):
        ax = fig.add_subplot(gs[1, i])
        arrow_panel(ax, chunks[m], color_for(m),
                    f"{short(m)} — planned path\n(top-down, color = height)",
                    lims=lims, cbar=(i == len(models) - 1), fig=fig)

    ax = fig.add_subplot(gs[1, len(models)])
    ax_g = fig.add_subplot(gs[1, len(models) + 1])
    bars_panel(ax, ax_g, chunks)

    if ncol > len(models) + 2:
        ax = fig.add_subplot(gs[1, len(models) + 2:])
        ax.axis("off")
        lines = ["DECISION\n"]
        for m in models:
            lines.append(f"{short(m)}\n   {describe(chunks[m][0])}\n"
                         f"   chunk length {len(chunks[m])}\n")
        if len(models) >= 2:
            d = np.linalg.norm(chunks[models[0]][0] - chunks[models[1]][0])
            lines.append(f"\n||Δ first action|| = {d:.4f}")
        ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=11,
                family="monospace")

    wrapped = "\n".join(textwrap.wrap(f'INSTRUCTION:  "{instr}"', 110))
    fig.suptitle(f"frame {ki}   (episode step {step})\n{wrapped}",
                 fontsize=13, y=.99)
    save(fig, f"story_frame{ki:02d}")


# ---------------------------------------------------------------- 개요
def overview(runs):
    models = list(runs)
    m0 = models[0]
    n = len(runs[m0]["frame_idx"])
    instr = str(runs[m0]["instruction"])

    ncol = 1 + len(models)
    fig, axes = plt.subplots(n, ncol, figsize=(3.6 * ncol, 3.3 * n),
                             squeeze=False)
    for ki in range(n):
        step = int(runs[m0]["frame_idx"][ki])
        axes[ki, 0].imshow(runs[m0]["images"][ki]); axes[ki, 0].axis("off")
        axes[ki, 0].set_ylabel(f"step {step}")
        axes[ki, 0].set_title(f"frame {ki} · episode step {step}", fontsize=9)
        lims = common_lims([runs[m]["actions"][ki] for m in models])
        for i, m in enumerate(models):
            ax = axes[ki, 1 + i]
            ch = runs[m]["actions"][ki]
            arrow_panel(ax, ch, color_for(m), "", lims=lims)
            ax.set_title(f"{short(m)}\n{describe(ch[0])}", fontsize=8)
            ax.set_xlabel(""); ax.set_ylabel("")
    wrapped = "\n".join(textwrap.wrap(f'"{instr}"', 100))
    fig.suptitle(f"Episode storyboard — same observations, side by side\n{wrapped}",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    save(fig, "story_overview")


# ---------------------------------------------------------------- 언어 카드
def language_card(runs):
    models = [m for m in runs if "lang_variants" in runs[m]]
    if not models:
        return
    m0 = models[0]
    fidx = runs[m0]["frame_idx"].tolist()
    kk = int(runs[m0]["probe_frame"])
    ki = fidx.index(kk) if kk in fidx else 0
    variants = json.loads(str(runs[m0]["lang_variants"]))

    # 모든 프롬프트 변형 x 모델의 경로를 같은 축에 (안 그러면 비교 불가)
    lang_lims = common_lims([runs[m][f"lang_{k}"] for m in models
                             for k in LANG_ORDER if f"lang_{k}" in runs[m]])

    ncol = 1 + len(models)
    fig, axes = plt.subplots(len(LANG_ORDER), ncol,
                             figsize=(4.0 * ncol, 3.0 * len(LANG_ORDER)),
                             squeeze=False)
    for r, key in enumerate(LANG_ORDER):
        ax = axes[r, 0]
        ax.imshow(runs[m0]["images"][ki]); ax.axis("off")
        txt = variants.get(key, "")
        shown = f'"{txt}"' if txt else "(empty prompt)"
        ax.set_title(f"[{key}]\n" + "\n".join(textwrap.wrap(shown, 44)),
                     fontsize=8.5, loc="left")
        for i, m in enumerate(models):
            ax = axes[r, 1 + i]
            k = f"lang_{key}"
            if k not in runs[m]:
                ax.axis("off"); continue
            ch = runs[m][k]
            arrow_panel(ax, ch, color_for(m), "", lims=lang_lims)
            d = json.loads(str(runs[m]["lang_dist"])).get(key, float("nan"))
            ax.set_title(f"{short(m)}   ||Δ|| vs original = {d:.3f}\n{describe(ch[0])}",
                         fontsize=8.5)
            ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Same image, different instruction — does the plan change?\n"
                 "identical paths across rows ⇒ the model is not reading the text",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    save(fig, "story_language")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=None,
                    help="이 프레임만 상세 카드. 미지정 시 전부")
    a = ap.parse_args()

    runs = load()
    print(f"[sb] models: {list(runs)}")
    n = len(runs[list(runs)[0]]["frame_idx"])

    overview(runs)
    language_card(runs)
    for ki in ([a.frame] if a.frame is not None else range(n)):
        frame_card(runs, ki)
    print(f"\n[done] {FIG}")
