# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 01. 데이터 탐색과 평가 표본 고정 (nuScenes-mini)
#
# ## 목적
# nuScenes의 센서 구성, 캘리브레이션, ego pose를 이해하고, 이후 모든 노트북이 같은 프레임으로 평가하도록
# 평가 표본을 고정해 `results/outdoor/01_data_exploration/eval_samples.json`에 저장합니다.
# 탐지 평가가 4개 장면 부분집합(`eval4`)을 쓸 수 있도록 devkit 사용자 분할 파일(`v1.0-mini/splits.json`)도 씁니다.
#
# ## 비교 모델과 원리
# 이 노트북은 모델을 돌리지 않습니다. 비교의 바탕이 되는 센서와 좌표계를 정리합니다.
# - 카메라 6대(1600x900, 약 12Hz)와 32채널 LiDAR 1대(20Hz). 주석이 있는 키프레임은 2Hz입니다.
# - 좌표계: 센서 → ego(그 센서의 타임스탬프) → global. 카메라와 LiDAR는 촬영 시각이 달라 ego pose도 다르므로,
#   LiDAR를 카메라에 투영할 때는 global을 거쳐야 합니다(devkit `map_pointcloud_to_image`와 같은 방식).
# - LiDAR `.bin`은 점마다 float32 5개를 저장하고, devkit `LidarPointCloud.from_file`은 앞 4개(x, y, z, intensity)만 씁니다.
#
# ## 원 논문
# nuScenes: Caesar et al., CVPR 2020, https://arxiv.org/abs/1903.11027 .
# Panoptic nuScenes(lidarseg): Fong et al., 2021, https://arxiv.org/abs/2109.03805
#
# ## 커널
# `3dv`
#
# ## 예상 VRAM·시간 (추정)
# GPU를 쓰지 않습니다. lidarseg 라벨 집계를 포함해 수 분 정도로 추정합니다.
#
# ## 이 노트북이 답하려는 질문
# - 카메라와 LiDAR는 어떻게 정렬되며, 투영하면 어디가 비는가(하늘, 원거리, 가려진 면)?
# - 평가 장면 4개(`eval4`)는 공식 체크포인트의 학습 데이터와 겹치지 않는가?
# - 장면 표현(07)에 쓸 "정적에 가까운" 장면은 어느 것인가?

# %%
FORCE_RERUN = False
NOTEBOOK = "01_data_exploration"

import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bench3d.config import load_config
from bench3d.io import result_dir
from bench3d.metrics.detection import write_custom_splits
from bench3d.nusc import CAMERAS, load_lidar, project_lidar_to_camera, scene_samples
from bench3d.runtime import env_report, set_seed
from bench3d.sampling import select_windows
from bench3d.viz import bev, overlay_points
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nusplits
from PIL import Image
from pyquaternion import Quaternion

cfg = load_config("outdoor")
set_seed(cfg["seed"])
ENV = env_report(["numpy", "nuscenes-devkit", "matplotlib"])
print(json.dumps(ENV, indent=1))
OUT = result_dir(cfg, NOTEBOOK)
DATA_ROOT = cfg.path(cfg["dataset"]["data_root"])
EVAL4, PROBE6 = cfg["splits"]["eval4"], cfg["splits"]["probe6"]

# %%
nusc = NuScenes(version=cfg["dataset"]["version"], dataroot=str(DATA_ROOT), verbose=False)
print(len(nusc.scene), "scenes,", len(nusc.sample), "samples, lidarseg records:", len(getattr(nusc, "lidarseg", [])))

# %% [markdown]
# ## 1. 센서 구성과 캘리브레이션
# 첫 평가 장면의 가운데 키프레임 기준입니다. `dt_ms`는 같은 sample 안에서 LiDAR 대비 각 센서의 촬영 시각 차이입니다.
# 카메라는 LiDAR가 그 방향을 지나가는 시각에 맞춰 찍히지만 완전히 같지는 않습니다.

# %%
_toks0 = scene_samples(nusc, EVAL4[0])
s0 = nusc.get("sample", _toks0[len(_toks0) // 2])
t_lidar = nusc.get("sample_data", s0["data"]["LIDAR_TOP"])["timestamp"]
rows = []
for ch in ["LIDAR_TOP", *CAMERAS]:
    sd = nusc.get("sample_data", s0["data"][ch])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    yaw, pitch, roll = np.degrees(Quaternion(cs["rotation"]).yaw_pitch_roll)
    K = np.asarray(cs["camera_intrinsic"]) if cs["camera_intrinsic"] else None
    rows.append({"channel": ch, "x": cs["translation"][0], "y": cs["translation"][1], "z": cs["translation"][2],
                 "yaw": yaw, "pitch": pitch, "roll": roll,
                 "fx": K[0, 0] if K is not None else None, "cx": K[0, 2] if K is not None else None,
                 "cy": K[1, 2] if K is not None else None,
                 "size": f"{sd['width']}x{sd['height']}" if sd["width"] else "-",
                 "dt_ms": (sd["timestamp"] - t_lidar) / 1e3})
pd.DataFrame(rows).round(3)

# %% [markdown]
# ## 2. LiDAR를 6개 카메라에 투영
# 색은 카메라 기준 깊이(m)입니다. 이 희소 깊이가 02(단안 깊이)의 정답입니다.
# 하늘과 원거리에는 점이 없고, 센서 위치 차이(시차) 때문에 물체 뒤쪽 면의 점이 앞 물체 위에 찍히는 경우가 있습니다.

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 7))
for ax, ch in zip(axes.ravel(), ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                                 "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]):
    cam_sd = s0["data"][ch]
    proj = project_lidar_to_camera(nusc, s0["data"]["LIDAR_TOP"], cam_sd)
    img = Image.open(DATA_ROOT / nusc.get("sample_data", cam_sd)["filename"])
    overlay_points(ax, np.asarray(img), proj["uv"], proj["depth"], vmax=60, s=1.5,
                   title=f"{ch}: {len(proj['depth'])} pts")
plt.tight_layout()
fig.savefig(OUT / "lidar_on_cameras.png", dpi=110)

# %% [markdown]
# ## 3. BEV와 3D 박스
# LiDAR 좌표계 BEV(x 전방, y 좌측)에 주석 박스 바닥면을 겹쳐 그립니다. 오른쪽은 devkit 렌더러로 CAM_FRONT에 3D 박스를 그린 것입니다.

# %%
lidar_sd = s0["data"]["LIDAR_TOP"]
pts = load_lidar(nusc, lidar_sd)
_, boxes, _ = nusc.get_sample_data(lidar_sd)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8))
bev(a1, pts, extent=50, color="#56B4E9", s=0.3, boxes_xy=[b.bottom_corners()[:2].T for b in boxes])
a1.set_title(f"LIDAR_TOP BEV + {len(boxes)} boxes")
nusc.render_sample_data(s0["data"]["CAM_FRONT"], ax=a2)
plt.tight_layout()
fig.savefig(OUT / "bev_and_boxes.png", dpi=110)

# %% [markdown]
# ## 4. 장면 통계와 분할
# - `ego_path_m`: LIDAR_TOP 키프레임 ego 위치 사이 거리의 합(장면 동안 이동한 거리)
# - `moving_ann_ratio`: devkit `box_velocity`로 구한 속도가 0.5 m/s를 넘는 주석의 비율(정적 장면 판단용, NaN 속도는 제외)
# - `official`: 공식 trainval 분할에서의 소속. 공식 체크포인트는 `train`으로 학습됐으므로 평가는 `val`인 장면만 씁니다.

# %%
official_train = set(nusplits.train)
stats, keyframes = {}, {}
for sc in nusc.scene:
    name = sc["name"]
    toks = scene_samples(nusc, name)
    keyframes[name] = toks
    pos = np.array([nusc.get("ego_pose", nusc.get("sample_data", nusc.get("sample", t)["data"]["LIDAR_TOP"])
                             ["ego_pose_token"])["translation"] for t in toks])
    speeds, n_ann = [], 0
    for t in toks:
        for a in nusc.get("sample", t)["anns"]:
            n_ann += 1
            v = nusc.box_velocity(a)
            if np.all(np.isfinite(v)):
                speeds.append(np.linalg.norm(v[:2]))
    path = float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum())
    stats[name] = {"n_keyframes": len(toks), "ego_path_m": round(path, 1),
                   "duration_s": round((nusc.get("sample", toks[-1])["timestamp"]
                                        - nusc.get("sample", toks[0])["timestamp"]) / 1e6, 1),
                   "n_ann": n_ann,
                   "moving_ann_ratio": round(float(np.mean(np.array(speeds) > 0.5)) if speeds else float("nan"), 3),
                   "split_here": "eval4" if name in EVAL4 else ("probe6" if name in PROBE6 else "-"),
                   "official": "train" if name in official_train else "val",
                   "description": sc["description"]}
scene_df = pd.DataFrame(stats).T
scene_df

# %%
leak = [s for s in EVAL4 if s in official_train]
assert not leak, f"eval4 contains official-train scenes: {leak}"
print("eval4 ∩ official train = ∅  ✓")

# %% [markdown]
# ## 5. 장면 표현(07)에 쓸 장면 선택
# NeRF/3DGS는 카메라가 움직여야 하고(시차), 움직이는 물체가 적을수록 유리합니다.
# 그래서 `eval4` 중 이동 거리가 30 m 이상인 장면에서 움직이는 주석 비율이 가장 낮은 장면을 고릅니다.
# 기준을 만족하는 장면이 없으면 이동 거리가 가장 긴 장면을 씁니다.

# %%
cand = scene_df.loc[EVAL4].copy()
moving = cand[cand["ego_path_m"].astype(float) >= 30]
recon_scene = (moving.sort_values("moving_ann_ratio").index[0] if len(moving)
               else cand.sort_values("ego_path_m", ascending=False).index[0])
print("recon_scene:", recon_scene, dict(stats[recon_scene]))

# %% [markdown]
# ## 6. lidarseg 클래스 분포
# 06(포인트 분할)의 선형 탐침은 `probe6`으로 학습하고 `eval4`로 평가합니다. 두 분할의 클래스 분포 차이를 미리 봅니다.

# %%
def label_hist(scenes):
    cnt = Counter()
    for name in scenes:
        for t in keyframes[name]:
            sd_tok = nusc.get("sample", t)["data"]["LIDAR_TOP"]
            rec = nusc.get("lidarseg", sd_tok)
            lab = np.fromfile(DATA_ROOT / rec["filename"], dtype=np.uint8)
            cnt.update(dict(zip(*np.unique(lab, return_counts=True))))
    return cnt


if getattr(nusc, "lidarseg", None):
    h_eval, h_probe = label_hist(EVAL4), label_hist(PROBE6)
    idx2name = nusc.lidarseg_idx2name_mapping
    seg_df = pd.DataFrame({"eval4": {idx2name[k]: v for k, v in h_eval.items()},
                           "probe6": {idx2name[k]: v for k, v in h_probe.items()}}).fillna(0).astype(int)
    seg_df["eval4_%"] = (100 * seg_df["eval4"] / seg_df["eval4"].sum()).round(2)
    seg_df["probe6_%"] = (100 * seg_df["probe6"] / seg_df["probe6"].sum()).round(2)
    display(seg_df.sort_values("eval4", ascending=False))  # noqa: F821
else:
    print("lidarseg not loaded -> check outdoor/scripts/verify_nuscenes_mini.py")

# %% [markdown]
# ## 7. 프로파일별 다시점 창(03) 미리보기
# `bench3d.sampling.select_windows`는 결정적이라 03에서도 같은 프레임을 고릅니다.

# %%
P_mv = cfg.p["multiview"]
win_rows = []
for name in EVAL4:
    for w in select_windows(keyframes[name], P_mv["n_frames"], P_mv["windows_per_scene"]):
        p = np.array([nusc.get("ego_pose", nusc.get("sample_data", nusc.get("sample", t)["data"]["CAM_FRONT"])
                               ["ego_pose_token"])["translation"] for t in w])
        win_rows.append({"scene": name, "first": w[0][:8], "n": len(w),
                         "travel_m": round(float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()), 1)})
pd.DataFrame(win_rows)

# %% [markdown]
# ## 8. 저장
# - `eval_samples.json`: 이후 노트북이 읽는 평가 표본(스키마는 `docs/notebook_conventions.md` §5)
# - `v1.0-mini/splits.json`: devkit 사용자 분할(`eval4`, `probe6`). 05의 탐지 평가가 사용합니다.

# %%
eval_samples = {"version": cfg["dataset"]["version"], "eval4": EVAL4, "probe6": PROBE6,
                "keyframes": {k: keyframes[k] for k in EVAL4 + PROBE6},
                "scene_stats": stats,
                "recon_scene": recon_scene}
(OUT / "eval_samples.json").write_text(json.dumps(eval_samples, indent=2, ensure_ascii=False, default=float))
splits_path = write_custom_splits(DATA_ROOT, cfg["dataset"]["version"], {"eval4": EVAL4, "probe6": PROBE6})
print("saved", OUT / "eval_samples.json", "and", splits_path)

# %% [markdown]
# ## 해석
# > (실행 후 작성) 투영에서 비는 영역, 장면별 특성(야간/비, 이동 거리, 움직이는 물체 비율), 클래스 분포 차이를 적습니다.
