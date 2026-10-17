# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 04. Pseudo-LiDAR — 단안 깊이를 LiDAR 형식 점군으로
#
# ## 목적
# 02에서 가장 정확했던 **metric 단안 깊이**를 eval4 장면의 **모든 키프레임 × 카메라**에 대해 3D로 역투영하고,
# 같은 sample의 `LIDAR_TOP` 좌표계로 옮겨 nuScenes LiDAR `.bin` 형식으로 저장한다. 05에서 LiDAR용 3D 탐지기에 이 점군을 그대로
# 넣어 "카메라로 만든 점군으로 LiDAR 탐지기가 얼마나 버티는가"를 본다. 이 노트북은 그 입력을 만들고, 실제 LiDAR와
# **점 밀도(거리 링별)**·**높이 분포**·**BEV 모양**을 같은 카메라 시야(FOV) 안에서 비교한다.
#
# ## 비교 모델과 원리
# - **Pseudo-LiDAR (Wang et al., CVPR 2019)**: 깊이맵의 각 픽셀 (u, v, d)를 `d·K⁻¹[u, v, 1]ᵀ`로 카메라 좌표 3D 점으로 바꾸고,
#   보정 행렬로 LiDAR 좌표계에 옮긴다. 핵심 주장은 "카메라 3D 탐지의 약점은 깊이 자체보다 **표현**(이미지 격자)에 있다" —
#   점군 표현으로 바꾸면 LiDAR용 탐지기를 그대로 쓸 수 있다는 것. 깊이 오차는 거리 제곱으로 커지는 경향이 있어 먼 물체가 길게 번진다.
# - **깊이 공급 모델**: 02의 metric 모델(DA V2 Metric-VKITTI, DA3METRIC-LARGE, high 프로필의 DA3NESTED) 중
#   **primary AbsRel이 가장 낮은 모델**을 자동 선택한다(`config.yaml`의 `pseudo_lidar_source`로 덮어쓸 수 있음). 각 모델 원리는 02 첫 셀 참고.
#   상대 깊이 모델(DA3MONO)은 장면마다 GT로 스케일을 맞춰야 하므로 후보에서 뺀다.
# - **기준: 실제 LIDAR_TOP 키프레임** (32빔 회전형). 카메라 시야 밖 점은 비교에서 제외해 공정하게 맞춘다(`frustum_mask`).
#
# ## 원 논문
# - Pseudo-LiDAR: Wang et al., "Pseudo-LiDAR from Visual Depth Estimation", CVPR 2019 — https://arxiv.org/abs/1812.07179
#   (TODO(verify): 이 환경은 arxiv.org 접근이 막혀 id를 직접 확인하지 못함)
# - Depth Anything V2 — https://arxiv.org/abs/2406.09414 / Depth Anything 3 — https://arxiv.org/abs/2511.10647
#
# ## 커널
# `3dv` (torch 2.1.2, cu121)
#
# ## 예상 VRAM·시간 (추정)
# **추정치**다. 02의 예측 캐시(stride 2)를 재사용하므로, mid 프로필에서는 나머지 절반 키프레임만 새로 추론한다.
#
# | 프로필 | 카메라 | 추론(02 캐시 미스분) | 역투영·저장 | VRAM(추정) | 디스크(추정) |
# |---|---|---|---|---|---|
# | low | CAM_FRONT | ~80장, 1–3분 | 1분 이내 | 2–4 GB | ~0.3 GB |
# | mid | 6캠 | ~490장, 3–8분 | 수 분 | 2–4 GB | ~1.5–2 GB (.bin) + 예측 캐시 ~1 GB |
# | high | 6캠 | 02가 stride 1이면 0장 | 수 분 | Nested면 8–12 GB | 위와 같음 |
#
# (`pixel_stride=4`면 카메라당 최대 400×225=9만 점, 6캠이면 sample당 최대 54만 점 × 20 B ≈ 11 MB)
#
# ## 이 노트북이 답하려는 질문
# 1. 어떤 깊이 모델을 pseudo-LiDAR 소스로 쓰는가, 그 근거는? (02 결과)
# 2. pseudo-LiDAR는 실제 LiDAR와 비교해 거리별로 점 밀도가 어떻게 다른가? (가까운 곳은 과밀, 먼 곳은?)
# 3. 높이 분포(지면·차량 높이)가 실제 LiDAR와 맞는가? 지면이 휘거나 두꺼워지는가?
# 4. BEV에서 물체(GT 박스) 주변 점이 어떻게 번지는가? 05의 탐지 결과를 해석할 단서는?

# %% [markdown]
# ## 1. 설정 로드

# %%
FORCE_RERUN = False            # True: ignore cached results
NOTEBOOK = "04_pseudo_lidar"

import json
import time
from pathlib import Path

import numpy as np
from bench3d.config import load_config
from bench3d.io import is_cached, result_dir, save_summary
from bench3d.runtime import Measure, env_report, free_cuda, set_seed

cfg = load_config("outdoor")                 # profile: config.yaml or $THREEDCHECK_PROFILE
P = cfg.p["pseudo_lidar"]                    # this notebook's profile block
FORCE_RERUN = FORCE_RERUN or cfg.get("force_rerun", False)
set_seed(cfg["seed"])
ENV = env_report(["torch", "numpy", "nuscenes-devkit", "huggingface_hub", "depth_anything_3", "xformers",
                  "opencv-python", "scipy"])
print(json.dumps(ENV, indent=1))
print("profile:", cfg["profile"], P)

# %%
import warnings

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from bench3d.io import load_nuscenes_bin, load_summary, save_nuscenes_bin
from bench3d.metrics.pointcloud import frustum_mask
from bench3d.models.depth import (
    cached_prediction_ok,
    checkpoint_name,
    depth_kind,
    load_depth_model,
    load_prediction,
    pred_cache_path,
    save_prediction,
)
from bench3d.nusc import (
    backproject,
    intrinsic,
    lidar_to_camera,
    load_lidar,
    sample_cameras,
    scene_samples,
    sensor_to_ego,
    sensor_to_global,
    transform_points,
)
from bench3d.sampling import middle, subsample
from bench3d.viz import bev
from nuscenes.nuscenes import NuScenes

DEPTH_NB = "02_monocular_depth"
CAMS = list(P["cams"])
PIXEL_STRIDE = int(P["pixel_stride"])
MAX_DEPTH = float(P["max_depth"])
ENCODER = cfg.p["depth"].get("dav2_encoder")       # DA V2 size comes from the 02 profile block
MODEL_SPECS = cfg["models"]["depth"]
# DA3 sky head: sky pixels get the 99th-percentile depth (model/da3.py _process_mono_sky_estimation) -> drop them
DROP_SKY = True
print("cams:", CAMS, "| pixel_stride:", PIXEL_STRIDE, "| max_depth:", MAX_DEPTH)

# %% [markdown]
# ## 2. 데이터 로드와 소스 모델 선택
#
# - 표본: `eval_samples.json`의 `eval4` 장면 **전체 키프레임(stride 1)**. 05의 탐지 평가는 분할의 모든 sample이 필요하므로 프로필의
#   `depth.sample_stride`를 쓰지 않는다. `bench3d.nusc.scene_samples`로 키프레임 목록이 빠짐없는지 확인한다.
# - 소스 모델: `pseudo_lidar_source`가 설정돼 있으면 그 값, 아니면 `results/outdoor/02_monocular_depth/<model>/summary.json`
#   (metric 모델의 primary = 원 출력, median scaling 없음) 중 `abs_rel`이 가장 낮은 모델.
#   02와 프로필이 다르면(예: DA V2 인코더) 체크포인트가 달라지므로 경고하고, 그 경우 예측 캐시는 재사용하지 않는다.

# %%
nusc = NuScenes(version=cfg["dataset"]["version"], dataroot=str(cfg.path(cfg["dataset"]["data_root"])), verbose=False)

ES_PATH = cfg.path(cfg["results_root"]) / "01_data_exploration" / "eval_samples.json"
assert ES_PATH.exists(), f"{ES_PATH} not found: run 01_data_exploration first"
ES = json.loads(ES_PATH.read_text())
assert ES["version"] == cfg["dataset"]["version"], (ES["version"], cfg["dataset"]["version"])
EVAL_SCENES = list(cfg["splits"]["eval4"])
assert set(ES["eval4"]) == set(EVAL_SCENES), f"eval4 mismatch: {ES['eval4']} vs config {EVAL_SCENES}"

SAMPLES = {sc: subsample(ES["keyframes"][sc], 1) for sc in EVAL_SCENES}  # stride 1: every keyframe
for sc in EVAL_SCENES:
    assert SAMPLES[sc] == scene_samples(nusc, sc), f"{sc}: eval_samples.json keyframes are not the full scene"
TOKENS = [(sc, tok) for sc in EVAL_SCENES for tok in SAMPLES[sc]]
print({sc: len(v) for sc, v in SAMPLES.items()}, "->", len(TOKENS), "samples x", len(CAMS), "cams")

# %%
cand_rows = []
for mid, spec in MODEL_SPECS.items():
    f = cfg.path(cfg["results_root"]) / DEPTH_NB / mid / "summary.json"
    if depth_kind(spec) != "metric" or not f.exists():
        continue
    s = load_summary(f)
    cand_rows.append({"model": mid, "abs_rel": s["metrics"].get("abs_rel"), "delta1": s["metrics"].get("delta1"),
                      "rmse": s["metrics"].get("rmse"), "profile": s.get("profile"), "checkpoint": s.get("checkpoint"),
                      "variant": s.get("variant")})
CANDS = pd.DataFrame(cand_rows)
if len(CANDS):
    display(CANDS.sort_values("abs_rel"))

SRC = cfg.get("pseudo_lidar_source")
if SRC:
    SELECTION = "config pseudo_lidar_source"
else:
    ok = CANDS.dropna(subset=["abs_rel"]) if len(CANDS) else CANDS
    assert len(ok), f"no metric-model summary in {DEPTH_NB}: run 02 first or set pseudo_lidar_source in config.yaml"
    SRC = str(ok.sort_values("abs_rel").iloc[0]["model"])
    SELECTION = f"lowest primary abs_rel among metric models in {DEPTH_NB}"
SRC_SPEC = MODEL_SPECS[SRC]
assert depth_kind(SRC_SPEC) == "metric", f"{SRC} is not a metric model; pseudo-LiDAR needs metric depth"
CKPT = checkpoint_name(SRC, SRC_SPEC, ENCODER)
if len(CANDS) and SRC in set(CANDS["model"]):
    ck02 = CANDS.set_index("model").loc[SRC, "checkpoint"]
    if ck02 != CKPT:
        warnings.warn(f"02 evaluated {ck02} but the current profile gives {CKPT}; 02 caches will not be reused",
                      stacklevel=1)
print(f"pseudo-LiDAR source: {SRC} ({CKPT}) — {SELECTION}")

DIR02 = cfg.path(cfg["results_root"]) / DEPTH_NB / SRC      # read-only: 02's prediction cache
OUT = result_dir(cfg, NOTEBOOK, SRC)
PSEUDO_ROOT = OUT / "pseudo"                                # mmdet3d-style data root for notebook 05


def pred_path(tok: str, cam: str) -> Path:
    """02's cached prediction if it matches the checkpoint, else this notebook's own cache."""
    p02 = pred_cache_path(DIR02, tok, cam)
    return p02 if cached_prediction_ok(p02, CKPT) else pred_cache_path(OUT, tok, cam)


# %% [markdown]
# ## 3. 모델 실행 (02 캐시에 없는 예측만)
#
# 02와 같은 래퍼(`bench3d.models.depth`)와 같은 캐시 형식(float16 npz, 원본 해상도)을 쓴다. DA3METRIC의 focal 처리(처리 해상도 기준)도
# 02와 같다. 새로 계산한 예측은 이 노트북 폴더의 `pred/`에 저장하고 02 폴더에는 쓰지 않는다.

# %%
def load_rgb(cam_sd: str) -> np.ndarray:
    sd = nusc.get("sample_data", cam_sd)
    bgr = cv2.imread(str(Path(nusc.dataroot) / sd["filename"]), cv2.IMREAD_COLOR)
    assert bgr is not None, sd["filename"]
    return np.ascontiguousarray(bgr[..., ::-1])


reuse = sum(cached_prediction_ok(pred_cache_path(DIR02, tok, cam), CKPT) for _sc, tok in TOKENS for cam in CAMS)
todo = [(tok, cam) for _sc, tok in TOKENS for cam in CAMS
        if not cached_prediction_ok(pred_cache_path(DIR02, tok, cam), CKPT)
        and (FORCE_RERUN or not cached_prediction_ok(pred_cache_path(OUT, tok, cam), CKPT))]
print(time.strftime("%H:%M:%S"), f"{reuse} predictions reused from {DEPTH_NB}, {len(todo)} to run")

rt_file = OUT / "runtime_inference.json"
INFER_RT = json.loads(rt_file.read_text()) if is_cached(rt_file, FORCE_RERUN) else {}
if todo:
    times, peaks = [], []
    with free_cuda():
        with Measure() as m_load:
            model = load_depth_model(SRC, SRC_SPEC, encoder=ENCODER)
        for i, (tok, cam) in enumerate(todo):
            cam_sd = nusc.get("sample", tok)["data"][cam]
            rgb, K = load_rgb(cam_sd), intrinsic(nusc, cam_sd)
            with Measure() as m:
                pred = model.predict(rgb, K)
            times.append(m.seconds)
            peaks.append(m.peak_vram_mb)
            save_prediction(pred_cache_path(OUT, tok, cam), pred,
                            {"model": SRC, "checkpoint": CKPT, "sample_token": tok, "cam": cam})
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}  {np.median(times):.3f} s/img")
        del model
    peaks = [p for p in peaks if p is not None]
    INFER_RT = {"checkpoint": CKPT, "load_s": m_load.seconds, "n_images_timed": len(times),
                "median_s_per_image": float(np.median(times[1:] if len(times) > 1 else times)),
                "peak_vram_mb": float(max(peaks)) if peaks else None}
    rt_file.write_text(json.dumps(INFER_RT, indent=1))
print("inference runtime:", INFER_RT or "(all predictions reused)")

# %% [markdown]
# ### 3.1 역투영 → LIDAR_TOP 좌표 → `.bin` 저장
#
# - 카메라마다 `bench3d.nusc.backproject(depth, K, stride=pixel_stride, max_depth=max_depth)`: `pixel_stride` 간격 픽셀만 쓰고,
#   카메라 z(깊이)가 0.1 m 이하이거나 `max_depth`(프로필, mid=60 m) 이상인 점은 버린다. DA3 계열은 하늘 마스크 픽셀도 버린다.
# - 카메라 좌표 → **같은 sample의 LIDAR_TOP** 좌표: `inv(lidar_to_camera(lidar_sd, cam_sd))`. 두 센서의 ego pose(타임스탬프)를 각각
#   거쳐 global로 연결하므로 devkit의 투영과 정확히 역관계다. 그래서 점은 실제 LiDAR와 같은 프레임·같은 시각 기준이 된다.
# - 6개 카메라 점을 이어 붙여 `save_nuscenes_bin`으로 float32 × 5채널 (x, y, z, intensity, ring)로 저장한다.
#   **intensity와 ring은 0으로 채운다.** nuScenes용 mmdet3d 탐지기는 intensity를 입력 특징으로 쓰는 경우가 많아(설정의 `use_dim`),
#   이 자체가 도메인 차이다 → 05 해석 때 고려.
# - 경로: `results/outdoor/04_pseudo_lidar/<model>/pseudo/samples/LIDAR_TOP/<실제 LIDAR_TOP 파일과 같은 이름>.pcd.bin`.
#   05는 `<...>/pseudo`를 점군 data root로 쓰면 된다. 이 루트에는 **키프레임만** 있고 sweeps는 없다(멀티 sweep 설정의 탐지기는
#   sweep 0개로 돌리거나, 실제 sweep을 섞지 않도록 05에서 처리해야 한다).
# - 카메라 사이 겹침 영역에서는 두 카메라의 점이 모두 들어간다(중복 제거 안 함).

# %%
def pseudo_points(tok: str):
    """Pseudo-LiDAR of one sample in its LIDAR_TOP frame: (N, 3) float32 and per-camera counts."""
    s = nusc.get("sample", tok)
    lidar_sd = s["data"]["LIDAR_TOP"]
    pts, n_cam = [], {}
    for cam in CAMS:
        cam_sd = s["data"][cam]
        pr = load_prediction(pred_path(tok, cam))
        mask = ~pr.sky if (DROP_SKY and pr.sky is not None) else None
        xyz_cam, _ = backproject(pr.depth, intrinsic(nusc, cam_sd), stride=PIXEL_STRIDE, max_depth=MAX_DEPTH, mask=mask)
        cam_to_lidar = np.linalg.inv(lidar_to_camera(nusc, lidar_sd, cam_sd))
        pts.append(transform_points(cam_to_lidar, xyz_cam))
        n_cam[cam] = int(len(xyz_cam))
    return np.concatenate(pts).astype(np.float32), n_cam


man_file = OUT / "manifest.json"
old = json.loads(man_file.read_text()).get("samples", {}) if is_cached(man_file, FORCE_RERUN) else {}
SAMPLE_INFO, gen_times = {}, []
for i, (sc, tok) in enumerate(TOKENS):
    lidar_sd = nusc.get("sample", tok)["data"]["LIDAR_TOP"]
    fname = nusc.get("sample_data", lidar_sd)["filename"]       # samples/LIDAR_TOP/<name>.pcd.bin
    assert fname.startswith("samples/LIDAR_TOP/"), fname
    out_bin = PSEUDO_ROOT / fname
    if is_cached(out_bin, FORCE_RERUN) and tok in old:
        n_pts, n_cam = out_bin.stat().st_size // (5 * 4), old[tok].get("n_points_per_cam")
    else:
        t0 = time.perf_counter()
        xyz, n_cam = pseudo_points(tok)
        save_nuscenes_bin(out_bin, xyz)                          # intensity = ring = 0
        gen_times.append(time.perf_counter() - t0)
        n_pts = len(xyz)
    SAMPLE_INFO[tok] = {"scene": sc, "lidar_sd_token": lidar_sd, "file": fname, "n_points": int(n_pts),
                        "n_points_per_cam": n_cam, "cams": CAMS}
    if (i + 1) % 40 == 0 or i + 1 == len(TOKENS):
        print(f"  {i + 1}/{len(TOKENS)} samples, {len(gen_times)} generated")
print(f"mean pseudo points / sample: {np.mean([v['n_points'] for v in SAMPLE_INFO.values()]):.0f}")

# %% [markdown]
# ## 4. 평가: 실제 LiDAR와 비교 (같은 카메라 FOV 안)
#
# **공정 비교 영역**: 두 점군 모두 global 좌표로 옮긴 뒤 `bench3d.metrics.pointcloud.frustum_mask(points, cams, max_depth)`로
# 사용한 카메라(`pseudo_lidar.cams`) 중 하나라도 보이는 점, 카메라 z < `max_depth`인 점만 남긴다. pseudo-LiDAR는 구성상 이미 이 영역 안이다.
#
# **지표**
# - **거리 링별 점 밀도 [점/m²]**: LIDAR_TOP 기준 수평 거리 r = √(x²+y²)를 10 m 링(0–10, 10–20, …, max_depth까지)으로 나눠
#   sample당 점 수를 링 면적으로 나눈다. 링 면적은 **카메라 FOV가 덮는 방위각 비율**을 곱한 근사값이다(LiDAR 높이에서 방위각
#   0.25° 간격으로 시야 포함 여부를 계산). 두 점군을 같은 면적으로 나누므로 **pseudo/real 밀도 비율은 이 근사에 영향받지 않는다.**
# - **높이 분포**: ego 좌표계 z(ego 원점은 후륜 차축 아래 지면 부근)의 히스토그램(−2~6 m, 0.1 m 간격), 전체 sample 합산 후 정규화.
#   실제 LiDAR는 지면에서 날카로운 봉우리가 나와야 하고, pseudo-LiDAR의 지면 봉우리가 넓거나 치우치면 깊이 오차·기울기 오차의 신호다.
# - 실제 LiDAR는 sample마다 키프레임 1스캔(sweep 합치지 않음)만 쓴다.

# %%
RING_EDGES = np.arange(0.0, MAX_DEPTH + 1e-6, 10.0)
RINGS = [(float(lo), float(hi)) for lo, hi in zip(RING_EDGES[:-1], RING_EDGES[1:])]
RING_LABELS = [f"{lo:g}-{hi:g}m" for lo, hi in RINGS]
Z_EDGES = np.arange(-2.0, 6.0 + 1e-6, 0.1)


def fov_ring_area(tok: str) -> np.ndarray:
    """Ring areas [m^2] times the azimuth fraction covered by the cameras (probed at LiDAR height)."""
    lidar_sd = nusc.get("sample", tok)["data"]["LIDAR_TOP"]
    cams = sample_cameras(nusc, tok, CAMS)
    az = np.deg2rad(np.arange(0.0, 360.0, 0.25))
    areas = []
    for lo, hi in RINGS:
        r = 0.5 * (lo + hi)
        probe = np.stack([r * np.cos(az), r * np.sin(az), np.zeros_like(az)], 1)
        frac = frustum_mask(transform_points(sensor_to_global(nusc, lidar_sd), probe), cams).mean()
        areas.append(np.pi * (hi**2 - lo**2) * frac)
    return np.asarray(areas)


def fov_stats(tok: str, pts_lidar: np.ndarray) -> dict:
    """FOV-restricted ring counts and ego-frame height histogram of a cloud in the LIDAR_TOP frame."""
    lidar_sd = nusc.get("sample", tok)["data"]["LIDAR_TOP"]
    cams = sample_cameras(nusc, tok, CAMS)
    keep = frustum_mask(transform_points(sensor_to_global(nusc, lidar_sd), pts_lidar), cams, max_depth=MAX_DEPTH)
    p = pts_lidar[keep]
    r = np.hypot(p[:, 0], p[:, 1])
    ring_n = np.array([int(((r >= lo) & (r < hi)).sum()) for lo, hi in RINGS])
    z_ego = transform_points(sensor_to_ego(nusc, lidar_sd), p)[:, 2]
    return {"n_fov": int(keep.sum()), "n_all": int(len(pts_lidar)), "ring_n": ring_n,
            "z_hist": np.histogram(z_ego, Z_EDGES)[0], "z_q": np.percentile(z_ego, [5, 50, 95]) if len(p) else
            np.full(3, np.nan)}


AREA = fov_ring_area(TOKENS[0][1])          # rig is fixed -> same for every sample
STATS = {"pseudo": [], "real": []}
for _sc, tok in TOKENS:
    lidar_sd = nusc.get("sample", tok)["data"]["LIDAR_TOP"]
    pseudo = load_nuscenes_bin(PSEUDO_ROOT / SAMPLE_INFO[tok]["file"])[:, :3]
    STATS["pseudo"].append(fov_stats(tok, pseudo))
    STATS["real"].append(fov_stats(tok, load_lidar(nusc, lidar_sd)[:, :3]))

DENS = {k: np.stack([s["ring_n"] for s in v]) / AREA for k, v in STATS.items()}      # (n_samples, n_rings)
ZH = {k: np.sum([s["z_hist"] for s in v], 0) for k, v in STATS.items()}
dens_df = pd.DataFrame({"fov_area_m2": AREA, "real_pts_per_m2": DENS["real"].mean(0),
                        "pseudo_pts_per_m2": DENS["pseudo"].mean(0)}, index=RING_LABELS)
dens_df["pseudo/real"] = dens_df["pseudo_pts_per_m2"] / dens_df["real_pts_per_m2"]
display(dens_df.round(3))
for k in ("real", "pseudo"):
    q = np.nanmedian(np.stack([s["z_q"] for s in STATS[k]]), 0)
    print(f"{k:6s}: points/sample in FOV {np.mean([s['n_fov'] for s in STATS[k]]):9.0f} | "
          f"ego z p5/p50/p95 (median over samples) {q.round(2)}")

# %% [markdown]
# ## 5. 결과 저장
#
# - `manifest.json`: 05가 읽는 목록. sample_token → `.bin` 상대 경로(`pseudo/` 기준), 점 수, 카메라. 헤더에 소스 모델·체크포인트·선택 근거·
#   역투영 설정·채널 의미(intensity=0, ring=0)를 기록한다.
# - `summary.json`: 밀도·높이 통계(평평한 숫자). `runtime_s`는 sample당 역투영+변환+저장 시간 중앙값(이번 실행에서 생성한 것만),
#   추론 시간·VRAM은 `extra.inference`(02 캐시를 재사용한 예측은 포함되지 않음). `extra.selected_for_detection=true`는 05가
#   pseudo-LiDAR 소스를 고를 때 읽는 표시이며, 다른 모델의 이전 04 출력에서는 false로 내린다.

# %%
MANIFEST = {
    "model": SRC, "checkpoint": CKPT, "selection": SELECTION, "profile": cfg["profile"],
    "data_root": str(PSEUDO_ROOT.relative_to(cfg.root)),
    "format": "nuScenes LIDAR_TOP .bin, float32 x5 (x, y, z, intensity, ring); same basename as the real file",
    "zero_filled_channels": {"intensity": 0.0, "ring": 0.0},
    "frame": "LIDAR_TOP sensor frame of the same sample (camera -> global -> LIDAR_TOP through each ego pose)",
    "cams": CAMS, "pixel_stride": PIXEL_STRIDE, "max_depth": MAX_DEPTH, "drop_sky": DROP_SKY,
    "scenes": EVAL_SCENES, "n_samples": len(SAMPLE_INFO), "sweeps": "none (keyframes only)",
    "samples": SAMPLE_INFO,
}
man_file.write_text(json.dumps(MANIFEST, indent=1))

metrics = {"n_samples": len(SAMPLE_INFO),
           "n_points_pseudo_mean": float(np.mean([v["n_points"] for v in SAMPLE_INFO.values()])),
           "n_points_pseudo_fov_mean": float(np.mean([s["n_fov"] for s in STATS["pseudo"]])),
           "n_points_real_fov_mean": float(np.mean([s["n_fov"] for s in STATS["real"]])),
           "n_points_real_all_mean": float(np.mean([s["n_all"] for s in STATS["real"]]))}
for j, lab in enumerate(RING_LABELS):
    metrics[f"density_pseudo@{lab}"] = float(DENS["pseudo"][:, j].mean())
    metrics[f"density_real@{lab}"] = float(DENS["real"][:, j].mean())
    metrics[f"density_ratio@{lab}"] = float(DENS["pseudo"][:, j].mean() / max(DENS["real"][:, j].mean(), 1e-12))
for k in ("pseudo", "real"):
    q = np.nanmedian(np.stack([s["z_q"] for s in STATS[k]]), 0)
    metrics[f"z_ego_p05_{k}"], metrics[f"z_ego_p50_{k}"], metrics[f"z_ego_p95_{k}"] = (float(x) for x in q)

save_summary(
    OUT, part=cfg["_part"], notebook=NOTEBOOK, profile=cfg["profile"], variant=f"{len(CAMS)}cam/stride{PIXEL_STRIDE}",
    model=SRC, metrics=metrics,
    runtime_s=float(np.median(gen_times)) if gen_times else None,
    peak_vram_mb=INFER_RT.get("peak_vram_mb"),
    checkpoint=CKPT, packages=ENV["packages"], seed=cfg["seed"],
    extra={"selection": SELECTION, "candidates": cand_rows, "runtime_s_meaning": "median s/sample for backproject+save",
           "inference": INFER_RT, "n_predictions_reused_from_02": int(reuse), "ring_edges_m": RING_EDGES.tolist(),
           "fov_ring_area_m2": AREA.tolist(), "z_edges_m": [float(Z_EDGES[0]), float(Z_EDGES[-1]), 0.1],
           "z_hist_pseudo": ZH["pseudo"].tolist(), "z_hist_real": ZH["real"].tolist(),
           "manifest": str(man_file.relative_to(cfg.root)), "zero_filled_channels": ["intensity", "ring"],
           "selected_for_detection": True},       # read by 05 to pick the pseudo-LiDAR source
)
# Only the latest source stays flagged for 05 (older 04 outputs of other models are kept but unflagged).
for f in OUT.parent.glob("*/summary.json"):
    s = json.loads(f.read_text())
    if s.get("model") != SRC and s.get("extra", {}).get("selected_for_detection"):
        s["extra"]["selected_for_detection"] = False
        f.write_text(json.dumps(s, indent=2, ensure_ascii=False))
print("saved:", man_file.relative_to(cfg.root), "and", (OUT / "summary.json").relative_to(cfg.root))

# %% [markdown]
# ## 6. 시각화
#
# ### 6.1 BEV 겹쳐 그리기
# 회색 = 실제 LiDAR(카메라 FOV 안), 파랑 = pseudo-LiDAR, 주황 박스 = GT 3D 박스(LIDAR_TOP 좌표). x 전방, y 좌측.

# %%
BEV_EXTENT = min(MAX_DEPTH, 50.0)
BEV_MAX_PTS = 150_000
rng = np.random.default_rng(cfg["seed"])
VIZ = [(sc, middle(SAMPLES[sc])) for sc in EVAL_SCENES[:3]]

fig, axes = plt.subplots(1, len(VIZ), figsize=(6 * len(VIZ), 6))
for ax, (sc, tok) in zip(np.atleast_1d(axes), VIZ):
    lidar_sd = nusc.get("sample", tok)["data"]["LIDAR_TOP"]
    cams = sample_cameras(nusc, tok, CAMS)
    real = load_lidar(nusc, lidar_sd)[:, :3]
    real = real[frustum_mask(transform_points(sensor_to_global(nusc, lidar_sd), real), cams, max_depth=MAX_DEPTH)]
    pseudo = load_nuscenes_bin(PSEUDO_ROOT / SAMPLE_INFO[tok]["file"])[:, :3]
    if len(pseudo) > BEV_MAX_PTS:
        pseudo = pseudo[rng.choice(len(pseudo), BEV_MAX_PTS, replace=False)]
    _, boxes, _ = nusc.get_sample_data(lidar_sd)                     # boxes in the LIDAR_TOP sensor frame
    boxes_xy = [b.bottom_corners()[:2].T for b in boxes]
    bev(ax, pseudo, extent=BEV_EXTENT, color="#0072B2", s=0.15, label="pseudo-LiDAR")
    bev(ax, real, extent=BEV_EXTENT, color="#555555", s=0.6, label="LiDAR (camera FOV)", boxes_xy=boxes_xy)
    ax.set_title(f"{sc} {tok[:8]}", fontsize=9)
    ax.legend(fontsize=8, markerscale=10, loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2 거리 링별 점 밀도와 높이 분포

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
x = np.arange(len(RING_LABELS))
axes[0].bar(x - 0.2, dens_df["real_pts_per_m2"], 0.4, color="#555555", label="LiDAR (FOV)")
axes[0].bar(x + 0.2, dens_df["pseudo_pts_per_m2"], 0.4, color="#0072B2", label="pseudo-LiDAR")
axes[0].set_yscale("log")
axes[0].set_ylabel("points / m² (FOV-restricted ring area)")
axes[0].set_title("density per range ring (mean over samples)")
axes[1].plot(x, dens_df["pseudo/real"], "o-", color="#0072B2")
axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
axes[1].set_yscale("log")
axes[1].set_title("density ratio pseudo / real (area-independent)")
for ax in axes[:2]:
    ax.set_xticks(x)
    ax.set_xticklabels(RING_LABELS, fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
axes[0].legend(fontsize=8)
zc = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
for k, col in (("real", "#555555"), ("pseudo", "#0072B2")):
    axes[2].plot(zc, ZH[k] / max(ZH[k].sum(), 1), color=col, label=k)
axes[2].set_xlabel("ego-frame z [m]")
axes[2].set_ylabel("fraction of points")
axes[2].set_title("height distribution (camera FOV, r < max_depth)")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. 해석
#
# > (실행 후 작성) 아래 질문에 표·그래프의 숫자로 답한다.
#
# 1. **소스 모델**: 선택된 모델과 02에서의 AbsRel·δ1, 2위 모델과의 차이.
#    > (실행 후 작성)
# 2. **밀도**: 가까운 링에서 pseudo-LiDAR가 몇 배 과밀한가? 먼 링(40–60 m)에서 비율은? (`pixel_stride`가 밀도를 정하므로
#    05 탐지기의 voxel/pillar 상한과 함께 본다)
#    > (실행 후 작성)
# 3. **높이 분포**: 지면 봉우리 위치·폭이 실제 LiDAR와 얼마나 다른가? 차량 높이(0.5–2 m) 구간 비율은?
#    > (실행 후 작성)
# 4. **BEV 모양**: GT 박스 주변에서 점이 시선 방향으로 늘어지는가(깊이 오차의 방향성)? 물체 경계의 "꼬리(flying pixel)"는?
#    > (실행 후 작성)
# 5. **05로 넘길 주의점**: intensity·ring 0 채움, sweep 없음, 카메라 겹침 영역 중복 점 — 탐지 성능 저하 원인 후보.
#    > (실행 후 작성)
