# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 02. 단안 깊이 추정 (Monocular Depth) — 카메라 한 장 vs LiDAR
#
# ## 목적
# nuScenes 카메라 이미지 **한 장**만으로 추정한 깊이가 LiDAR 실측과 얼마나 맞는지 정량 비교한다.
# 전체 평균뿐 아니라 **거리 구간별**, **카메라별** 오차와 "거리에 따라 오차가 얼마나 빨리 커지는가"를 본다.
# 여기서 고른 모델과 캐시한 깊이는 `04_pseudo_lidar`의 입력이 된다.
#
# ## 비교 모델과 원리
# - **Depth Anything V2 Metric — Virtual KITTI 2 (`dav2_metric_vkitti`)**
#   DINOv2 ViT 인코더 + DPT 디코더. 대규모 의사 라벨로 학습한 상대 깊이 모델을 합성 주행 데이터(VKITTI2)로 metric 미세조정했다.
#   마지막 층이 `Sigmoid × max_depth`라서 출력은 (0, 80) m로 **상한이 있다** (`metric_depth/depth_anything_v2/dpt.py`).
#   카메라 내부 파라미터를 입력받지 않으므로, 화각이 학습 카메라와 다르면 스케일이 틀어질 수 있다 → 카메라별 분석에서 확인.
#   코드 기본값 `max_depth=20`은 실내용이라 반드시 80을 넘긴다.
# - **Depth Anything 3 Metric (`da3_metric_large`, DA3METRIC-LARGE)**
#   DA3의 단안 metric 전용 모델(ViT-L + DPT, 하늘 분할 헤드 포함). 네트워크는 초점거리에 무관한 "정규화 깊이"를 내고,
#   미터 = `focal_px × 출력 / 300`으로 사후 변환한다(README FAQ). 즉 화각 정보를 **보정값으로 곱해** 카메라 차이를 흡수한다.
#   어느 해상도의 focal을 써야 하는지는 아래 3절에서 소스 근거로 정한다.
# - **Depth Anything 3 Nested (`da3_nested_giant_large`, DA3NESTED-GIANT-LARGE-1.1, high 프로필만)**
#   any-view giant 모델(깊이 + 카메라 추정)과 위 metric 모델을 중첩한 구조. metric 분기 출력을 giant가 **예측한** 초점거리로
#   미터화한 뒤 최소제곱 스케일로 giant 깊이에 맞춰, 출력이 바로 미터다. `-1.1`은 주행 장면 학습 버그를 고친 재학습판.
# - **Depth Anything 3 Mono (`da3_mono_large`, DA3MONO-LARGE)**
#   상대 깊이 전용 모델(시차가 아닌 깊이를 직접 예측). 스케일이 없으므로 **영상마다 LiDAR 중앙값으로 스케일을 맞춘 뒤(median
#   scaling)** 평가한다. 이 값은 "모양(상대 구조)" 정확도이고, 정답 스케일을 몰래 쓴 결과이므로 metric 모델의 원 출력과 같은 줄에서
#   비교하면 안 된다. 표에는 항상 "median-scaled"로 표시한다.
#
# ## 원 논문
# - Depth Anything V2 — https://arxiv.org/abs/2406.09414
# - Depth Anything 3 — https://arxiv.org/abs/2511.10647 (arXiv id는 `Depth-Anything-3/README.md` 배지·인용에서 확인)
#
# ## 커널
# `3dv` (torch 2.1.2, cu121). DA V2는 `third_party/Depth-Anything-V2/metric_depth`가 `.pth`로 sys.path에 올라가 있고,
# DA3는 `depth_anything_3` 패키지(editable)로 설치돼 있다.
#
# ## 예상 VRAM·시간 (추정)
# 아래는 **추정치**다(실측 아님). 실측은 이 노트북이 `runtime.json`/`summary.json`에 기록한다.
#
# | 프로필 | 입력 | 모델 | 피크 VRAM(추정) | 시간(추정) |
# |---|---|---|---|---|
# | low (≤8GB) | CAM_FRONT, stride 4 (~40장) | DAV2-B, DA3METRIC-L, DA3MONO-L | 각 2–3 GB | 모델당 1–2분 |
# | mid (RTX 3060 12GB) | 6캠, stride 2 (~490장) | DAV2-L, DA3METRIC-L, DA3MONO-L | 각 2–4 GB | 모델당 3–8분 + 평가 수 분 |
# | high (24GB+) | 6캠, stride 1 (~970장) | + DA3NESTED-GIANT-LARGE-1.1 | Nested 8–12 GB | 전체 30–60분 |
#
# 디스크: 예측 캐시가 float16 npz(1600×900)라 장당 약 1.5–3 MB, mid 프로필 기준 모델당 약 1 GB(추정).
#
# ## 이 노트북이 답하려는 질문
# 1. metric 모델은 nuScenes에서 **스케일까지** 맞추는가? (원 출력 vs median-scaled의 차이 = 스케일 오차, 나머지 = 모양 오차)
# 2. 거리 구간(0–10, 10–20, 20–30, 30–50, 50 m+)별로 어디서 무너지는가?
# 3. 오차가 거리에 따라 어떤 거듭제곱으로 커지는가? (log-log 기울기 k)
# 4. 카메라(전방/측방/후방, 화각 차이)에 따라 결과가 달라지는가?
# 5. 04 pseudo-LiDAR에는 어느 모델을 쓰는 것이 타당한가?

# %% [markdown]
# ## 1. 설정 로드

# %%
FORCE_RERUN = False            # True: ignore cached results
NOTEBOOK = "02_monocular_depth"

import json
import time
from pathlib import Path

import numpy as np
from bench3d.config import load_config
from bench3d.io import is_cached, result_dir, save_summary
from bench3d.runtime import Measure, env_report, free_cuda, set_seed

cfg = load_config("outdoor")                 # profile: config.yaml or $THREEDCHECK_PROFILE
P = cfg.p["depth"]                           # this notebook's profile block
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
from bench3d.align import median_scale
from bench3d.metrics.depth import DEFAULT_BINS, bin_label, binned_depth_metrics, depth_metrics, error_vs_distance
from bench3d.models.depth import (
    cached_prediction_ok,
    checkpoint_name,
    depth_kind,
    load_depth_model,
    load_prediction,
    pred_cache_path,
    save_prediction,
)
from bench3d.nusc import intrinsic, load_lidar, project_lidar_to_camera, sparse_depth_map
from bench3d.sampling import middle, subsample
from bench3d.viz import depth_panel, error_vs_distance_plot, model_colors, overlay_points
from nuscenes.nuscenes import NuScenes
from scipy.ndimage import minimum_filter

# Pinned upstream commits (envs/third_party.lock) are logged into every summary.
_lock = [ln.split() for ln in cfg.path("envs/third_party.lock").read_text().splitlines()
         if ln.strip() and not ln.startswith("#")]
TP_COMMITS = {r[0]: r[2] for r in _lock if r[0] in ("Depth-Anything-V2", "Depth-Anything-3")}
print("third_party:", TP_COMMITS, "| dir:", cfg.third_party())

# The env's .pth must expose the METRIC Depth-Anything-V2 (its DepthAnythingV2 takes max_depth).
import depth_anything_v2.dpt as _dav2_dpt

assert "metric_depth" in Path(_dav2_dpt.__file__).parts, f"non-metric DA V2 on sys.path: {_dav2_dpt.__file__}"
if not Path(_dav2_dpt.__file__).resolve().is_relative_to(cfg.third_party("Depth-Anything-V2").resolve()):
    warnings.warn(f"depth_anything_v2 imported from {_dav2_dpt.__file__}, not from cfg.third_party()", stacklevel=1)

MODELS = list(P["models"])
MODEL_SPECS = cfg["models"]["depth"]
ENCODER = P.get("dav2_encoder")
CAMS = list(P["cams"])
MAXD = float(P["max_eval_depth"])
print("models:", MODELS, "| dav2 encoder:", ENCODER, "| cams:", CAMS, "| max eval depth:", MAXD)

# %% [markdown]
# ## 2. 데이터 로드
#
# - **평가 표본**: `01_data_exploration`이 저장한 `eval_samples.json`의 `eval4` 장면(모두 공식 val 분할 → 학습 누수 없음,
#   `docs/step0_research.md` 2절)에서 키프레임을 `subsample(keyframes, sample_stride)`로 고른다. 카메라는 프로필의 `cams`.
# - **정답(GT)**: 같은 sample의 `LIDAR_TOP` 키프레임을 각 카메라로 투영한 희소 깊이(`bench3d.nusc.project_lidar_to_camera`,
#   `min_depth=1.0` = devkit `map_pointcloud_to_image` 기본값, 자차 차체 점 제거). 한 픽셀에 여러 점이 떨어지면 가장 가까운 점을
#   남긴다(`sparse_depth_map`의 z-buffer). 카메라와 LiDAR는 타임스탬프가 조금 다르므로 각자의 ego pose를 거쳐 global 좌표로 연결된다.
#
# > **LiDAR–카메라 시차(parallax)와 가림(occlusion) 주의**
# > LIDAR_TOP은 카메라보다 높고(지붕 중앙) 옆·뒤로도 떨어져 있다. 그래서 LiDAR는 보이지만 카메라에서는 **앞 물체에 가려진 뒷면/배경**
# > 점이 이미지에 투영되어, 물체 경계 근처에서 "앞 물체 픽셀에 먼 깊이 GT"가 붙는 경우가 생긴다. 이 점들은 모델이 맞게 예측해도
# > 큰 오차로 잡힌다(특히 측방 카메라, 가까운 차량·기둥 주변).
# > 이번 노트북은 **필터 없는 결과를 주 결과(primary)** 로 둔다. 보조로 간단한 휴리스틱 필터를 적용한 결과(`occlusion_filtered`)도
# > 저장한다: 각 GT 점에 대해 주변 창(세로 7 × 가로 15 px) 안의 가장 가까운 GT 깊이보다 10% 넘게 멀면 가려진 점으로 보고 제거한다.
# > 가로 창 15 px는 같은 LiDAR 링에서 이웃 점(수평 각해상도 약 0.2–0.3° ≈ 수~7 px)을 포함하도록 잡은 값이며, 실제 깊이 경계의
# > 정상 점도 일부 지운다(보수적으로 해석). 두 결과의 차이가 크면 그만큼 GT 자체의 노이즈가 메트릭에 섞여 있다는 뜻이다.

# %%
nusc = NuScenes(version=cfg["dataset"]["version"], dataroot=str(cfg.path(cfg["dataset"]["data_root"])), verbose=False)

ES_PATH = cfg.path(cfg["results_root"]) / "01_data_exploration" / "eval_samples.json"
assert ES_PATH.exists(), f"{ES_PATH} not found: run 01_data_exploration first"
ES = json.loads(ES_PATH.read_text())
assert ES["version"] == cfg["dataset"]["version"], (ES["version"], cfg["dataset"]["version"])
EVAL_SCENES = list(cfg["splits"]["eval4"])
assert set(ES["eval4"]) == set(EVAL_SCENES), f"eval4 mismatch: {ES['eval4']} vs config {EVAL_SCENES}"

SAMPLES = {sc: subsample(ES["keyframes"][sc], P["sample_stride"]) for sc in EVAL_SCENES}
ITEMS = [(sc, tok, cam) for sc in EVAL_SCENES for tok in SAMPLES[sc] for cam in CAMS]
print({sc: len(v) for sc, v in SAMPLES.items()}, "->", len(ITEMS), "images (sample x camera)")

# Intrinsics differ per camera (field of view matters for models that do not take K).
_s0 = nusc.get("sample", SAMPLES[EVAL_SCENES[0]][0])
for cam in CAMS:
    K = intrinsic(nusc, _s0["data"][cam])
    sd = nusc.get("sample_data", _s0["data"][cam])
    print(f"{cam:16s} fx={K[0, 0]:7.1f} fy={K[1, 1]:7.1f} cx={K[0, 2]:6.1f} cy={K[1, 2]:6.1f}  {sd['width']}x{sd['height']}")


def load_rgb(cam_sd: str) -> np.ndarray:
    """RGB uint8 image of a camera sample_data."""
    sd = nusc.get("sample_data", cam_sd)
    bgr = cv2.imread(str(Path(nusc.dataroot) / sd["filename"]), cv2.IMREAD_COLOR)
    assert bgr is not None, sd["filename"]
    return np.ascontiguousarray(bgr[..., ::-1])


# %%
GT_MIN_DEPTH = 1.0    # devkit map_pointcloud_to_image default (drops ego-car points)
OCC_WIN = tuple(P["occlusion_window"])  # (rows, cols) window of the occlusion heuristic (see markdown above)
OCC_REL = float(P["occlusion_rel"])  # drop a GT point if it is this much farther than the nearest in its window


def occlusion_keep(v, u, d, H, W, win=OCC_WIN, rel=OCC_REL):
    """True for GT points not flagged as seen-by-LiDAR-but-occluded-in-camera."""
    zb = np.full((H, W), np.inf, np.float32)
    zb[v, u] = d
    near = minimum_filter(zb, size=win, mode="constant", cval=np.inf)
    return d <= near[v, u] * (1.0 + rel)


def lidar_gt(sample_token: str, cams) -> dict:
    """Sparse LiDAR depth per camera: dict cam -> {v, u, d, occ_keep} (z-buffered, one point per pixel)."""
    s = nusc.get("sample", sample_token)
    lidar_sd = s["data"]["LIDAR_TOP"]
    pts = load_lidar(nusc, lidar_sd)
    out = {}
    for cam in cams:
        cam_sd = s["data"][cam]
        sd = nusc.get("sample_data", cam_sd)
        pr = project_lidar_to_camera(nusc, lidar_sd, cam_sd, min_depth=GT_MIN_DEPTH, points=pts)
        dm = sparse_depth_map(pr["uv"], pr["depth"], sd["width"], sd["height"])
        v, u = np.nonzero(dm)
        d = dm[v, u]
        out[cam] = {"v": v, "u": u, "d": d, "occ_keep": occlusion_keep(v, u, d, sd["height"], sd["width"])}
    return out


GT = {}
for sc in EVAL_SCENES:
    for tok in SAMPLES[sc]:
        for cam, g in lidar_gt(tok, CAMS).items():
            GT[(tok, cam)] = g
_n = np.array([len(g["d"]) for g in GT.values()])
_occ = np.array([1 - g["occ_keep"].mean() for g in GT.values() if len(g["d"])])
print(f"GT points / image: mean {_n.mean():.0f}, min {_n.min()}, max {_n.max()} | "
      f"occlusion heuristic removes {100 * _occ.mean():.1f}% on average")

# %% [markdown]
# ## 3. 모델별 실행
#
# 모델 하나씩 로드 → 전체 (sample, camera) 추론 → 캐시 저장 → 메모리 해제. 예측은 원본 해상도(1600×900) float16 npz로
# `results/outdoor/02_monocular_depth/<model>/pred/<sample_token>_<cam>.npz`에 저장한다(04가 재사용). 캐시에는 체크포인트 이름이
# 들어 있어, 프로필(예: `dav2_encoder`)이 바뀌면 자동으로 다시 계산한다. 공통 래퍼는 `bench3d.models.depth`.
#
# **입력 규칙**
# - DA V2: `infer_image(BGR, input_size=518)` — 짧은 변을 518(14의 배수)로 맞춰 추론하고, 결과를 원래 H×W로 bilinear 보간해 돌려준다
#   (`metric_depth/depth_anything_v2/dpt.py:188-195`). `max_depth=80`을 생성자에 넘긴다(기본 20, `dpt.py:160`).
# - DA3: `model.inference([image])`에 **이미지만** 넘긴다. extrinsics/intrinsics를 넘기면 pose-conditioned 경로로 바뀐다
#   (`api.py:203-219`, `_normalize_extrinsics`, `_align_to_input_extrinsics_intrinsics`). 한 번에 한 장씩 넣어 다시점 상호작용을 막는다.
#
# **DA3METRIC의 focal 처리 — 소스 근거** (Depth-Anything-3 @ `3d835ec`)
# 1. 출력 해상도: `inference()` 기본값 `process_res=504`, `process_res_method="upper_bound_resize"` (`api.py:145-146`).
#    긴 변을 504로 줄이고(`utils/io/input_processor.py:324-334`) 각 변을 14의 배수로 반올림한다(`input_processor.py:365-383`).
#    1600×900 → 504×284 → **504×280** (sx = 0.3150, sy = 0.3111; 정적 호출로 확인). 깊이 헤드는 이 처리 해상도 그대로 출력하고
#    (`model/dpt.py:233-237`, `down_ratio=1`), `OutputProcessor`도 리사이즈하지 않는다(`utils/io/output_processor.py:87`).
#    → **깊이는 처리 해상도로 반환된다.** 평가를 위해 1600×900으로 되돌린다(값은 미터라 공간 리사이즈만, `cv2.INTER_LINEAR`).
# 2. focal 기준: 공식 코드에서 이 변환을 직접 쓰는 유일한 곳은 Nested 모델이고, 그 metric 분기는 DA3METRIC-LARGE와 같은 구성이다
#    (`configs/da3nested-giant-large.yaml:9-10`, `__inherit__: da3metric-large`). `NestedDepthAnything3Net.forward`는 **같은 처리 해상도
#    텐서 `x`** 를 metric 분기에 넣고(`model/da3.py:365`), 그 출력에 `apply_metric_scaling(depth, output.intrinsics)`를 적용한다
#    (`da3.py:379-382`, `utils/alignment.py:132-133`: `depth × ((fx+fy)/2) / 300`). 이 `intrinsics`는
#    `pose_encoding_to_extri_intri(pose_enc, (H, W))`에서 **처리 해상도 H, W**로 만든 값이다(`da3.py:136, 224`,
#    `model/utils/transform.py:55-57`). README FAQ(`README.md:235`)는 "focal length in pixels"라고만 한다.
#    → **처리 해상도 기준 focal**을 쓴다: `f_proc = (fx·sx + fy·sy)/2` (원본 fx≈1266이면 f_proc≈399).
#    원본 해상도 focal을 쓰면 깊이가 약 3.2배(1/0.315) 커진다. 아래 평가 셀에서 LiDAR 중앙값과의 비율로 런타임 확인을 하고,
#    2배 넘게 어긋나면 경고한다.

# %%
RUNTIME = {}

for model_id in MODELS:
    spec = MODEL_SPECS[model_id]
    ckpt = checkpoint_name(model_id, spec, ENCODER)
    mdir = result_dir(cfg, NOTEBOOK, model_id)
    rt_file = mdir / "runtime.json"
    todo = [(sc, tok, cam) for sc, tok, cam in ITEMS
            if FORCE_RERUN or not cached_prediction_ok(pred_cache_path(mdir, tok, cam), ckpt)]
    print(time.strftime("%H:%M:%S"),
          f"[{model_id}] {ckpt} ({depth_kind(spec)}): {len(ITEMS) - len(todo)} cached, {len(todo)} to run")
    if not todo:  # timings of the run that produced the cache
        RUNTIME[model_id] = json.loads(rt_file.read_text()) if is_cached(rt_file) else {"checkpoint": ckpt}
        continue
    times, peaks = [], []
    with free_cuda():
        with Measure() as m_load:
            model = load_depth_model(model_id, spec, encoder=ENCODER)
        for i, (_sc, tok, cam) in enumerate(todo):
            cam_sd = nusc.get("sample", tok)["data"][cam]
            rgb, K = load_rgb(cam_sd), intrinsic(nusc, cam_sd)
            with Measure() as m:
                pred = model.predict(rgb, K)
            times.append(m.seconds)
            peaks.append(m.peak_vram_mb)
            save_prediction(pred_cache_path(mdir, tok, cam), pred,
                            {"model": model_id, "checkpoint": ckpt, "sample_token": tok, "cam": cam})
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}  {np.median(times):.3f} s/img")
        del model
    peaks = [p for p in peaks if p is not None]
    RUNTIME[model_id] = {
        "checkpoint": ckpt,
        "load_s": m_load.seconds,
        "n_images_timed": len(times),
        "median_s_per_image": float(np.median(times[1:] if len(times) > 1 else times)),  # skip warm-up image
        "mean_s_per_image": float(np.mean(times)),
        "peak_vram_mb": float(max(peaks)) if peaks else None,
        "per_image_s": times,
    }
    rt_file.write_text(json.dumps(RUNTIME[model_id], indent=1))

print({m: {k: RUNTIME[m].get(k) for k in ("median_s_per_image", "peak_vram_mb", "load_s")} for m in MODELS})

# %% [markdown]
# ## 4. 평가
#
# **메트릭 정의**
# - 평가 픽셀: LiDAR GT가 있는 픽셀 중 `1.0 m < GT < max_eval_depth`(프로필, mid=80 m). 예측은 GT 픽셀 위치에서 읽는다
#   (원본 해상도, 보간 없음). 예측은 `min_depth=1e-3`으로 아래쪽만 자른다(`bench3d.metrics.depth.depth_metrics`).
# - 지표: AbsRel, SqRel, RMSE, RMSE_log, δ<1.25ᵏ (Eigen et al. 2014). **모든 (sample, camera)의 유효 픽셀을 한데 모아 계산**한다
#   (pixel-pooled). 참고용으로 영상별 평균(KITTI 방식) AbsRel·δ1도 `*_imgmean`으로 남긴다.
# - 거리 구간: `DEFAULT_BINS` = 0–10, 10–20, 20–30, 30–50, 50 m+ (GT 거리 기준). `max_eval_depth`가 80이므로 "50m+"는 실제로 50–80 m다.
#   DA V2 metric은 출력 상한이 80 m(`Sigmoid × max_depth`)라 먼 구간에서 구조적으로 과소추정될 수 있다.
# - 오차-거리 곡선: GT 거리를 2–80 m 로그 간격 20구간으로 나눠 구간별 평균 |오차|를 구하고 `|err| ≈ c·d^k`를 log-log 직선으로 맞춘다
#   (`error_vs_distance`, 구간당 50점 미만은 제외). **k ≈ 2는 스테레오 삼각측량의 법칙**이다: 깊이 z = f·B/disparity라서
#   disparity 오차가 일정하면 깊이 오차 ∝ z². 단안 네트워크에는 이런 물리 법칙이 내장돼 있지 않으므로 여기서의 k는 **오차 증가 양상을
#   요약하는 기술 통계일 뿐** 모델의 원리를 검증하는 값이 아니다. k ≈ 1이면 상대 오차가 거리에 무관하게 일정하다는 뜻이다.
#
# **스케일 정렬 (변형별 정의)**
#
# | 변형 (`summary*.json`) | metric 모델 | 상대 깊이 모델(DA3MONO) |
# |---|---|---|
# | `primary` → `summary.json` | **원 출력 그대로 (주 결과)** | 영상별 median scaling (주 결과, "median-scaled") |
# | `median_scaled` → `summary_median_scaled.json` | 영상별 median scaling (**진단용**) | — |
# | `occlusion_filtered` → `summary_occlusion_filtered.json` | primary와 같은 예측, 가림 필터 GT | 같음 |
#
# median scaling: 영상마다 `s = median(GT) / median(pred)`를 평가 픽셀(GT < max_eval_depth)에서 구해 곱한다(`bench3d.align.median_scale`).
# GT를 써서 스케일을 맞추므로 **오라클**이다. metric 모델의 primary와 median_scaled 차이가 "스케일 오차", median_scaled에 남은 오차가
# "모양 오차"다. 모델 간 순위는 같은 변형끼리만 비교한다.
#
# **스케일 런타임 확인 (sanity check)**: metric 모델마다 영상별 `median(pred)/median(GT)`의 중앙값을 출력하고, 0.5–2 범위를 벗어나면
# 경고한다. DA3METRIC은 "원본 해상도 focal을 썼다면" 얼마였을지도 함께 출력한다.

# %%
KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")
SANITY_RANGE = (0.5, 2.0)


def gather(model_id: str) -> dict:
    """Prediction sampled at GT pixels for every ITEM, concatenated in ITEMS order."""
    mdir = result_dir(cfg, NOTEBOOK, model_id)
    out = {k: [] for k in ("pred", "gt", "cam", "img", "occ")}
    scales, infos = [], []
    for k, (_sc, tok, cam) in enumerate(ITEMS):
        g = GT[(tok, cam)]
        pr = load_prediction(pred_cache_path(mdir, tok, cam))
        p = pr.depth[g["v"], g["u"]]
        try:
            s = median_scale(p, g["d"], mask=g["d"] < MAXD)
        except ValueError:  # no valid GT in this image
            s = np.nan
        scales.append(s)
        infos.append(pr.info)
        out["pred"].append(p)
        out["gt"].append(g["d"])
        out["cam"].append(np.full(len(p), CAMS.index(cam), np.int8))
        out["img"].append(np.full(len(p), k, np.int32))
        out["occ"].append(g["occ_keep"])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["scale"] = np.asarray(scales, float)
    out["infos"] = infos
    return out


def evaluate(pred, gt, cam, img) -> dict:
    overall = depth_metrics(pred, gt, max_depth=MAXD)
    bins = binned_depth_metrics(pred, gt, bins=DEFAULT_BINS, max_depth=MAXD)
    curve = error_vs_distance(pred, gt, mask=gt < MAXD, d_min=2.0, d_max=MAXD)
    per_cam = {c: depth_metrics(pred[cam == i], gt[cam == i], max_depth=MAXD) for i, c in enumerate(CAMS)}
    cuts = np.flatnonzero(np.diff(img)) + 1  # img is sorted (ITEMS order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_img = [depth_metrics(p_, g_, max_depth=MAXD) for p_, g_ in zip(np.split(pred, cuts), np.split(gt, cuts))]
        img_mean = {k: float(np.nanmean([d[k] for d in per_img])) for k in ("abs_rel", "delta1")}
    return {"overall": overall, "bins": bins, "curve": curve, "per_cam": per_cam, "img_mean": img_mean}


def flat_metrics(ev: dict) -> dict:
    """summary.json metrics: overall + per-bin (`abs_rel@0-10m`) + per-camera + reference values, numbers only."""
    m = {k: ev["overall"][k] for k in KEYS}
    m["n"] = ev["overall"]["n"]
    for b, d in ev["bins"].items():
        for k in KEYS:
            m[f"{k}@{b}"] = d[k]
        m[f"n@{b}"] = d["n"]
    for c, d in ev["per_cam"].items():
        m[f"abs_rel@{c}"] = d["abs_rel"]
        m[f"delta1@{c}"] = d["delta1"]
    m["abs_rel_imgmean"] = ev["img_mean"]["abs_rel"]
    m["delta1_imgmean"] = ev["img_mean"]["delta1"]
    m["err_slope_k"] = ev["curve"]["slope_k"]
    return m


VARIANTS, EVALS, SANITY, DATA = {}, {}, {}, {}
for model_id in MODELS:
    kind = depth_kind(MODEL_SPECS[model_id])
    D = gather(model_id)
    DATA[model_id] = D
    pred_ms = D["pred"] * D["scale"][D["img"]]  # per-image median-scaled
    if kind == "metric":
        VARIANTS[(model_id, "primary")] = {"pred": D["pred"], "sel": None, "scaling": "none", "gt_filter": "none",
                                           "primary": True, "label": "metric (raw output)"}
        VARIANTS[(model_id, "median_scaled")] = {"pred": pred_ms, "sel": None, "scaling": "median", "gt_filter": "none",
                                                 "primary": False, "label": "median-scaled (diagnostic)"}
        ratio = float(np.nanmedian(1.0 / D["scale"]))  # median over images of median(pred)/median(gt)
        SANITY[model_id] = {"pred_over_gt_median_ratio": ratio, "ok": bool(SANITY_RANGE[0] <= ratio <= SANITY_RANGE[1])}
        f = [i for i in D["infos"] if "focal_proc_px" in i]
        if f:  # DA3METRIC: what the ratio would be with the original-resolution focal
            SANITY[model_id]["focal_proc_px_median"] = float(np.median([i["focal_proc_px"] for i in f]))
            SANITY[model_id]["ratio_if_orig_focal"] = ratio * float(np.median([i["focal_orig_px"] / i["focal_proc_px"]
                                                                                for i in f]))
        print(f"[sanity] {model_id}: median(pred)/median(GT) = {ratio:.3f}", SANITY[model_id])
        if not SANITY[model_id]["ok"]:
            warnings.warn(f"{model_id}: metric scale off by >2x vs LiDAR (ratio {ratio:.2f}); "
                          "check the focal / max_depth handling before trusting the metric results", stacklevel=1)
    else:
        VARIANTS[(model_id, "primary")] = {"pred": pred_ms, "sel": None, "scaling": "median", "gt_filter": "none",
                                           "primary": True, "label": "median-scaled (relative model)"}
    prim = VARIANTS[(model_id, "primary")]
    VARIANTS[(model_id, "occlusion_filtered")] = {**prim, "sel": D["occ"], "gt_filter": "occlusion_heuristic",
                                                  "primary": False, "label": prim["label"] + ", occlusion-filtered GT"}

for (model_id, variant), V in VARIANTS.items():
    D = DATA[model_id]
    sel = V["sel"] if V["sel"] is not None else slice(None)
    EVALS[(model_id, variant)] = evaluate(V["pred"][sel], D["gt"][sel], D["cam"][sel], D["img"][sel])
    ov = EVALS[(model_id, variant)]["overall"]
    print(f"{model_id:24s} {variant:20s} AbsRel {ov['abs_rel']:.3f}  RMSE {ov['rmse']:6.2f}  "
          f"d1 {ov['delta1']:.3f}  k {EVALS[(model_id, variant)]['curve']['slope_k']:.2f}  n {ov['n']}")

# %% [markdown]
# ## 5. 결과 저장
#
# 모델 폴더 `results/outdoor/02_monocular_depth/<model>/`에:
# - `summary.json` (primary), `summary_median_scaled.json` (metric 모델만), `summary_occlusion_filtered.json`
#   — `metrics`는 전체 + 구간별(`abs_rel@0-10m` …) + 카메라별(`abs_rel@CAM_FRONT` …) 평평한 숫자. `runtime_s`는 **장당 중앙값(초)**,
#   `peak_vram_mb`는 추론 중 최대값(모델 로드 시간은 `extra.load_s`).
# - `bins_<variant>.json`: 구간별·카메라별 전체 지표와 오차-거리 곡선.
# - `pairs.npz`: 99 종합 노트북용 (pred, GT) 픽셀 쌍 무작위 표본(최대 20만, seed 고정). primary 예측과 median-scaled 예측, 카메라 인덱스 포함.
# - `runtime.json`, `pred/` (예측 캐시).

# %%
N_PAIRS = 200_000


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


SETTING = {"scenes": EVAL_SCENES, "sample_stride": P["sample_stride"], "cams": CAMS, "n_images": len(ITEMS),
           "max_eval_depth": MAXD, "gt": f"LIDAR_TOP keyframe -> camera, min_depth {GT_MIN_DEPTH}, nearest per pixel",
           "aggregation": "pixel-pooled over all images", "bins": [bin_label(lo, hi) for lo, hi in DEFAULT_BINS],
           "occlusion_heuristic": {"window_rows_cols": list(OCC_WIN), "rel": OCC_REL}}

for (model_id, variant), V in VARIANTS.items():
    ev = EVALS[(model_id, variant)]
    mdir = result_dir(cfg, NOTEBOOK, model_id)
    rt = RUNTIME.get(model_id, {})
    save_summary(
        mdir, part=cfg["_part"], notebook=NOTEBOOK, profile=cfg["profile"], variant=variant,
        filename="summary.json" if variant == "primary" else f"summary_{variant}.json",
        model=model_id, metrics=flat_metrics(ev),
        runtime_s=rt.get("median_s_per_image"), peak_vram_mb=rt.get("peak_vram_mb"),
        checkpoint=checkpoint_name(model_id, MODEL_SPECS[model_id], ENCODER), packages=ENV["packages"], seed=cfg["seed"],
        extra={"label": V["label"], "is_primary": V["primary"], "output_kind": depth_kind(MODEL_SPECS[model_id]),
               "scaling": V["scaling"], "gt_filter": V["gt_filter"], "runtime_s_meaning": "median s/image, excl. load",
               "load_s": rt.get("load_s"), "n_images_timed": rt.get("n_images_timed"),
               "dav2_encoder": ENCODER if model_id.startswith("dav2") else None,
               "sanity": SANITY.get(model_id), "setting": SETTING, "third_party": TP_COMMITS, "env": {
                   k: ENV.get(k) for k in ("torch", "torch_cuda", "gpu")}},
    )
    (mdir / f"bins_{variant}.json").write_text(json.dumps(_json_safe(
        {"model": model_id, "variant": variant, "label": V["label"], "overall": ev["overall"], "bins": ev["bins"],
         "per_cam": ev["per_cam"], "img_mean": ev["img_mean"], "curve": ev["curve"]}), indent=1))

rng = np.random.default_rng(cfg["seed"])
for model_id in MODELS:
    D = DATA[model_id]
    prim = VARIANTS[(model_id, "primary")]["pred"]
    valid = np.flatnonzero((D["gt"] > 0) & (D["gt"] < MAXD) & np.isfinite(prim))
    idx = np.sort(rng.choice(valid, min(N_PAIRS, len(valid)), replace=False))
    np.savez_compressed(result_dir(cfg, NOTEBOOK, model_id) / "pairs.npz",
                        pred=prim[idx].astype(np.float32), gt=D["gt"][idx].astype(np.float32),
                        pred_median_scaled=(D["pred"] * D["scale"][D["img"]])[idx].astype(np.float32),
                        cam=D["cam"][idx], cams=np.array(CAMS),
                        primary_scaling=np.array(VARIANTS[(model_id, "primary")]["scaling"]))
print("saved:", sorted(str(p.relative_to(cfg.root)) for p in (cfg.path(cfg["results_root"]) / NOTEBOOK).glob("*/*.json")))

# %% [markdown]
# ## 6. 시각화
#
# ### 6.1 거리 구간별 지표 표
# 행 = (모델, 변형). "median-scaled"가 붙은 행은 GT로 스케일을 맞춘 결과이므로 metric 원 출력 행과 직접 비교하지 않는다.

# %%
def bin_table(metric: str) -> pd.DataFrame:
    rows = []
    for (model_id, variant), ev in EVALS.items():
        row = {"model": model_id, "variant": VARIANTS[(model_id, variant)]["label"]}
        row.update({b: ev["bins"][b][metric] for b in ev["bins"]})
        row["all"] = ev["overall"][metric]
        rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "variant"])


for metric in ("abs_rel", "rmse", "delta1"):
    print(f"\n{metric} by GT distance bin")
    display(bin_table(metric).round(3))

display(pd.DataFrame({b: {m: EVALS[(m, "primary")]["bins"][b]["n"] for m in MODELS}
                      for b in EVALS[(MODELS[0], "primary")]["bins"]}).rename_axis("valid pixels"))

# %% [markdown]
# ### 6.2 오차-거리 곡선 (log-log)
# 왼쪽: 주 결과(primary). 오른쪽: 모든 모델을 median-scaled로 맞춘 결과(스케일 오차 제거 후 모양 오차만).
# 점선은 기울기 k=2 참조선(스테레오 법칙)이며 단안 모델이 이를 따를 이유는 없다.

# %%
def add_k2_reference(ax, curves):
    c = next((c for c in curves.values() if c["centers"]), None)
    if c is None:
        return
    x = np.asarray(c["centers"])
    ax.plot(x, c["mean_abs_err"][0] * (x / x[0]) ** 2, "k--", lw=0.8, label="k=2 reference (stereo law)")
    ax.legend(fontsize=8)


curves_primary = {f"{m} ({VARIANTS[(m, 'primary')]['label']})": EVALS[(m, "primary")]["curve"] for m in MODELS}
curves_ms = {}
for m in MODELS:
    v = "median_scaled" if (m, "median_scaled") in EVALS else "primary"
    curves_ms[f"{m} (median-scaled)"] = EVALS[(m, v)]["curve"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
error_vs_distance_plot(axes[0], curves_primary)
add_k2_reference(axes[0], curves_primary)
axes[0].set_title("primary")
error_vs_distance_plot(axes[1], curves_ms)
add_k2_reference(axes[1], curves_ms)
axes[1].set_title("all models median-scaled (shape error only)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.3 카메라별 AbsRel (primary)

# %%
cols = model_colors(MODELS)
fig, ax = plt.subplots(figsize=(10, 3.6))
w = 0.8 / len(MODELS)
for j, m in enumerate(MODELS):
    vals = [EVALS[(m, "primary")]["per_cam"][c]["abs_rel"] for c in CAMS]
    ax.bar(np.arange(len(CAMS)) + j * w, vals, w, color=cols[m], label=f"{m} ({VARIANTS[(m, 'primary')]['label']})")
ax.set_xticks(np.arange(len(CAMS)) + w * (len(MODELS) - 1) / 2)
ax.set_xticklabels(CAMS, fontsize=8)
ax.set_ylabel("AbsRel")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.4 정성 비교
# 위 줄: RGB | LiDAR GT | 모델별 깊이(같은 색 범위 0–max_eval_depth m). 아래 줄: 가림 필터가 지운 GT 점(빨강) | 모델별 GT 픽셀 |오차|(m).
# DA3MONO는 영상별 median scaling 후의 깊이를 보여준다.

# %%
ERR_VMAX = 10.0  # m, color range of the |error| panels
VIZ_CAM = "CAM_FRONT" if "CAM_FRONT" in CAMS else CAMS[0]
VIZ = [(sc, middle(SAMPLES[sc]), VIZ_CAM) for sc in EVAL_SCENES[:3]]

for sc, tok, cam in VIZ:
    cam_sd = nusc.get("sample", tok)["data"][cam]
    rgb = load_rgb(cam_sd)
    gray = np.repeat(rgb.mean(-1, keepdims=True), 3, -1).astype(np.uint8)
    g = GT[(tok, cam)]
    uv = np.c_[g["u"], g["v"]]
    ncol = 2 + len(MODELS)
    fig, axes = plt.subplots(2, ncol, figsize=(3.3 * ncol, 4.4))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{sc} {cam}", fontsize=9)
    axes[0, 0].set_axis_off()
    sc_gt = overlay_points(axes[0, 1], rgb, uv, g["d"], vmax=MAXD, s=1.5, title="LiDAR GT")
    axes[1, 0].set_axis_off()
    drop = ~g["occ_keep"]
    axes[1, 1].imshow(gray)
    axes[1, 1].scatter(g["u"][drop], g["v"][drop], s=2, c="#D55E00", linewidths=0)
    axes[1, 1].set_title(f"occlusion-flagged GT ({drop.mean() * 100:.1f}%)", fontsize=9)
    axes[1, 1].set_axis_off()
    for j, m in enumerate(MODELS):
        pr = load_prediction(pred_cache_path(result_dir(cfg, NOTEBOOK, m), tok, cam))
        d, tag = pr.depth, ""
        if VARIANTS[(m, "primary")]["scaling"] == "median":
            d, tag = d * median_scale(d[g["v"], g["u"]], g["d"], mask=g["d"] < MAXD), " (median-scaled)"
        depth_panel(axes[0, 2 + j], d, vmax=MAXD, title=m + tag)
        err = np.abs(d[g["v"], g["u"]] - g["d"])
        ok = g["d"] < MAXD
        absrel = float(np.mean(err[ok] / g["d"][ok])) if ok.any() else float("nan")
        sc_err = overlay_points(axes[1, 2 + j], gray, uv[ok], err[ok], vmax=ERR_VMAX, s=1.5,
                                title=f"|err| AbsRel={absrel:.3f}")
    fig.colorbar(sc_gt, ax=axes[0, :].tolist(), fraction=0.012, pad=0.01, label="depth [m]")
    fig.colorbar(sc_err, ax=axes[1, :].tolist(), fraction=0.012, pad=0.01, label="|error| [m]")
    plt.show()

# %% [markdown]
# ## 7. 해석
#
# > (실행 후 작성) 아래 질문에 표·그래프의 숫자로 답한다.
#
# 1. **스케일**: metric 모델의 primary와 median-scaled AbsRel 차이는? 스케일 sanity 비율(median(pred)/median(GT))은 1에서 얼마나 떨어졌나?
#    DA3METRIC의 처리 해상도 focal 선택이 맞았는지(비율 ≈ 1, "원본 focal이었다면" 비율 ≈ 3)도 여기서 확인한다.
#    > (실행 후 작성)
# 2. **거리 구간**: 어느 구간부터 δ1이 급락하나? DA V2의 80 m 상한이 50m+ 구간에 영향을 주는가?
#    > (실행 후 작성)
# 3. **오차-거리 기울기 k**: 모델별 k와, median scaling 후 k가 어떻게 변하는지. (k는 기술 통계일 뿐)
#    > (실행 후 작성)
# 4. **카메라별**: CAM_BACK(화각이 다른 렌즈)이나 측방 카메라에서 intrinsics를 쓰지 않는 DA V2가 특히 나빠지는가?
#    > (실행 후 작성)
# 5. **GT 노이즈**: occlusion_filtered와 primary의 차이(특히 0–10 m, 측방 카메라).
#    > (실행 후 작성)
# 6. **04 pseudo-LiDAR 선택**: primary AbsRel이 가장 낮은 metric 모델(04가 자동 선택)과 그 근거.
#    > (실행 후 작성)
#
# 주의: 평가 장면은 4개(공식 val)뿐이라 통계적으로 작고, 공식 벤치마크 수치와 직접 비교할 수 없다.
# DA3의 학습 데이터에 nuScenes가 포함되는지는 저장소에서 확인하지 못했다(TODO(verify): DA3 논문 학습 데이터 표). 포함된다면 val 장면이라도
# 학습에 쓰였을 가능성을 배제할 수 없다.
