# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 07. 장면 표현 (NeRF / Instant-NGP / 3D Gaussian Splatting) — 주행 로그에서 어디서 무너지는가
#
# ## 목적
# 범용 장면 표현 방법(Nerfacto, Instant-NGP, Splatfacto=3DGS)을 nuScenes-mini 한 장면의 전방 카메라 키프레임으로
# 학습하고, 남겨 둔 프레임에서 새 시점 합성 품질을 잰다. 목표는 순위 경쟁이 아니라 **주행 데이터 특유의 조건**
# (전진하는 단일 카메라, 움직이는 물체, 하늘·원거리)에서 이 방법들이 어디서, 왜 무너지는지를 보여 주는 것이다.
#
# ## 비교 모델과 원리
# - **Nerfacto** (nerfstudio): NeRF 계열. 해시 그리드 + 작은 MLP로 밀도·색을 표현하고, proposal 네트워크로 광선 샘플을
#   좁힌다. 무한 장면은 scene contraction으로 반경 2 구 안에 압축한다. 외관 임베딩으로 노출 차이를 흡수한다.
#   tiny-cuda-nn이 없으면 같은 구조를 torch 구현으로 돌린다(느려질 뿐 방법은 같음).
# - **Instant-NGP** (nerfstudio 구현 `instant-ngp`): 다중 해상도 해시 인코딩 + occupancy grid(nerfacc)로 빈 공간을 건너뛰는
#   빠른 NeRF. nerfstudio 구현은 원본 CUDA 코드가 아니라 NerfactoField + nerfacc 기반 재구현이다. 원거리는 contraction으로 처리.
# - **Splatfacto** (nerfstudio의 3D Gaussian Splatting, gsplat 래스터라이저): 장면을 수십만 개의 3D 가우시안(위치·공분산·
#   불투명도·SH 색)으로 표현하고 타일 래스터화로 렌더링한다. 학습 중 가우시안을 복제·분할·제거한다. 이 파서는 SfM 점을
#   주지 않으므로 **무작위 초기화**(정규화 좌표의 한 변 10 정육면체에 5만 개)로 시작한다.
# - 공통: 모두 **정적 장면**을 가정한다. 움직이는 물체는 손실에서 빼는 것(마스크) 말고는 모델링하지 않는다.
#
# ## 원 논문
# - NeRF: https://arxiv.org/abs/2003.08934
# - Instant-NGP: https://arxiv.org/abs/2201.05989
# - Nerfstudio (Nerfacto): https://arxiv.org/abs/2302.04264
# - 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079
# - gsplat: https://arxiv.org/abs/2409.06765
# - (07b 선택) OmniRe: https://arxiv.org/abs/2408.16760
#
# ## 커널
# `3dv` (nerfstudio 1.1.5, gsplat 1.4.0). 학습·평가는 같은 env의 `ns-train`/`ns-eval`/`ns-render`를 subprocess로 호출하므로
# 커널 프로세스는 GPU를 잡지 않는다.
#
# ## 예상 VRAM·시간 (추정)
# 아래는 **추정치**다(실측 아님). 한 장면, 전방 카메라 키프레임 약 40장(학습 36 / 평가 4), RTX 3060 기준.
#
# | 프로파일 | 해상도 | 반복 | nerfacto | instant-ngp | splatfacto |
# |---|---|---|---|---|---|
# | low (≤8GB) | 400×225 | 7k | 2–4GB, 15–40분 | 2–4GB, 15–40분 | 1–3GB, 5–10분 |
# | mid (12GB) | 800×450 | 15k | 3–6GB, 30–90분 | 3–6GB, 30–90분 | 2–5GB, 10–25분 |
# | high (24GB+) | 1600×900 | 30k | 6–10GB, 1–3시간 | 6–10GB, 1–3시간 | 4–10GB, 30–60분 |
#
# NeRF 두 방법은 tiny-cuda-nn이 없으면 시간 범위의 상한 쪽이다. 마스크 생성 1–2분, 방법당 평가·렌더링 수 분.
# `RUN_NO_MASK=True`면 학습 시간이 두 배다.
#
# ## 이 노트북이 답하려는 질문
# 1. 같은 장면·같은 반복 수에서 세 방법의 보류 프레임 PSNR/SSIM/LPIPS, 학습 시간, 렌더 속도, 크기는 어떻게 다른가?
# 2. **전진 운동**: 진행 방향(소실점) 근처는 시차가 거의 없다. 렌더 깊이가 LiDAR와 얼마나 어긋나며, 그 오차가 진행 방향과의
#    각도에 따라 어떻게 변하는가?
# 3. **움직이는 물체**: 동적 물체 마스크를 쓰면 정적 영역 품질이 좋아지는가? 마스크를 안 쓰면 어떤 잔상이 생기는가?
# 4. **하늘·원거리**: 이미지 위쪽(하늘·먼 건물)과 먼 거리 구간에서 색과 깊이가 어떻게 무너지는가?

# %% [markdown]
# ## 1. 설정 로드
# `FORCE_RERUN`은 캐시를 무시한다. 학습 캐시는 방법·변형별 `eval.json` 존재 여부로 판단한다.
# `RUN_NO_MASK=True`면 마스크 없는 변형도 학습해 움직이는 물체의 영향을 비교한다(기본 꺼짐).

# %%
FORCE_RERUN = False  # True: ignore cached results (retrain and re-evaluate)
NOTEBOOK = "07_scene_representation"
RUN_NO_MASK = False  # True: also train every method without dynamic-object masks (2x training time)
RECON_SCENE = None  # None: "recon_scene" from 01's eval_samples.json; or a scene name such as "scene-0103"
METHODS = None  # None: cfg["models"]["recon"]["methods"]; or a subset such as ["splatfacto"]

import json
import math
import shlex
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bench3d.config import load_config
from bench3d.io import is_cached, result_dir, save_summary
from bench3d.metrics.depth import DEFAULT_BINS, bin_label, binned_depth_metrics, depth_metrics
from bench3d.metrics.nvs import psnr
from bench3d.models import nerfstudio_tools as nst
from bench3d.nusc import intrinsic, project_lidar_to_camera, sensor_to_global, sparse_depth_map
from bench3d.runtime import env_report, set_seed
from bench3d.viz import depth_panel, model_colors, overlay_points
from IPython.display import display
from nuscenes.nuscenes import NuScenes
from PIL import Image

cfg = load_config("outdoor")  # profile: config.yaml or $THREEDCHECK_PROFILE
P = cfg.p["recon"]  # this notebook's profile block
R = cfg["models"]["recon"]
FORCE_RERUN = FORCE_RERUN or cfg.get("force_rerun", False)
set_seed(cfg["seed"])
ENV = env_report(["torch", "nerfstudio", "gsplat", "nerfacc", "tinycudann", "nuscenes-devkit", "tensorboard", "numpy"])
print(json.dumps(ENV, indent=1))
print("profile:", cfg["profile"], P)

METHODS = list(METHODS or R["methods"])
VARIANTS = ["masked"] + (["nomask"] if RUN_NO_MASK else [])
COLORS = model_colors(METHODS)
assert R["dataparser"] == "nuscenes-data", R["dataparser"]
if ENV["packages"].get("tinycudann") == "not installed":
    print("NOTE: tiny-cuda-nn not installed -> nerfacto / instant-ngp fall back to torch encodings (slower).")
if not ENV.get("cuda_available"):
    print("WARNING: no CUDA GPU visible; ns-train needs one.")

# %% [markdown]
# ## 2. 데이터 로드
# nerfstudio nuScenes 파서(`nuscenes-data`, `nerfstudio/data/dataparsers/nuscenes_dataparser.py`)에서 확인한 사실:
# - **한 장면, 카메라 하나만** 쓴다(`len(cameras) == 1` assert). 쓰는 프레임은 장면의 **키프레임(약 2Hz)** 뿐이다.
# - 포즈는 nuScenes 전역 좌표의 `ego_pose @ calibrated_sensor`이고, 카메라 위치를 평균 중심으로 옮긴 뒤 모든 축 중
#   최대 |t|로 나눠 **[-1, 1]로 정규화**한다. scene box도 [-1, 1]이다. `dataparser_scale`은 1.0으로 남겨 두므로
#   `ns-render`의 raw-depth는 미터가 아니다. 아래 `MPU`(정규화 1단위당 미터)로 직접 환산한다.
# - 이미지 크기는 1600×900으로 고정돼 있고 **다운스케일 옵션이 없다**. 해상도는 데이터매니저의
#   `--pipeline.datamanager.camera-res-scale-factor`(= 1 / `P["downscale"]`)로 조절한다. 이미지·마스크·내부 파라미터가
#   함께 줄어든다(`data/datasets/base_dataset.py`).
# - 학습/평가 분할: 키프레임 n장 중 `ceil(n·0.9)`장을 `linspace(0, n-1)`로 고르게 뽑아 학습에 쓰고, **나머지가 평가**다.
#   n≈40이면 평가 프레임은 4장뿐이라 수치의 분산이 크다. 아래 `nst.nuscenes_split`이 파서와 같은 식으로 인덱스를 재현한다.
#
# 장면은 01 노트북이 eval4 중에서 가장 정적인 장면(자차 이동 거리와 움직이는 주석 비율 기준)으로 고른 `recon_scene`을 쓴다.
# 움직이는 물체의 영향을 더 크게 보고 싶으면 `RECON_SCENE`에 `moving_ann_ratio`가 큰 장면을 넣는다.
# 사전학습 가중치를 쓰지 않는 장면별 최적화이므로 학습 데이터 누수 문제는 없다(LPIPS의 AlexNet만 사전학습).

# %%
DATA_ROOT = cfg.path(cfg["dataset"]["data_root"])
VERSION = cfg["dataset"]["version"]
CAMERA = R["camera"]  # parser literal: FRONT, FRONT_LEFT, ... (one camera)
CHANNEL = f"CAM_{CAMERA}"
SPLIT_FRAC = float(R["train_split_fraction"])
SCALE = 1.0 / float(P["downscale"])  # -> --pipeline.datamanager.camera-res-scale-factor
MAX_ITERS = int(P["max_num_iterations"])
W_IMG, H_IMG = int(1600 * SCALE), int(900 * SCALE)  # parser hard-codes 1600x900; rescale floors

eval_samples_file = cfg.path(cfg["results_root"]) / "01_data_exploration" / "eval_samples.json"
EVAL_SAMPLES = json.loads(eval_samples_file.read_text()) if eval_samples_file.exists() else {}
if RECON_SCENE is None:
    assert "recon_scene" in EVAL_SAMPLES, f"run 01_data_exploration first ({eval_samples_file}) or set RECON_SCENE"
    RECON_SCENE = EVAL_SAMPLES["recon_scene"]
SCENE = RECON_SCENE
if EVAL_SAMPLES.get("scene_stats"):
    display(pd.DataFrame(EVAL_SAMPLES["scene_stats"]).T.sort_values("moving_ann_ratio"))

nusc = NuScenes(version=VERSION, dataroot=str(DATA_ROOT), verbose=False)
FRAMES = nst.nuscenes_frames(nusc, SCENE, CHANNEL)  # parser order (sorted by timestamp)
I_TRAIN, I_EVAL = nst.nuscenes_split(len(FRAMES), SPLIT_FRAC)
EVAL_FRAMES = [FRAMES[i] for i in I_EVAL]
MPU = nst.nuscenes_metres_per_unit(nusc, FRAMES)  # metres per normalized nerfstudio unit
print(f"scene {SCENE}: {len(FRAMES)} {CHANNEL} keyframes, train {len(I_TRAIN)}, eval {len(I_EVAL)} -> {I_EVAL.tolist()}")
print(f"render size {W_IMG}x{H_IMG}, {MAX_ITERS} iterations, 1 unit = {MPU:.1f} m")

# %%
# Camera trajectory: train vs eval keyframes, arrows = optical axis (forward motion => small parallax ahead)
C2W = np.stack([sensor_to_global(nusc, f["sd_token"]) for f in FRAMES])
xy, fwd = C2W[:, :2, 3], C2W[:, :2, 2]
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(xy[:, 0], xy[:, 1], "-", color="0.7", lw=1)
ax.quiver(xy[I_TRAIN, 0], xy[I_TRAIN, 1], fwd[I_TRAIN, 0], fwd[I_TRAIN, 1], color="#0072B2", scale=25, width=0.004,
          label="train")
ax.quiver(xy[I_EVAL, 0], xy[I_EVAL, 1], fwd[I_EVAL, 0], fwd[I_EVAL, 1], color="#D55E00", scale=25, width=0.006,
          label="eval")
ax.set_aspect("equal")
ax.set_xlabel("global x [m]")
ax.set_ylabel("global y [m]")
ax.set_title(f"{SCENE} {CHANNEL} keyframes")
ax.legend()
plt.show()

# %% [markdown]
# ### 2.1 동적 물체 마스크 (`process_nuscenes_masks.py`)
# nerfstudio 1.1.5의 `nerfstudio/scripts/datasets/process_nuscenes_masks.py`를 그대로 쓴다. 콘솔 진입점(`ns-...`)이
# 등록돼 있지 않아서(`entry_points.txt`) `python -m nerfstudio.scripts.datasets.process_nuscenes_masks`로 실행한다.
# 인자: `--data-dir --output-dir --version --cameras --velocity-thresh(기본 0.75) --verbose`.
#
# 무엇을 가리는가
# - **인스턴스 단위**로 판단한다. 한 인스턴스의 주석 중 하나라도 속도 노름이 0.75 m/s 이상이면 "동적"이고, 그 인스턴스는
#   **모든 프레임에서** 가려진다(잠시 멈춘 순간 포함). 속도는 devkit `box_velocity`(앞뒤 주석의 중심 차분)이다.
# - 속도를 추정할 수 없으면(주석이 하나뿐이거나 간격이 너무 김) `box_velocity`가 NaN을 돌려주고,
#   `not (NaN < 0.75)`가 참이라 **동적으로 처리**된다(보수적).
# - 이미지에 조금이라도 보이는 박스(`BoxVisibility.ANY`)의 8개 꼭짓점을 투영해 6개 면을 `fillPoly`로 0으로 칠한다.
#   직육면체 투영이라 실제 실루엣보다 넓게 가린다. `CAM_BACK`은 자차 트렁크 때문에 아래 100행을 가린다.
# - 출력은 255(사용)/0(무시) PNG이고 경로 규칙 `<out>/masks/CAM_X/<이미지이름>.png`를 파서가 그대로 따른다.
#
# 알려진 한계
# - 스크립트의 TODO: `BoxVisibility.ANY`는 **일부가 카메라 뒤에 있는 박스를 놓친다**(바로 옆을 지나는 큰 차량 등).
#   꼭짓점이 카메라 뒤에 있으면 투영이 뒤집혀 면 채우기도 틀어질 수 있다.
# - 주석이 없는 동적 요소(그림자, 유리 반사, 주석 범위 밖 물체, 신호 변화)는 가리지 않는다.
# - 키프레임 주석 시각(LiDAR 기준)과 카메라 노출 시각이 수십 ms 다르다. 빠른 물체는 가장자리가 어긋난다.
# - 장면 필터가 없어 version 전체 샘플을 처리한다(mini는 약 400 샘플이라 문제없음).
#
# 학습에서의 사용: nerfacto/instant-ngp는 픽셀 샘플러가 마스크가 참인 픽셀에서만 광선을 뽑고
# (`data/pixel_samplers.py`), splatfacto는 손실 계산 전에 GT와 렌더를 마스크로 곱한다(`models/splatfacto.py:get_loss_dict`).
# **`ns-eval` 메트릭은 마스크를 쓰지 않는다**(전체 이미지). 그래서 4절에서 정적 영역 PSNR을 따로 계산한다.

# %%
MASK_DIR = result_dir(cfg, NOTEBOOK) / "nuscenes_masks"  # parser reads <MASK_DIR>/masks/<CHANNEL>/<name>.png
MASK_VELOCITY_THRESH = R.get("mask_velocity_thresh")  # None -> script default 0.75 m/s
mask_files = [nst.mask_path(MASK_DIR, CHANNEL, f["filename"]) for f in FRAMES]
if FORCE_RERUN or not all(p.exists() for p in mask_files):
    r = nst.run(nst.mask_cmd(DATA_ROOT, MASK_DIR, VERSION, cameras=(CAMERA,), velocity_thresh=MASK_VELOCITY_THRESH),
                MASK_DIR / "masks.log")
    print(f"masks written in {r['seconds']:.0f}s")
missing = [p for p in mask_files if not p.exists()]
assert not missing, f"{len(missing)} masks missing, e.g. {missing[0]}"


def load_mask(frame, w=W_IMG, h=H_IMG):
    """True = static pixel (used), False = dynamic object (mask script writes 255 / 0)."""
    m = Image.open(nst.mask_path(MASK_DIR, CHANNEL, frame["filename"])).resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(m) > 0


def load_rgb(frame, w=W_IMG, h=H_IMG):
    im = Image.open(DATA_ROOT / frame["filename"]).convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


DYN_RATIO = np.array([1.0 - load_mask(f).mean() for f in FRAMES])
print(f"dynamic-pixel ratio: mean {DYN_RATIO.mean():.3f}, max {DYN_RATIO.max():.3f}; "
      f"eval frames {np.round(DYN_RATIO[I_EVAL], 3).tolist()}")

fig, axes = plt.subplots(1, 4, figsize=(20, 3.5), gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})
axes[0].plot(DYN_RATIO, "-", color="0.4")
axes[0].plot(I_EVAL, DYN_RATIO[I_EVAL], "o", color="#D55E00", label="eval")
axes[0].set_xlabel("keyframe index")
axes[0].set_ylabel("masked pixel ratio")
axes[0].legend()
for ax, i in zip(axes[1:], np.argsort(-DYN_RATIO)[:3]):
    img, keep = load_rgb(FRAMES[i]), load_mask(FRAMES[i])
    ax.imshow(np.where(keep[..., None], img, 0.35 * img + 0.65 * np.array([1.0, 0.0, 0.0])))
    ax.set_title(f"frame {i} ({'eval' if i in I_EVAL else 'train'}), masked {DYN_RATIO[i]:.2f}", fontsize=9)
    ax.set_axis_off()
plt.show()

# %% [markdown]
# ## 3. 모델별 학습 (`ns-train`)
# 명령 구조(nerfstudio 1.1.5 `ns-train nerfacto nuscenes-data --help`로 확인, tyro 서브커맨드라 **트레이너 옵션은
# `nuscenes-data` 앞**, 파서 옵션은 뒤에 온다):
#
# ```
# ns-train <method> --output-dir <run>/ns --experiment-name <scene>_<variant> --timestamp bench
#     --max-num-iterations <P> --machine.seed <seed> --vis tensorboard --viewer.quit-on-train-completion True
#     --steps-per-eval-batch 0 --steps-per-eval-image 0 --steps-per-eval-all-images 0
#     --pipeline.datamanager.camera-res-scale-factor <1/downscale> [method options]
#   nuscenes-data --data <scene> --data-dir <nuScenes root> --version v1.0-mini --cameras FRONT
#     --train-split-fraction 0.9 [--mask-dir <masks>]
# ```
#
# 결정 사항
# - `--vis tensorboard`: 뷰어(viser 서버)를 띄우지 않고, 트레이너가 기록하는 `GPU Memory (MB)`
#   (`torch.cuda.max_memory_allocated`) 스칼라를 이벤트 파일에서 읽어 **피크 VRAM**으로 쓴다(nvidia-smi 값보다 작다).
# - 학습 중 평가를 끈다(`step_check(step, 0)`은 항상 거짓). 그래서 측정한 벽시계 시간은 학습 + 데이터 로딩 + 저장이다.
# - nerfacto는 기본으로 포즈 보정(`camera-optimizer.mode SO3xR3`)을 켠다. 평가 카메라는 보정되지 않으므로 학습 포즈만
#   움직이면 보류 프레임과 어긋난다. nuScenes 포즈는 측위 결과라 정확하므로 **끈다**(splatfacto 기본값도 off,
#   instant-ngp에는 이 옵션이 없다).
# - 출력: `<run>/ns/<experiment>/<method-name>/bench/config.yml`, 체크포인트는 그 아래 `nerfstudio_models/`.
#   학습은 셀을 오래 막는다. 진행은 `<run>/train.log`로 본다.

# %%
METHOD_ARGS = {
    # nuScenes poses come from localization; eval cameras are never optimized -> keep pose refinement off
    "nerfacto": ["--pipeline.model.camera-optimizer.mode", "off"],
    "instant-ngp": [],
    "splatfacto": [],  # camera optimizer already off by default; random init (parser provides no SfM points)
}


def run_dir(method, variant):
    return result_dir(cfg, NOTEBOOK, method) / variant


def experiment(variant):
    return f"{SCENE}_{variant}"


def build_train_cmd(method, variant):
    return nst.train_cmd(
        method, scene=SCENE, data_dir=DATA_ROOT, output_dir=run_dir(method, variant) / "ns",
        experiment_name=experiment(variant), max_num_iterations=MAX_ITERS, seed=cfg["seed"], version=VERSION,
        cameras=(CAMERA,), mask_dir=MASK_DIR if variant == "masked" else None, train_split_fraction=SPLIT_FRAC,
        scale_factor=SCALE, method_args=METHOD_ARGS.get(method, []))


for m in METHODS:
    for v in VARIANTS:
        print(shlex.join(build_train_cmd(m, v)), "\n")

# %%
RUNS = {}  # (method, variant) -> train meta
for method in METHODS:
    for variant in VARIANTS:
        rd = run_dir(method, variant)
        meta_file, eval_json = rd / "train_meta.json", rd / "eval.json"
        if is_cached(eval_json, FORCE_RERUN) and meta_file.exists():
            print(f"[cached] {method}/{variant}")
        elif is_cached(meta_file, FORCE_RERUN):
            print(f"[trained] {method}/{variant}: evaluation pending")
        else:
            shutil.rmtree(rd, ignore_errors=True)  # stale run (fixed timestamp) and its derived outputs
            rd.mkdir(parents=True)
            print(f"[train] {method}/{variant} ... (log: {rd / 'train.log'})")
            r = nst.run(build_train_cmd(method, variant), rd / "train.log")
            cfg_yml = nst.find_config(rd / "ns", experiment(variant))
            assert cfg_yml is not None, f"config.yml not found under {rd / 'ns'}"
            ckpt = nst.latest_checkpoint(cfg_yml.parent)
            meta = {"train_time_s": r["seconds"], "cmd": r["cmd"],
                    "config": str(cfg_yml.relative_to(cfg.root)),
                    "checkpoint": str(ckpt.relative_to(cfg.root)) if ckpt else None,
                    "peak_vram_mb": nst.peak_vram_from_tensorboard(cfg_yml.parent)}
            meta_file.write_text(json.dumps(meta, indent=2))
            print(f"  done in {r['seconds'] / 60:.1f} min, peak VRAM {meta['peak_vram_mb']} MB")
        RUNS[(method, variant)] = json.loads(meta_file.read_text())

# %% [markdown]
# ## 4. 평가
# **평가 프레임**: 파서가 학습에서 뺀 키프레임(`I_EVAL`). 같은 장면의 보류 시점 보간이다.
#
# **메트릭 정의**
# - `psnr`, `ssim`, `lpips`: `ns-eval --load-config <config.yml> --output-path eval.json`의 값. 평가 프레임마다
#   **전체 이미지**에서 계산한 뒤 평균한다(`pipelines/base_pipeline.py:get_average_image_metrics`). 각 모델의
#   `get_image_metrics_and_images`는 마스크를 쓰지 않는다. LPIPS는 torchmetrics AlexNet(`normalize=True`).
# - `eval_fps_ns`: `ns-eval`의 `fps` 키. 렌더링 + 메트릭 계산 시간이 섞여 있다. 그래서 별도로
#   `render_fps`를 잰다: 같은 평가 카메라에서 `get_outputs_for_camera`만 CUDA 동기화해서 반복 측정(워밍업 2회, 3회 반복,
#   별도 프로세스, `bench3d.models.nerfstudio_tools fps`).
# - `psnr_static` / `psnr_dynamic`: `ns-eval --render-output-path`가 저장한 `eval_img_XXXX.png`([GT | 렌더], 8비트)를
#   반으로 나눠 동적 마스크 밖(정적) / 안(동적, 마스크 픽셀 ≥ 200개인 프레임만)에서 `bench3d.metrics.nvs.psnr`로 계산.
#   8비트 양자화 때문에 `ns-eval` PSNR과 소수점 둘째 자리 정도 다를 수 있다(`psnr_png`로 확인).
# - `psnr_top` / `psnr_bottom`: 정적 픽셀 중 이미지 위쪽 `TOP_FRAC` 비율 행(하늘·먼 건물의 **대리 영역**, 하늘 분할 아님)과
#   그 아래.
# - 깊이: `ns-render dataset --split test --rendered-output-names raw-depth`의 렌더 깊이 × `MPU`(미터).
#   nerfacto/instant-ngp 깊이는 정규화된 광선 방향을 따라 잰 **광선 거리**라서 z-깊이로 바꾸고, splatfacto는
#   gsplat `RGB+ED`(기대 **z-깊이**)를 그대로 쓴다. 기준은 같은 샘플의 LIDAR_TOP 점을 카메라로 투영한 희소 z-깊이
#   (`bench3d.nusc.project_lidar_to_camera`, 최근접 z-buffer)이고, **동적 마스크 밖 픽셀만** 쓴다.
#   지표는 AbsRel, δ<1.25, GT 거리 구간별(0–10/10–20/20–30/30–50/50m+)과 **진행 방향과의 각도 구간별**로 낸다.
#   하늘에는 LiDAR 반사가 없어서 하늘 깊이는 이 지표로 잴 수 없다.
# - 크기: 마지막 체크포인트의 학습 파라미터 수와 MB(LPIPS 백본 제외), 파일 크기(Adam 상태 포함이라 약 3배).
#   splatfacto는 `_model.gauss_params.means`의 행 수 = **가우시안 개수**.

# %%
for (method, variant), meta in RUNS.items():
    rd = run_dir(method, variant)
    cfg_yml = cfg.path(meta["config"])
    if not is_cached(rd / "eval.json", FORCE_RERUN):
        shutil.rmtree(rd / "eval_images", ignore_errors=True)
        r = nst.run(nst.eval_cmd(cfg_yml, rd / "eval.json", rd / "eval_images"), rd / "eval.log")
        print(f"[eval] {method}/{variant} {r['seconds']:.0f}s")
    if not is_cached(rd / "render" / "test" / "raw-depth", FORCE_RERUN):
        shutil.rmtree(rd / "render", ignore_errors=True)
        r = nst.run(nst.render_cmd(cfg_yml, rd / "render", split="test", names=("raw-depth",)), rd / "render.log")
        print(f"[render depth] {method}/{variant} {r['seconds']:.0f}s")
    if not is_cached(rd / "fps.json", FORCE_RERUN):
        nst.run(nst.fps_cmd(cfg_yml, rd / "fps.json"), rd / "fps.log")
    if not is_cached(rd / "ckpt_stats.json", FORCE_RERUN):
        ckpt = nst.latest_checkpoint(cfg_yml.parent)
        stats = nst.checkpoint_stats(ckpt) | {"checkpoint": str(ckpt.relative_to(cfg.root))}
        (rd / "ckpt_stats.json").write_text(json.dumps(stats, indent=2))
    print(method, variant, {k: round(v, 4) for k, v in nst.parse_eval_json(rd / "eval.json").items()})

# %%
TOP_FRAC = 0.35  # rows above this fraction of the height: sky / far-range proxy (no sky segmentation)
MIN_DYN_PIX = 200  # psnr_dynamic only for eval frames with at least this many masked pixels
ANGLE_BINS = (0, 5, 10, 20, 40, 180)  # degrees between pixel ray and camera motion direction
MIN_BASELINE_M = 0.5  # below this ego displacement the motion direction is undefined
# nerfacto / instant-ngp: distance along the unit ray (nerfacc / ray samplers); splatfacto: gsplat "RGB+ED" z-depth
DEPTH_IS_RAY_DISTANCE = {"nerfacto": True, "instant-ngp": True, "splatfacto": False}


def scaled_K(frame):
    K = intrinsic(nusc, frame["sd_token"]).copy()
    K[:2] *= SCALE  # fx, cx, fy, cy scale like Cameras.rescale_output_resolution
    return K


def pixel_rays(K, w, h):
    """Unit camera-frame rays through pixel centres (nerfstudio uses a 0.5 pixel offset)."""
    v, u = np.mgrid[0:h, 0:w] + 0.5
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def motion_geometry(frame_idx, K, w, h):
    """Angle [deg] between each pixel ray and the camera motion direction, plus the focus of expansion (FOE) pixel.

    Motion = displacement between the neighbouring keyframes. Pixels near the FOE have almost no parallax.
    """
    j0, j1 = max(frame_idx - 1, 0), min(frame_idx + 1, len(FRAMES) - 1)
    t = C2W[j1, :3, 3] - C2W[j0, :3, 3]
    if np.linalg.norm(t) < MIN_BASELINE_M:
        return None, None
    t_cam = C2W[frame_idx, :3, :3].T @ (t / np.linalg.norm(t))
    ang = np.degrees(np.arccos(np.clip(pixel_rays(K, w, h) @ t_cam, -1.0, 1.0)))
    foe = None
    if t_cam[2] > 0:
        p = K @ t_cam
        foe = p[:2] / p[2]
    return ang, foe


def load_depth_m(method, variant, frame, K):
    f = run_dir(method, variant) / "render" / "test" / "raw-depth" / (Path(frame["filename"]).stem + ".npy.gz")
    d = np.squeeze(nst.load_npy_gz(f)).astype(np.float64) * MPU
    if DEPTH_IS_RAY_DISTANCE.get(method, True):
        d = d * pixel_rays(K, d.shape[1], d.shape[0])[..., 2]  # ray distance -> z-depth
    return d


def lidar_depth(frame, w, h):
    proj = project_lidar_to_camera(nusc, frame["lidar_sd_token"], frame["sd_token"])
    return sparse_depth_map(proj["uv"] * SCALE, proj["depth"], w, h)


def evaluate_views(method, variant):
    rd = run_dir(method, variant)
    rows, pool = [], {"pred": [], "gt": [], "ang": []}
    for k, frame in enumerate(EVAL_FRAMES):
        gt, pred = nst.split_eval_image(rd / "eval_images" / f"eval_img_{k:04d}.png")
        h, w = gt.shape[:2]
        assert (w, h) == (W_IMG, H_IMG), f"eval image {w}x{h} != {W_IMG}x{H_IMG}"
        static = load_mask(frame, w, h)
        top = np.zeros((h, w), bool)
        top[: int(TOP_FRAC * h)] = True
        row = {"k": k, "frame_idx": int(I_EVAL[k]), "sample_token": frame["sample_token"],
               "dyn_ratio": float(1.0 - static.mean()), "psnr_png": psnr(pred, gt),
               "psnr_static": psnr(pred, gt, static),
               "psnr_dynamic": psnr(pred, gt, ~static) if (~static).sum() >= MIN_DYN_PIX else math.nan,
               "psnr_top": psnr(pred, gt, top & static), "psnr_bottom": psnr(pred, gt, ~top & static)}
        K = scaled_K(frame)
        depth = load_depth_m(method, variant, frame, K)
        lid = lidar_depth(frame, w, h)
        valid = (lid > 0) & static
        dm = depth_metrics(depth, lid, valid)
        row.update({"depth_abs_rel": dm["abs_rel"], "depth_delta1": dm["delta1"], "depth_n": dm["n"]})
        ang, _ = motion_geometry(int(I_EVAL[k]), K, w, h)
        pool["pred"].append(depth[valid])
        pool["gt"].append(lid[valid])
        pool["ang"].append(ang[valid] if ang is not None else np.full(int(valid.sum()), np.nan))
        rows.append(row)
    pred, gt, ang = (np.concatenate(pool[key]) for key in ("pred", "gt", "ang"))
    depth_rep = {"all": depth_metrics(pred, gt), "by_distance": binned_depth_metrics(pred, gt, bins=DEFAULT_BINS),
                 "by_angle": {f"{lo}-{hi}deg": depth_metrics(pred, gt, (ang >= lo) & (ang < hi))
                              for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:])}}
    return rows, depth_rep


PER_VIEW, DEPTH = {}, {}
for key in RUNS:
    rd = run_dir(*key)
    pv_file, dep_file = rd / "per_view.json", rd / "depth_vs_lidar.json"
    if is_cached(pv_file, FORCE_RERUN) and dep_file.exists():
        PER_VIEW[key], DEPTH[key] = json.loads(pv_file.read_text()), json.loads(dep_file.read_text())
        continue
    PER_VIEW[key], DEPTH[key] = evaluate_views(*key)
    pv_file.write_text(json.dumps(PER_VIEW[key], indent=1, allow_nan=True))
    dep_file.write_text(json.dumps(DEPTH[key], indent=1, allow_nan=True))

# %% [markdown]
# ## 5. 결과 저장 (`summary.json`)
# 방법별 `results/outdoor/07_scene_representation/<method>/summary.json`(마스크 사용)과, `RUN_NO_MASK`면
# `summary_nomask.json`. `runtime_s` = 학습 벽시계 시간, `peak_vram_mb` = 학습 중 `torch.cuda.max_memory_allocated`
# (텐서보드 이벤트, 없으면 null). 커널 안에서 GPU를 쓰지 않으므로 `bench3d.runtime.Measure`는 쓰지 않는다.


# %%
def nanmean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else math.nan


SUMMARY = {}
for (method, variant), meta in RUNS.items():
    rd = run_dir(method, variant)
    ev = nst.parse_eval_json(rd / "eval.json")
    fps = json.loads((rd / "fps.json").read_text())
    ck = json.loads((rd / "ckpt_stats.json").read_text())
    pv, dep = PER_VIEW[(method, variant)], DEPTH[(method, variant)]
    metrics = {
        "psnr": ev["psnr"], "ssim": ev["ssim"], "lpips": ev["lpips"],
        "psnr_png": nanmean([r["psnr_png"] for r in pv]),
        "psnr_static": nanmean([r["psnr_static"] for r in pv]),
        "psnr_dynamic": nanmean([r["psnr_dynamic"] for r in pv]),
        "psnr_top": nanmean([r["psnr_top"] for r in pv]),
        "psnr_bottom": nanmean([r["psnr_bottom"] for r in pv]),
        "depth_abs_rel": dep["all"]["abs_rel"], "depth_delta1": dep["all"]["delta1"],
        "train_time_s": meta["train_time_s"], "render_fps": fps["render_fps"], "eval_fps_ns": ev["fps"],
        "n_params": ck["n_params"], "model_mb": ck["model_mb"], "ckpt_mb": ck["ckpt_mb"],
    }
    if ck["n_gaussians"] is not None:
        metrics["n_gaussians"] = ck["n_gaussians"]
    SUMMARY[(method, variant)] = metrics
    save_summary(
        result_dir(cfg, NOTEBOOK, method), part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model=method,
        variant=f"{SCENE}/{CHANNEL}/{variant}", metrics=metrics, runtime_s=meta["train_time_s"],
        peak_vram_mb=meta["peak_vram_mb"], checkpoint=ck["checkpoint"], packages=ENV["packages"], seed=cfg["seed"],
        filename="summary.json" if variant == "masked" else f"summary_{variant}.json",
        extra={"scene": SCENE, "camera": CHANNEL, "masks": variant == "masked", "eval_frame_idx": I_EVAL.tolist(),
               "n_train": len(I_TRAIN), "train_split_fraction": SPLIT_FRAC, "max_num_iterations": MAX_ITERS,
               "camera_res_scale_factor": SCALE, "render_size": [W_IMG, H_IMG], "metres_per_unit": MPU,
               "ns_eval": ev, "render_timing": fps, "depth_vs_lidar": dep, "train_cmd": meta["cmd"],
               "method_args": METHOD_ARGS.get(method, []), "top_frac": TOP_FRAC})

TABLE = pd.DataFrame(SUMMARY).T
TABLE.index.names = ["method", "variant"]
display(TABLE.round(3))

# %% [markdown]
# ## 6. 시각화

# %%
cols = ["psnr", "psnr_static", "psnr_top", "psnr_bottom"]
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
sub = TABLE.xs("masked", level="variant")
x = np.arange(len(cols))
for i, m in enumerate(sub.index):
    axes[0].bar(x + i * 0.25, sub.loc[m, cols].astype(float), 0.25, color=COLORS[m], label=m)
axes[0].set_xticks(x + 0.25, cols)
axes[0].set_ylabel("PSNR [dB]")
axes[0].legend()
axes[1].bar(sub.index, sub["train_time_s"].astype(float) / 60, color=[COLORS[m] for m in sub.index])
axes[1].set_ylabel("training time [min]")
axes[2].bar(sub.index, sub["render_fps"].astype(float), color=[COLORS[m] for m in sub.index])
axes[2].set_ylabel(f"render fps @ {W_IMG}x{H_IMG}")
axes[2].set_yscale("log")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. 실패 분석
# ### 7.0 가장 나쁜 평가 시점 (렌더 vs GT)
# 방법마다 PSNR이 가장 낮은 평가 프레임 `N_WORST`장. 오른쪽은 |렌더 − GT|(채널 평균), 하늘색 윤곽은 동적 마스크,
# 점선은 `TOP_FRAC` 경계.

# %%
N_WORST = 2
for method in METHODS:
    rows = sorted(PER_VIEW[(method, "masked")], key=lambda r: r["psnr_png"])[:N_WORST]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 3.2 * len(rows)), squeeze=False)
    for ax_row, r in zip(axes, rows):
        gt, pred = nst.split_eval_image(run_dir(method, "masked") / "eval_images" / f"eval_img_{r['k']:04d}.png")
        static = load_mask(EVAL_FRAMES[r["k"]], gt.shape[1], gt.shape[0])
        ax_row[0].imshow(gt)
        ax_row[0].set_title(f"GT frame {r['frame_idx']}", fontsize=9)
        ax_row[1].imshow(pred)
        ax_row[1].set_title(f"{method}  PSNR {r['psnr_png']:.2f} (static {r['psnr_static']:.2f})", fontsize=9)
        im = ax_row[2].imshow(np.abs(pred - gt).mean(-1), cmap="magma", vmin=0, vmax=0.3)
        if (~static).any():
            ax_row[2].contour((~static).astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        ax_row[2].axhline(TOP_FRAC * gt.shape[0], color="w", ls="--", lw=0.8)
        ax_row[2].set_title("|render - GT|", fontsize=9)
        for ax in ax_row:
            ax.set_axis_off()
        fig.colorbar(im, ax=ax_row[2], fraction=0.03)
    plt.show()

# %% [markdown]
# ### 7.1 (a) 전진 운동과 깊이 모호성
# 카메라가 앞으로만 움직이면 이웃 프레임 사이의 시차는 진행 방향(확장 초점, FOE)에서 0이고 바깥으로 갈수록 커진다
# (같은 깊이에서 시차 ∝ sin(광선과 이동 방향 사이 각)/깊이). 그래서 화면 중앙의 먼 물체는 거의 삼각측량이 안 된다.
# 아래는 (1) 진행 방향과의 각도 구간별 깊이 AbsRel(LiDAR 기준, 정적 픽셀), (2) 한 평가 프레임에서 LiDAR와 각 방법의
# 렌더 깊이, × 표시는 FOE다. 평가 프레임이 4장 안팎이라 구간별 표본이 작다(`n` 확인).

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for method in METHODS:
    ba = DEPTH[(method, "masked")]["by_angle"]
    ax.plot(list(ba), [ba[b]["abs_rel"] for b in ba], "o-", color=COLORS[method], label=method)
ax.set_xlabel("angle between pixel ray and motion direction")
ax.set_ylabel("depth AbsRel vs LiDAR")
ax.grid(alpha=0.3)
ax.legend()
plt.show()
display(pd.DataFrame({m: {b: v["n"] for b, v in DEPTH[(m, "masked")]["by_angle"].items()} for m in METHODS}))

k_show = len(EVAL_FRAMES) // 2  # middle eval frame
frame = EVAL_FRAMES[k_show]
K = scaled_K(frame)
_, foe = motion_geometry(int(I_EVAL[k_show]), K, W_IMG, H_IMG)
proj = project_lidar_to_camera(nusc, frame["lidar_sd_token"], frame["sd_token"])
fig, axes = plt.subplots(1, 1 + len(METHODS), figsize=(5 * (1 + len(METHODS)), 3.2))
overlay_points(axes[0], load_rgb(frame), proj["uv"] * SCALE, proj["depth"], vmax=60, s=1.5,
               title=f"LiDAR depth, frame {I_EVAL[k_show]}")
for ax, method in zip(axes[1:], METHODS):
    depth_panel(ax, load_depth_m(method, "masked", frame, K), vmax=60, title=f"{method} render z-depth [m]")
for ax in axes:
    if foe is not None and 0 <= foe[0] < W_IMG and 0 <= foe[1] < H_IMG:
        ax.plot(foe[0], foe[1], "wx", ms=10, mew=2)
plt.show()

# %% [markdown]
# > (실행 후 작성) 각도 구간별 AbsRel이 FOE 근처(0–5°)에서 가장 큰가? 방법 간 차이는? 중앙의 먼 차선·건물이 평면처럼
# > 뭉개지거나 앞으로 당겨졌는가?

# %% [markdown]
# ### 7.2 (b) 움직이는 물체 (마스크 사용 vs 미사용)
# 마스크 변형은 동적 픽셀을 학습에서 아예 보지 않으므로, 평가 프레임의 동적 영역은 "물체 뒤 배경"을 추측해 그린다
# (그 영역 PSNR은 낮을 수밖에 없다). 마스크 미사용 변형은 여러 시점에서 다른 위치에 있던 물체를 정적 장면 하나로
# 설명하려다 반투명 잔상·부유물이 생기고, 그 영향이 **정적 영역**까지 번질 수 있다. 비교는 `RUN_NO_MASK=True`일 때만 나온다.
# 01에서 고른 장면은 가장 정적인 장면이라 효과가 작을 수 있다.

# %%
k_dyn = int(np.argmax([r["dyn_ratio"] for r in PER_VIEW[(METHODS[0], "masked")]]))
print(f"eval frame with most dynamic pixels: k={k_dyn}, frame {I_EVAL[k_dyn]}, "
      f"masked ratio {DYN_RATIO[I_EVAL[k_dyn]]:.3f}")
if "nomask" not in VARIANTS:
    print("RUN_NO_MASK is False -> only the masked variant is shown. Set it to True to compare.")
else:
    display(TABLE[["psnr", "psnr_static", "psnr_dynamic"]].unstack("variant").round(2))
if DYN_RATIO[I_EVAL].max() == 0:
    print("No dynamic objects in the eval frames of this scene; set RECON_SCENE to a scene with moving objects.")

for method in METHODS:
    variants = [v for v in VARIANTS if (method, v) in PER_VIEW]
    fig, axes = plt.subplots(1, 1 + 2 * len(variants), figsize=(4.5 * (1 + 2 * len(variants)), 3))
    gt, _ = nst.split_eval_image(run_dir(method, "masked") / "eval_images" / f"eval_img_{k_dyn:04d}.png")
    static = load_mask(EVAL_FRAMES[k_dyn], gt.shape[1], gt.shape[0])
    axes[0].imshow(gt)
    if (~static).any():
        axes[0].contour((~static).astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    axes[0].set_title("GT + dynamic mask", fontsize=9)
    for j, v in enumerate(variants):
        _, pred = nst.split_eval_image(run_dir(method, v) / "eval_images" / f"eval_img_{k_dyn:04d}.png")
        r = PER_VIEW[(method, v)][k_dyn]
        axes[1 + 2 * j].imshow(pred)
        axes[1 + 2 * j].set_title(f"{method} [{v}] static {r['psnr_static']:.2f} / dyn {r['psnr_dynamic']:.2f}",
                                  fontsize=9)
        axes[2 + 2 * j].imshow(np.abs(pred - gt).mean(-1), cmap="magma", vmin=0, vmax=0.3)
        axes[2 + 2 * j].set_title(f"|render - GT| [{v}]", fontsize=9)
    for ax in axes:
        ax.set_axis_off()
    plt.show()

# %% [markdown]
# > (실행 후 작성) 마스크를 쓰면 정적 영역 PSNR이 올라가는가? 미사용 변형의 잔상은 어디에 생기는가(차선 위, 하늘)?
# > 마스크가 놓친 물체(카메라 옆을 지나는 차량 등)가 있는가?

# %% [markdown]
# ### 7.3 (c) 하늘과 원거리
# 하늘은 무한히 멀고 질감이 없으며, 먼 건물은 시차가 매우 작다. NeRF는 contraction 공간의 바깥 껍질에 색을 칠하거나
# 카메라 가까이에 "하늘 막"을 만들기 쉽고, 3DGS는 무작위 초기화된 가우시안이 하늘을 덮을 만큼 멀리 가지 못해
# 가까운 큰 가우시안으로 하늘을 칠하기 쉽다. LiDAR는 하늘을 재지 못하므로 (1) 거리 구간별 깊이 AbsRel(먼 구간),
# (2) 위쪽 영역 렌더 깊이 분포(하늘이 몇 m에 놓였는가), (3) 위/아래 PSNR 차이로 본다.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
bins = [bin_label(lo, hi) for lo, hi in DEFAULT_BINS]
for i, method in enumerate(METHODS):
    bd = DEPTH[(method, "masked")]["by_distance"]
    axes[0].bar(np.arange(len(bins)) + 0.25 * i, [bd[b]["abs_rel"] for b in bins], 0.25, color=COLORS[method],
                label=method)
axes[0].set_xticks(np.arange(len(bins)) + 0.25, bins)
axes[0].set_ylabel("depth AbsRel vs LiDAR")
axes[0].legend()

top_rows = slice(0, int(TOP_FRAC * H_IMG))
for method in METHODS:
    d_top = np.concatenate([load_depth_m(method, "masked", f, scaled_K(f))[top_rows].ravel() for f in EVAL_FRAMES])
    d_top = d_top[np.isfinite(d_top) & (d_top > 0)]
    axes[1].hist(np.log10(d_top), bins=80, histtype="step", color=COLORS[method], label=method, density=True)
axes[1].set_xlabel("log10 rendered z-depth [m], top band")
axes[1].legend()

sub = TABLE.xs("masked", level="variant")
axes[2].bar(sub.index, (sub["psnr_bottom"] - sub["psnr_top"]).astype(float), color=[COLORS[m] for m in sub.index])
axes[2].set_ylabel("PSNR bottom - top [dB]")
plt.tight_layout()
plt.show()

frame = EVAL_FRAMES[k_show]
fig, axes = plt.subplots(2, len(METHODS), figsize=(5.5 * len(METHODS), 6), squeeze=False)
for j, method in enumerate(METHODS):
    gt, pred = nst.split_eval_image(run_dir(method, "masked") / "eval_images" / f"eval_img_{k_show:04d}.png")
    axes[0, j].imshow(np.abs(pred - gt).mean(-1)[top_rows], cmap="magma", vmin=0, vmax=0.3)
    axes[0, j].set_title(f"{method} |render - GT|, top band", fontsize=9)
    depth_panel(axes[1, j], load_depth_m(method, "masked", frame, scaled_K(frame))[top_rows], vmax=150,
                title=f"{method} z-depth top band (0-150 m)")
    axes[0, j].set_axis_off()
plt.show()

# %% [markdown]
# > (실행 후 작성) 하늘 깊이가 수십 m 안쪽에 몰려 있는가(가짜 하늘 막)? 50m+ 구간 AbsRel은 가까운 구간보다 얼마나 큰가?
# > 위쪽 영역의 색 오차는 주로 하늘인가, 먼 건물·나무 가장자리인가?

# %% [markdown]
# ## 8. 해석
# > (실행 후 작성) 방법별 PSNR/SSIM/LPIPS·학습 시간·렌더 fps·크기 요약.
#
# > (실행 후 작성) 전진 운동: 진행 방향 근처 깊이 오차와 그 원인.
#
# > (실행 후 작성) 움직이는 물체: 마스크 효과와 남은 문제(놓친 박스, 그림자).
#
# > (실행 후 작성) 하늘·원거리: 하늘 표현 방식과 원거리 깊이 오차.
#
# > (실행 후 작성) 이 결과가 주행 전용 방법(아래 07b)이 필요한 이유를 어떻게 보여 주는가.
#
# 주의: 평가 프레임이 한 장면의 4장 안팎이고 보간(학습 프레임 사이) 시점이라, 수치는 이 설정 안에서의 상대 비교로만 읽는다.

# %% [markdown]
# ## 07b (선택, 이 노트북에서 구현하지 않음): 주행 전용 장면 표현
# DriveStudio(https://github.com/ziyc/drivestudio)는 nuScenes 6캠과 v1.0-mini를 공식 지원하며 OmniRe, StreetGS,
# DeformableGS, PVG를 한 코드베이스에서 돌린다. OmniRe는 장면을 하늘(별도 환경맵) + 정적 배경 가우시안 + 강체 물체
# (주석 박스 궤적에 고정된 가우시안) + 비강체(사람, SMPL 기반)로 나눠 위 7.1–7.3의 실패를 정면으로 다룬다.
# 여기서 구현하지 않는 이유: 공식 요구 환경이 torch 2.0.0/cu117 + gsplat 1.3.0이고, 하늘 마스크용 SegFormer는
# torch 1.8 **별도 env**가 필요하다(docs/step0_research.md). 우리 `3dv` env와 맞지 않아 env 두 개를 더 만들어야 하므로,
# 필요하면 RunPod에서 `3dv-drive` env로 따로 돌리고 결과만 `results/outdoor/07b_.../summary.json`으로 가져온다.
