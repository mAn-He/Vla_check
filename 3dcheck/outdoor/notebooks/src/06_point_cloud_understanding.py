# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv-point (torch 2.5.0, cu124)
#     language: python
#     name: 3dv-point
# ---

# %% [markdown]
# # 06. 포인트 클라우드 이해: PointNet, Concerto, PTv3, Sonata (nuScenes-mini)
#
# ## 목적
# LiDAR 점군을 "이해"하는 모델을 두 층위에서 비교합니다.
# - **6a 객체 분류(교육용)**: PointNet을 직접 구현해, GT 박스 안의 LiDAR 점만으로 10개 탐지 클래스를 구분합니다.
# - **6b 장면 의미 분할**: 최신 사전학습 백본(PTv3 계열)으로 nuScenes-lidarseg 16클래스를 분할합니다.
#   Concerto 실외 백본 + 선형 탐침이 주 실험이고, PTv3 공식 nuScenes 체크포인트는 로드를 시도만 하며,
#   실내 전용 Sonata는 특징 PCA로 도메인 불일치를 보여주기만 합니다.
# - **6c pseudo-LiDAR**: 04에서 카메라 깊이로 만든 pseudo-LiDAR를 같은 모델에 넣고 무엇을 잃는지 봅니다(정성 + 대리 지표).
#
# ## 비교 모델과 원리
# - **PointNet**: 모든 점에 같은 MLP(shared MLP)를 적용한 뒤 max-pool로 모읍니다. max는 순서에 무관한 대칭 함수라서
#   점의 순서가 바뀌어도 출력이 같습니다. 선택적으로 T-Net이 입력 좌표를 정렬하는 3x3 변환을 예측합니다.
#   여기서는 박스 좌표계로 옮긴 점 `P["pointnet_points"]`개를 입력으로, probe6 장면에서 처음부터 학습합니다.
# - **Point Transformer V3 (PTv3)**: 점을 공간 채움 곡선(z-order, Hilbert)으로 직렬화해 1차원으로 늘어놓고,
#   1024점 패치 안에서만 attention을 합니다. 이웃 탐색(kNN)을 없애 대규모 점군에서 빠릅니다. 인코더-디코더 U-Net 구조입니다.
#   nuScenes 공식 가중치는 README에 "temporarily invalid"(Pointcept v1.5.1용)로 적혀 있어 로드 시도만 합니다.
# - **Sonata**: PTv3 인코더를 라벨 없이 자기증류(self-distillation)로 사전학습한 모델입니다.
#   사전학습 데이터가 실내(ScanNet, ScanNet++, S3DIS, ARKitScenes, HM3D, Structured3D)뿐이라 실외 분할에는 쓰지 않고,
#   LiDAR에서 특징이 어떻게 보이는지(PCA 색)만 확인합니다.
# - **Concerto**: Sonata의 3D 자기증류에 2D 이미지 특징과의 교차 모달 예측을 더한 공동 2D-3D 자기지도 학습입니다.
#   실외용 `concerto_large_outdoor` 백본을 고정하고 선형층(점마다 1728차원 → 16클래스) 하나만 probe6으로 학습합니다.
#   공개된 nuScenes 분할 헤드는 없습니다(docs/step0_research.md).
#
# ## 원 논문
# - PointNet: Qi et al., CVPR 2017, https://arxiv.org/abs/1612.00593 (고정 저장소 README에는 없는 표준 인용)
# - Point Transformer V3: Wu et al., CVPR 2024, https://arxiv.org/abs/2312.10035 (PointTransformerV3, Pointcept README)
# - Sonata: Wu et al., CVPR 2025, https://arxiv.org/abs/2503.16429 (sonata README)
# - Concerto: Zhang et al., NeurIPS 2025, https://arxiv.org/abs/2510.23607 (Concerto README)
#
# ## 커널
# `3dv-point` (torch 2.5.0, CUDA 12.4). Pointcept 학습·평가는 같은 env의 python으로 `tools/train.py`, `tools/test.py`를
# subprocess로 실행합니다. PTv3 시도만 `PYTHONPATH`로 Pointcept-v1.5.1 체크아웃을 앞세웁니다.
#
# ## 예상 VRAM·시간 (추정)
# 아래는 모두 **추정치**입니다(RTX 3060에서 실측하지 않음). flash-attn이 없으면 attention이 fp32로 돌아 더 느리고 메모리를 더 씁니다.
#
# | 단계 | mid (12GB) | low (6~8GB) |
# |---|---|---|
# | 6a PointNet 학습 (약 1만 객체, 30 epoch) | 1~2GB, 수 분 | 같음 |
# | 6b Concerto 선형 탐침 학습 (PTv3-large 고정, probe6 약 240스캔 x 20 epoch) | 8~10GB, 30~60분 | batch 1, flash 끔: 6GB 내외, 1~2시간 |
# | 6b 평가 (eval4 약 160스캔 x TTA 10회) | 4~6GB, 10~20분 | 20~40분 |
# | 6b PTv3 시도 | 실패하면 수 초, 성공하면 평가와 비슷 | 같음 |
# | Sonata/Concerto PCA (스캔 2개) | 3~5GB, 1~2분 | 같음 |
# | 6c pseudo-LiDAR 평가 (원천 모델마다) | 점 수에 비례, 10~30분 | 더 느림 |
#
# ## 이 노트북이 답하려는 질문
# - 박스 안 점만 보는 PointNet은 10개 클래스를 얼마나 구분하고, 점이 적은 원거리·소형 객체에서 어떻게 무너지는가?
# - 실외 자기지도 백본(Concerto)의 특징은 선형 분류기 하나로 nuScenes 16클래스를 얼마나 분리하는가? (mini 6장면만으로 학습)
# - 실내에서만 학습한 Sonata의 특징은 LiDAR에서 어떻게 보이는가?
# - 카메라로 만든 pseudo-LiDAR를 같은 모델에 넣으면 어떤 클래스가 먼저 무너지는가?

# %% [markdown]
# ## 0. 설정 로드
# 하이퍼파라미터와 경로는 `outdoor/config.yaml`의 `models.point`와 `profiles.<profile>.point`에서 읽습니다.
# config에 아직 없는 값(`pointnet_min_points` 등)은 `P.get(키, 기본값)`으로 읽고 기본값을 이 셀에 드러냅니다.

# %%
FORCE_RERUN = False  # True: ignore cached results
NOTEBOOK = "06_point_cloud_understanding"

import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sonata
import torch
from bench3d.config import load_config
from bench3d.io import collect_summaries, is_cached, load_nuscenes_bin, load_summary, result_dir, save_summary
from bench3d.metrics.segmentation import confusion_matrix, iou_from_confusion
from bench3d.models import point as bp
from bench3d.nusc import load_lidar, scene_samples
from bench3d.runtime import Measure, env_report, free_cuda, set_seed
from bench3d.sampling import middle
from concerto.transform import Compose as ConcertoCompose
from huggingface_hub import hf_hub_download, list_repo_files
from matplotlib.patches import Patch
from nuscenes.nuscenes import NuScenes
from pointcept.utils.config import Config as PCConfig

cfg = load_config("outdoor")  # profile: config.yaml or $THREEDCHECK_PROFILE
P = cfg.p["point"]  # this notebook's profile block
M = cfg["models"]["point"]
FORCE_RERUN = FORCE_RERUN or cfg.get("force_rerun", False)
set_seed(cfg["seed"])
ENV = env_report(["torch", "numpy", "spconv-cu124", "torch-scatter", "nuscenes-devkit", "open3d", "huggingface-hub",
                  "flash-attn", "sonata", "concerto", "scipy"])
ENV["third_party"] = {n: bp.git_head(cfg.third_party(n)) for n in ["Pointcept", "Pointcept-v1.5.1", "Concerto", "sonata"]}
print(json.dumps(ENV, indent=1))
print("profile:", cfg["profile"], P)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FLASH = bool(P["enable_flash"]) and bp.flash_available()
if P["enable_flash"] and not USE_FLASH:
    print("WARNING: config enable_flash=true but flash_attn is not importable -> running with enable_flash=False")

# Values not (yet) in config.yaml: defaults shown here, requested as profile keys
PN_MIN_PTS = P.get("pointnet_min_points", 5)  # objects with fewer LiDAR points in the box are dropped
PN_BATCH = P.get("pointnet_batch_size", 32)
PN_LR = P.get("pointnet_lr", 1e-3)
PN_FEATURE_TNET = P.get("pointnet_feature_transform", False)
NUM_WORKER = P.get("num_worker", 4)  # Pointcept DataLoader uses persistent_workers=True -> must be >= 1
TIMING_SCANS = P.get("timing_scans", 10)
PROXY_MAX_DIST = P.get("proxy_max_dist", 0.5)  # metres, 6c label transfer radius
BEV_EXTENT = 50.0  # plot only

PC_ROOT = cfg.third_party("Pointcept")  # main checkout (also on sys.path through the env's .pth)
PC151_ROOT = cfg.third_party(M["ptv3_nuscenes"]["pointcept_checkout"])
DATA_ROOT = cfg.path(cfg["dataset"]["data_root"])
PC_DATA = cfg.path(cfg["dataset"].get("pointcept_root", "data/pointcept_nuscenes_mini"))
SWEEPS = 10  # NuScenesDataset(sweeps=10) default -> info file name (pointcept/datasets/nuscenes.py)
VAL_PKL = PC_DATA / "info" / f"nuscenes_infos_{SWEEPS}sweeps_val.pkl"
TRAIN_PKL = PC_DATA / "info" / f"nuscenes_infos_{SWEEPS}sweeps_train.pkl"

# %% [markdown]
# ## 1. 데이터 로드
# - 평가 표본은 01 노트북의 `eval_samples.json`에서 읽습니다. **평가는 eval4 장면의 모든 키프레임**, 학습(PointNet, 선형 탐침)은
#   **probe6 장면의 모든 키프레임**만 씁니다.
# - Pointcept용 info pkl은 `outdoor/scripts/pointcept_nuscenes_mini.py`로 만듭니다. 공식 전처리 스크립트는 v1.0-trainval/test만
#   허용하고 장면 수를 assert하므로, 공식 함수(`get_available_scenes`, `fill_trainval_infos`)를 import해 mini에 맞춘
#   **비공식 변형**입니다(train = probe6, val = eval4 키프레임, test pkl 없음; 차이는 스크립트 docstring).
# - 클래스 이름과 라벨 매핑(원본 32클래스 → 16클래스, 나머지는 무시 -1)은 Pointcept 소스에서 직접 읽습니다
#   (`configs/concerto/semseg-ptv3-large-v1m1-nuscenes-5a-lin.py`의 `data.names`, `NuScenesDataset.get_learning_map`).

# %%
EVAL_SAMPLES = cfg.path(cfg["results_root"]) / "01_data_exploration" / "eval_samples.json"
EVAL = json.loads(EVAL_SAMPLES.read_text())
EVAL4, PROBE6 = cfg["splits"]["eval4"], cfg["splits"]["probe6"]
assert sorted(EVAL["eval4"]) == sorted(EVAL4) and sorted(EVAL["probe6"]) == sorted(PROBE6), "rerun notebook 01"

nusc = NuScenes(version=cfg["dataset"]["version"], dataroot=str(DATA_ROOT), verbose=False)
assert getattr(nusc, "lidarseg", None), "lidarseg-mini tables not found (outdoor/scripts/verify_nuscenes_mini.py)"
EVAL_KF = [t for s in EVAL4 for t in EVAL["keyframes"][s]]
TRAIN_KF = [t for s in PROBE6 for t in (EVAL["keyframes"].get(s) or scene_samples(nusc, s))]
print(f"eval4 keyframes: {len(EVAL_KF)}  probe6 keyframes: {len(TRAIN_KF)}")

# Pointcept info pkls (unofficial mini adaptation)
if FORCE_RERUN or not (VAL_PKL.exists() and TRAIN_PKL.exists()):
    cmd = [sys.executable, str(cfg.root / "outdoor" / "scripts" / "pointcept_nuscenes_mini.py"),
           "--out", str(PC_DATA), "--eval-samples", str(EVAL_SAMPLES), "--max-sweeps", str(SWEEPS)]
    subprocess.run(cmd + (["--force"] if FORCE_RERUN else []), check=True, cwd=cfg.root)
VAL_INFOS = pickle.loads(VAL_PKL.read_bytes())
TRAIN_INFOS = pickle.loads(TRAIN_PKL.read_bytes())
assert {i["token"] for i in VAL_INFOS} == set(EVAL_KF), "val infos differ from eval_samples.json keyframes"
print(json.loads((PC_DATA / "info" / "mini_split_meta.json").read_text()))

# class names / label map / config straight from the pinned Pointcept sources
LIN_CFG_PATH = PC_ROOT / M["concerto_outdoor"]["pointcept_config"]
LIN = PCConfig.fromfile(str(LIN_CFG_PATH))
NAMES16, NUM16, IGN = list(LIN.data.names), int(LIN.data.num_classes), int(LIN.data.ignore_index)
LUT16 = bp.lidarseg16_lut(IGN)
PAL16 = bp.class_palette(NUM16)
print(NUM16, "classes:", NAMES16)

# pseudo-LiDAR sources written by notebook 04 (same relative lidar_path layout under <model>/pseudo/)
# TODO(verify): 04 stores each cloud at pseudo/<real LIDAR_TOP filename> (samples/LIDAR_TOP/<basename>.bin);
#               the 04 manifest json is not parsed here, files are matched by that relative path only.
PSEUDO_ROOT = cfg.path(cfg["results_root"]) / "04_pseudo_lidar"
if cfg.get("pseudo_lidar_source"):
    PSEUDO_SRCS = [cfg["pseudo_lidar_source"]]
else:
    PSEUDO_SRCS = sorted(p.name for p in PSEUDO_ROOT.glob("*") if (p / "pseudo" / "samples" / "LIDAR_TOP").is_dir())
print("pseudo-LiDAR sources:", PSEUDO_SRCS or "none found (6c will be skipped)")

VIS_SAMPLE = middle(EVAL["keyframes"][EVAL4[0]])  # one fixed keyframe for all qualitative panels
VIS_INFO = next(i for i in VAL_INFOS if i["token"] == VIS_SAMPLE)

# %% [markdown]
# ## 2. 6a. PointNet 객체 분류 (직접 구현)
# **데이터**: 키프레임 LiDAR(`LIDAR_TOP`, sweep 없음)에서 GT 박스 안의 점을 devkit `points_in_box`로 잘라냅니다.
# 카테고리는 devkit `category_to_detection_name`으로 10개 탐지 클래스에 매핑하고, 매핑되지 않는 것(동물, 유모차 등)은 버립니다.
# 점은 박스 좌표계(중심 원점, 박스 heading이 +x)로 옮깁니다. 위치와 방향은 지워지지만 **크기 정보는 남습니다**(GT 박스를 쓰는 oracle 설정).
# 박스 안 점이 `PN_MIN_PTS`개 미만이면 버리고, 나머지는 `P["pointnet_points"]`개로 다시 뽑습니다(부족하면 중복 허용).
#
# **분할**: 학습 probe6 키프레임, 평가 eval4 키프레임(박스는 장면마다 따로라 객체 단위 누수는 없지만, 같은 장면의 같은 객체가
# 여러 키프레임에 반복 등장하므로 표본이 서로 독립은 아닙니다).
#
# **지표**: 정확도(전체), 균형 정확도(클래스별 정확도 평균), 클래스별 정확도, 혼동 행렬, 박스 안 점 수 구간별 정확도.
# 클래스 불균형이 큽니다(car, pedestrian이 대부분, construction_vehicle, trailer 등은 수십 개 이하). 그래서 전체 정확도는
# 다수 클래스에 끌려가므로 균형 정확도를 함께 봅니다. 손실은 가중치 없는 cross-entropy입니다.

# %%
PN_DIR = result_dir(cfg, NOTEBOOK, "pointnet")
CLASSES10 = bp.detection_classes()
crop_file = PN_DIR / "crops.pkl"
if is_cached(crop_file, FORCE_RERUN):
    CROPS = pickle.loads(crop_file.read_bytes())
else:
    CROPS = {"train": bp.object_crops(nusc, TRAIN_KF, P["pointnet_points"], PN_MIN_PTS, seed=cfg["seed"]),
             "test": bp.object_crops(nusc, EVAL_KF, P["pointnet_points"], PN_MIN_PTS, seed=cfg["seed"] + 1)}
    crop_file.write_bytes(pickle.dumps(CROPS))
for split, c in CROPS.items():
    counts = np.bincount(c["labels"], minlength=len(CLASSES10))
    print(f"{split}: {len(c['labels'])} objects, skipped {c['skipped']}")
    print("   ", dict(zip(CLASSES10, counts.tolist())))

# %%
pn_summary_file = PN_DIR / "summary.json"
pn_pred_file = PN_DIR / "pred.npz"
if is_cached(pn_summary_file, FORCE_RERUN) and pn_pred_file.exists():
    PN_SUM = load_summary(pn_summary_file)
    pn_pred = np.load(pn_pred_file)["pred"]
else:
    set_seed(cfg["seed"])
    with free_cuda():
        pn = bp.PointNetCls(len(CLASSES10), in_dim=3, input_transform=True, feature_transform=PN_FEATURE_TNET)
        with Measure() as m_pn:
            pn_hist = bp.fit_pointnet(pn, CROPS["train"]["points"], CROPS["train"]["labels"],
                                      epochs=P["pointnet_epochs"], batch_size=PN_BATCH, lr=PN_LR, device=DEVICE,
                                      seed=cfg["seed"])
        pn_pred, pn_prob, pn_inf_s = bp.predict_pointnet(pn, CROPS["test"]["points"], device=DEVICE)
        torch.save(pn.state_dict(), PN_DIR / "pointnet.pt")
        del pn
    gt = CROPS["test"]["labels"]
    np.savez(pn_pred_file, pred=pn_pred, prob=pn_prob, labels=gt)
    conf10 = confusion_matrix(pn_pred, gt, len(CLASSES10))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_cls = np.diag(conf10) / conf10.sum(1)
    n_in = CROPS["test"]["n_in_box"]
    bins = [PN_MIN_PTS, 20, 50, 200, np.inf]
    acc_by_pts = {f"[{lo},{hi})": (float((pn_pred[(n_in >= lo) & (n_in < hi)] == gt[(n_in >= lo) & (n_in < hi)]).mean())
                                   if ((n_in >= lo) & (n_in < hi)).any() else None)
                  for lo, hi in zip(bins[:-1], bins[1:])}
    pn_metrics = {"accuracy": float((pn_pred == gt).mean()), "balanced_accuracy": float(np.nanmean(per_cls)),
                  "n_test_objects": int(len(gt)), "n_train_objects": int(len(CROPS["train"]["labels"])),
                  "infer_ms_per_object": 1000 * pn_inf_s / max(len(gt), 1)}
    save_summary(PN_DIR, part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model="pointnet",
                 variant=f"gt_box_crops/{P['pointnet_points']}pts/min{PN_MIN_PTS}/probe6->eval4",
                 metrics=pn_metrics, runtime_s=m_pn.seconds, peak_vram_mb=m_pn.peak_vram_mb,
                 checkpoint="none (trained from scratch in this notebook)", packages=ENV["packages"], seed=cfg["seed"],
                 extra={"classes": CLASSES10, "per_class_acc": dict(zip(CLASSES10, per_cls.tolist())),
                        "confusion": conf10.tolist(), "acc_by_points_in_box": acc_by_pts,
                        "train_counts": np.bincount(CROPS["train"]["labels"], minlength=10).tolist(),
                        "test_counts": np.bincount(gt, minlength=10).tolist(),
                        "skipped": {k: v["skipped"] for k, v in CROPS.items()},
                        "input_transform": True, "feature_transform": PN_FEATURE_TNET,
                        "batch_size": PN_BATCH, "lr": PN_LR, "history": pn_hist})
    PN_SUM = load_summary(pn_summary_file)
print(json.dumps(PN_SUM["metrics"], indent=1))
print(json.dumps(PN_SUM["extra"]["per_class_acc"], indent=1))

# %% [markdown]
# ## 3. 6b-1. Concerto 실외 백본 + 선형 탐침 (Pointcept 공식 config)
# **config**: `configs/concerto/semseg-ptv3-large-v1m1-nuscenes-5a-lin.py`(아래 셀이 고정 소스에서 직접 읽음).
# - 모델: `DefaultSegmentorV2`, 백본 `PT-v3m2`(`in_channels=9` = 좌표 3 + 색 3 + 법선 3, `enc_mode=True`),
#   `freeze_backbone=True`, 헤드는 `nn.Linear(1728, 16)` 하나(인코더 5단계 특징 64+128+256+512+768을 원 해상도로 이어 붙임).
# - 데이터: `NuScenesColorNormalDataset`은 **색과 법선을 0으로** 넣습니다. 좌표는 `RandomScale(0.2)`로 1/5로 줄인 뒤
#   `GridSample(0.01)`이므로 실제로는 5cm 격자입니다. 평가는 축척·뒤집기 TTA 10회의 softmax 합입니다.
# - 손실: CrossEntropy + Lovasz.
#
# **바꾼 값 (`--options`)**, 근거는 `bench3d/models/point.py` docstring의 소스 경로:
#
# | 키 | 공식 | 여기 | 이유 |
# |---|---|---|---|
# | `hooks.0.keywords` | `module.student.backbone` | `module` | Pointcept `pointcept/models/concerto/README.md`: HF의 `concerto_*.pth`는 이렇게 바꿔야 로드됨 |
# | `weight` | 없음 | HF `Pointcept/Concerto/concerto_large_outdoor.pth` | `concerto.load`와 같은 `hf_hub_download` 경로 |
# | `data.{train,val,test}.data_root` | `data/nuscenes` | 준비한 mini 폴더 | `data_root` 변수는 파싱 시점에 펼쳐지므로 각 split에 직접 넣음 |
# | `batch_size` / `epoch` | 64 / 50 | `P["probe_batch_size"]` / `P["probe_epochs"]` | 12GB GPU, mini |
# | `eval_epoch` | 50 | 1 | `Trainer.max_epoch = eval_epoch`, 데이터는 `epoch // eval_epoch`번 반복 → 학습 끝에 한 번만 평가 |
# | `evaluate` | True | False | eval4로 best 체크포인트를 고르지 않음(평가셋 선택 누수 방지). `model_last.pth`만 저장 |
# | `hooks.5.test_last` | False | True | `PreciseEvaluator`가 `model_best` 대신 마지막 가중치로 테스트 |
# | `model.backbone.enable_flash` | True | `USE_FLASH` | flash-attn이 없으면 False (Concerto README: patch 1024 유지 가능) |
# | `num_worker` / `enable_wandb` / `seed` | 112 / True / None | `NUM_WORKER` / False / config seed | 단일 GPU, 재현성 |
#
# 학습률(AdamW 0.002, OneCycle)은 batch 64 기준 공식 값을 그대로 둡니다. batch가 작아 업데이트 횟수가 늘어나는 점은 공식 결과와 다른 조건입니다.
#
# **누수 점검**: 선형 헤드는 probe6로만 학습하고 eval4로 체크포인트를 고르지 않습니다. 다만 `concerto_large_outdoor`의
# 자기지도 사전학습 데이터 구성(nuScenes 사용 여부, val 장면 포함 여부)은 Concerto README와 Pointcept config에서 확인하지 못했습니다
# (TODO(verify)). 라벨은 쓰지 않는 사전학습이지만, eval4 장면의 점군을 사전학습에서 봤을 가능성은 남습니다.

# %%
CC = M["concerto_outdoor"]  # TODO(verify): pretraining scene list of concerto_large_outdoor (eval4 overlap?)
CC_DIR = result_dir(cfg, NOTEBOOK, "concerto_outdoor")
CC_EXP = CC_DIR / "probe"  # Pointcept save_path (model/, result/, train.log, config.py)
assert LIN.hooks[0]["type"] == "CheckpointLoader" and LIN.hooks[5]["type"] == "PreciseEvaluator", LIN.hooks

# same call as concerto/model.py load() -> shares the ~/.cache/concerto/ckpt download with the PCA section
CC_CKPT = Path(hf_hub_download(repo_id=CC["hf_repo"], filename=f"{CC['name']}.pth", repo_type="model",
                               revision="main", local_dir=str(Path.home() / ".cache" / "concerto" / "ckpt")))

# architecture check: HF checkpoint config vs. the linear-probe config backbone (CheckpointLoader is strict=False)
_ck = torch.load(CC_CKPT, map_location="cpu", weights_only=True)
_bb = dict(LIN.model.backbone)
ARCH_KEYS = ["in_channels", "order", "stride", "enc_depths", "enc_channels", "enc_num_head", "mlp_ratio", "qkv_bias",
             "pre_norm", "enable_rpe", "layer_scale", "enc_mode"]
_norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v  # noqa: E731
CC_MISMATCH = {k: {"ckpt": _ck["config"].get(k), "lin_cfg": _bb.get(k)} for k in ARCH_KEYS
               if k in _ck["config"] and _norm(_ck["config"][k]) != _norm(_bb.get(k))}
print("ckpt keys:", list(_ck), " state_dict prefixes:", sorted({k.split(".")[0] for k in _ck["state_dict"]}))
print("architecture mismatches (ckpt vs lin config):", CC_MISMATCH or "none")
# layer_scale absent from the lin config means PT-v3m2 default None
assert not CC_MISMATCH, "linear-probe backbone != checkpoint backbone; fix the config before training"
del _ck

cc_opts = {
    "save_path": CC_EXP, "weight": CC_CKPT,
    "data.train.data_root": PC_DATA, "data.val.data_root": PC_DATA, "data.test.data_root": PC_DATA,
    "batch_size": P["probe_batch_size"], "epoch": P["probe_epochs"], "eval_epoch": 1, "evaluate": False,
    "num_worker": NUM_WORKER, "enable_wandb": False, "seed": cfg["seed"],
    "hooks.0.keywords": "module", "hooks.0.replacement": "module.backbone", "hooks.5.test_last": True,
    "model.backbone.enable_flash": USE_FLASH,
}
cc_run_file = CC_DIR / "run_train.json"
n_pred = len(list((CC_EXP / "result").glob("*_pred.npy")))
if is_cached(CC_EXP / "model" / "model_last.pth", FORCE_RERUN) and n_pred == len(VAL_INFOS) and cc_run_file.exists():
    CC_RUN = json.loads(cc_run_file.read_text())
    print("cached:", CC_EXP)
else:
    if CC_EXP.exists():
        shutil.rmtree(CC_EXP)  # SemSegTester re-uses existing *_pred.npy -> start clean
    CC_RUN = bp.run_pointcept("train", LIN_CFG_PATH, cc_opts, pointcept_root=PC_ROOT,
                              log_path=CC_DIR / "train_subprocess.log")
    print(bp.tail(CC_RUN["log"]))
    assert CC_RUN["returncode"] == 0, f"tools/train.py failed, see {CC_RUN['log']}"
    cc_run_file.write_text(json.dumps(CC_RUN, indent=1))

CC_LOG = "\n".join(p.read_text(errors="replace") for p in [Path(CC_RUN["log"]), CC_EXP / "train.log", CC_EXP / "test.log"]
                   if p.exists())
CC_MISSING = bp.parse_missing_keys(CC_LOG)
print("missing keys after weight load:", CC_MISSING)
assert CC_MISSING is not None and all(k.startswith("seg_head") for k in CC_MISSING), \
    "backbone weights were not loaded (CheckpointLoader keyword mapping)"
CC_LOGRES = bp.parse_semseg_log(CC_LOG, NAMES16)
print("Pointcept log:", None if CC_LOGRES is None else {k: CC_LOGRES[k] for k in ("miou", "macc", "allacc")})

# %% [markdown]
# ### 평가 (eval4, 실제 LiDAR)
# - 대상: eval4 키프레임의 **모든 LiDAR 점**(거리 제한 없음). 정답은 lidarseg 원본 라벨을 Pointcept 매핑으로 16클래스로 바꾼 것이며,
#   매핑이 없는 클래스(ego 차량, 소음, 동물 등)는 무시(-1)합니다.
# - 예측: Pointcept `SemSegTester`가 저장한 점별 예측(`result/<lidar_token>_pred.npy`, TTA 10회).
# - 혼동 행렬은 스캔 전체에 걸쳐 누적합니다(데이터셋 단위).
#   `miou`(bench3d)는 GT나 예측에 나타난 클래스만 평균하고, `miou_all16`은 Pointcept 로그와 같이 16클래스 모두를 평균합니다
#   (eval4에 없는 클래스는 IoU 0으로 들어가 값을 낮춤). 둘 다 저장합니다.
# - 추론 시간: (1) Pointcept 로그의 스캔당 평균(TTA 10회 + 데이터 처리 포함), (2) 이 노트북에서 잰 단일 패스(TTA 없음, val 변환)
#   `TIMING_SCANS`개 스캔의 중앙값(첫 스캔은 워밍업으로 제외). 피크 VRAM은 (2)의 값이며 학습 subprocess의 VRAM은 재지 않았습니다.

# %%
def eval_pointcept_preds(pred_dir: Path, infos: list[dict]):
    """Dataset-level confusion of Pointcept *_pred.npy files against lidarseg (16 classes)."""
    conf = np.zeros((NUM16, NUM16), np.int64)
    missing = []
    for info in infos:
        f = pred_dir / f"{info['lidar_token']}_pred.npy"
        if not f.exists():
            missing.append(info["token"])
            continue
        pred = np.load(f).reshape(-1)
        gt = bp.load_lidarseg16(nusc, info["lidar_token"], LUT16)
        assert len(pred) == len(gt), (f.name, len(pred), len(gt))
        conf += confusion_matrix(pred, gt, NUM16, ignore_index=IGN)
    res = iou_from_confusion(conf, NAMES16)
    tp, union = np.diag(conf), conf.sum(0) + conf.sum(1) - np.diag(conf)
    res["miou_all16"] = float(np.mean(tp / (union + 1e-10)))  # Pointcept SemSegTester definition
    return conf, res, missing


@torch.no_grad()
def single_pass_timing(exp_dir: Path, infos: list[dict], data_root: Path, n: int):
    """Median forward time of the trained segmentor on n scans (val transform, no TTA), plus peak VRAM."""
    from pointcept.datasets.transform import Compose
    from pointcept.models import build_model

    pc = PCConfig.fromfile(str(exp_dir / "config.py"))
    model = build_model(pc.model).to(DEVICE).eval()
    sd = torch.load(exp_dir / "model" / "model_last.pth", map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    tf = Compose(pc.data.val.transform)
    times = []
    with Measure() as m:  # one Measure only: nested Measure() would reset the peak-memory counter
        for info in infos[:n]:
            pts = np.fromfile(data_root / "raw" / info["lidar_path"], dtype=np.float32).reshape(-1, 5)
            xyz = pts[:, :3].copy()
            d = tf({"coord": xyz, "color": np.zeros_like(xyz), "normal": np.zeros_like(xyz),
                    "segment": np.full(len(xyz), IGN, np.int64)})
            for k in ("segment", "origin_segment"):  # no labels -> DefaultSegmentorV2 takes the test branch
                d.pop(k, None)
            d = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(d)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    del model
    return float(np.median(times[1:] if len(times) > 1 else times)), m.peak_vram_mb


cc_summary_file = CC_DIR / "summary.json"
if is_cached(cc_summary_file, FORCE_RERUN):
    CC_SUM = load_summary(cc_summary_file)
else:
    CC_CONF, CC_RES, cc_missing = eval_pointcept_preds(CC_EXP / "result", VAL_INFOS)
    assert not cc_missing, f"{len(cc_missing)} eval scans without predictions"
    with free_cuda():
        cc_t1, cc_vram = single_pass_timing(CC_EXP, VAL_INFOS, PC_DATA, TIMING_SCANS)
    save_summary(CC_DIR, part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model="concerto_outdoor",
                 variant="linear_probe/probe6->eval4/lidar",
                 metrics={"miou": CC_RES["miou"], "miou_all16": CC_RES["miou_all16"],
                          "overall_acc": CC_RES["overall_acc"],
                          "miou_pointcept_log": CC_LOGRES["miou"] if CC_LOGRES else None,
                          "infer_s_per_scan_single": cc_t1,
                          "infer_s_per_scan_tta10": CC_LOGRES["scan_time_avg_s"] if CC_LOGRES else None,
                          "train_and_test_s": CC_RUN["seconds"], "n_eval_scans": len(VAL_INFOS)},
                 runtime_s=CC_RUN["seconds"], peak_vram_mb=cc_vram,
                 checkpoint=f"{CC['hf_repo']}/{CC['name']}.pth + linear head trained on probe6",
                 packages=ENV["packages"], seed=cfg["seed"],
                 extra={"per_class_iou": CC_RES["per_class_iou"], "classes": NAMES16, "confusion": CC_CONF.tolist(),
                        "per_class_iou_pointcept_log": CC_LOGRES["per_class_iou"] if CC_LOGRES else None,
                        "pointcept_config": M["concerto_outdoor"]["pointcept_config"],
                        "pointcept_options": {k: str(v) for k, v in cc_opts.items() if k not in ("save_path", "weight")},
                        "missing_keys": CC_MISSING, "use_flash": USE_FLASH, "third_party": ENV["third_party"],
                        "peak_vram_note": "single-pass inference in-notebook; training subprocess not measured",
                        "leakage_note": "eval4 never used for training or model selection (evaluate=False, test_last)"})
    CC_SUM = load_summary(cc_summary_file)
print(json.dumps(CC_SUM["metrics"], indent=1))

# %% [markdown]
# ## 4. 6b-2. PTv3 공식 nuScenes 체크포인트 로드 시도 (Pointcept v1.5.1)
# - README 모델 표의 nuScenes 행(80.3 mIoU)은 HF `Pointcept/PointTransformerV3/nuscenes-semseg-pt-v3m1-0-base`를 가리키지만,
#   같은 README가 **"Released model weights are temporarily invalid as the model structure of PTv3 is adjusted."**라고 적고 있습니다.
#   그래서 v1.5.1 체크아웃의 `configs/nuscenes/semseg-pt-v3m1-0-base.py`와 `tools/test.py`로 시도하고, 실패하면 오류를 기록하고 넘어갑니다.
# - **데이터 형식은 호환됩니다.** v1.5.1과 main의 `preprocessing/nuscenes/preprocess_nuscenes_info.py`는 빈 줄 하나만 다르고,
#   v1.5.1 `NuScenesDataset`도 같은 info 파일명, `raw/` 경로, `lidar_path`·`gt_segment_path`·`lidar_token` 키, 같은 16클래스 매핑을 씁니다.
#   입력은 `in_channels=4`(좌표 + 반사강도/255)라 Concerto와 달리 intensity를 씁니다.
# - **정적 점검에서 이미 찾은 장애**: v1.5.1의 `pointcept/models/__init__.py`가 OctFormer를 무조건 import하는데,
#   `ocnn`이 없으면 `octformer_v1m1_base.py`의 `class OctreeT(Octree)`가 `Octree=None`을 상속하려다 `TypeError`가 납니다.
#   3dv-point env에는 `ocnn`이 없으므로 `pip install ocnn`(v1.5.1 README) 없이는 모델 import 단계에서 실패합니다.
# - 누수 점검: 이 체크포인트는 공식 train으로 학습했고 eval4는 모두 공식 val이라 누수가 없습니다.
# - flash-attn이 없으면 PTv3 README대로 `enable_flash=False`, `enc/dec_patch_size=128`로 바꿉니다.

# %%
PT = M["ptv3_nuscenes"]
PT_DIR = result_dir(cfg, NOTEBOOK, "ptv3_nuscenes")
PT_CFG_PATH = PC151_ROOT / PT["pointcept_config"]
PT_EXP = PT_DIR / "exp"
pt_summary_file = PT_DIR / "summary.json"
if is_cached(pt_summary_file, FORCE_RERUN):
    PT_SUM = load_summary(pt_summary_file)
else:
    pt_status = {"status": "not_run", "errors": [], "checkpoint_file": None}
    pt_metrics, pt_extra, pt_seconds = {}, {}, None
    try:
        if importlib.util.find_spec("ocnn") is None:
            pt_status["errors"].append("preflight: ocnn not installed -> Pointcept v1.5.1 models import is expected to "
                                       "fail (octformer_v1m1_base.py: class OctreeT(Octree) with Octree=None)")
        hf_files = [f for f in list_repo_files(PT["hf_repo"]) if f.startswith(PT["hf_subdir"] + "/")]
        pt_status["hf_files"] = hf_files
        # TODO(verify): weight file name inside the HF folder (Pointcept exp layout suggests model/model_best.pth)
        pths = sorted(f for f in hf_files if f.endswith(".pth"))
        if not pths:
            raise FileNotFoundError(f"no .pth under {PT['hf_repo']}/{PT['hf_subdir']}: {hf_files}")
        rel = next((f for f in pths if f.endswith("model_best.pth")), pths[0])
        pt_status["checkpoint_file"] = rel
        pt_w = hf_hub_download(repo_id=PT["hf_repo"], filename=rel, local_dir=str(PT_DIR / "ckpt"))
        pt_opts = {"save_path": PT_EXP, "weight": pt_w, "data.test.data_root": PC_DATA,
                   "model.backbone.enable_flash": USE_FLASH, "enable_wandb": False, "seed": cfg["seed"]}
        if not USE_FLASH:  # PointTransformerV3 README, "Flash Attention" section
            pt_opts.update({"model.backbone.enc_patch_size": (128,) * 5, "model.backbone.dec_patch_size": (128,) * 4})
        if PT_EXP.exists():
            shutil.rmtree(PT_EXP)
        run = bp.run_pointcept("test", PT_CFG_PATH, pt_opts, pointcept_root=PC151_ROOT, pythonpath=PC151_ROOT,
                               log_path=PT_DIR / "test_subprocess.log")
        pt_seconds = run["seconds"]
        pt_status["returncode"] = run["returncode"]
        if run["returncode"] != 0:
            raise RuntimeError("tools/test.py (v1.5.1) failed:\n" + bp.tail(run["log"], 12))
        pt_conf, pt_res, pt_missing = eval_pointcept_preds(PT_EXP / "result", VAL_INFOS)
        pt_log = bp.parse_semseg_log(Path(run["log"]).read_text(errors="replace"), NAMES16)
        pt_metrics = {"miou": pt_res["miou"], "miou_all16": pt_res["miou_all16"], "overall_acc": pt_res["overall_acc"],
                      "miou_pointcept_log": pt_log["miou"] if pt_log else None,
                      "infer_s_per_scan_tta10": pt_log["scan_time_avg_s"] if pt_log else None,
                      "n_eval_scans": len(VAL_INFOS) - len(pt_missing)}
        pt_extra = {"per_class_iou": pt_res["per_class_iou"], "confusion": pt_conf.tolist()}
        pt_status["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - record and continue (expected: invalid weights / import failure)
        pt_status["status"] = "failed"
        pt_status["errors"].append(f"{type(e).__name__}: {e}")
    save_summary(PT_DIR, part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model="ptv3_nuscenes",
                 variant="official_ckpt/pointcept_v1.5.1/eval4/lidar", metrics=pt_metrics, runtime_s=pt_seconds,
                 peak_vram_mb=None, checkpoint=f"{PT['hf_repo']}/{PT['hf_subdir']}", packages=ENV["packages"],
                 seed=cfg["seed"], extra={**pt_extra, **pt_status, "classes": NAMES16,
                                          "pointcept_config": PT["pointcept_config"],
                                          "readme_note": "Released model weights are temporarily invalid"})
    PT_SUM = load_summary(pt_summary_file)
print("PTv3 status:", PT_SUM["extra"]["status"])
for err in PT_SUM["extra"].get("errors", []):
    print(" -", err[:800])
print(json.dumps(PT_SUM["metrics"], indent=1))

# %% [markdown]
# ## 5. 6b-3. Sonata와 Concerto 특징 PCA (도메인 불일치 시각화)
# 학습 없이 사전학습 인코더의 점별 특징을 뽑아 PCA 3성분을 RGB로 칠합니다. 비슷한 색 = 비슷한 특징입니다.
# 색 자체는 모델마다 기저가 달라 서로 비교할 수 없고, **구조가 얼마나 의미 단위로 나뉘는지**만 봅니다.
#
# | | Sonata | Concerto outdoor |
# |---|---|---|
# | 가중치 | HF `facebook/sonata`, `sonata` (실내 전용) | HF `Pointcept/Concerto`, `concerto_large_outdoor` |
# | 변환 | `sonata.transform.default()`: CenterShift(z 포함), GridSample 2cm | Concerto `demo/0_pca_outdoor.py`: 좌표 x0.2, GridSample 0.01(=5cm), CenterShift(z 제외) |
# | 색 입력 | 0 (LiDAR에 RGB 없음) | 0 (선형 탐침 데이터셋과 같음, demo의 `--wo_color`) |
# | 법선 입력 | open3d 추정, 센서 쪽으로 뒤집음(Pointcept `NuScenesImagePointDataset.get_normals` 방식) | 0 (선형 탐침 데이터셋과 같음, demo의 `--wo_normal`) |
# | PCA | demo: 6성분, 0.6/0.4 가중, 밝기 1.2 | demo: 12성분, 0.2/0.2/0.1/0.5 가중, 밝기 1.0 |
#
# Sonata는 RGB-D/메시에서 얻은 색·법선으로 학습했으므로, 색 0과 LiDAR 추정 법선은 그 자체로 입력 분포가 다릅니다.
# 같은 키프레임의 실제 LiDAR와 pseudo-LiDAR(있으면 첫 번째 원천) 둘 다에 적용합니다.

# %%
SN = M["sonata"]
SN_DIR = result_dir(cfg, NOTEBOOK, "sonata")
VIS_LIDAR_SD = VIS_INFO["lidar_token"]
PCA_CLOUDS = {"lidar": load_lidar(nusc, VIS_LIDAR_SD)[:, :3]}
if PSEUDO_SRCS:
    _pf = PSEUDO_ROOT / PSEUDO_SRCS[0] / "pseudo" / VIS_INFO["lidar_path"]
    if _pf.exists():
        PCA_CLOUDS[f"pseudo:{PSEUDO_SRCS[0]}"] = load_nuscenes_bin(_pf)[:, :3]
PCA_MODELS = {
    "sonata": dict(package="sonata", name=SN["name"], repo=SN["hf_repo"], transform=sonata.transform.default,
                   normal="estimate", pca=bp.PCA_SONATA, out=SN_DIR),
    "concerto_outdoor": dict(package="concerto", name=CC["name"], repo=CC["hf_repo"],
                             transform=lambda: ConcertoCompose(bp.CONCERTO_OUTDOOR_TRANSFORM), normal="zeros",
                             pca=bp.PCA_CONCERTO_OUTDOOR, out=CC_DIR),
}
PCA = {}
for mname, spec in PCA_MODELS.items():
    pca_file = spec["out"] / "pca_colors.npz"
    if is_cached(pca_file, FORCE_RERUN):
        z = np.load(pca_file)
        PCA[mname] = {c: z[c] for c in PCA_CLOUDS if c in z.files}
        continue
    set_seed(cfg["seed"])  # PCA (torch.pca_lowrank) is randomized
    with free_cuda():
        model, custom = bp.load_ptv3_package(spec["package"], spec["name"], spec["repo"], P["enable_flash"])
        cols, timing, n_grid = {}, {}, {}
        with Measure() as m_pca:
            for cname, xyz in PCA_CLOUDS.items():
                feat, inv, sec = bp.ptv3_features(model, bp.point_input(xyz, spec["normal"]), spec["transform"](),
                                                  DEVICE)
                cols[cname] = (bp.pca_colors(feat, **spec["pca"])[inv].cpu().numpy() * 255).astype(np.uint8)
                timing[cname], n_grid[cname] = sec, int(feat.shape[0])
                feat_dim = int(feat.shape[1])
                del feat, inv
        del model
    np.savez_compressed(pca_file, **cols)
    PCA[mname] = cols
    save_summary(spec["out"], part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model=mname,
                 variant=f"pca_only/{VIS_SAMPLE}", filename="summary.json" if mname == "sonata" else "summary_pca.json",
                 metrics={"feat_s_per_scan_lidar": timing["lidar"], "n_grid_points_lidar": n_grid["lidar"],
                          "feat_dim": feat_dim},
                 runtime_s=m_pca.seconds, peak_vram_mb=m_pca.peak_vram_mb, checkpoint=f"{spec['repo']}/{spec['name']}.pth",
                 packages=ENV["packages"], seed=cfg["seed"],
                 extra={"sample_token": VIS_SAMPLE, "clouds": list(PCA_CLOUDS), "timing_s": timing, "n_grid": n_grid,
                        "inputs": {"color": "zeros", "normal": spec["normal"]}, "custom_config": custom,
                        "pca": spec["pca"], "note": "qualitative only; sonata is indoor-pretrained"})
print({m: list(v) for m, v in PCA.items()})

# %% [markdown]
# ## 6. 6c. pseudo-LiDAR에 같은 Concerto 탐침 적용
# - 입력: 04가 저장한 `results/outdoor/04_pseudo_lidar/<model>/pseudo/samples/LIDAR_TOP/*.bin`(각 샘플의 LIDAR_TOP 좌표계,
#   반사강도·ring은 0). 6b와 **같은 가중치(`model_last.pth`)와 같은 테스트 파이프라인(TTA 10회)**을 쓰도록, `raw/`가 pseudo 폴더를
#   가리키고 라벨 경로를 뺀 info를 만들어 `tools/test.py`로 돌립니다(`bench3d.models.point.write_infos_without_labels`).
# - Concerto 선형 탐침의 입력 특징은 좌표·색(0)·법선(0)뿐이라 반사강도가 0인 것은 이 모델에 영향을 주지 않습니다.
#   (반사강도를 쓰는 PTv3 v1.5.1 config에는 pseudo-LiDAR를 넣지 않습니다.)
#
# **대리(proxy) 지표 — 진짜 정답이 아닙니다.** pseudo 점에는 라벨이 없으므로, 같은 샘플의 실제 LiDAR 점 중
# **`PROXY_MAX_DIST`(0.5m) 안에서 가장 가까운 점의 라벨**을 옮겨 붙이고, 그런 점이 없으면 무시합니다. 그 위에서 mIoU를 계산합니다.
# 편향:
# 1. **선택 편향(낙관적)**: 깊이가 크게 틀린 점은 근처에 LiDAR 점이 없어 무시됩니다. 틀린 점이 벌점을 받지 않으므로
#    proxy mIoU는 "깊이가 대략 맞은 영역에서의 분할 품질"이지 pseudo-LiDAR 전체의 품질이 아닙니다. 그래서 `label_coverage`(라벨을
#    받은 pseudo 점 비율)를 함께 봐야 합니다.
# 2. **거리 편향**: LiDAR는 멀수록 성겨서 원거리 pseudo 점은 라벨을 덜 받습니다. 평가가 근거리 위주가 됩니다.
# 3. **경계 오류**: 0.5m 안의 최근접 라벨은 물체 경계에서 틀릴 수 있습니다(예: 차 옆 도로 점).
# 4. **영역 차이**: pseudo-LiDAR는 카메라 시야와 04의 최대 깊이 안에만 있습니다. 비교 기준으로 같은 영역(실제 LiDAR 점 중
#    0.5m 안에 pseudo 점이 있는 점)에서의 실제 LiDAR mIoU(`lidar_same_region_miou`)를 함께 보고합니다.

# %%
PSEUDO_RES = {}
for src in PSEUDO_SRCS:
    ps_dir = CC_DIR / f"pseudo_{src}"
    ps_raw = PSEUDO_ROOT / src / "pseudo"
    ps_summary_file = CC_DIR / f"summary_pseudo_{src}.json"
    if is_cached(ps_summary_file, FORCE_RERUN):
        PSEUDO_RES[src] = load_summary(ps_summary_file)
        continue
    ps_data = ps_dir / "pointcept_data"
    ps_info = bp.write_infos_without_labels(VAL_PKL, ps_raw, ps_data, sweeps=SWEEPS)
    print(src, ps_info)
    if ps_info["n_kept"] == 0:
        print(f"  no pseudo clouds matched the eval4 keyframes under {ps_raw}; skipped")
        continue
    ps_infos = pickle.loads((ps_data / "info" / VAL_PKL.name).read_bytes())
    ps_exp = ps_dir / "exp"
    if ps_exp.exists():
        shutil.rmtree(ps_exp)  # SemSegTester re-uses *_pred.npy
    ps_opts = {"save_path": ps_exp, "weight": CC_EXP / "model" / "model_last.pth", "data.test.data_root": ps_data,
               "model.backbone.enable_flash": USE_FLASH, "enable_wandb": False, "seed": cfg["seed"]}
    ps_run = bp.run_pointcept("test", LIN_CFG_PATH, ps_opts, pointcept_root=PC_ROOT,
                              log_path=ps_dir / "test_subprocess.log")
    if ps_run["returncode"] != 0:
        print(bp.tail(ps_run["log"]))
        raise RuntimeError(f"tools/test.py on pseudo-LiDAR {src} failed")
    ps_log = bp.parse_semseg_log(Path(ps_run["log"]).read_text(errors="replace"), NAMES16)

    conf_proxy = np.zeros((NUM16, NUM16), np.int64)
    conf_region = np.zeros((NUM16, NUM16), np.int64)
    n_pseudo = n_labeled = 0
    for info in ps_infos:
        pred_ps = np.load(ps_exp / "result" / f"{info['lidar_token']}_pred.npy").reshape(-1)
        xyz_ps = load_nuscenes_bin(ps_raw / info["lidar_path"])[:, :3]
        xyz_li = load_lidar(nusc, info["lidar_token"])[:, :3]
        gt_li = bp.load_lidarseg16(nusc, info["lidar_token"], LUT16)
        lab_ps, _ = bp.transfer_labels_nn(xyz_li, gt_li, xyz_ps, PROXY_MAX_DIST, IGN)
        conf_proxy += confusion_matrix(pred_ps, lab_ps, NUM16, ignore_index=IGN)
        n_pseudo += len(xyz_ps)
        n_labeled += int((lab_ps != IGN).sum())
        # reference: real-LiDAR prediction restricted to the region the pseudo cloud covers
        pred_li = np.load(CC_EXP / "result" / f"{info['lidar_token']}_pred.npy").reshape(-1)
        _, d_li = bp.transfer_labels_nn(xyz_ps, np.zeros(len(xyz_ps), np.int64), xyz_li, PROXY_MAX_DIST, IGN)
        covered = np.isfinite(d_li)
        conf_region += confusion_matrix(pred_li[covered], gt_li[covered], NUM16, ignore_index=IGN)
    res_proxy = iou_from_confusion(conf_proxy, NAMES16)
    res_region = iou_from_confusion(conf_region, NAMES16)
    save_summary(CC_DIR, part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model="concerto_outdoor",
                 variant=f"pseudo_lidar:{src}/proxy_nn{PROXY_MAX_DIST}m", filename=ps_summary_file.name,
                 metrics={"proxy_miou": res_proxy["miou"], "proxy_overall_acc": res_proxy["overall_acc"],
                          "label_coverage": n_labeled / max(n_pseudo, 1),
                          "lidar_same_region_miou": res_region["miou"],
                          "infer_s_per_scan_tta10": ps_log["scan_time_avg_s"] if ps_log else None,
                          "n_scans": len(ps_infos), "pseudo_points_per_scan": n_pseudo / max(len(ps_infos), 1)},
                 runtime_s=ps_run["seconds"], peak_vram_mb=None,
                 checkpoint=f"{CC['hf_repo']}/{CC['name']}.pth + linear head trained on probe6 (real LiDAR)",
                 packages=ENV["packages"], seed=cfg["seed"],
                 extra={"per_class_proxy_iou": res_proxy["per_class_iou"],
                        "per_class_lidar_same_region_iou": res_region["per_class_iou"],
                        "confusion_proxy": conf_proxy.tolist(), "pseudo_source": src,
                        "n_missing_pseudo_scans": ps_info["n_missing"],
                        "proxy_definition": f"GT of nearest real LiDAR point within {PROXY_MAX_DIST} m, else ignored",
                        "proxy_biases": ["optimistic selection: far-off pseudo points are ignored, not penalised",
                                         "distance: sparse far LiDAR -> near-range dominated",
                                         "boundary label noise from nearest-neighbour transfer",
                                         "region: camera FOV and 04 max depth only"]})
    PSEUDO_RES[src] = load_summary(ps_summary_file)
for src, s in PSEUDO_RES.items():
    print(src, json.dumps(s["metrics"], indent=1))

# %% [markdown]
# ## 7. 결과 요약
# 이 노트북의 `summary*.json`을 모아 한 표로 봅니다(지표 이름은 모델마다 다를 수 있음).

# %%
df = collect_summaries(cfg.path(cfg["results_root"]) / NOTEBOOK)
if len(df):
    print(df.pivot_table(index=["model", "variant"], columns="metric", values="value", aggfunc="first").T.to_string())

# %% [markdown]
# ## 8. 시각화
# ### 8-1. PointNet: 혼동 행렬(행 정규화), 클래스별 정확도, 박스 안 점 수별 정확도

# %%
conf10 = np.asarray(PN_SUM["extra"]["confusion"], float)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
with np.errstate(invalid="ignore", divide="ignore"):
    cn = conf10 / conf10.sum(1, keepdims=True)
im = axes[0].imshow(cn, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks(range(10), CLASSES10, rotation=60, ha="right", fontsize=8)
axes[0].set_yticks(range(10), CLASSES10, fontsize=8)
axes[0].set_xlabel("predicted")
axes[0].set_ylabel("ground truth")
axes[0].set_title("PointNet confusion (row-normalised)")
fig.colorbar(im, ax=axes[0], fraction=0.046)
pc_acc = PN_SUM["extra"]["per_class_acc"]
cnt = PN_SUM["extra"]["test_counts"]
axes[1].bar(range(10), [np.nan if pc_acc[c] is None else pc_acc[c] for c in CLASSES10], color="#0072B2")
axes[1].set_xticks(range(10), [f"{c}\n(n={n})" for c, n in zip(CLASSES10, cnt)], rotation=60, ha="right", fontsize=7)
axes[1].set_ylim(0, 1)
axes[1].set_title(f"per-class accuracy (balanced {PN_SUM['metrics']['balanced_accuracy']:.2f}, "
                  f"overall {PN_SUM['metrics']['accuracy']:.2f})")
abp = PN_SUM["extra"]["acc_by_points_in_box"]
axes[2].bar(list(abp), [np.nan if v is None else v for v in abp.values()], color="#E69F00")
axes[2].set_ylim(0, 1)
axes[2].set_xlabel("LiDAR points in GT box")
axes[2].set_title("accuracy vs. points in box")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8-2. 같은 키프레임의 BEV: GT vs Concerto 예측 (실제 LiDAR / pseudo-LiDAR), PTv3(성공 시)
# 회색은 무시 라벨(-1)입니다. 좌표는 LIDAR_TOP 좌표계(x 앞, y 왼쪽), 범위 ±50m.

# %%
def seg_colors(labels: np.ndarray) -> np.ndarray:
    c = np.full((len(labels), 3), 0.8, np.float32)
    ok = labels >= 0
    c[ok] = PAL16[labels[ok]]
    return c


def bev_panel(ax, xyz: np.ndarray, colors: np.ndarray, title: str, extent: float = BEV_EXTENT):
    m = (np.abs(xyz[:, 0]) < extent) & (np.abs(xyz[:, 1]) < extent)
    ax.scatter(xyz[m, 0], xyz[m, 1], c=colors[m], s=0.3, linewidths=0)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


vis_tok = VIS_INFO["lidar_token"]
xyz_vis = load_lidar(nusc, vis_tok)[:, :3]
gt_vis = bp.load_lidarseg16(nusc, vis_tok, LUT16)
panels = [("GT (LiDAR)", xyz_vis, seg_colors(gt_vis)),
          ("Concerto probe (LiDAR)", xyz_vis, seg_colors(np.load(CC_EXP / "result" / f"{vis_tok}_pred.npy")))]
if PT_SUM["extra"]["status"] == "ok":
    panels.append(("PTv3 v1.5.1 (LiDAR)", xyz_vis, seg_colors(np.load(PT_EXP / "result" / f"{vis_tok}_pred.npy"))))
for src in PSEUDO_RES:
    f_pred = CC_DIR / f"pseudo_{src}" / "exp" / "result" / f"{vis_tok}_pred.npy"
    f_pts = PSEUDO_ROOT / src / "pseudo" / VIS_INFO["lidar_path"]
    if f_pred.exists() and f_pts.exists():
        xyz_ps = load_nuscenes_bin(f_pts)[:, :3]
        lab_ps, _ = bp.transfer_labels_nn(xyz_vis, gt_vis, xyz_ps, PROXY_MAX_DIST, IGN)
        panels.append((f"proxy GT ({src})", xyz_ps, seg_colors(lab_ps)))
        panels.append((f"Concerto probe (pseudo {src})", xyz_ps, seg_colors(np.load(f_pred))))
ncol = min(4, len(panels))
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.5 * nrow), squeeze=False)
for ax, (title, xyz, col) in zip(axes.flat, panels):
    bev_panel(ax, xyz, col, title)
for ax in axes.flat[len(panels):]:
    ax.set_axis_off()
fig.legend(handles=[Patch(color=PAL16[i], label=n) for i, n in enumerate(NAMES16)] + [Patch(color="0.8", label="ignore")],
           loc="lower center", ncol=9, fontsize=7)
fig.suptitle(f"sample {VIS_SAMPLE}", fontsize=10)
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()

# %% [markdown]
# ### 8-3. 3D 보기 (plotly): Concerto 예측, 실제 LiDAR

# %%
_pred_vis = np.load(CC_EXP / "result" / f"{vis_tok}_pred.npy")
_sel = np.random.default_rng(cfg["seed"]).choice(len(xyz_vis), min(60000, len(xyz_vis)), replace=False)
_rgb = (seg_colors(_pred_vis[_sel]) * 255).astype(int)
fig3d = go.Figure(go.Scatter3d(x=xyz_vis[_sel, 0], y=xyz_vis[_sel, 1], z=xyz_vis[_sel, 2], mode="markers",
                               marker={"size": 1.2, "color": [f"rgb({r},{g},{b})" for r, g, b in _rgb]}))
fig3d.update_layout(scene={"aspectmode": "data"}, height=600, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                    title="Concerto linear probe prediction (LiDAR)")
fig3d.show()

# %% [markdown]
# ### 8-4. 특징 PCA: Sonata(실내 사전학습) vs Concerto(실외 사전학습)

# %%
clouds = [c for c in PCA_CLOUDS if all(c in PCA[m] for m in PCA_MODELS)]
fig, axes = plt.subplots(len(clouds), len(PCA_MODELS), figsize=(6 * len(PCA_MODELS), 6 * len(clouds)), squeeze=False)
for r, cname in enumerate(clouds):
    for c, mname in enumerate(PCA_MODELS):
        bev_panel(axes[r, c], PCA_CLOUDS[cname], PCA[mname][cname] / 255.0, f"{mname} PCA | {cname}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8-5. 클래스별 IoU 비교

# %%
series = {"Concerto probe (LiDAR)": CC_SUM["extra"]["per_class_iou"]}
if PT_SUM["extra"]["status"] == "ok":
    series["PTv3 v1.5.1 (LiDAR)"] = PT_SUM["extra"]["per_class_iou"]
for src, s in PSEUDO_RES.items():
    series[f"Concerto, LiDAR same region ({src})"] = s["extra"]["per_class_lidar_same_region_iou"]
    series[f"Concerto, pseudo PROXY ({src})"] = s["extra"]["per_class_proxy_iou"]
fig, ax = plt.subplots(figsize=(16, 4.5))
w = 0.8 / len(series)
for k, (name, d) in enumerate(series.items()):
    vals = [np.nan if d.get(n) is None else d[n] for n in NAMES16]
    ax.bar(np.arange(NUM16) + k * w - 0.4 + w / 2, vals, w, label=name, color=bp.class_palette(20)[2 * k % 20])
ax.set_xticks(range(NUM16), NAMES16, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("IoU")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.set_title("per-class IoU on eval4 (missing bar = class absent in GT and prediction)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. 해석
# ### 6a PointNet
# > (실행 후 작성) 균형 정확도와 전체 정확도의 차이, 가장 많이 헷갈린 클래스 쌍(예: truck/construction_vehicle/trailer,
# > bicycle/motorcycle), 점이 적은 구간에서의 정확도 하락.
#
# ### 6b Concerto 선형 탐침 / PTv3 / Sonata
# > (실행 후 작성) Concerto mIoU(`miou`와 `miou_all16`의 차이 원인 포함), 잘 되는 클래스(도로, 차, 식생)와 안 되는 클래스(소형·희소),
# > mini 6장면 학습과 작은 batch의 한계. PTv3 시도 결과(오류 내용). Sonata PCA가 지면/물체를 나누는지, Concerto와 비교.
#
# ### 6c pseudo-LiDAR
# > (실행 후 작성) proxy mIoU와 label_coverage를 함께 해석. 같은 영역의 실제 LiDAR mIoU와의 차이, 먼저 무너지는 클래스,
# > 깊이 오차(물체 가장자리 번짐, 원거리 왜곡)가 분할에 주는 영향. proxy 지표의 낙관적 편향을 감안.
