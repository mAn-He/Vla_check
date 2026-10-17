# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 03. 다시점 기하: DUSt3R · MASt3R · VGGT · Depth Anything 3 (nuScenes-mini)
#
# ## 목적
# 카메라 이미지 여러 장만으로 **카메라 자세와 3D 점군을 함께 복원**하는 다시점 모델 4종을 nuScenes-mini 평가 장면
# (`eval4`, 모두 공식 val 분할)에서 비교한다. 입력 설정은 두 가지다.
# - **(A) 전방 연속 키프레임** `front`: 장면마다 CAM_FRONT 키프레임 `n_frames`장(2 Hz). 전진 주행이라 시차가 작고 카메라 중심이 거의 일직선이다.
# - **(B) 서라운드 리그** `surround`: 장면 가운데 키프레임 한 개의 6개 카메라. 시야 겹침이 적고 카메라 중심끼리 1~2 m밖에 떨어져 있지 않다.
#
# GT 자세는 nuScenes의 `ego_pose × calibrated_sensor`(카메라→전역), GT 기하는 같은 샘플들의 LIDAR_TOP 키프레임 점군이다.
#
# ## 비교 모델과 원리
# **DUSt3R** (`naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`)
# - 이미지 두 장을 ViT 인코더와 교차 어텐션 디코더에 넣어, 두 장 모두의 픽셀별 3D 점(pointmap)을 첫 이미지 좌표계에서 직접 회귀한다. 카메라 파라미터를 입력으로 받지 않는다.
# - N장이면 가능한 모든 쌍(complete graph, 대칭)을 추론한 뒤 쌍별 pointmap을 하나의 좌표계로 맞추는 **전역 정렬**을 경사하강으로 푼다(`PointCloudOptimizer`, MST 초기화).
# - 출력: 카메라→월드 자세, 초점거리, 깊이맵, 픽셀별 신뢰도. 스케일은 임의(상대)다.
#
# **MASt3R** (`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`)
# - DUSt3R 구조에 지역 특징(descriptor) 헤드를 더해 픽셀 매칭을 하고, metric 스케일 pointmap으로 학습했다.
# - 다시점은 **sparse global alignment**: 매칭점의 3D 정합(coarse, `lr1`/`niter1`) → 2D 재투영 오차 정제(fine, `lr2`/`niter2`, `refine+depth`면 깊이도 최적화) 순서로 푼다.
# - 출력이 이론상 metric이므로 Sim(3) 스케일 `s`가 1에 가까운지도 본다.
#
# **VGGT** (`facebook/VGGT-1B`)
# - 1B 파라미터 트랜스포머가 N장을 한꺼번에 받아(프레임 내/프레임 간 어텐션 교대) 카메라, 깊이맵, pointmap, 트랙을 **한 번의 순전파**로 예측한다. 최적화 단계가 없다.
# - 외부 파라미터는 첫 프레임 기준 world→camera(OpenCV)이고, 스케일은 정규화된 상대 스케일이다.
# - 3D 점은 README 권고("depth map + camera로 만든 점이 point map branch보다 보통 더 정확")에 따라 depth 헤드 + 카메라 역투영으로 만든다.
#
# **Depth Anything 3 any-view** (`depth-anything/DA3-LARGE-1.1`)
# - 하나의 ViT(DINOv2 계열)가 임의 개수 이미지에서 깊이와 광선(ray)을 예측하는 depth-ray 표현을 쓴다. 카메라 디코더가 자세와 내부 파라미터를 낸다.
# - 여기서는 자세·내부 파라미터를 넣지 않고(pose-free) 호출한다. 기준 뷰는 내부에서 자동 선택(`saddle_balanced`)되고 출력 순서는 입력 순서로 복원된다.
# - `-1.1`은 학습 버그를 고친 재학습판이며, README는 주행 장면에서 더 좋다고 적고 있다.
#
# ## 원 논문
# - DUSt3R: Geometric 3D Vision Made Easy — https://arxiv.org/abs/2312.14132
# - MASt3R: Grounding Image Matching in 3D with MASt3R — https://arxiv.org/abs/2406.09756
# - VGGT: Visual Geometry Grounded Transformer — https://arxiv.org/abs/2503.11651
# - Depth Anything 3 — https://arxiv.org/abs/2511.10647 (Depth-Anything-3 README의 arXiv 배지 기준)
#
# ## 커널
# `3dv` (Python 3.10, torch 2.1.2 + CUDA 12.1). 모델 코드는 `envs/third_party.lock`의 고정 커밋을 쓴다
# (mast3r `f5209af` + dust3r 서브모듈, vggt `a288dd0`, Depth-Anything-3 `3d835ec`). 래퍼: `bench3d.models.multiview`.
#
# ## 예상 VRAM·시간 (추정)
# 아래는 **추정치**이며 이 노트북이 모델·창(window)별 실측값(`Measure`: 벽시계 시간, `max_memory_allocated`)을 저장한다.
#
# | 모델 | VRAM (추정) | 시간 (추정) | 근거 |
# |---|---|---|---|
# | VGGT | mid 10장: 3.2~5.6 GB 사이, high 20장: 약 5.6 GB | 미측정 | 과거 README 표(커밋 44b3afb, **H100, aggregator만**): 8장 3.23 GB, 20장 5.58 GB, 50장 11.41 GB. 현재 README에서는 삭제됨. 헤드·해상도 차이로 실제와 다를 수 있음 |
# | DUSt3R | 미상 (실측) | 미상 (실측) | 쌍 N(N−1)개 추론(mid 10장 = 90쌍, high 20장 = 380쌍) + 전역 정렬 `dust3r_niter`회. 쌍 수에 비례해 커짐 |
# | MASt3R | 미상 (실측) | 미상 (실측) | 쌍 추론 + sparse GA `niter1 + niter2`회. 창마다 쌍 캐시를 디스크에 씀(창 처리 후 삭제) |
# | DA3-LARGE-1.1 | 미상 (실측) | 미상 (실측) | 순전파 1회(process_res 504) |
#
# 프로파일: `low` 6장·서라운드 없음·DA3 제외 / `mid` 10장·창 1개·서라운드 포함 / `high` 20장·창 2개.
#
# ## 이 노트북이 답하려는 질문
# 1. 전진 주행(낮은 시차)에서 각 모델의 자세 정확도(ATE, RPE, RRA/RTA, AUC@5/15/30)는 어느 정도인가?
# 2. 겹침이 적은 서라운드 6캠에서도 카메라 간 상대 자세를 맞추는가?
# 3. 복원 점군이 LiDAR와 얼마나 맞는가(정확도·완전성·Chamfer·F-score)? LiDAR가 못 보는 영역은 어떻게 읽어야 하는가?
# 4. 최적화 기반(DUSt3R/MASt3R)과 순전파(VGGT/DA3)의 속도–정확도 트레이드오프는?
# 5. metric으로 학습한 MASt3R의 정렬 스케일 `s`는 1에 가까운가?
# 6. 주행 데이터에서 순전파 모델은 어디서 실패하는가: 낮은 시차의 전진 운동, 겹침이 적은 서라운드 리그, 동적 물체, 하늘·원거리?

# %% [markdown]
# ## 1. 설정 로드
# 모든 하이퍼파라미터는 `outdoor/config.yaml`의 `models.multiview`(공식 데모 기본값)와 `profiles.<profile>.multiview`에서 읽는다.
# `KEEP_MAST3R_CACHE=False`면 MASt3R의 쌍별 캐시(창마다 수백 MB~GB 수준일 수 있음)를 창 처리 직후 지운다.

# %%
FORCE_RERUN = False            # True: ignore cached results
KEEP_MAST3R_CACHE = False      # True: keep MASt3R forward/canonical caches under the result dir
NOTEBOOK = "03_multiview_geometry"

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bench3d.config import load_config
from bench3d.io import is_cached, load_ply, result_dir, save_ply, save_summary
from bench3d.models import multiview as mv
from bench3d.runtime import Measure, env_report, free_cuda, set_seed

cfg = load_config("outdoor")                 # profile: config.yaml or $THREEDCHECK_PROFILE
P = cfg.p["multiview"]                       # this notebook's profile block
MCFG = cfg["models"]["multiview"]            # official model ids / demo defaults
FORCE_RERUN = FORCE_RERUN or cfg.get("force_rerun", False)
set_seed(cfg["seed"])
ENV = env_report(["torch", "numpy", "nuscenes-devkit", "vggt", "depth-anything-3", "roma", "huggingface-hub",
                  "xformers", "scipy", "plotly"])


def git_commit(name: str) -> str:
    """Commit of a pinned third-party checkout (DUSt3R/MASt3R are not pip packages)."""
    try:
        r = subprocess.run(["git", "-C", str(cfg.third_party(name)), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


THIRD_PARTY = {n: git_commit(n) for n in ["mast3r", "mast3r/dust3r", "vggt", "Depth-Anything-3"]}
PACKAGES = {**ENV["packages"], **{f"git:{k}": v for k, v in THIRD_PARTY.items()}}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: no CUDA device; DUSt3R/MASt3R/VGGT/DA3 on CPU will be extremely slow")

MODELS = list(P["models"])
EXPS = ["front"] + (["surround"] if P.get("surround", False) else [])
for m in MODELS:
    assert m in mv.MODEL_IDS and m in MCFG, f"unknown multiview model {m}"
print(json.dumps(ENV, indent=1))
print("third_party:", THIRD_PARTY)
print("profile:", cfg["profile"], P)
print("models:", MODELS, "| experiments:", EXPS, "| device:", DEVICE)

# %% [markdown]
# ## 2. 데이터 로드
# - 평가 장면과 키프레임 목록은 `01_data_exploration`이 저장한 `eval_samples.json`을 읽는다(모든 노트북이 같은 프레임 사용).
# - (A) `front`: `select_windows(keyframes, n_frames, windows_per_scene)`로 장면마다 연속 CAM_FRONT 키프레임 창을 고른다.
# - (B) `surround`: `middle(keyframes)` 샘플의 6개 카메라(`bench3d.nusc.CAMERAS` 순서). 같은 샘플이라도 카메라마다 촬영 시각이 조금씩 달라
#   각 카메라의 GT 자세는 **자기 sample_data의 ego pose**로 계산한다(`bench3d.nusc.camera_info`).
# - GT 카메라 중심의 배치도 함께 출력한다. `sv2/sv1`(중심 분포의 둘째/첫째 특이값 비)가 작으면 중심이 거의 일직선이다.

# %%
from bench3d import nusc as bn
from bench3d.align import centers_degenerate
from bench3d.sampling import middle, select_windows
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version=cfg["dataset"]["version"], dataroot=str(cfg.path(cfg["dataset"]["data_root"])), verbose=False)
EVAL_FILE = cfg.path(cfg["results_root"]) / "01_data_exploration" / "eval_samples.json"
assert EVAL_FILE.exists(), f"{EVAL_FILE} missing: run 01_data_exploration first"
ES = json.loads(EVAL_FILE.read_text())
assert ES["version"] == cfg["dataset"]["version"], ES["version"]
assert set(ES["eval4"]) == set(cfg["splits"]["eval4"]), "eval4 in eval_samples.json differs from config"


def make_window(win_id: str, scene: str, exp: str, sample_tokens: list[str], channels: list[str]) -> dict:
    cams, lidar = [], []
    for st in sample_tokens:
        s = nusc.get("sample", st)
        cams += [bn.camera_info(nusc, s["data"][ch]) for ch in channels]
        lidar.append(s["data"]["LIDAR_TOP"])
    return {"id": win_id, "scene": scene, "exp": exp, "samples": list(sample_tokens), "cams": cams,
            "lidar_sd": lidar, "image_paths": [Path(nusc.dataroot) / c["filename"] for c in cams]}


def center_stats(cams: list[dict]) -> dict:
    c = np.stack([x["c2w"][:3, 3] for x in cams])
    sv = np.linalg.svd(c - c.mean(0), compute_uv=False)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    return {"max_baseline_m": float(d.max()), "sv2_over_sv1": float(sv[1] / sv[0]) if sv[0] > 1e-9 else 0.0,
            "degenerate": centers_degenerate(c)}


WINDOWS = {e: [] for e in EXPS}
for scene in cfg["splits"]["eval4"]:
    kf = ES["keyframes"][scene]
    for wi, win in enumerate(select_windows(kf, P["n_frames"], P["windows_per_scene"])):
        WINDOWS["front"].append(make_window(f"{scene}_w{wi}", scene, "front", win, ["CAM_FRONT"]))
    if "surround" in EXPS:
        WINDOWS["surround"].append(make_window(f"{scene}_mid", scene, "surround", [middle(kf)], list(bn.CAMERAS)))

rows = [{"exp": e, "window": w["id"], "n_views": len(w["cams"]), **center_stats(w["cams"])}
        for e in EXPS for w in WINDOWS[e]]
GEOM = pd.DataFrame(rows)
print(GEOM.to_string(index=False))
if "surround" in EXPS:
    print("NOTE surround: GT camera centers span only ~1-2 m -> Sim(3) scale/translation (and ATE, Chamfer) are "
          "ill-conditioned; pairwise RRA/RTA/AUC are the primary surround metrics.")

# %% [markdown]
# ### GT 점군 (LiDAR)
# - 창에 속한 샘플들의 LIDAR_TOP **키프레임** 점을 각자의 `sensor_to_global`로 전역 좌표에 옮겨 합친다(`front`는 `n_frames`개 스윕의 합집합, `surround`는 1개).
# - 자차 지붕에 맞은 점은 devkit의 `LidarPointCloud.remove_close`와 같은 규칙(센서 좌표 |x|<r 이고 |y|<r)으로 지운다.
#   `r`은 `P["gt_ego_radius"]`(없으면 1.0 m).
# - 합집합이므로 움직이는 물체는 궤적처럼 번진다(동적 물체 문제). 카메라 점군은 이런 번짐이 없으므로 그 근처에서 오차가 커질 수 있다.

# %%
GT_EGO_RADIUS = float(P.get("gt_ego_radius", 1.0))


def gt_lidar_world(win: dict) -> np.ndarray:
    pts = []
    for lsd in win["lidar_sd"]:
        p = bn.load_lidar(nusc, lsd)
        far = ~((np.abs(p[:, 0]) < GT_EGO_RADIUS) & (np.abs(p[:, 1]) < GT_EGO_RADIUS))
        pts.append(bn.transform_points(bn.sensor_to_global(nusc, lsd), p[far]))
    return np.concatenate(pts)


print({e: [len(w["lidar_sd"]) for w in WINDOWS[e]] for e in EXPS}, "LiDAR sweeps per window")

# %% [markdown]
# ## 3. 모델별 실행
# 모델 하나씩: 로드 → 모든 창(front, surround) 추론 → 창별 저장 → 메모리 해제(`free_cuda`). 결과가 있으면 건너뛴다(`FORCE_RERUN`).
#
# 공통 출력(`bench3d.models.multiview`): `c2w`(N,4,4, 모델 자체 좌표계), `points`(N,H,W,3), `conf`, `mask`(모델 데모의 기본 신뢰도 필터),
# `intrinsics`(처리 해상도 기준), `image_hw`, `images`. 창마다 `<model>/<exp>/<window>_pred.npz`와 `_run.json`(시간·VRAM·설정)을 저장한다.
#
# | 모델 | 호출 (고정 커밋 소스에서 확인) | 반복 횟수 |
# |---|---|---|
# | DUSt3R | `load_images(size=512)` → `make_pairs(complete, prefilter=None, symmetrize=True)` → `inference(batch_size=1)` → `global_aligner(PointCloudOptimizer)` → `compute_global_alignment(init=mst, schedule, lr)` | `dust3r_niter` |
# | MASt3R | `load_images(size=512)` → `make_pairs` → `sparse_global_alignment(lr1, niter1, lr2, niter2, opt_depth, shared_intrinsics, matching_conf_thr)` → `get_dense_pts3d(clean_depth=True)` | `mast3r_niter1/2` |
# | VGGT | `load_and_preprocess_images(mode=crop)` → bf16(CC≥8)/fp16 autocast 순전파 → `pose_encoding_to_extri_intri`(w2c 3×4) → 역행렬 → `unproject_depth_map_to_point_map` | — |
# | DA3 | `DepthAnything3.from_pretrained` → `inference(paths, process_res=504)` (자세 입력 없음) → w2c(3×4 또는 4×4) 역행렬 → 깊이 역투영 | — |
#
# - 시간은 이미지 로드·전처리·추론·(최적화)를 포함하고 모델 로드는 제외한다(로드 시간은 `_run.json`의 `load_s`).
#   첫 창에는 CUDA 워밍업이 섞이므로 요약에는 평균과 중앙값을 함께 남긴다.
# - 피크 VRAM은 창 단위 `torch.cuda.max_memory_allocated`이며 모델 가중치를 포함한다.
# - OOM이 나면 그 창을 실패로 기록하고 다음 창으로 넘어간다(평가에서 제외, 요약 `extra.failed`에 표시).

# %%
def pred_files(model_id: str, win: dict):
    d = result_dir(cfg, NOTEBOOK, model_id) / win["exp"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{win['id']}_pred.npz", d / f"{win['id']}_run.json"


def run_kwargs(model_id: str, win: dict) -> dict:
    if model_id == "dust3r":
        return {"niter": P["dust3r_niter"]}
    if model_id == "mast3r":
        cache = result_dir(cfg, NOTEBOOK, model_id) / win["exp"] / f"{win['id']}_cache"  # unique per image set
        return {"niter1": P["mast3r_niter1"], "niter2": P["mast3r_niter2"], "cache_dir": cache,
                "keep_cache": KEEP_MAST3R_CACHE}
    return {}


ALL_WINDOWS = [w for e in EXPS for w in WINDOWS[e]]
FRESH = set()  # (model, exp, window) predicted in this session -> re-evaluate
for model_id in MODELS:
    todo = [w for w in ALL_WINDOWS if not is_cached(pred_files(model_id, w)[0], FORCE_RERUN)]
    print(f"[{model_id}] cached {len(ALL_WINDOWS) - len(todo)}, to run {len(todo)}")
    if not todo:
        continue
    with free_cuda():
        with Measure() as m_load:
            model = mv.load_model(model_id, MCFG[model_id], DEVICE)
        print(f"  loaded {MCFG[model_id]['hf_repo']} in {m_load.seconds:.1f}s")
        for w in todo:
            npz, js = pred_files(model_id, w)
            try:
                with Measure() as m:
                    out = mv.run_model(model_id, model, w["image_paths"], MCFG[model_id], DEVICE,
                                       **run_kwargs(model_id, w))
            except torch.cuda.OutOfMemoryError as e:
                js.write_text(json.dumps({"failed": "CUDA OOM", "error": str(e)[:500], "n_views": len(w["cams"])}))
                print(f"  {w['exp']}/{w['id']}: CUDA OOM -> skipped")
                torch.cuda.empty_cache()
                continue
            mv.save_prediction(npz, out)
            js.write_text(json.dumps({"seconds": m.seconds, "peak_vram_mb": m.peak_vram_mb, "load_s": m_load.seconds,
                                      "n_views": len(w["cams"]), "image_hw": list(out["image_hw"]),
                                      "meta": out["meta"]}, indent=1, default=str))
            FRESH.add((model_id, w["exp"], w["id"]))
            print(f"  {w['exp']}/{w['id']}: {m.seconds:.1f}s, peak {m.peak_vram_mb or 0:.0f} MB, "
                  f"hw={out['image_hw']}, kept(conf)={out['mask'].mean():.2f}")
            del out
        del model

# %% [markdown]
# ## 4. 평가
#
# ### 자세 정렬: Sim(3)과 `used_method`
# 예측 자세는 모델마다 임의의 좌표계·스케일이므로 `bench3d.align.align_poses_sim3(pred_c2w, gt_c2w, method="auto")`로
# Sim(3) `(s, R, t)`를 구해 GT 전역 좌표계로 옮긴다(`x_gt ≈ s·R·x_pred + t`).
# - **Umeyama**(카메라 중심만 사용)는 중심들이 거의 **일직선**이면(앞만 보고 직진하는 CAM_FRONT 창) 궤적 축을 중심으로 한
#   회전(roll)이 정해지지 않는다. 이 회전은 ATE를 바꾸지 않지만 **점군을 통째로 돌려서** Chamfer를 망가뜨린다.
# - 그래서 `auto`는 `centers_degenerate`(중심 분포의 sv2/sv1 < 0.05)이면 **`rot+center`**로 바꾼다:
#   회전은 `R_gt·R_predᵀ`의 chordal 평균(SO(3) 투영), 스케일·이동은 그 회전을 적용한 중심의 최소제곱으로 구한다.
# - 창마다 `used_method`와 `s`를 저장한다. 비교용으로 두 방법의 스케일(`s_umeyama`, `s_rot_center`)도 남긴다. 둘이 크게 다르면 정렬이 불안정하다는 신호다.
# - MASt3R는 metric으로 학습했으므로 `s≈1`이면 metric 스케일도 맞춘 것이다. DUSt3R/VGGT/DA3-LARGE의 `s`는 의미가 없다(상대 스케일).
#
# ### 자세 메트릭 (`bench3d.metrics.pose.pose_metrics`, 정렬 후)
# - **ATE**: 카메라 중심 위치 오차의 RMSE/평균/중앙값 [m]. **RPE**: 연속 뷰 i→i+1 상대 자세의 이동 RMSE [m]·회전 평균 [°].
# - **RRA / RTA**: 모든 뷰 쌍의 상대 회전 오차 [°] / 상대 이동 **방향** 오차 [°]. 스케일과 정렬에 무관하다.
# - **AUC@T**: 쌍별 `max(회전 오차, 이동 방향 오차)`가 1..T° 정수 임계값 아래인 비율을 평균한 값(0~1).
#
# ### 서라운드(B)의 주 지표
# 서라운드의 GT 카메라 중심은 1~2 m 안에 모여 있다. 이 짧은 기준선으로 스케일과 이동을 추정하면 작은 중심 오차가 `s`를 크게 흔든다
# (Umeyama든 `rot+center`든 마찬가지). 따라서 **서라운드는 RRA/RTA/AUC를 주 지표**로, ATE와 Chamfer 계열은 **참고 지표**로만 읽는다.
# 또한 6캠 각각이 자기 시각의 ego pose를 쓰므로 주행 중에는 GT 중심 배치에도 차량 이동(최대 수십 cm)이 섞인다.
#
# ### 기하 메트릭 (`bench3d.metrics.pointcloud`)
# 1. **신뢰도 필터**: 모델 데모의 기본 규칙을 그대로 쓴다(창별 `_run.json`의 `meta.conf_rule`).
#    DUSt3R: 원시 conf > log(3.0)(데모가 `conf_trf`를 거친 값을 원시 conf와 비교함) / MASt3R: `clean_depth` 후 dense conf > 1.5 /
#    VGGT: 창 전체 depth conf 하위 50% 제거 / DA3: conf ≥ clamp(1.05, p40, p90).
#    규칙이 모델마다 달라 남는 점의 비율(`kept_conf_frac`)도 함께 보고한다.
# 2. 남은 예측 점을 위 Sim(3)로 전역 좌표에 옮긴다.
# 3. **크롭**: 예측 점과 LiDAR 점을 모두 입력 카메라(GT 자세·원본 K)의 시야 안, 깊이 `crop_max_depth` m 이내로 자른다(`frustum_mask`).
# 4. **밀도 균일화**: 두 점군을 `voxel` m 격자로 다운샘플한다.
# 5. `reconstruction_metrics`: **accuracy** = 평균 d(예측→GT), **completeness** = 평균 d(GT→예측), **chamfer** = 둘의 평균,
#    precision/recall/**F-score@0.25/0.5/1.0 m**, 그리고 **중앙값**(`accuracy_median`, `completeness_median`).
#
# **해석 주의**: LiDAR는 희소하고(32채널), 수직 시야가 좁아 높은 건물 윗부분 등 카메라가 보는 면을 못 본다. 유리·검은 물체도 반사가 약하다.
# 그런 곳의 예측 점은 맞아도 가장 가까운 LiDAR 점이 멀어서 **accuracy(예측→GT)가 불리하게** 나온다. 하늘을 가까운 거리에 놓는 모델도 accuracy가 나빠진다.
# 평균은 이런 꼬리에 끌려가므로 **중앙값을 함께** 본다. completeness(GT→예측)는 LiDAR가 본 면을 예측이 덮었는지를 본다.

# %%
from bench3d.align import align_poses_sim3, apply_sim3, apply_sim3_to_poses
from bench3d.metrics.pointcloud import frustum_mask, reconstruction_metrics, voxel_downsample
from bench3d.metrics.pose import pose_metrics

TAUS = (0.25, 0.5, 1.0)
GT_DIR = result_dir(cfg, NOTEBOOK, "lidar_gt")
_GT = {}


def gt_cloud(win: dict) -> np.ndarray:
    """Frustum-cropped LiDAR union in the global frame (cached in memory; voxelized .ply saved for viewing)."""
    key = (win["exp"], win["id"])
    if key not in _GT:
        pts = gt_lidar_world(win)
        _GT[key] = pts[frustum_mask(pts, win["cams"], P["crop_max_depth"])]
        ply = GT_DIR / win["exp"] / f"{win['id']}.ply"
        if not is_cached(ply, FORCE_RERUN):
            save_ply(ply, voxel_downsample(_GT[key], P["voxel"]))
    return _GT[key]


def evaluate_window(model_id: str, win: dict) -> dict:
    npz, js = pred_files(model_id, win)
    run = json.loads(js.read_text())
    pred = mv.load_prediction(npz)
    gt_c2w = np.stack([c["c2w"] for c in win["cams"]])
    s, R, t, used = align_poses_sim3(pred["c2w"], gt_c2w, method="auto")
    s_alt = {}
    for meth in ("umeyama", "rot+center"):  # diagnostics only
        try:
            s_alt[meth] = float(align_poses_sim3(pred["c2w"], gt_c2w, method=meth)[0])
        except (ValueError, np.linalg.LinAlgError):
            s_alt[meth] = float("nan")
    c2w_al = apply_sim3_to_poses(pred["c2w"], s, R, t)
    pm = pose_metrics(c2w_al, gt_c2w)

    m = pred["mask"].reshape(-1)
    xyz = pred["points"].reshape(-1, 3)[m].astype(np.float64)
    rgb = pred["images"].reshape(-1, 3)[m]
    ok = np.isfinite(xyz).all(1)
    xyz = apply_sim3(xyz[ok], s, R, t)
    keep = frustum_mask(xyz, win["cams"], P["crop_max_depth"])
    pred_c = np.c_[xyz[keep], rgb[ok][keep]]
    rm = reconstruction_metrics(pred_c[:, :3], gt_cloud(win), taus=TAUS, voxel=P["voxel"])
    pv = voxel_downsample(pred_c, P["voxel"])  # subsampled aligned cloud for viewing
    save_ply(npz.with_name(f"{win['id']}_aligned.ply"), pv[:, :3], rgb=pv[:, 3:6].astype(np.uint8))
    return {"model": model_id, "exp": win["exp"], "window": win["id"], "scene": win["scene"],
            "n_views": len(win["cams"]), "used_method": used, "s": float(s), "s_umeyama": s_alt["umeyama"],
            "s_rot_center": s_alt["rot+center"], "R": R.tolist(), "t": np.asarray(t).tolist(),
            "gt_centers": center_stats(win["cams"]), "pose": pm, "recon": rm,
            "kept_conf_frac": float(m.mean()), "kept_crop_frac": float(keep.mean()) if len(keep) else 0.0,
            "runtime_s": run["seconds"], "peak_vram_mb": run["peak_vram_mb"], "load_s": run.get("load_s"),
            "conf_rule": run["meta"]["conf_rule"], "point_source": run["meta"].get("point_source"),
            "image_hw": run["image_hw"], "c2w_aligned": c2w_al.tolist(), "c2w_gt": gt_c2w.tolist()}


RECS, FAILED = [], []
for model_id in MODELS:
    for w in ALL_WINDOWS:
        npz, js = pred_files(model_id, w)
        out_js = npz.with_name(f"{w['id']}.json")
        if not npz.exists():
            FAILED.append({"model": model_id, "exp": w["exp"], "window": w["id"],
                           "reason": json.loads(js.read_text()).get("failed") if js.exists() else "not run"})
            continue
        stale = (model_id, w["exp"], w["id"]) in FRESH or (out_js.exists() and npz.stat().st_mtime > out_js.stat().st_mtime)
        if is_cached(out_js, FORCE_RERUN) and not stale:
            rec = json.loads(out_js.read_text())
        else:
            rec = evaluate_window(model_id, w)
            out_js.write_text(json.dumps(rec, indent=1, default=float))
        RECS.append(rec)
        print(f"{model_id:12s} {w['exp']:8s} {w['id']:18s} {rec['used_method']:10s} s={rec['s']:.3f} "
              f"AUC@30={rec['pose']['auc@30']:.3f} ATE={rec['pose']['ate_rmse']:.2f}m "
              f"chamfer={rec['recon'].get('chamfer', float('nan')):.2f}m")
if FAILED:
    print("FAILED / missing:", FAILED)

# %% [markdown]
# ## 5. 결과 저장
# 모델·실험마다 `results/outdoor/03_multiview_geometry/<model>/summary_<exp>.json`(`save_summary`, 평평한 숫자 metrics)을 쓴다.
# - `metrics`: 창 평균. 자세(ATE, RPE, RRA/RTA, AUC), 기하(accuracy, completeness, chamfer, 중앙값, precision/recall/F-score),
#   `sim3_scale_mean/median`, `rot_center_frac`(rot+center 정렬을 쓴 창 비율), `kept_conf_frac`.
# - `runtime_s`: 창당 평균 시간(로드 제외), `peak_vram_mb`: 창 중 최대.
# - `extra`: 주 지표 목록(서라운드는 RRA/RTA/AUC), 창별 `s`·`used_method`·시간, 신뢰도 규칙, 설정, 실패 창.

# %%
POSE_KEYS = ["ate_rmse", "ate_mean", "ate_median", "rpe_trans_rmse", "rpe_rot_mean_deg", "rra_mean_deg",
             "rta_mean_deg", "auc@5", "auc@15", "auc@30"]
RECON_KEYS = ["accuracy", "completeness", "chamfer", "accuracy_median", "completeness_median", "n_pred", "n_gt"] + \
    [f"{k}@{t:g}" for t in TAUS for k in ("precision", "recall", "fscore")]
PRIMARY = {"front": ["auc@30", "rra_mean_deg", "rta_mean_deg", "ate_rmse", "chamfer", "fscore@0.5"],
           "surround": ["auc@30", "auc@15", "rra_mean_deg", "rta_mean_deg"]}
VARIANT = {"front": f"front: {P['n_frames']} consecutive CAM_FRONT keyframes x {P['windows_per_scene']} window(s)/scene, eval4",
           "surround": "surround: 6 cameras of the middle keyframe per scene, eval4"}


def nanmean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


SUMMARIES = {}
for model_id in MODELS:
    for exp in EXPS:
        recs = [r for r in RECS if r["model"] == model_id and r["exp"] == exp]
        if not recs:
            continue
        metrics = {k: nanmean([r["pose"].get(k) for r in recs]) for k in POSE_KEYS}
        metrics.update({k: nanmean([r["recon"].get(k) for r in recs]) for k in RECON_KEYS})
        ss = [r["s"] for r in recs]
        metrics.update({"sim3_scale_mean": float(np.mean(ss)), "sim3_scale_median": float(np.median(ss)),
                        "rot_center_frac": float(np.mean([r["used_method"] == "rot+center" for r in recs])),
                        "kept_conf_frac": nanmean([r["kept_conf_frac"] for r in recs]), "n_windows": len(recs)})
        secs = [r["runtime_s"] for r in recs]
        vram = [r["peak_vram_mb"] for r in recs if r["peak_vram_mb"] is not None]
        extra = {
            "primary_metrics": PRIMARY[exp],
            "scale_ill_conditioned": exp == "surround",
            "alignment": "bench3d.align.align_poses_sim3(method='auto'); points transformed by the same Sim(3)",
            "windows": [{k: r[k] for k in ("window", "scene", "n_views", "used_method", "s", "s_umeyama",
                                          "s_rot_center", "runtime_s", "peak_vram_mb", "kept_conf_frac")}
                        | {"gt_max_baseline_m": r["gt_centers"]["max_baseline_m"]} for r in recs],
            "runtime_median_s": float(np.median(secs)), "load_s": recs[0].get("load_s"),
            "conf_rule": sorted({r["conf_rule"].split(" = ")[0] for r in recs}),
            "point_source": recs[0]["point_source"], "image_hw": recs[0]["image_hw"],
            "settings": {"model_cfg": MCFG[model_id], "n_frames": P["n_frames"], "windows_per_scene": P["windows_per_scene"],
                         "dust3r_niter": P.get("dust3r_niter"), "mast3r_niter1": P.get("mast3r_niter1"),
                         "mast3r_niter2": P.get("mast3r_niter2"), "crop_max_depth": P["crop_max_depth"],
                         "voxel": P["voxel"], "gt_ego_radius": GT_EGO_RADIUS, "taus": list(TAUS)},
            "gt": "LIDAR_TOP keyframes of the window's samples -> global; frustum-cropped to the input cameras",
            "failed": [f for f in FAILED if f["model"] == model_id and f["exp"] == exp],
        }
        out = save_summary(result_dir(cfg, NOTEBOOK, model_id), part=cfg["_part"], notebook=NOTEBOOK,
                           profile=cfg["profile"], model=model_id, variant=VARIANT[exp], metrics=metrics,
                           runtime_s=float(np.mean(secs)), peak_vram_mb=float(max(vram)) if vram else None,
                           checkpoint=MCFG[model_id]["hf_repo"], packages=PACKAGES, seed=cfg["seed"], extra=extra,
                           filename=f"summary_{exp}.json")
        SUMMARIES[(model_id, exp)] = {**metrics, "runtime_s": float(np.mean(secs)),
                                      "peak_vram_mb": float(max(vram)) if vram else float("nan")}
        print("saved", out.relative_to(cfg.root))

# %%
TABLE_COLS = {"front": ["auc@30", "rra_mean_deg", "rta_mean_deg", "ate_rmse", "rpe_trans_rmse", "chamfer",
                        "accuracy_median", "completeness_median", "fscore@0.5", "sim3_scale_median", "rot_center_frac",
                        "runtime_s", "peak_vram_mb"],
              "surround": ["auc@30", "auc@15", "rra_mean_deg", "rta_mean_deg", "ate_rmse", "chamfer", "fscore@0.5",
                           "sim3_scale_median", "runtime_s", "peak_vram_mb"]}
TABLES = {}
for exp in EXPS:
    df = pd.DataFrame({m: SUMMARIES[(m, exp)] for m in MODELS if (m, exp) in SUMMARIES}).T
    if df.empty:
        continue
    TABLES[exp] = df[[c for c in TABLE_COLS[exp] if c in df.columns]]
    print(f"\n=== {exp} (primary: {', '.join(PRIMARY[exp])}) ===")
    print(TABLES[exp].round(3).to_string())

# %% [markdown]
# ## 6. 시각화
# ### 6.1 궤적 (위에서 본 모습)
# GT(검정)와 Sim(3) 정렬 후 각 모델의 카메라 중심. `front`는 창마다, `surround`는 장면마다 카메라 중심과 광축(z축) 방향을 그린다.
# 좌표는 창의 첫 GT 카메라 기준 전역 x/y [m]이다.

# %%
import matplotlib.pyplot as plt
from bench3d.viz import model_colors, plotly_points

COLORS = model_colors(MODELS)


def plot_topview(exp: str, arrows: bool):
    wins = WINDOWS[exp]
    fig, axes = plt.subplots(1, len(wins), figsize=(4.2 * len(wins), 4.2), squeeze=False)
    for ax, w in zip(axes[0], wins):
        gt = np.stack([c["c2w"] for c in w["cams"]])
        o = gt[0, :3, 3]
        series = [("GT", gt, "#000000")]
        series += [(r["model"], np.asarray(r["c2w_aligned"]), COLORS[r["model"]])
                   for r in RECS if r["exp"] == exp and r["window"] == w["id"]]
        span = max(np.ptp(gt[:, 0, 3]), np.ptp(gt[:, 1, 3]), 1.0)
        for name, T, col in series:
            c = T[:, :3, 3] - o
            ax.plot(c[:, 0], c[:, 1], "o-" if not arrows else "o", ms=3, lw=1, color=col, label=name)
            if arrows:
                z = T[:, :3, 2]
                ax.quiver(c[:, 0], c[:, 1], z[:, 0], z[:, 1], color=col, angles="xy", scale_units="xy",
                          scale=1.0 / (0.25 * span), width=0.004)
        ax.set_aspect("equal")
        ax.set_title(w["id"], fontsize=9)
        ax.set_xlabel("x [m]")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("y [m]")
    axes[0, -1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{exp}: GT vs Sim(3)-aligned camera centers (top view)")
    fig.tight_layout()
    return fig


plot_topview("front", arrows=False)
if "surround" in EXPS:
    plot_topview("surround", arrows=True)
plt.show()

# %% [markdown]
# ### 6.2 3D 점군: LiDAR vs 각 모델 (front 첫 창)
# 크롭·복셀화한 LiDAR(GT)와 각 모델의 정렬된 점군(`*_aligned.ply`)을 같은 좌표계에서 겹쳐 본다. 범례를 눌러 모델별로 켜고 끌 수 있다.

# %%
VIEW_WIN = WINDOWS["front"][0]
origin = np.stack([c["c2w"] for c in VIEW_WIN["cams"]])[0, :3, 3]
clouds = {"LiDAR (GT)": voxel_downsample(gt_cloud(VIEW_WIN), P["voxel"]) - origin}
for model_id in MODELS:
    ply = pred_files(model_id, VIEW_WIN)[0].with_name(f"{VIEW_WIN['id']}_aligned.ply")
    if ply.exists():
        clouds[model_id] = load_ply(ply)["xyz"] - origin
fig3d = plotly_points(clouds, max_points=40000, colors={"LiDAR (GT)": "#000000", **COLORS})
fig3d.update_layout(title=f"{VIEW_WIN['id']} (front): LiDAR vs aligned predictions, origin = first GT camera")
fig3d.show()

# %% [markdown]
# ### 6.3 속도–정확도
# x축은 창당 평균 실행 시간(로그), y축은 front의 Chamfer(낮을수록 좋음)와 AUC@30(높을수록 좋음).
# 최적화 기반(DUSt3R/MASt3R)과 순전파(VGGT/DA3)가 어디에 놓이는지 본다.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for m in MODELS:
    f = SUMMARIES.get((m, "front"))
    if f:
        axes[0].scatter(f["runtime_s"], f["chamfer"], s=60, color=COLORS[m], label=m)
        axes[1].scatter(f["runtime_s"], f["auc@30"], s=60, color=COLORS[m], marker="o", label=f"{m} front")
    sr = SUMMARIES.get((m, "surround"))
    if sr:
        axes[1].scatter(sr["runtime_s"], sr["auc@30"], s=60, color=COLORS[m], marker="^", label=f"{m} surround")
for ax, yl in zip(axes[:2], ["front Chamfer [m] (lower better)", "AUC@30 (higher better)"]):
    ax.set_xscale("log")
    ax.set_xlabel("runtime per window [s] (log)")
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
xs = np.arange(len(MODELS))
for k, exp in enumerate(EXPS):
    vals = [SUMMARIES.get((m, exp), {}).get("auc@30", np.nan) for m in MODELS]
    axes[2].bar(xs + (k - 0.5 * (len(EXPS) - 1)) * 0.38, vals, width=0.38, label=exp,
                color=[COLORS[m] for m in MODELS], alpha=1.0 if exp == "front" else 0.5, edgecolor="k")
axes[2].set_xticks(xs, MODELS, rotation=20)
axes[2].set_ylabel("AUC@30")
axes[2].set_title("AUC@30: front (solid) vs surround (faded)")
axes[2].grid(alpha=0.3, axis="y")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 6.4 창별 정렬 진단
# `used_method`, 스케일 `s`(두 방법 비교), GT 중심 배치. `s_umeyama`와 `s_rot_center`가 크게 다르면 그 창의 스케일·ATE·Chamfer는 믿기 어렵다.

# %%
DIAG = pd.DataFrame([{"model": r["model"], "exp": r["exp"], "window": r["window"], "used_method": r["used_method"],
                      "s": r["s"], "s_umeyama": r["s_umeyama"], "s_rot_center": r["s_rot_center"],
                      "gt_baseline_m": r["gt_centers"]["max_baseline_m"], "gt_sv2/sv1": r["gt_centers"]["sv2_over_sv1"],
                      "auc@30": r["pose"]["auc@30"], "chamfer": r["recon"].get("chamfer", np.nan),
                      "kept_conf": r["kept_conf_frac"], "sec": r["runtime_s"]} for r in RECS])
print(DIAG.round(3).to_string(index=False) if not DIAG.empty else "no results")

# %% [markdown]
# ## 7. 해석 (실행 후 작성)
#
# ### 7.1 전방 연속 키프레임 (A)
# > (실행 후 작성) 자세: 모델별 AUC@30·RRA·RTA·ATE 순위와 차이. `rot+center`가 쓰인 창 비율과 그 이유(직진 구간).
#
# > (실행 후 작성) 기하: Chamfer·F-score@0.5와 중앙값. accuracy(예측→GT)가 completeness보다 나쁜 경우, LiDAR 미관측 영역(높은 건물, 먼 배경) 때문인지 3D 시각화로 확인.
#
# ### 7.2 서라운드 리그 (B)
# > (실행 후 작성) RRA/RTA/AUC(주 지표)로 본 상대 자세 정확도. 겹침이 적은 뒤쪽 카메라(CAM_BACK 계열)에서 실패하는지. ATE·Chamfer는 스케일이 불안정하므로 참고만.
#
# ### 7.3 스케일
# > (실행 후 작성) MASt3R의 `s`가 1에 가까운가(metric 학습 효과). 다른 모델의 `s`는 상대 스케일이므로 해석하지 않는다.
#
# ### 7.4 속도–정확도
# > (실행 후 작성) 최적화 기반(DUSt3R/MASt3R)의 추가 시간이 정확도로 보상되는가. 순전파 모델(VGGT/DA3)의 창당 시간과 VRAM.
#
# ### 7.5 주행 데이터에서 순전파 모델은 어디서 실패하는가?
# > (실행 후 작성) 아래 네 가지 후보 각각에 대해 근거(지표·그림)를 적는다.
# > - 낮은 시차의 전진 운동: 깊이 방향 이동만 있어 삼각측량 기준선이 작다 → 깊이/스케일 드리프트, RTA 증가?
# > - 겹침이 적은 서라운드 리그: 이웃 카메라끼리 겹침이 좁다 → 상대 회전이 틀어지거나 카메라가 겹쳐 놓이는가?
# > - 동적 물체: 움직이는 차량이 정적 장면 가정을 깬다(LiDAR 합집합에서는 궤적처럼 번짐) → 그 주변 오차.
# > - 하늘·원거리: 하늘을 유한 거리에 놓거나 먼 배경 깊이가 붕괴 → accuracy 꼬리, 신뢰도 필터가 얼마나 막는가.
#
# ### 7.6 한계
# > (실행 후 작성) 장면 4개·창 1~2개로 통계적으로 작다. 신뢰도 필터가 모델별 데모 기본값이라 남는 점의 양이 다르다.
# > 학습 데이터 누수: DUSt3R README의 학습 데이터 목록에 Waymo는 있으나 nuScenes는 없다(docs/step0_research.md). VGGT·DA3·MASt3R의 nuScenes 사용 여부는 미확인(TODO(verify)).
