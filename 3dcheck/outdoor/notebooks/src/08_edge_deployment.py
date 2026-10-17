# ---
# jupyter:
#   kernelspec:
#     display_name: 3dv (torch 2.1.2, cu121)
#     language: python
#     name: 3dv
# ---

# %% [markdown]
# # 08. 엣지 배포: PointPillars ONNX → TensorRT, voxelization CPU vs GPU
#
# ## 목적
# nuScenes PointPillars(FPN) 공식 체크포인트를 엣지(Jetson Orin Nano 등)에 올린다고 가정하고, 파이프라인을
# **voxelize → H2D 복사 → 신경망 → 디코드·NMS** 단계로 나눠 각 단계의 지연시간과 정확도 손실을 잰다.
# 핵심 질문은 "CPU가 약한 기기에서 voxelization이 병목이 되는가"이다(docs/step0_research.md 4절).
#
# ## 비교 모델과 원리
# - **PointPillars (nuScenes, FPN)**: 0.25 m 기둥(pillar)마다 최대 64점을 모아 `voxels[P,64,4]`(x, y, z, 시간차)를 만들고,
#   PointNet(HardVFE) → BEV 캔버스 scatter → SECOND 백본 + FPN → 앵커 헤드(클래스·상자·방향) → 디코드·NMS.
# - **분할 배포**: voxelization은 출력 크기가 데이터마다 달라지고 해시·scatter 기반이라 표준 ONNX로 옮기기 어렵다.
#   그래서 그래프 밖에서 voxelize하고, 신경망(PFN → scatter → backbone → neck → head 원시 출력)만 **최대 기둥 수 P로 고정**해
#   `torch.onnx.export` 한다. 빈 기둥은 `num_points=0`으로 패딩하고 scatter에서 버림 칸으로 보낸다.
# - **voxelization 3종**: (1) numpy CPU, (2) torch GPU 벡터화(좌표 → 선형 인덱스 → `unique`), (3) 기준: mmcv CUDA 커널
#   (`hard_voxelize`, mmdet3d `VoxelizationByGridShape`와 `mmcv.ops.Voxelization`이 같은 커널을 씀).
# - **런타임**: PyTorch FP32, ONNX Runtime(CPU, 가능하면 CUDA EP), TensorRT FP32/FP16(설치된 경우).
# - **디코드·NMS**: 모델 헤드의 `predict_by_feat`(mmdet3d 1.4.0 `Base3DDenseHead`)를 그대로 써서 후처리 정확성을 유지한다.
#
# ## 원 논문
# - PointPillars: https://arxiv.org/abs/1812.05784
# - 참고 구현(직접 비교 대상 아님): NVIDIA-AI-IOT/CUDA-PointPillars (KITTI/OpenPCDet 모델, Orin FP16 voxelization 0.18 ms,
#   디코드·NMS 1.79 ms를 README에 보고)
#
# ## 커널
# `3dv` (torch 2.1.2 + cu121, mmcv 2.1.0, mmdet3d 1.4.0, onnx, onnxruntime). TensorRT는 선택(없으면 해당 절은 설치 안내만 출력).
# mmdeploy는 mmdet3d 1.3.0에 고정·정체 상태라 쓰지 않는다.
#
# ## 예상 VRAM·시간 (추정)
# **추정치**다. PointPillars 추론 피크 VRAM 1~3 GB(모든 프로파일 공통), ONNX 내보내기 1분 이내,
# 단계별 벤치마크(warmup·iters는 config `edge`) 수 분, ORT CPU 신경망은 샘플당 수백 ms~수 s라 반복 수를 줄인다.
# 끝단 mAP 확인(eval4 첫 장면, 키프레임 약 40개 × 백엔드 2~4개) 5~15분. TensorRT 엔진 빌드는 FP32/FP16 각각 수 분.
#
# ## 이 노트북이 답하려는 질문
# 1. numpy·torch GPU voxelization은 mmcv 커널과 같은 결과(기둥 수, 기둥별 점 수)를 내는가? 다르면 왜인가?
# 2. 신경망만 고정 크기로 ONNX에 담을 수 있는가? 막히는 연산은 무엇인가? ORT/TRT 출력은 PyTorch와 얼마나 같은가?
# 3. 단계별 지연시간에서 무엇이 병목인가(CPU voxelize, 패딩된 기둥 텐서의 H2D, 신경망, NMS)?
# 4. FP16 TensorRT 경로의 끝단 정확도(mAP/NDS) 손실은 얼마인가?

# %% [markdown]
# ## 1. 설정 로드

# %%
FORCE_RERUN = False            # True: ignore cached results
NOTEBOOK = "08_edge_deployment"
MODEL_ID = "pointpillars"
OPSET = 13                     # ONNX opset (TensorRT 8.6 parses up to 17)
RUN_E2E = True                 # end-to-end mAP of the split pipeline on one eval4 scene
ORT_CPU_ITERS = 10             # ORT on CPU is slow; fewer timing iterations

import json
import shutil
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from bench3d.config import load_config
from bench3d.io import is_cached, result_dir, save_summary
from bench3d.runtime import Measure, env_report, free_cuda, set_seed

cfg = load_config("outdoor")                 # profile: config.yaml or $THREEDCHECK_PROFILE
P = cfg.p["edge"]                            # this notebook's profile block
M = cfg["models"]["detection"][MODEL_ID]
FORCE_RERUN = FORCE_RERUN or cfg.get("force_rerun", False)
set_seed(cfg["seed"])
ENV = env_report(["torch", "numpy", "mmcv", "mmdet3d", "mmengine", "onnx", "onnxruntime", "onnxruntime-gpu",
                  "tensorrt", "nuscenes-devkit"])
print(json.dumps(ENV, indent=1))
print("profile:", cfg["profile"], P)
assert torch.cuda.is_available(), "this notebook times GPU stages; run on a CUDA machine"
DEV = torch.device("cuda:0")

from bench3d.models import mmdet3d_tools as mt

MMDET3D = mt.add_repo_to_path(cfg.third_party("mmdetection3d"))
OUT = result_dir(cfg, NOTEBOOK, MODEL_ID)

# %% [markdown]
# ## 2. 데이터와 모델 로드
# - 05와 같은 eval4 info(`nuscenes_infos_eval4.pkl`, `outdoor/scripts/prepare_mmdet3d_infos.sh`)와 공식 테스트 파이프라인
#   (10 sweep, `PointsRangeFilter`)을 그대로 써서 입력 점군을 만든다. 벤치마크 샘플은 첫 eval4 장면의 가운데 키프레임.
# - `mmdet3d.apis.init_model`은 `naiveSyncBN`을 일반 BN으로 바꾸고 체크포인트를 읽어 eval 모드로 둔다.

# %%
from bench3d.nusc import scene_samples
from bench3d.sampling import middle
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from nuscenes.nuscenes import NuScenes

DATA_ROOT = cfg.path(cfg["dataset"]["data_root"])
VERSION = cfg["dataset"]["version"]
EVAL4_PKL = cfg["dataset"]["mmdet3d_info_eval4"]
assert (DATA_ROOT / EVAL4_PKL).exists(), "run outdoor/scripts/prepare_mmdet3d_infos.sh first"

TEST_CFG = mt.configure_test(mt.load_config(MMDET3D, M["config"]), data_root=DATA_ROOT, ann_file=EVAL4_PKL,
                             work_dir=OUT / "work_dirs", checkpoint=M["checkpoint"], jsonfile_prefix=None,
                             format_only=False, seed=cfg["seed"])   # evaluator unused here
init_default_scope("mmdet3d")
dataset = DATASETS.build(TEST_CFG.test_dataloader.dataset)
model = init_model(TEST_CFG, M["checkpoint"], device=str(DEV))
vl = model.data_preprocessor.voxel_layer
VOX = dict(voxel_size=list(vl.voxel_size), pc_range=list(vl.point_cloud_range), max_points=int(vl.max_num_points),
           max_voxels=int(vl.max_voxels[1]))   # index 1 = test-time limit (mmcv Voxelization.forward)
print(type(vl).__name__, VOX)
assert P["max_pillars"] >= VOX["max_voxels"], "max_pillars must cover the test-time max_voxels"
NP = int(P["max_pillars"])

nusc = NuScenes(version=VERSION, dataroot=str(DATA_ROOT), verbose=False)
E2E_SCENE = cfg["splits"]["eval4"][0]
scene_tokens = scene_samples(nusc, E2E_SCENE)
token_to_idx = {dataset.get_data_info(i)["token"]: i for i in range(len(dataset))}
BENCH_IDX = token_to_idx[middle(scene_tokens)]
data = dataset[BENCH_IDX]
points = data["inputs"]["points"]           # (N, 4): x, y, z, time lag  (use_dim [0,1,2,4] of the 10-sweep cloud)
meta = data["data_samples"].metainfo
print(f"bench sample {BENCH_IDX}: {points.shape[0]} points, features {points.shape[1]}")

# %% [markdown]
# ## 3. Voxelization 비교 (numpy CPU / torch GPU / mmcv CUDA)
# mmcv `hard_voxelize`(deterministic=True, `mmcv/ops/csrc/common/cuda/voxelization_cuda_kernel.cuh`) 규칙:
# - 좌표 `c = floor((p - min) / size)`, 격자 크기 `round((max - min) / size)`, 범위 밖 점은 버림. 좌표 순서는 (z, y, x).
# - 기둥 번호는 **점 순서에서 처음 나타난 순서**, 기둥 안의 점도 점 순서대로 최대 64개, 기둥이 `max_voxels`를 넘으면 뒤에 나온 기둥을 버림.
#
# numpy 구현은 `np.unique(return_index)`로 같은 "처음 등장 순서"를 재현하므로 결과가 같아야 한다.
# torch 구현은 `torch.unique`가 키를 **정렬**하므로 기본 순서가 다르다(`order="sorted"`). `order="first"`는 각 기둥의 첫 점
# 인덱스를 `scatter_reduce(amin)`로 구해 순서를 맞춘다. 신경망 출력은 scatter 때문에 기둥 순서와 무관하고, 순서 차이가
# 결과를 바꾸는 경우는 기둥 수가 `max_voxels`를 넘어 **어떤 기둥을 버리는지**가 달라질 때뿐이다.
# 경계에 정확히 걸린 점은 부동소수점 연산 순서 차이로 다른 칸에 들어갈 수 있다(아주 드묾).

# %%
def _key_parts(c, grid):
    return (c[:, 2] * grid[1] + c[:, 1]) * grid[0] + c[:, 0]          # linear index over (z, y, x)


def voxelize_numpy(pts: np.ndarray, voxel_size, pc_range, max_points: int, max_voxels: int):
    """Hard voxelization on CPU with mmcv's (deterministic) semantics. Returns voxels, coors (z,y,x), num_points."""
    pts = np.ascontiguousarray(pts, np.float32)
    vs, lo, hi = (np.asarray(v, np.float32) for v in (voxel_size, pc_range[:3], pc_range[3:]))
    grid = np.round((hi - lo) / vs).astype(np.int64)
    c = np.floor((pts[:, :3] - lo) / vs).astype(np.int64)
    ok = np.all((c >= 0) & (c < grid), axis=1)
    pts, c = pts[ok], c[ok]
    key = _key_parts(c, grid)
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    rank = np.empty(len(uniq), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(uniq))     # voxel id = order of first appearance
    vid = rank[inv]
    order = np.argsort(vid, kind="stable")                            # points grouped by voxel, point order kept
    svid = vid[order]
    counts = np.bincount(vid, minlength=len(uniq))
    slot = np.arange(len(order)) - (np.cumsum(counts) - counts)[svid]
    keep = (slot < max_points) & (svid < max_voxels)
    n_vox = min(len(uniq), max_voxels)
    voxels = np.zeros((n_vox, max_points, pts.shape[1]), np.float32)
    voxels[svid[keep], slot[keep]] = pts[order[keep]]
    ukeys = uniq[np.argsort(first, kind="stable")][:n_vox]
    coors = np.stack([ukeys // (grid[0] * grid[1]), (ukeys // grid[0]) % grid[1], ukeys % grid[0]], 1).astype(np.int32)
    return voxels, coors, np.minimum(counts[:n_vox], max_points).astype(np.int32)   # counts are indexed by voxel id


def voxelize_torch(pts: torch.Tensor, voxel_size, pc_range, max_points: int, max_voxels: int, order: str = "first"):
    """Vectorized GPU hard voxelization: coords -> linear index -> torch.unique (+ optional first-appearance order)."""
    dev = pts.device
    vs = torch.tensor(voxel_size, device=dev, dtype=torch.float32)
    lo = torch.tensor(pc_range[:3], device=dev, dtype=torch.float32)
    hi = torch.tensor(pc_range[3:], device=dev, dtype=torch.float32)
    grid = torch.round((hi - lo) / vs).long()
    c = torch.floor((pts[:, :3] - lo) / vs).long()
    ok = ((c >= 0) & (c < grid)).all(1)
    pts, c = pts[ok], c[ok]
    n = pts.shape[0]
    key = _key_parts(c, grid)
    uniq, inv = torch.unique(key, return_inverse=True)                # sorted keys
    if order == "first":
        first = torch.full((uniq.numel(),), n, device=dev, dtype=torch.long)
        first = first.scatter_reduce(0, inv, torch.arange(n, device=dev), reduce="amin", include_self=True)
        perm = torch.argsort(first)
        rank = torch.empty_like(perm)
        rank[perm] = torch.arange(perm.numel(), device=dev)
        vid, ukeys = rank[inv], uniq[perm]
    else:
        vid, ukeys = inv, uniq
    svid, sorder = torch.sort(vid, stable=True)
    counts = torch.bincount(vid, minlength=uniq.numel())
    slot = torch.arange(n, device=dev) - (torch.cumsum(counts, 0) - counts)[svid]
    keep = (slot < max_points) & (svid < max_voxels)
    n_vox = min(uniq.numel(), max_voxels)
    voxels = pts.new_zeros((n_vox, max_points, pts.shape[1]))
    voxels[svid[keep], slot[keep]] = pts[sorder[keep]]
    ukeys = ukeys[:n_vox]
    coors = torch.stack([ukeys // (grid[0] * grid[1]), (ukeys // grid[0]) % grid[1], ukeys % grid[0]], 1).int()
    return voxels, coors, counts[:n_vox].clamp(max=max_points).int()


def compare_voxels(name, a, b) -> dict:
    """a, b: (voxels, coors, num_points) as numpy. Aligns voxels by coordinate, then compares."""
    (va, ca, na), (vb, cb, nb) = a, b
    ka = {tuple(x): i for i, x in enumerate(ca.tolist())}
    kb = {tuple(x): i for i, x in enumerate(cb.tolist())}
    common = sorted(set(ka) & set(kb))
    ia, ib = [ka[k] for k in common], [kb[k] for k in common]
    out = {"n_vox_a": len(ca), "n_vox_b": len(cb), "n_common": len(common),
           "same_order": bool(len(ca) == len(cb) and np.array_equal(ca, cb)),
           "num_points_equal": bool(np.array_equal(na[ia], nb[ib])),
           "voxel_values_maxabs": float(np.abs(va[ia] - vb[ib]).max()) if common else None,
           "points_a": int(na.sum()), "points_b": int(nb.sum())}
    print(f"{name:28s} {out}")
    return out


pts_np = points.numpy()
pts_gpu = points.to(DEV)
mmcv_vox = [t.cpu().numpy() for t in vl(pts_gpu)]                       # model's own layer (mmdet3d wrapper)
from mmcv.ops import Voxelization

mmcv_ops_vox = [t.cpu().numpy() for t in Voxelization(VOX["voxel_size"], VOX["pc_range"], VOX["max_points"],
                                                        (VOX["max_voxels"], VOX["max_voxels"])).to(DEV).eval()(pts_gpu)]
np_vox = voxelize_numpy(pts_np, VOX["voxel_size"], VOX["pc_range"], VOX["max_points"], VOX["max_voxels"])
tf_vox = [t.cpu().numpy() for t in voxelize_torch(pts_gpu, VOX["voxel_size"], VOX["pc_range"], VOX["max_points"],
                                                  VOX["max_voxels"], "first")]
ts_vox = [t.cpu().numpy() for t in voxelize_torch(pts_gpu, VOX["voxel_size"], VOX["pc_range"], VOX["max_points"],
                                                  VOX["max_voxels"], "sorted")]
VOX_CMP = {
    "mmcv_ops_vs_mmdet3d_layer": compare_voxels("mmcv.ops vs model layer", mmcv_ops_vox, mmcv_vox),
    "numpy_vs_mmcv": compare_voxels("numpy vs mmcv", np_vox, mmcv_vox),
    "torch_first_vs_mmcv": compare_voxels("torch(first) vs mmcv", tf_vox, mmcv_vox),
    "torch_sorted_vs_mmcv": compare_voxels("torch(sorted) vs mmcv", ts_vox, mmcv_vox),
}
N_POINTS, N_VOXELS = int(pts_np.shape[0]), int(len(mmcv_vox[1]))
pp = mmcv_vox[2]
print(f"{N_POINTS} points -> {N_VOXELS} pillars (cap {VOX['max_voxels']}); points/pillar median {np.median(pp):.0f}, "
      f"saturated (=64) {np.mean(pp == VOX['max_points']):.1%}")

# %% [markdown]
# ## 4. 신경망만 고정 크기 모듈로 감싸기
# `PillarNet`은 체크포인트의 모듈(`pts_voxel_encoder.vfe_layers`, `pts_backbone`, `pts_neck`, `pts_bbox_head`)을 그대로 쓰고,
# 두 부분만 내보내기용으로 다시 쓴다.
# - **HardVFE 특징 장식**(클러스터 중심·기둥 중심 오프셋): 원본은 `num_points`로 나누므로 패딩 기둥(0점)에서 0/0 = NaN이
#   되고 마스크 곱(NaN×0)으로도 지워지지 않는다. `clamp(min=1)`로 막는다(실제 기둥은 값이 같음).
# - **PointPillarsScatter**: 원본 `canvas[:, idx] = feats`(ONNX ScatterND) 대신 크기 `ny·nx + 1` 캔버스에
#   `scatter`(ONNX ScatterElements)하고, 패딩 기둥은 마지막 버림 칸으로 보낸 뒤 잘라낸다.
#
# 원본 경로(mmcv voxelize → `extract_pts_feat` → 헤드)와 같은 기둥을 넣어 출력 차이를 확인한다.

# %%
import torch.nn as nn


class PillarNet(nn.Module):
    """voxels (P,M,C) f32, num_points (P,) i32, coors (P,4) i32 [batch,z,y,x] -> 3 levels x (cls, box, dir) maps."""

    def __init__(self, det):
        super().__init__()
        enc, mid = det.pts_voxel_encoder, det.pts_middle_encoder
        assert type(enc).__name__ == "HardVFE" and type(mid).__name__ == "PointPillarsScatter"
        self.vfe_layers = enc.vfe_layers
        self.cluster, self.center, self.dist = enc._with_cluster_center, enc._with_voxel_center, enc._with_distance
        self.vx, self.vy, self.vz = enc.vx, enc.vy, enc.vz
        self.xo, self.yo, self.zo = enc.x_offset, enc.y_offset, enc.z_offset
        self.nx, self.ny, self.C = mid.nx, mid.ny, mid.in_channels
        self.backbone, self.neck, self.head = det.pts_backbone, det.pts_neck, det.pts_bbox_head

    def forward(self, voxels, num_points, coors):
        npts = num_points.float().clamp(min=1.0).view(-1, 1, 1)
        feats = [voxels]
        if self.cluster:
            feats.append(voxels[:, :, :3] - voxels[:, :, :3].sum(dim=1, keepdim=True) / npts)
        if self.center:
            cf = coors.float()
            feats.append(torch.stack([voxels[:, :, 0] - (cf[:, 3:4] * self.vx + self.xo),
                                      voxels[:, :, 1] - (cf[:, 2:3] * self.vy + self.yo),
                                      voxels[:, :, 2] - (cf[:, 1:2] * self.vz + self.zo)], dim=-1))
        if self.dist:
            feats.append(torch.norm(voxels[:, :, :3], 2, 2, keepdim=True))
        x = torch.cat(feats, dim=-1)
        mask = torch.arange(voxels.shape[1], device=voxels.device).view(1, -1) < num_points.view(-1, 1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        for vfe in self.vfe_layers:
            x = vfe(x)                                                  # -> (P, C)
        idx = coors[:, 2].long() * self.nx + coors[:, 3].long()
        idx = torch.where(num_points > 0, idx, torch.full_like(idx, self.nx * self.ny))
        canvas = x.new_zeros(self.nx * self.ny + 1, self.C).scatter(0, idx.view(-1, 1).expand(-1, self.C), x)
        bev = canvas[:-1].t().reshape(1, self.C, self.ny, self.nx)
        cls, box, dirc = self.head(self.neck(self.backbone(bev)))
        return (*cls, *box, *dirc)


def pad_inputs(voxels, coors_zyx, num_points, n: int = NP):
    """Pad to n pillars (num_points 0) and prepend the batch column -> numpy (voxels, num_points, coors[b,z,y,x])."""
    k = len(coors_zyx)
    assert k <= n, (k, n)
    v = np.zeros((n, *voxels.shape[1:]), np.float32)
    c = np.zeros((n, 4), np.int32)
    p = np.zeros((n,), np.int32)
    v[:k], c[:k, 1:], p[:k] = voxels, coors_zyx, num_points
    return v, p, c


def split_outs(outs):
    k = len(outs) // 3
    return list(outs[:k]), list(outs[k:2 * k]), list(outs[2 * k:])


net = PillarNet(model).eval().to(DEV)
OUT_NAMES = [f"{kind}_{lvl}" for kind in ("cls", "box", "dir") for lvl in range(3)]
IN_NAMES = ["voxels", "num_points", "coors"]

with torch.no_grad():
    vd = model.data_preprocessor.voxelize([pts_gpu], [data["data_samples"]])
    ref_outs = model.pts_bbox_head(model.extract_pts_feat(vd, points=[pts_gpu], batch_input_metas=[meta]))
    ref_flat = [t for group in ref_outs for t in group]
    inp_mmcv = [torch.from_numpy(a).to(DEV) for a in pad_inputs(*mmcv_vox)]
    inp_np = [torch.from_numpy(a).to(DEV) for a in pad_inputs(*np_vox)]
    wrap_mmcv = net(*inp_mmcv)
    wrap_np = net(*inp_np)
MAXABS = {
    "wrapper_vs_mmdet3d": max(float((a - b).abs().max()) for a, b in zip(wrap_mmcv, ref_flat)),
    "wrapper_numpyvox_vs_mmdet3d": max(float((a - b).abs().max()) for a, b in zip(wrap_np, ref_flat)),
}
print({n: tuple(t.shape) for n, t in zip(OUT_NAMES, wrap_mmcv)})
print("max |diff|:", MAXABS)

# %% [markdown]
# ## 5. ONNX 내보내기
# `torch.onnx.export`(TorchScript 기반, opset `OPSET`)로 정적 크기(P = config `edge.max_pillars`) 그래프를 만든다.
# 실패하거나 경고(TracerWarning 등)가 나면 **숨기지 않고** 그대로 출력·저장한다.

# %%
import onnx

ONNX_PATH = OUT / f"pointpillars_nus_fpn_P{NP}_op{OPSET}.onnx"
EXPORT = {"path": str(ONNX_PATH), "error": None, "warnings": [], "ops": {}}
if not is_cached(ONNX_PATH, FORCE_RERUN):
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            torch.onnx.export(net, tuple(inp_np), str(ONNX_PATH), opset_version=OPSET, input_names=IN_NAMES,
                              output_names=OUT_NAMES, do_constant_folding=True)
        except Exception:  # noqa: BLE001  (report any export failure instead of hiding it)
            EXPORT["error"] = traceback.format_exc()
    EXPORT["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for w in wlist})
if EXPORT["error"]:
    print("EXPORT FAILED:\n", EXPORT["error"])
elif ONNX_PATH.exists():
    om = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(om)
    EXPORT["ops"] = dict(Counter(n.op_type for n in om.graph.node))
    print(f"exported {ONNX_PATH.name} ({ONNX_PATH.stat().st_size / 2**20:.1f} MB)")
    print("op types:", EXPORT["ops"])
    flagged = [op for op in EXPORT["ops"] if op in ("ScatterND", "ScatterElements", "NonZero", "Loop", "If")]
    print("ops to watch for TensorRT:", flagged)
for w in EXPORT["warnings"]:
    print("WARN", w)

# %% [markdown]
# ## 6. ONNX Runtime 실행과 PyTorch 비교
# 같은 입력(numpy voxelize 결과를 P로 패딩)으로 ORT 출력과 PyTorch 래퍼 출력의 최대 절대 오차를 본다.
# 이 env의 onnxruntime은 CPU 휠이라 CUDA EP가 없으면 CPU만 잰다(GPU용은 `onnxruntime-gpu`, 버전은 TODO(verify)).

# %%
import onnxruntime as ort

ORT = {}
ort_inputs = dict(zip(IN_NAMES, pad_inputs(*np_vox)))
avail = ort.get_available_providers()
print("ORT providers:", avail)
for ep in ["CPUExecutionProvider"] + (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else []):
    if EXPORT["error"] or not ONNX_PATH.exists():
        break
    try:
        sess = ort.InferenceSession(str(ONNX_PATH), providers=[ep])
        outs = sess.run(OUT_NAMES, ort_inputs)
        diff = max(float(np.abs(o - t.cpu().numpy()).max()) for o, t in zip(outs, wrap_np))
        ORT[ep] = {"session": sess, "maxabs_vs_torch": diff}
        print(f"{ep}: max |ORT - torch| = {diff:.3e}")
    except Exception:  # noqa: BLE001  (unsupported op / EP errors are reported)
        ORT[ep] = {"session": None, "error": traceback.format_exc()}
        print(f"{ep} FAILED:\n{ORT[ep]['error']}")

# %% [markdown]
# ## 7. TensorRT FP32 / FP16 (선택)
# - `tensorrt` 파이썬 패키지가 있으면 ONNX 파서로 엔진을 만들고(FP32, FP16), 같은 입력으로 출력 오차와 지연시간을 잰다.
# - 패키지가 없고 `trtexec`만 있으면 그것으로 빌드·시간 측정만 한다(입력은 실제 기둥 데이터 `.bin`을 `--loadInputs`로 넣음.
#   무작위 입력은 scatter 인덱스가 범위를 벗어날 수 있음).
# - 둘 다 없으면 설치 안내만 출력한다. mmdeploy 문서는 ScatterND 관련 수정 때문에 TensorRT ≥ 8.6을 요구한다.

# %%
TRT = {}
try:
    import tensorrt as trt
except ImportError:
    trt = None
TRTEXEC = shutil.which("trtexec")
INPUT_BINS = {}
for name, arr in ort_inputs.items():                                  # raw inputs for trtexec / Jetson
    INPUT_BINS[name] = OUT / f"input_{name}.bin"
    arr.tofile(INPUT_BINS[name])

if trt is None and TRTEXEC is None:
    print("TensorRT not found. Install, then rerun this section:\n"
          "  desktop: pip install tensorrt==<x.y>  # TODO(verify): wheel/version for CUDA 12.1 + torch 2.1.2; need >= 8.6\n"
          "  Jetson : JetPack ships TensorRT and /usr/src/tensorrt/bin/trtexec  # TODO(verify): version per JetPack")


def trt_build(onnx_path: Path, fp16: bool, workspace_gb: int = 2) -> bytes:
    """Build a serialized engine with the TensorRT Python API (8.6 / 10.x)."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    major = int(trt.__version__.split(".")[0])
    flags = 0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("build_serialized_network returned None (see TensorRT log above)")
    return bytes(blob)


class TrtRunner:
    """Run an engine with torch CUDA tensors (name-based I/O API, TensorRT >= 8.5).

    TODO(verify): written against the TensorRT 8.6/10 Python API docs; not executed here (no TensorRT in 3dv).
    """

    def __init__(self, blob: bytes):
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(blob)
        self.ctx = self.engine.create_execution_context()
        to_torch = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                self.outputs[n] = torch.empty(tuple(self.engine.get_tensor_shape(n)),
                                              dtype=to_torch[self.engine.get_tensor_dtype(n)], device=DEV)
                self.ctx.set_tensor_address(n, self.outputs[n].data_ptr())

    def __call__(self, inputs: dict) -> list:
        for n, t in inputs.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return [self.outputs[n] for n in OUT_NAMES]


def trtexec_time(onnx_path: Path, fp16: bool) -> dict:
    """Fallback: build + time with trtexec. TODO(verify): output line format of your TensorRT version."""
    import re
    import subprocess

    engine = onnx_path.with_suffix(f".{'fp16' if fp16 else 'fp32'}.engine")
    load = ",".join(f"'{n}':{p}" for n, p in INPUT_BINS.items())
    cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={engine}", f"--loadInputs={load}",
           f"--iterations={P['iters']}", "--warmUp=500"] + (["--fp16"] if fp16 else [])
    log = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms", log)
    return {"cmd": " ".join(cmd), "median_ms": float(m.group(4)) if m else None, "log_tail": log[-2000:]}


trt_inputs = {n: torch.from_numpy(a).to(DEV) for n, a in ort_inputs.items()}
for prec in ("fp32", "fp16"):
    if EXPORT["error"] or not ONNX_PATH.exists():
        break
    if trt is not None:
        eng = ONNX_PATH.with_suffix(f".{prec}.engine")
        try:
            if not is_cached(eng, FORCE_RERUN):
                eng.write_bytes(trt_build(ONNX_PATH, fp16=prec == "fp16"))
            runner = TrtRunner(eng.read_bytes())
            outs = runner(trt_inputs)
            torch.cuda.synchronize()
            diff = max(float((o.float() - t).abs().max()) for o, t in zip(outs, wrap_np))
            TRT[prec] = {"runner": runner, "maxabs_vs_torch": diff, "engine": str(eng)}
            print(f"TensorRT {trt.__version__} {prec}: max |TRT - torch| = {diff:.3e}")
        except Exception:  # noqa: BLE001  (parser / builder errors are reported)
            TRT[prec] = {"runner": None, "error": traceback.format_exc()}
            print(f"TensorRT {prec} FAILED:\n{TRT[prec]['error']}")
    elif TRTEXEC:
        TRT[prec] = {"runner": None, **trtexec_time(ONNX_PATH, prec == "fp16")}
        print(prec, TRT[prec]["median_ms"], "ms (trtexec GPU compute)")

# %% [markdown]
# ## 8. 단계별 지연시간
# 같은 샘플을 `warmup`회 돌린 뒤 `iters`회 재서 중앙값(ms)을 쓴다. GPU 단계는 매 회 `cuda.synchronize()` 후 잰다.
#
# | 단계 | 측정 내용 |
# |---|---|
# | voxelize | numpy(CPU), torch 벡터화(GPU), mmcv 커널(GPU, 기준) |
# | H2D | GPU voxelize 경로: 원시 점 `(N,4)` 복사(일반/고정 메모리) · CPU voxelize 경로: 패딩된 `voxels(P,64,4)`+`coors`+`num_points` 복사 |
# | network | torch FP32(래퍼), ORT CPU(반복 `ORT_CPU_ITERS`회), ORT CUDA(있으면), TRT FP32/FP16(있으면) |
# | decode+NMS | `pts_bbox_head.predict_by_feat`(GPU, 앵커 디코드 + 회전 NMS) |
#
# CPU voxelize 경로는 P=40000 기준 약 41 MB(`40000×64×4×4 B`)를 매번 GPU로 보내야 한다는 점이 GPU 경로와의 구조적 차이다.

# %%
def bench(fn, warmup: int = P["warmup"], iters: int = P["iters"]) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


LAT = {}
pinned = points.pin_memory()
vox_np_padded = pad_inputs(*np_vox)
vox_pinned = [torch.from_numpy(a).pin_memory() for a in vox_np_padded]
gpu_inputs = [torch.from_numpy(a).to(DEV) for a in vox_np_padded]
with Measure() as m_bench, torch.no_grad():
    LAT["vox_numpy_cpu"] = bench(lambda: voxelize_numpy(pts_np, VOX["voxel_size"], VOX["pc_range"],
                                                        VOX["max_points"], VOX["max_voxels"]))
    LAT["vox_torch_gpu"] = bench(lambda: voxelize_torch(pts_gpu, VOX["voxel_size"], VOX["pc_range"],
                                                        VOX["max_points"], VOX["max_voxels"], "first"))
    LAT["vox_torch_gpu_sorted"] = bench(lambda: voxelize_torch(pts_gpu, VOX["voxel_size"], VOX["pc_range"],
                                                               VOX["max_points"], VOX["max_voxels"], "sorted"))
    LAT["vox_mmcv_gpu"] = bench(lambda: vl(pts_gpu))
    LAT["h2d_points"] = bench(lambda: points.to(DEV))
    LAT["h2d_points_pinned"] = bench(lambda: pinned.to(DEV, non_blocking=True))
    LAT["h2d_voxels_padded"] = bench(lambda: [torch.from_numpy(a).to(DEV) for a in vox_np_padded])
    LAT["h2d_voxels_padded_pinned"] = bench(lambda: [t.to(DEV, non_blocking=True) for t in vox_pinned])
    LAT["net_torch_fp32"] = bench(lambda: net(*gpu_inputs))
    for ep, r in ORT.items():
        if r.get("session") is not None:
            it = ORT_CPU_ITERS if ep == "CPUExecutionProvider" else P["iters"]
            LAT[f"net_ort_{ep.replace('ExecutionProvider', '').lower()}"] = bench(
                lambda s=r["session"]: s.run(OUT_NAMES, ort_inputs), warmup=2, iters=it)
    for prec, r in TRT.items():
        if r.get("runner") is not None:
            LAT[f"net_trt_{prec}"] = bench(lambda rr=r["runner"]: rr(trt_inputs))
        elif r.get("median_ms") is not None:
            LAT[f"net_trt_{prec}"] = r["median_ms"]                     # trtexec GPU compute time
    cls_l, box_l, dir_l = split_outs(net(*gpu_inputs))
    LAT["decode_nms"] = bench(lambda: model.pts_bbox_head.predict_by_feat(cls_l, box_l, dir_l, batch_input_metas=[meta]))
    LAT["official_test_step"] = bench(lambda: model.test_step(pseudo_collate([data])))   # incl. its voxelize, no I/O

for k, v in LAT.items():
    print(f"{k:28s} {v:9.2f} ms")
print(f"bench wall {m_bench.seconds:.1f} s, peak VRAM {m_bench.peak_vram_mb:.0f} MB")

# %% [markdown]
# ### 파이프라인 조합 (단계 중앙값의 합, 추정)
# 단계별 중앙값을 더한 값이라 실제 연속 실행(겹침, 캐시 효과)과 다를 수 있다.

# %%
import pandas as pd

net_keys = [k for k in LAT if k.startswith("net_")]
PIPE = {}
for nk in net_keys:
    if "ort_cpu" in nk:
        continue                                                     # CPU network is not an edge-GPU option
    PIPE[f"cpu_vox + {nk}"] = {"voxelize": LAT["vox_numpy_cpu"], "h2d": LAT["h2d_voxels_padded_pinned"],
                               "network": LAT[nk], "decode_nms": LAT["decode_nms"]}
    PIPE[f"gpu_vox + {nk}"] = {"voxelize": LAT["vox_torch_gpu"], "h2d": LAT["h2d_points_pinned"],
                               "network": LAT[nk], "decode_nms": LAT["decode_nms"]}
PIPE_DF = pd.DataFrame(PIPE).T
PIPE_DF["total_ms"] = PIPE_DF.sum(1)
PIPE_DF.round(2)

# %% [markdown]
# ## 9. 끝단 정확도 (eval4 첫 장면)
# 분할 파이프라인(numpy voxelize → P 패딩 → ORT/TRT → `predict_by_feat`)과 공식 경로(`model.test_step`)를 같은 장면의
# 모든 키프레임에 돌리고, `NuScenesMetric(format_only=True)`로 nuScenes json을 만든 뒤 bench3d `evaluate_detection`으로
# 그 장면만 평가한다(devkit 커스텀 분할 `edge_scene`을 `splits.json`에 추가). 장면 하나라 절대값보다 **백엔드 간 차이**를 본다.
# ORT CPU는 느리므로 GPU EP나 TensorRT가 없으면 ORT CPU로 돈다(수 분).

# %%
from bench3d.metrics.detection import evaluate_detection, write_custom_splits
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmengine.structures import InstanceData

E2E = {}
if RUN_E2E:
    write_custom_splits(DATA_ROOT, VERSION, {"edge_scene": [E2E_SCENE]})
    idxs = [token_to_idx[t] for t in scene_tokens]

    def run_split(net_fn):
        """numpy voxelize + padded network + head decode -> list of Det3DDataSample."""
        out = []
        for i in idxs:
            d = dataset[i]
            v = pad_inputs(*voxelize_numpy(d["inputs"]["points"].numpy(), VOX["voxel_size"], VOX["pc_range"],
                                           VOX["max_points"], VOX["max_voxels"]))
            c, b, r = split_outs([o.float() for o in net_fn(v)])
            res = model.pts_bbox_head.predict_by_feat(c, b, r, batch_input_metas=[d["data_samples"].metainfo])
            out += model.add_pred_to_datasample([d["data_samples"]], res)
        return out

    backends = {"torch_official": lambda: [s for i in idxs for s in model.test_step(pseudo_collate([dataset[i]]))]}
    ort_ep = "CUDAExecutionProvider" if ORT.get("CUDAExecutionProvider", {}).get("session") else "CPUExecutionProvider"
    if ORT.get(ort_ep, {}).get("session") is not None:
        s_ort = ORT[ort_ep]["session"]
        backends[f"numpy_ort_{ort_ep[:-17].lower()}"] = lambda: run_split(
            lambda v: [torch.from_numpy(o).to(DEV) for o in s_ort.run(OUT_NAMES, dict(zip(IN_NAMES, v)))])
    for prec, r in TRT.items():
        if r.get("runner") is not None:
            backends[f"numpy_trt_{prec}"] = lambda rr=r["runner"]: run_split(
                lambda v: [o.clone() for o in rr({n: torch.from_numpy(a).to(DEV) for n, a in zip(IN_NAMES, v)})])

    for name, fn in backends.items():
        cache = OUT / f"e2e_{name}.json"
        if is_cached(cache, FORCE_RERUN):
            E2E[name] = json.loads(cache.read_text())
            continue
        with free_cuda(), torch.no_grad(), Measure() as m_e2e:
            samples = fn()
        metric = NuScenesMetric(data_root=str(DATA_ROOT) + "/", ann_file=str(DATA_ROOT / EVAL4_PKL), metric="bbox",
                                format_only=True, jsonfile_prefix=str(OUT / f"json_e2e_{name}"))
        metric.dataset_meta = dataset.metainfo
        for s in samples:
            s.pred_instances = s.get("pred_instances", InstanceData())
        metric.process({}, [s.to_dict() for s in samples])
        metric.compute_metrics(metric.results)
        r = evaluate_detection(nusc, mt.result_json(OUT / f"json_e2e_{name}"), "edge_scene", OUT / f"devkit_e2e_{name}")
        E2E[name] = {"mAP": r["mAP"], "NDS": r["NDS"], "n_gt_boxes": r["n_gt_boxes"], "seconds": m_e2e.seconds,
                     "per_class_ap": r["per_class_ap"]}
        cache.write_text(json.dumps(E2E[name], indent=1))
    for name, r in E2E.items():
        print(f"{name:24s} mAP {r['mAP']:.4f} NDS {r['NDS']:.4f} ({r['seconds']:.0f} s for {len(idxs)} samples)")

# %% [markdown]
# ## 10. 결과 저장

# %%
metrics = {f"lat_{k}_ms": v for k, v in LAT.items()}
metrics.update({"n_points": N_POINTS, "n_voxels": N_VOXELS, "max_pillars": NP,
                "maxabs_wrapper_vs_mmdet3d": MAXABS["wrapper_vs_mmdet3d"],
                "maxabs_wrapper_numpyvox_vs_mmdet3d": MAXABS["wrapper_numpyvox_vs_mmdet3d"]})
metrics.update({f"maxabs_ort_{ep[:-17].lower()}_vs_torch": r["maxabs_vs_torch"] for ep, r in ORT.items()
                if "maxabs_vs_torch" in r})
metrics.update({f"maxabs_trt_{p}_vs_torch": r["maxabs_vs_torch"] for p, r in TRT.items() if "maxabs_vs_torch" in r})
for name, r in E2E.items():
    metrics[f"e2e_mAP_{name}"], metrics[f"e2e_NDS_{name}"] = r["mAP"], r["NDS"]
extra = {
    "bench_sample_token": dataset.get_data_info(BENCH_IDX)["token"], "e2e_scene": E2E_SCENE, "voxel_cfg": VOX,
    "voxel_compare": VOX_CMP, "onnx": {k: v for k, v in EXPORT.items()}, "opset": OPSET,
    "ort_providers": avail, "ort_errors": {ep: r["error"] for ep, r in ORT.items() if "error" in r},
    "tensorrt": {p: {k: v for k, v in r.items() if k != "runner"} for p, r in TRT.items()},
    "tensorrt_version": getattr(trt, "__version__", None), "trtexec": TRTEXEC,
    "pipelines_ms": PIPE_DF.round(3).to_dict(orient="index"), "e2e": E2E, "gpu": ENV.get("gpu"),
}
save_summary(OUT, part="outdoor", notebook=NOTEBOOK, profile=cfg["profile"], model=MODEL_ID,
             variant=f"edge/P{NP}/{ENV.get('gpu')}", metrics=metrics, runtime_s=m_bench.seconds,
             peak_vram_mb=m_bench.peak_vram_mb, checkpoint=M["checkpoint"], packages=ENV["packages"],
             seed=cfg["seed"], extra=extra)

# %% [markdown]
# ## 11. 시각화: 단계별 지연시간

# %%
import matplotlib.pyplot as plt
from bench3d.viz import PALETTE

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
vk = [k for k in LAT if k.startswith("vox_")]
axes[0].barh(vk, [LAT[k] for k in vk], color=PALETTE[0])
axes[0].set_xscale("log")
axes[0].set_xlabel("median ms (log)")
axes[0].set_title("voxelization", fontsize=9)
left = np.zeros(len(PIPE_DF))
for i, stage in enumerate(["voxelize", "h2d", "network", "decode_nms"]):
    axes[1].barh(PIPE_DF.index, PIPE_DF[stage], left=left, color=PALETTE[i], label=stage)
    left += PIPE_DF[stage].to_numpy()
axes[1].set_xlabel("ms (sum of stage medians)")
axes[1].legend(fontsize=8)
axes[1].set_title("pipelines", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Jetson Orin Nano 재현 절차
# 이 노트북의 ONNX와 입력 `.bin`(결과 폴더의 `input_voxels.bin`, `input_num_points.bin`, `input_coors.bin`)을 그대로 옮겨
# 같은 단계를 잰다. 버전 관련 내용은 모두 TODO(verify)다.
#
# 1. **JetPack 설치**: Orin Nano용 JetPack(TensorRT, CUDA, cuDNN 포함). TODO(verify): JetPack 버전별 TensorRT 버전
#    (ScatterElements/ScatterND 처리 때문에 TensorRT ≥ 8.6 권장 — mmdeploy 문서 기준).
# 2. **전력 모드 고정**: `sudo nvpmodel -m <mode>`, `sudo jetson_clocks`. TODO(verify): Orin Nano 모드 번호(최대 전력 모드 이름).
#    측정마다 모드를 기록한다.
# 3. **파일 복사**: `pointpillars_nus_fpn_P40000_op13.onnx`, `input_*.bin`.
# 4. **엔진 빌드 (FP16 / FP32)**:
#    ```bash
#    /usr/src/tensorrt/bin/trtexec --onnx=pointpillars_nus_fpn_P40000_op13.onnx \
#        --saveEngine=pp_fp16.engine --fp16 --memPoolSize=workspace:2048   # TODO(verify): flag name per TRT version
#    /usr/src/tensorrt/bin/trtexec --onnx=pointpillars_nus_fpn_P40000_op13.onnx --saveEngine=pp_fp32.engine
#    ```
# 5. **신경망 시간**: 무작위 입력은 scatter 인덱스가 범위를 벗어날 수 있어 실제 입력을 넣는다.
#    ```bash
#    /usr/src/tensorrt/bin/trtexec --loadEngine=pp_fp16.engine --iterations=200 --warmUp=500 \
#        --loadInputs='voxels':input_voxels.bin,'num_points':input_num_points.bin,'coors':input_coors.bin
#    ```
#    출력의 `GPU Compute Time`(median)을 network 단계로 기록한다. `--useCudaGraph`도 따로 재 본다.
# 6. **나머지 단계**:
#    - voxelize CPU: 이 노트북의 `voxelize_numpy`를 단독 스크립트로 옮겨 ARM CPU에서 같은 `.bin` 점군으로 잰다.
#    - voxelize GPU: Jetson용 PyTorch(TODO(verify): NVIDIA 배포 휠/버전)로 `voxelize_torch`, 또는 CUDA-PointPillars식 CUDA 커널.
#    - H2D: Orin은 CPU·GPU가 같은 DRAM을 쓰므로(통합 메모리) 복사 비용이 데스크톱과 다르다. pinned/zero-copy를 따로 잰다.
#    - decode+NMS: mmdet3d 전체를 Jetson에 설치하는 것은 무겁다. 후속 과제: 앵커 디코드 + 회전 NMS를 torch/CUDA로 이식.
# 7. **참고 기준**: NVIDIA CUDA-PointPillars는 Orin에서 voxelization 0.18 ms, 디코드·NMS 1.79 ms(FP16)를 보고하지만
#    KITTI/OpenPCDet 모델(입력 범위·격자·클래스 수가 다름)이라 이 nuScenes 모델과 직접 비교할 수 없다.
# 8. **추가 최적화 후보**: INT8 PTQ(eval4 밖 장면으로 보정), `max_pillars` 축소, sweep 축소(10 → 1~3, 05의 a1 결과로 정확도 손실 확인),
#    CUDA Graph, pinned/zero-copy 메모리.

# %% [markdown]
# ## 13. 해석
# - **voxelization 일치**: > (실행 후 작성) numpy·torch(first)가 mmcv와 기둥 수·점 수·값이 같은지, sorted 순서의 영향
# - **ONNX 내보내기**: > (실행 후 작성) 경고·실패 연산, ScatterElements가 TensorRT에서 문제없는지
# - **수치 오차**: > (실행 후 작성) ORT / TRT FP32 / TRT FP16의 max |diff|, FP16 끝단 mAP 변화
# - **병목**: > (실행 후 작성) CPU voxelize vs 신경망 비율, 패딩 기둥 H2D(약 41 MB) 비용, NMS 비중
# - **Jetson 예측**: > (실행 후 작성) 데스크톱 결과로부터 Orin에서 어떤 단계가 병목일지, 재현 후 비교
