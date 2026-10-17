"""mmdetection3d helpers for the outdoor detection notebooks (05, 08). 3dv env only.

What lives here (all verified against mmdet3d 1.4.0 / mmengine 0.10.7 installed in 3dv):
- repo path handling for the pinned mmdetection3d clone (configs, projects/BEVFusion)
- test-config surgery: data root / info pkl / points prefix / sweep count / FOV crop / format-only evaluator
- `CropPointsToCameraFOV`: pipeline transform registered into mmdet3d TRANSFORMS
- `LoopTimer`: mmengine hook that times the test loop (model-only and incl. data loading)
- `run_test`: Runner.from_cfg(cfg).test() with Measure + LoopTimer

Sources (paths relative to site-packages or the pinned repo):
- mmdet3d/datasets/det3d_dataset.py  parse_data_info: lidar path = data_prefix['pts'] + basename,
  sweeps -> data_prefix['sweeps'] + basename; images[cam] keep 'lidar2cam' / 'cam2img'.
- mmengine/dataset/base_dataset.py  _join_prefix: absolute data_prefix values are kept as is.
- mmdet3d/datasets/transforms/loading.py  LoadPointsFromMultiSweeps: column 4 of the keyframe is set to 0
  (time lag), sweeps get `ts - sweep_ts`; with sweeps_num=0 no sweep is loaded (choices = arange(0)).
- mmdet3d/evaluation/metrics/nuscenes_metric.py  NuScenesMetric(format_only, jsonfile_prefix) writes
  <jsonfile_prefix>/pred_instances_3d/results_nusc.json and indexes the ann_file data_list by `sample_idx`.
- mmengine/runner/runner.py  Runner.test() = build_test_loop + load_or_resume (no init_weights, so
  `init_cfg=Pretrained` backbones such as BEVFusion's Swin are not downloaded at test time).
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmengine.hooks import Hook

RESULT_JSON = "pred_instances_3d/results_nusc.json"  # relative to jsonfile_prefix (NuScenesMetric.format_results)
NUSCENES_IMAGE_SIZE = (1600, 900)  # (W, H) of all nuScenes cameras; checked against the devkit in notebook 05


def add_repo_to_path(repo: str | Path) -> Path:
    """Make `projects.*` (BEVFusion) and `tools.*` importable from the pinned clone.

    Appended (not prepended) so the installed mmdet3d package keeps precedence over the clone's source tree.
    """
    repo = Path(repo).resolve()
    if not (repo / "configs").is_dir():
        raise FileNotFoundError(f"not an mmdetection3d checkout: {repo}")
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    return repo


def load_config(repo: str | Path, rel_config: str):
    """mmengine Config of an official config file (custom_imports such as BEVFusion are imported here)."""
    from mmengine.config import Config

    repo = add_repo_to_path(repo)
    return Config.fromfile(str(repo / rel_config))


# ----------------------------------------------------------------------------- pipeline surgery
def iter_transforms(pipeline):
    """Yield every transform dict, including those nested in `transforms` (e.g. MultiScaleFlipAug3D)."""
    for t in pipeline:
        yield t
        if isinstance(t, dict) and isinstance(t.get("transforms"), list):
            yield from iter_transforms(t["transforms"])


def find_transforms(pipeline, type_name: str) -> list:
    return [t for t in iter_transforms(pipeline) if isinstance(t, dict) and t.get("type") == type_name]


def get_sweeps(pipeline) -> int | None:
    ts = find_transforms(pipeline, "LoadPointsFromMultiSweeps")
    return ts[0]["sweeps_num"] if ts else None


def set_sweeps(pipeline, n: int) -> int:
    """Set `sweeps_num` of LoadPointsFromMultiSweeps. Returns the number of transforms changed.

    The transform is kept (not removed) on purpose: it zeroes column 4 (ring index -> time lag 0 for the
    keyframe) and applies `use_dim`; removing it would feed the ring index where the model expects time.
    """
    ts = find_transforms(pipeline, "LoadPointsFromMultiSweeps")
    for t in ts:
        t["sweeps_num"] = int(n)
    return len(ts)


def insert_after(pipeline: list, after_type: str, new_t: dict) -> None:
    """Insert `new_t` right after the first top-level transform of `after_type`."""
    for i, t in enumerate(pipeline):
        if t.get("type") == after_type:
            pipeline.insert(i + 1, new_t)
            return
    raise KeyError(f"{after_type} not in top-level pipeline {[t.get('type') for t in pipeline]}")


def configure_test(cfg, *, data_root: str | Path, ann_file: str, work_dir: str | Path, checkpoint: str,
                   jsonfile_prefix: str | Path | None, format_only: bool = True, pts_prefix: str | Path | None = None,
                   sweeps: int | None = None, fov_crop_cams: list[str] | None = None, seed: int = 0,
                   num_workers: int | None = None):
    """Return a copy of an official config prepared for Runner.from_cfg(cfg).test().

    data_root:   nuScenes root (absolute); `ann_file` is a file name inside it (e.g. nuscenes_infos_eval4.pkl).
    pts_prefix:  absolute folder that replaces data_prefix['pts'] (pseudo-LiDAR with the samples/LIDAR_TOP layout).
    sweeps:      None = official, 0 = keyframe only.
    fov_crop_cams: crop input points to these cameras' image frustums (after sweep loading).
    """
    cfg = copy.deepcopy(cfg)
    data_root = str(Path(data_root).resolve()) + "/"
    cfg.work_dir = str(work_dir)
    cfg.load_from = checkpoint
    cfg.resume = False
    cfg.launcher = "none"
    cfg.randomness = dict(seed=seed)  # CenterPoint/BEVFusion test pipelines sample sweeps randomly (no test_mode)

    dl = cfg.test_dataloader
    ds = dl.dataset
    assert "pipeline" in ds, f"wrapped test dataset not supported: {ds.get('type')}"
    ds.data_root = data_root
    ds.ann_file = ann_file
    if pts_prefix is not None:
        prefix = dict(ds.get("data_prefix", {}))
        prefix["pts"] = str(Path(pts_prefix).resolve())
        ds.data_prefix = prefix
    if sweeps is not None and set_sweeps(ds.pipeline, sweeps) == 0:
        raise ValueError("pipeline has no LoadPointsFromMultiSweeps (camera-only model?)")
    if fov_crop_cams:
        after = "LoadPointsFromMultiSweeps" if find_transforms(ds.pipeline, "LoadPointsFromMultiSweeps") \
            else "LoadPointsFromFile"
        insert_after(ds.pipeline, after, dict(type="CropPointsToCameraFOV", cams=list(fov_crop_cams)))
    if num_workers is not None:
        dl.num_workers = int(num_workers)
        dl.persistent_workers = int(num_workers) > 0

    ev = cfg.test_evaluator
    assert isinstance(ev, dict) and ev.get("type") == "NuScenesMetric", ev
    ev.data_root = data_root
    ev.ann_file = data_root + ann_file
    ev.format_only = format_only
    ev.jsonfile_prefix = str(jsonfile_prefix) if jsonfile_prefix is not None else None
    return cfg


# ----------------------------------------------------------------------------- FOV crop transform
@TRANSFORMS.register_module(force=True)
class CropPointsToCameraFOV(BaseTransform):
    """Keep points whose projection lands inside the image of at least one of `cams`.

    Uses the info's per-camera `lidar2cam` (4x4, keyframe LiDAR -> camera, through the global frame so ego motion
    between the two timestamps is included) and `cam2img` (3x3). Multi-sweep points are already in the keyframe
    LiDAR frame when this runs (inserted after LoadPointsFromMultiSweeps), so one matrix per camera suffices.
    Same visibility rule as bench3d.metrics.pointcloud.frustum_mask (z > min_depth, 0 <= u < W, 0 <= v < H).
    """

    def __init__(self, cams, image_size=NUSCENES_IMAGE_SIZE, min_depth: float = 1e-3):
        self.cams = list(cams)
        self.image_size = tuple(image_size)
        self.min_depth = float(min_depth)

    def transform(self, results: dict) -> dict:
        points = results["points"]
        xyz = points.coord.numpy().astype(np.float64)
        homo = np.c_[xyz, np.ones(len(xyz))]
        keep = np.zeros(len(xyz), bool)
        W, H = self.image_size
        for cam in self.cams:
            img = results["images"][cam]
            l2c = np.asarray(img["lidar2cam"], np.float64)
            K = np.asarray(img["cam2img"], np.float64)[:3, :3]
            pc = homo @ l2c.T
            ok = pc[:, 2] > self.min_depth
            uvw = pc[ok, :3] @ K.T
            uv = uvw[:, :2] / uvw[:, 2:3]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            keep[np.flatnonzero(ok)[inside]] = True
        results["points"] = points[torch.from_numpy(keep)]
        results["fov_crop_ratio"] = float(keep.mean()) if len(keep) else 0.0
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cams={self.cams}, image_size={self.image_size})"


# ----------------------------------------------------------------------------- timing
class LoopTimer(Hook):
    """Times the mmengine TestLoop.

    model_s: sum over iterations of before_test_iter -> after_test_iter (data_preprocessor + forward + decode).
    loop_s:  before_test_epoch -> last after_test_iter (adds data loading / collation; excludes json formatting).
    """

    priority = "LOWEST"

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def before_test_epoch(self, runner) -> None:
        self._sync()
        self.t0 = self.t_last = time.perf_counter()
        self.model_s, self.n_iters, self.n_items = 0.0, 0, 0

    def before_test_iter(self, runner, batch_idx, data_batch=None) -> None:
        self._sync()
        self._ti = time.perf_counter()

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        self._sync()
        self.t_last = time.perf_counter()
        self.model_s += self.t_last - self._ti
        self.n_iters += 1
        self.n_items += len(outputs) if outputs is not None else 0

    def report(self, items_per_sample: int = 1) -> dict:
        """items_per_sample: 6 for FCOS3D (one dataset item per camera image)."""
        n = self.n_items / items_per_sample
        loop_s = self.t_last - self.t0
        return {"n_samples": n, "loop_s": loop_s, "model_s": self.model_s,
                "samples_per_s": n / loop_s if loop_s > 0 else None,
                "samples_per_s_model_only": n / self.model_s if self.model_s > 0 else None}


def run_test(cfg, items_per_sample: int = 1) -> dict:
    """Runner.from_cfg(cfg).test() with wall time / peak VRAM (bench3d Measure) and loop timing."""
    from mmengine.runner import Runner

    from ..runtime import Measure

    runner = Runner.from_cfg(cfg)
    timer = LoopTimer()
    runner.register_hook(timer, priority="LOWEST")
    with Measure() as m:
        metrics = runner.test()
    out = {"metrics": dict(metrics or {}), "runtime_s": m.seconds, "peak_vram_mb": m.peak_vram_mb,
           **timer.report(items_per_sample)}
    del runner
    return out


def result_json(jsonfile_prefix: str | Path) -> Path:
    return Path(jsonfile_prefix) / RESULT_JSON
