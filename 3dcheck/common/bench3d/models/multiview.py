"""Multi-view reconstruction wrappers (DUSt3R, MASt3R, VGGT, Depth Anything 3 any-view) for the `3dv` env.

All heavy imports are inside the functions, so `import bench3d.models.multiview` is cheap.

Common output contract of every `run_<model>(model, image_paths, cfg_model, device, ...)`:
    dict(
        c2w        = (N, 4, 4) float64  camera-to-world, OpenCV axes, in the model's own (arbitrary) world frame,
        points     = (N, H, W, 3) float32 world points per view (same frame as c2w),
        conf       = (N, H, W) float32  the model's own confidence (scale differs between models),
        mask       = (N, H, W) bool     the model's own default confidence filter (see `conf_rule`),
        intrinsics = (N, 3, 3) float64  at the processed resolution (H, W),
        image_hw   = (H, W)             processed resolution,
        images     = (N, H, W, 3) uint8 processed input images (for point colors),
        meta       = dict(conf_rule=str, point_source=str, ...),
    )
Views keep the order of `image_paths`. `cfg_model` is `cfg["models"]["multiview"][<model_id>]`.

Sources (pinned commits in envs/third_party.lock):
  DUSt3R  third_party/mast3r/dust3r: dust3r/demo.py (get_reconstructed_scene, get_3D_model_from_scene),
          dust3r/inference.py, dust3r/image_pairs.py, dust3r/utils/image.py,
          dust3r/cloud_opt/{__init__,base_opt,optimizer}.py
  MASt3R  third_party/mast3r: mast3r/demo.py, mast3r/cloud_opt/sparse_ga.py, mast3r/image_pairs.py, mast3r/model.py
  VGGT    third_party/vggt: README.md ("Detailed Usage"), demo_gradio.py, visual_util.py,
          vggt/utils/{load_fn,pose_enc,geometry}.py
  DA3     third_party/Depth-Anything-3: README.md, src/depth_anything_3/api.py, specs.py, model/da3.py,
          utils/export/glb.py (confidence threshold and unprojection)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

MODEL_IDS = ("dust3r", "mast3r", "vggt", "da3_anyview")


# ----------------------------------------------------------------------------- helpers
def _to_uint8(imgs) -> np.ndarray:
    """List/array of (H,W,3) float images in [0,1] -> (N,H,W,3) uint8."""
    arr = np.stack([np.asarray(x, dtype=np.float32) for x in imgs])
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _as_4x4(m: np.ndarray) -> np.ndarray:
    """(N,3,4) or (N,4,4) -> (N,4,4)."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape[-2:] == (4, 4):
        return m
    if m.shape[-2:] != (3, 4):
        raise ValueError(f"expected (N,3,4) or (N,4,4), got {m.shape}")
    out = np.tile(np.eye(4), (m.shape[0], 1, 1))
    out[:, :3, :4] = m
    return out


def unproject_depth(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """z-depth (N,H,W) + intrinsics (N,3,3) + c2w (N,4,4) -> world points (N,H,W,3)."""
    N, H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    pix = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3).astype(np.float64)
    out = np.empty((N, H, W, 3), np.float32)
    for i in range(N):
        rays = pix @ np.linalg.inv(K[i]).T
        pc = rays * depth[i].reshape(-1, 1)
        out[i] = (pc @ c2w[i, :3, :3].T + c2w[i, :3, 3]).reshape(H, W, 3)
    return out


def _paths(image_paths) -> list[str]:
    paths = [str(p) for p in image_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate image paths")  # MASt3R keys its cache by path
    return paths


# ----------------------------------------------------------------------------- DUSt3R
def load_dust3r(cfg_model: dict, device):
    from dust3r.model import AsymmetricCroCo3DStereo  # dust3r/demo.py

    return AsymmetricCroCo3DStereo.from_pretrained(cfg_model["hf_repo"]).to(device)


def run_dust3r(model, image_paths, cfg_model: dict, device, niter: int) -> dict:
    """dust3r/demo.py get_reconstructed_scene with PointCloudOptimizer (N > 2)."""
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    paths = _paths(image_paths)
    if len(paths) < 3:
        raise ValueError("PointCloudOptimizer needs >= 3 images (demo uses PairViewer for 2)")
    imgs = load_images(paths, size=cfg_model["image_size"], verbose=False, patch_size=model.patch_size)
    pairs = make_pairs(imgs, scene_graph=cfg_model["scene_graph"], prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    ga = cfg_model["global_alignment"]
    scene.compute_global_alignment(init=ga["init"], niter=int(niter), schedule=ga["schedule"], lr=ga["lr"])

    c2w = to_numpy(scene.get_im_poses()).astype(np.float64)
    pts = np.stack(to_numpy(scene.get_pts3d())).astype(np.float32)
    conf = np.stack(to_numpy([c for c in scene.im_conf])).astype(np.float32)
    # demo.get_3D_model_from_scene: min_conf_thr is passed through conf_trf ('log') and then compared with the
    # raw per-image confidence in get_masks(). Reproduced as-is.
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(cfg_model["min_conf_thr"])))
    mask = np.stack(to_numpy(scene.get_masks())).astype(bool)
    K = to_numpy(scene.get_intrinsics()).astype(np.float64)
    out = dict(c2w=c2w, points=pts, conf=conf, mask=mask, intrinsics=K, image_hw=tuple(pts.shape[1:3]),
               images=_to_uint8(scene.imgs),
               meta=dict(conf_rule=f"raw im_conf > log({cfg_model['min_conf_thr']}) = {scene.min_conf_thr:.4f} "
                                   "(dust3r/demo.py get_3D_model_from_scene)",
                         point_source="global-alignment depthmaps -> get_pts3d()", n_pairs=len(pairs),
                         niter=int(niter)))
    del scene, output, pairs
    return out


# ----------------------------------------------------------------------------- MASt3R
def load_mast3r(cfg_model: dict, device):
    from mast3r.model import AsymmetricMASt3R  # mast3r/demo.py (top-level demo.py)

    return AsymmetricMASt3R.from_pretrained(cfg_model["hf_repo"]).to(device)


def run_mast3r(model, image_paths, cfg_model: dict, device, niter1: int, niter2: int, cache_dir,
               keep_cache: bool = False) -> dict:
    """mast3r/demo.py get_reconstructed_scene (sparse_global_alignment) + dense points via get_dense_pts3d.

    cache_dir must be unique per image set: sparse_ga caches canonical views keyed only by image path.
    """
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs

    paths = _paths(image_paths)
    sg = cfg_model["sparse_ga"]
    imgs = load_images(paths, size=cfg_model["image_size"], verbose=False)
    pairs = make_pairs(imgs, scene_graph=cfg_model["scene_graph"], prefilter=None, symmetrize=True)
    if sg["optim_level"] == "coarse":
        niter2 = 0
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scene = sparse_global_alignment(paths, pairs, str(cache_dir), model, lr1=sg["lr1"], niter1=int(niter1),
                                    lr2=sg["lr2"], niter2=int(niter2), device=device,
                                    opt_depth="depth" in sg["optim_level"],
                                    shared_intrinsics=sg["shared_intrinsics"],
                                    matching_conf_thr=sg["matching_conf_thr"])
    # demo defaults: clean_depth=True, min_conf_thr=1.5 (mast3r/demo.py main_demo sliders)
    clean_depth = bool(cfg_model.get("clean_depth", True))
    min_conf_thr = float(cfg_model.get("min_conf_thr", 1.5))
    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    confs = [to_numpy(c).astype(np.float32) for c in confs]
    H, W = confs[0].shape
    pts = np.stack([to_numpy(p).reshape(H, W, 3) for p in pts3d]).astype(np.float32)
    conf = np.stack(confs)
    c2w = to_numpy(scene.get_im_poses()).astype(np.float64)
    K = np.stack([to_numpy(k) for k in scene.intrinsics]).astype(np.float64)
    out = dict(c2w=c2w, points=pts, conf=conf, mask=conf > min_conf_thr, intrinsics=K, image_hw=(H, W),
               images=_to_uint8(scene.imgs),
               meta=dict(conf_rule=f"dense conf > {min_conf_thr} after clean_depth={clean_depth} "
                                   "(mast3r/demo.py get_3D_model_from_scene)",
                         point_source="SparseGA.get_dense_pts3d()", n_pairs=len(pairs),
                         niter1=int(niter1), niter2=int(niter2), optim_level=sg["optim_level"]))
    del scene, pairs
    if not keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)  # forward/canonical caches can reach ~GB per window
    return out


# ----------------------------------------------------------------------------- VGGT
def load_vggt(cfg_model: dict, device):
    from vggt.models.vggt import VGGT  # README Quick Start

    return VGGT.from_pretrained(cfg_model["hf_repo"]).to(device).eval()


def run_vggt(model, image_paths, cfg_model: dict, device) -> dict:
    """README Quick Start + demo_gradio.run_model. Points from depth + camera unprojection (README: 'usually leads to
    more accurate 3D points than point map branch'); point_source='pointmap' uses the point head instead."""
    import torch
    from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = _paths(image_paths)
    images = load_and_preprocess_images(paths, mode=cfg_model["preprocess_mode"]).to(device)
    on_cuda = torch.device(device).type == "cuda"
    dtype = torch.bfloat16 if on_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=on_cuda):
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    w2c = extrinsic[0].float().cpu().numpy()  # (S,3,4) camera-from-world, OpenCV
    K = intrinsic[0].float().cpu().numpy().astype(np.float64)
    c2w = closed_form_inverse_se3(w2c).astype(np.float64)
    source = cfg_model.get("point_source", "depth")
    if source == "depth":
        depth = pred["depth"][0].float().cpu().numpy()  # (S,H,W,1)
        pts = unproject_depth_map_to_point_map(depth, w2c, K).astype(np.float32)
        conf = pred["depth_conf"][0].float().cpu().numpy()
    elif source == "pointmap":
        pts = pred["world_points"][0].float().cpu().numpy()
        conf = pred["world_points_conf"][0].float().cpu().numpy()
    else:
        raise ValueError(source)
    # visual_util.predictions_to_glb: drop the lowest conf_thres % over all frames (demo default 50) and conf <= 1e-5
    pct = float(cfg_model.get("conf_percentile", 50.0))
    thr = float(np.percentile(conf, pct)) if pct > 0 else 0.0
    mask = (conf >= thr) & (conf > 1e-5)
    imgs = images.permute(0, 2, 3, 1).float().cpu().numpy()
    del pred, images
    return dict(c2w=c2w, points=pts, conf=conf.astype(np.float32), mask=mask, intrinsics=K,
                image_hw=tuple(pts.shape[1:3]), images=_to_uint8(imgs),
                meta=dict(conf_rule=f"{source} conf >= {pct:g}th percentile over the window ({thr:.4f}) and > 1e-5 "
                                    "(vggt/visual_util.py)",
                          point_source=source, autocast_dtype=str(dtype)))


# ----------------------------------------------------------------------------- Depth Anything 3 (any-view)
def load_da3(cfg_model: dict, device):
    from depth_anything_3.api import DepthAnything3  # README Basic Usage

    return DepthAnything3.from_pretrained(cfg_model["hf_repo"]).to(device=device)


def run_da3(model, image_paths, cfg_model: dict, device) -> dict:
    """Pose-free any-view inference (no extrinsics/intrinsics passed). prediction.extrinsics is w2c: README says
    (N,3,4), specs.Prediction says (N,4,4) -> both accepted. Points by unprojecting depth with the returned
    intrinsics at the processed resolution (as utils/export/glb.py does)."""
    paths = _paths(image_paths)
    pred = model.inference(paths, process_res=cfg_model["process_res"])
    w2c = _as_4x4(pred.extrinsics)
    c2w = np.linalg.inv(w2c)
    K = np.asarray(pred.intrinsics, dtype=np.float64)
    depth = np.asarray(pred.depth, dtype=np.float32)
    pts = unproject_depth(depth, K, c2w)
    conf = np.asarray(pred.conf, dtype=np.float32) if pred.conf is not None else np.ones_like(depth)
    # utils/export/glb.py get_conf_thresh: clamp(conf_thresh=1.05, p40, p90) over all pixels (no sky head in DA3-LARGE)
    defaults = (("conf_thresh", 1.05), ("conf_thresh_percentile", 40.0), ("ensure_thresh_percentile", 90.0))
    base, lo_pct, hi_pct = (float(cfg_model.get(k, d)) for k, d in defaults)
    thr = min(max(base, float(np.percentile(conf, lo_pct))), float(np.percentile(conf, hi_pct)))
    mask = np.isfinite(depth) & (depth > 0) & (conf >= thr)
    images = np.asarray(pred.processed_images, dtype=np.uint8)
    return dict(c2w=c2w, points=pts, conf=conf, mask=mask, intrinsics=K, image_hw=tuple(depth.shape[1:3]),
                images=images,
                meta=dict(conf_rule=f"conf >= clamp({base:g}, p{lo_pct:g}, p{hi_pct:g}) = {thr:.4f} "
                                    "(depth_anything_3/utils/export/glb.py)",
                          point_source="depth unprojection with predicted K, w2c", is_metric=int(pred.is_metric)))


# ----------------------------------------------------------------------------- dispatch
_LOADERS = {"dust3r": load_dust3r, "mast3r": load_mast3r, "vggt": load_vggt, "da3_anyview": load_da3}
_RUNNERS = {"dust3r": run_dust3r, "mast3r": run_mast3r, "vggt": run_vggt, "da3_anyview": run_da3}


def load_model(model_id: str, cfg_model: dict, device):
    return _LOADERS[model_id](cfg_model, device)


def run_model(model_id: str, model, image_paths, cfg_model: dict, device, **kw) -> dict:
    """kw: dust3r -> niter; mast3r -> niter1, niter2, cache_dir[, keep_cache]; vggt / da3_anyview -> none."""
    out = _RUNNERS[model_id](model, image_paths, cfg_model, device, **kw)
    n = len(list(image_paths))
    assert out["c2w"].shape == (n, 4, 4), out["c2w"].shape
    assert out["points"].shape[:1] == (n,) and out["points"].shape[-1] == 3
    assert out["conf"].shape == out["points"].shape[:3] == out["mask"].shape
    return out


def save_prediction(path, out: dict) -> Path:
    """Compressed npz of the contract arrays (meta goes to the caller's json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, c2w=out["c2w"], points=out["points"], conf=out["conf"], mask=out["mask"],
                        intrinsics=out["intrinsics"], image_hw=np.asarray(out["image_hw"]), images=out["images"])
    return path


def load_prediction(path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["image_hw"] = tuple(int(x) for x in d["image_hw"])
    return d
