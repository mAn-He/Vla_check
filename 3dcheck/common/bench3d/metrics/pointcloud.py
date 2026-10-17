"""Point-cloud reconstruction metrics (accuracy / completeness / Chamfer / F-score) and cropping."""

from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one point (the first) per voxel. Used to equalize densities before comparing clouds."""
    if voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def nn_distance(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Distance from each src point to its nearest dst point."""
    from scipy.spatial import cKDTree

    if len(dst) == 0:
        return np.full(len(src), np.inf)
    d, _ = cKDTree(dst[:, :3]).query(src[:, :3], k=1, workers=-1)
    return d


def reconstruction_metrics(pred: np.ndarray, gt: np.ndarray, taus=(0.25, 0.5, 1.0), voxel: float = 0.0,
                           trunc: float | None = None) -> dict:
    """accuracy = mean d(pred->gt), completeness = mean d(gt->pred), chamfer = their mean,
    precision/recall/fscore@tau (Knapitsch et al. 2017).

    voxel>0 downsamples both clouds first (LiDAR vs dense camera clouds have very different densities).
    trunc caps per-point distances before averaging (limits outlier domination); None = no cap.
    """
    if voxel > 0:
        pred, gt = voxel_downsample(pred, voxel), voxel_downsample(gt, voxel)
    if len(pred) == 0 or len(gt) == 0:
        return {"n_pred": len(pred), "n_gt": len(gt)}
    d_pg, d_gp = nn_distance(pred, gt), nn_distance(gt, pred)
    acc_d, comp_d = (np.minimum(d_pg, trunc), np.minimum(d_gp, trunc)) if trunc else (d_pg, d_gp)
    out = {
        "accuracy": float(acc_d.mean()),
        "completeness": float(comp_d.mean()),
        "chamfer": float((acc_d.mean() + comp_d.mean()) / 2),
        "accuracy_median": float(np.median(d_pg)),
        "completeness_median": float(np.median(d_gp)),
        "n_pred": int(len(pred)),
        "n_gt": int(len(gt)),
    }
    for tau in taus:
        p, r = float((d_pg < tau).mean()), float((d_gp < tau).mean())
        out[f"precision@{tau:g}"] = p
        out[f"recall@{tau:g}"] = r
        out[f"fscore@{tau:g}"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return out


def frustum_mask(points_world: np.ndarray, cams: list[dict], max_depth: float | None = None) -> np.ndarray:
    """True for points visible in any camera.

    cams: dicts with K (3x3), w2c (4x4), width, height.
    """
    pts = np.asarray(points_world)[:, :3]
    keep = np.zeros(len(pts), bool)
    homo = np.c_[pts, np.ones(len(pts))]
    for c in cams:
        pc = (homo @ np.asarray(c["w2c"]).T)[:, :3]
        z = pc[:, 2]
        ok = z > 1e-3
        if max_depth is not None:
            ok &= z < max_depth
        uv = pc[ok] @ np.asarray(c["K"]).T
        uv = uv[:, :2] / uv[:, 2:3]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < c["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < c["height"])
        idx = np.flatnonzero(ok)[inside]
        keep[idx] = True
    return keep
