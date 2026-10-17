"""Depth metrics (Eigen et al. 2014 conventions) with distance binning.

All functions take metric depth arrays of the same shape; GT pixels <= 0 are invalid.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_BINS = ((0, 10), (10, 20), (20, 30), (30, 50), (50, math.inf))


def _valid(pred, gt, mask, min_depth, max_depth):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    m = (gt > min_depth) & np.isfinite(pred) & np.isfinite(gt)
    if max_depth is not None:
        m &= gt < max_depth
    if mask is not None:
        m &= np.asarray(mask, bool)
    return pred[m], gt[m]


def depth_metrics(
    pred, gt, mask=None, min_depth: float = 1e-3, max_depth: float | None = None, clip_pred: bool = True
) -> dict:
    """AbsRel, SqRel, RMSE, RMSE_log, delta<1.25^k, plus n (valid pixels).

    clip_pred clamps predictions to >= min_depth so log / ratio terms stay finite.
    """
    p, g = _valid(pred, gt, mask, min_depth, max_depth)
    n = int(g.size)
    if n == 0:
        return {k: math.nan for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")} | {"n": 0}
    if clip_pred:
        p = np.maximum(p, min_depth)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "n": n,
    }


def bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}m" if math.isfinite(hi) else f"{lo:g}m+"


def binned_depth_metrics(pred, gt, mask=None, bins=DEFAULT_BINS, **kw) -> dict[str, dict]:
    """depth_metrics per GT-distance bin. Keys are labels like '0-10m', '50m+'."""
    gt = np.asarray(gt, float)
    out = {}
    for lo, hi in bins:
        m = (gt >= lo) & (gt < hi)
        if mask is not None:
            m &= np.asarray(mask, bool)
        out[bin_label(lo, hi)] = depth_metrics(pred, gt, m, **kw)
    return out


def error_vs_distance(
    pred, gt, mask=None, n_bins: int = 20, d_min: float = 2.0, d_max: float = 80.0, min_count: int = 50
) -> dict:
    """Mean absolute error per log-spaced distance bin and a power-law fit |err| ~ c * d^k.

    k ~= 2 is the stereo-triangulation scaling (error grows with distance squared); a monocular
    network has no such law built in, so the fitted k is an empirical description, not a test of a model.
    Returns dict(centers, mean_abs_err, counts, slope_k, intercept_logc).
    """
    p, g = _valid(pred, gt, mask, 1e-3, None)
    edges = np.geomspace(d_min, d_max, n_bins + 1)
    centers, errs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g >= lo) & (g < hi)
        if m.sum() >= min_count:
            centers.append(math.sqrt(lo * hi))
            errs.append(float(np.mean(np.abs(p[m] - g[m]))))
            counts.append(int(m.sum()))
    centers, errs = np.array(centers), np.array(errs)
    ok = errs > 0
    if ok.sum() >= 2:
        k, logc = np.polyfit(np.log(centers[ok]), np.log(errs[ok]), 1)
    else:
        k, logc = math.nan, math.nan
    return {
        "centers": centers.tolist(),
        "mean_abs_err": errs.tolist(),
        "counts": counts,
        "slope_k": float(k),
        "intercept_logc": float(logc),
    }
