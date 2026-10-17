"""Camera pose metrics. Poses are camera-to-world 4x4 (c2w); align predictions first (bench3d.align)."""

from __future__ import annotations

import itertools

import numpy as np


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix in degrees."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def ate(pred_c2w_aligned: np.ndarray, gt_c2w: np.ndarray) -> dict:
    """Absolute trajectory error of camera centers (after Sim(3) alignment): RMSE / mean / median in GT units."""
    d = np.linalg.norm(pred_c2w_aligned[:, :3, 3] - gt_c2w[:, :3, 3], axis=1)
    return {"ate_rmse": float(np.sqrt(np.mean(d**2))), "ate_mean": float(d.mean()), "ate_median": float(np.median(d))}


def rpe(pred_c2w_aligned: np.ndarray, gt_c2w: np.ndarray, delta: int = 1) -> dict:
    """Relative pose error between frames i and i+delta: translation RMSE (GT units) and mean rotation error (deg)."""
    te, re = [], []
    for i in range(len(gt_c2w) - delta):
        rel_p = np.linalg.inv(pred_c2w_aligned[i]) @ pred_c2w_aligned[i + delta]
        rel_g = np.linalg.inv(gt_c2w[i]) @ gt_c2w[i + delta]
        err = np.linalg.inv(rel_g) @ rel_p
        te.append(np.linalg.norm(err[:3, 3]))
        re.append(rotation_angle_deg(err[:3, :3]))
    if not te:
        return {"rpe_trans_rmse": float("nan"), "rpe_rot_mean_deg": float("nan")}
    return {"rpe_trans_rmse": float(np.sqrt(np.mean(np.square(te)))), "rpe_rot_mean_deg": float(np.mean(re))}


def pairwise_relative_errors(pred_c2w: np.ndarray, gt_c2w: np.ndarray):
    """For every pair i<j: relative rotation error (deg) and relative translation *direction* error (deg).

    Scale-free and alignment-free (RRA/RTA protocol used by DUSt3R / VGGT evaluations).
    Returns (rot_err[], trans_err[]).
    """
    rot, trans = [], []
    for i, j in itertools.combinations(range(len(gt_c2w)), 2):
        rel_p = np.linalg.inv(pred_c2w[i]) @ pred_c2w[j]
        rel_g = np.linalg.inv(gt_c2w[i]) @ gt_c2w[j]
        rot.append(rotation_angle_deg(rel_g[:3, :3].T @ rel_p[:3, :3]))
        trans.append(angle_between_deg(rel_p[:3, 3], rel_g[:3, 3]))
    return np.array(rot), np.array(trans)


def pose_auc(rot_err: np.ndarray, trans_err: np.ndarray, max_threshold: int = 30) -> float:
    """AUC of the accuracy curve of max(rot_err, trans_err) over integer thresholds 1..max_threshold (deg).

    acc(t) = fraction of pairs with error < t;  AUC = mean_t acc(t)  (in [0, 1]).
    NaN translation errors (zero baseline) count as failures.
    """
    e = np.maximum(np.nan_to_num(rot_err, nan=np.inf), np.nan_to_num(trans_err, nan=np.inf))
    if e.size == 0:
        return float("nan")
    ts = np.arange(1, max_threshold + 1)
    return float(np.mean([(e < t).mean() for t in ts]))


def pose_metrics(pred_c2w_aligned: np.ndarray, gt_c2w: np.ndarray, auc_thresholds=(5, 15, 30)) -> dict:
    out = {**ate(pred_c2w_aligned, gt_c2w), **rpe(pred_c2w_aligned, gt_c2w)}
    r, t = pairwise_relative_errors(pred_c2w_aligned, gt_c2w)
    out["rra_mean_deg"] = float(np.mean(r)) if r.size else float("nan")
    out["rta_mean_deg"] = float(np.nanmean(t)) if t.size else float("nan")
    for th in auc_thresholds:
        out[f"auc@{th}"] = pose_auc(r, t, th)
    return out
