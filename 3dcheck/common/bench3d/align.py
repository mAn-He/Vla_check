"""Scale / similarity alignment between predictions and ground truth.

Conventions: poses are camera-to-world 4x4 (c2w). A Sim(3) (s, R, t) maps prediction coordinates
to ground-truth coordinates: x_gt ~= s * R @ x_pred + t.
"""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares (s, R, t) with dst ~= s R src + t (Umeyama 1991). src, dst: (N, 3)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need >=3 corresponding points")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt


def centers_degenerate(centers: np.ndarray, ratio: float = 0.05) -> bool:
    """True if camera centers are nearly collinear or coincident (2nd singular value small vs 1st)."""
    c = np.asarray(centers, float)
    c = c - c.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    return bool(sv[0] < 1e-9 or sv[1] / sv[0] < ratio)


def align_poses_sim3(pred_c2w: np.ndarray, gt_c2w: np.ndarray, method: str = "auto", with_scale: bool = True):
    """Sim(3) aligning predicted camera poses to GT.

    method:
      "umeyama"      - camera centers only (standard for ATE).
      "rot+center"   - rotation from the chordal mean of R_gt R_pred^T, then scale/translation from centers.
                       Needed when centers are (near) collinear, e.g. a forward-driving CAM_FRONT sequence,
                       where Umeyama leaves the roll about the trajectory axis undetermined. That roll does not
                       change ATE but does rotate the predicted point cloud, which would corrupt Chamfer metrics.
      "auto"         - "rot+center" if centers_degenerate(), else "umeyama".
    Returns (s, R, t, used_method).
    """
    pred_c2w, gt_c2w = np.asarray(pred_c2w, float), np.asarray(gt_c2w, float)
    pc, gc = pred_c2w[:, :3, 3], gt_c2w[:, :3, 3]
    if method == "auto":
        method = "rot+center" if centers_degenerate(gc) else "umeyama"
    if method == "umeyama":
        s, R, t = umeyama(pc, gc, with_scale)
        return s, R, t, method
    if method != "rot+center":
        raise ValueError(method)
    R = project_to_so3(sum(g[:3, :3] @ p[:3, :3].T for g, p in zip(gt_c2w, pred_c2w)))
    pc_r = pc @ R.T
    mu_p, mu_g = pc_r.mean(0), gc.mean(0)
    xp, xg = pc_r - mu_p, gc - mu_g
    denom = (xp**2).sum()
    s = float((xp * xg).sum() / denom) if (with_scale and denom > 1e-12) else 1.0
    t = mu_g - s * mu_p
    return s, R, t, method


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * np.asarray(points) @ np.asarray(R).T + t


def apply_sim3_to_poses(c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.array(c2w, float, copy=True)
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = apply_sim3(out[:, :3, 3], s, R, t)
    return out


def median_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Scale factor median(gt)/median(pred) over valid pixels (monocular relative-depth protocol)."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    m = (gt > 0) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        m &= mask
    if not m.any():
        raise ValueError("no valid pixels for median scaling")
    return float(np.median(gt[m]) / np.median(pred[m]))


def lstsq_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None):
    """Least-squares (a, b) with gt ~= a * pred + b (affine-invariant protocol, e.g. MiDaS). Returns (a, b)."""
    pred, gt = np.asarray(pred, float).ravel(), np.asarray(gt, float).ravel()
    m = (gt > 0) & np.isfinite(pred)
    if mask is not None:
        m &= np.asarray(mask).ravel()
    A = np.stack([pred[m], np.ones(m.sum())], 1)
    a, b = np.linalg.lstsq(A, gt[m], rcond=None)[0]
    return float(a), float(b)
