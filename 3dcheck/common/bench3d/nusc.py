"""nuScenes geometry helpers (requires nuscenes-devkit).

Frames: sensor -> ego (at the sensor's timestamp) -> global. Each sample_data has its own ego pose,
so camera and LiDAR of the same sample are related through the global frame, as in the devkit's
`NuScenesExplorer.map_pointcloud_to_image`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")


def _tm(translation, rotation, inverse: bool = False) -> np.ndarray:
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    return transform_matrix(np.asarray(translation), Quaternion(rotation), inverse=inverse)


def sensor_to_ego(nusc, sd_token: str) -> np.ndarray:
    cs = nusc.get("calibrated_sensor", nusc.get("sample_data", sd_token)["calibrated_sensor_token"])
    return _tm(cs["translation"], cs["rotation"])


def ego_to_global(nusc, sd_token: str) -> np.ndarray:
    ep = nusc.get("ego_pose", nusc.get("sample_data", sd_token)["ego_pose_token"])
    return _tm(ep["translation"], ep["rotation"])


def sensor_to_global(nusc, sd_token: str) -> np.ndarray:
    """4x4 sensor->global. For a camera this is its c2w pose (OpenCV axes: x right, y down, z forward)."""
    return ego_to_global(nusc, sd_token) @ sensor_to_ego(nusc, sd_token)


def intrinsic(nusc, sd_token: str) -> np.ndarray:
    cs = nusc.get("calibrated_sensor", nusc.get("sample_data", sd_token)["calibrated_sensor_token"])
    return np.asarray(cs["camera_intrinsic"], float)


def camera_info(nusc, sd_token: str) -> dict:
    """K, c2w, w2c, width, height, filename for a camera sample_data."""
    sd = nusc.get("sample_data", sd_token)
    c2w = sensor_to_global(nusc, sd_token)
    return {
        "sd_token": sd_token,
        "channel": sd["channel"],
        "filename": sd["filename"],
        "timestamp": sd["timestamp"],
        "K": intrinsic(nusc, sd_token),
        "c2w": c2w,
        "w2c": np.linalg.inv(c2w),
        "width": sd["width"],
        "height": sd["height"],
    }


def sample_cameras(nusc, sample_token: str, cams=CAMERAS) -> list[dict]:
    s = nusc.get("sample", sample_token)
    return [camera_info(nusc, s["data"][c]) for c in cams]


def load_lidar(nusc, lidar_sd_token: str) -> np.ndarray:
    """Raw LIDAR_TOP keyframe/sweep points, (N, 5) float32 in the sensor frame."""
    sd = nusc.get("sample_data", lidar_sd_token)
    pts = np.fromfile(Path(nusc.dataroot) / sd["filename"], dtype=np.float32)
    return pts.reshape(-1, 5)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts)
    return pts[:, :3] @ T[:3, :3].T + T[:3, 3]


def lidar_to_camera(nusc, lidar_sd_token: str, cam_sd_token: str) -> np.ndarray:
    """4x4 lidar(sensor, its timestamp) -> camera(sensor, its timestamp) through the global frame."""
    return np.linalg.inv(sensor_to_global(nusc, cam_sd_token)) @ sensor_to_global(nusc, lidar_sd_token)


def project_lidar_to_camera(
    nusc, lidar_sd_token: str, cam_sd_token: str, min_depth: float = 1.0, points: np.ndarray | None = None
) -> dict:
    """Project LiDAR points into a camera image.

    Returns dict(uv (M,2) float pixel coords, depth (M,), index (M,) into the LiDAR points).
    min_depth=1.0 follows the devkit's map_pointcloud_to_image default (drops points on the ego car).
    `points` overrides the file (e.g. a pseudo-LiDAR cloud in the same LIDAR_TOP frame).
    """
    pts = load_lidar(nusc, lidar_sd_token) if points is None else points
    pc = transform_points(lidar_to_camera(nusc, lidar_sd_token, cam_sd_token), pts)
    K = intrinsic(nusc, cam_sd_token)
    sd = nusc.get("sample_data", cam_sd_token)
    z = pc[:, 2]
    ok = z > min_depth
    uvw = pc[ok] @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= sd["width"] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= sd["height"] - 1)
    idx = np.flatnonzero(ok)[inside]
    return {"uv": uv[inside], "depth": z[idx], "index": idx}


def sparse_depth_map(uv: np.ndarray, depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize projected points; keeps the nearest point per pixel (z-buffer). 0 = no measurement."""
    dm = np.zeros((height, width), np.float32)
    u = np.round(uv[:, 0]).astype(int).clip(0, width - 1)
    v = np.round(uv[:, 1]).astype(int).clip(0, height - 1)
    order = np.argsort(-depth)  # far first, near overwrites
    dm[v[order], u[order]] = depth[order]
    return dm


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float | None = None,
    mask: np.ndarray | None = None,
):
    """Dense depth (HxW, metres) -> points in the camera frame. Returns (points (N,3), pixel_idx (N,2) as v,u)."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > min_depth)
    if max_depth is not None:
        ok &= d < max_depth
    if mask is not None:
        ok &= mask[vs, us]
    u, v, d = us[ok].astype(float), vs[ok].astype(float), d[ok].astype(float)
    Kinv = np.linalg.inv(K)
    rays = np.stack([u, v, np.ones_like(u)], 1) @ Kinv.T
    return rays * d[:, None], np.stack([vs[ok], us[ok]], 1)


def box_center_in_cameras(center_global, cams: list[dict], max_depth: float | None = None) -> bool:
    from .metrics.pointcloud import frustum_mask

    return bool(frustum_mask(np.asarray(center_global, float)[None], cams, max_depth)[0])


def scene_samples(nusc, scene_name: str) -> list[str]:
    """Ordered sample tokens of a scene."""
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    toks, t = [], scene["first_sample_token"]
    while t:
        toks.append(t)
        t = nusc.get("sample", t)["next"]
    return toks
