"""Point-cloud files: nuScenes .bin (float32 x N channels) and .ply."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_nuscenes_bin(path: str | Path, xyz: np.ndarray, extra: np.ndarray | None = None, n_channels: int = 5) -> Path:
    """Write points in the nuScenes LIDAR_TOP layout: float32, `n_channels` per point.

    Channels after xyz are taken from `extra` (N x k) and the rest are zero-filled
    (nuScenes: x, y, z, intensity, ring index; pseudo-LiDAR has neither, so both are 0).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    out = np.zeros((len(xyz), n_channels), dtype=np.float32)
    out[:, :3] = xyz
    if extra is not None:
        extra = np.asarray(extra, dtype=np.float32).reshape(len(xyz), -1)
        out[:, 3 : 3 + extra.shape[1]] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(path)
    return path


def load_nuscenes_bin(path: str | Path, n_channels: int = 5) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % n_channels:
        raise ValueError(f"{path}: size {pts.size} not divisible by {n_channels}")
    return pts.reshape(-1, n_channels)


def save_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray | None = None, scalars: dict | None = None) -> Path:
    """Binary PLY with optional uint8 colors and extra float32 per-vertex scalars."""
    from plyfile import PlyData, PlyElement

    xyz = np.asarray(xyz, dtype=np.float32)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    for k in scalars or {}:
        fields.append((k, "f4"))
    v = np.empty(len(xyz), dtype=fields)
    v["x"], v["y"], v["z"] = xyz.T
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        v["red"], v["green"], v["blue"] = rgb.T
    for k, s in (scalars or {}).items():
        v[k] = np.asarray(s, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))
    return path


def load_ply(path: str | Path) -> dict:
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    out = {"xyz": np.stack([v["x"], v["y"], v["z"]], 1)}
    names = v.dtype.names
    if {"red", "green", "blue"} <= set(names):
        out["rgb"] = np.stack([v["red"], v["green"], v["blue"]], 1)
    for n in names:
        if n not in ("x", "y", "z", "red", "green", "blue"):
            out[n] = np.asarray(v[n])
    return out
