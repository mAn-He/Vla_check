"""Point-cloud understanding helpers for outdoor 06 (3dv-point env only; imports torch at module load).

Contents: a from-scratch PointNet classifier, GT-box object crops from nuScenes LiDAR, lidarseg label mapping,
Sonata/Concerto feature extraction + PCA colors, nearest-neighbour label transfer, and Pointcept subprocess runners.

Upstream facts (pinned sources, paths relative to each repo under third_party/; nuscenes-devkit 1.2.0 installed):
- Pointcept (main)
  pointcept/datasets/nuscenes.py   NuScenesDataset: info = <data_root>/info/nuscenes_infos_{sweeps}sweeps_{split}.pkl,
      points = <data_root>/raw/<lidar_path> (float32 x 5), labels = <data_root>/raw/<gt_segment_path> (uint8 raw ids)
      mapped by get_learning_map(ignore_index) to 16 classes; get_data_name = info["lidar_token"].
      NuScenesColorNormalDataset feeds color = normal = zeros (in_channels 9 = coord + color + normal).
  pointcept/engines/defaults.py    default_config_parser: `--options` merged by Config.merge_from_dict (dotted keys,
      list indices allowed); data.train.loop = epoch // eval_epoch; default_setup asserts epoch % eval_epoch == 0.
  pointcept/utils/config.py        DictAction.__call__: every value goes through _parse_iterable, which strips spaces
      and splits on top-level commas -> option values must not contain spaces or commas (paths included).
  pointcept/engines/train.py       Trainer.max_epoch = eval_epoch (evaluation + checkpoint after each of those);
      DataLoader(persistent_workers=True) -> num_worker must be >= 1.
  pointcept/engines/hooks/misc.py  CheckpointLoader: key -> "module."+key, replace `keywords` by `replacement` once,
      load_state_dict(strict=False), logs "Missing keys: [...]". CheckpointSaver writes model/model_last.pth and,
      only when cfg.evaluate, model/model_best.pth. PreciseEvaluator(test_last) runs cfg.test after training.
  pointcept/engines/test.py        SemSegTester: per-scan argmax over TTA-summed softmax, mapped back to all original
      points, saved as <save_path>/result/<name>_pred.npy (re-used if present -> fresh save_path per run);
      logs "Val result: mIoU/mAcc/allAcc a/b/c" and "Class_i - <name> Result: iou/accuracy x/y" (mIoU = plain mean
      over all classes, absent classes count as 0); per scan "Batch <t> (<avg>)" incl. TTA and data time.
- Concerto: concerto/model.py load(name, repo_id, download_root, custom_config, ckpt_only) calls
  hf_hub_download(repo_id, filename=f"{name}.pth", repo_type="model", revision="main",
  local_dir=download_root or ~/.cache/concerto/ckpt); ckpt = {"config", "state_dict"}. MODELS includes
  "concerto_large_outdoor". demo/0_pca_outdoor.py: TRANSFORM_CONFIG (below), feature upcast loop, PCA q=12.
- Sonata: sonata/model.py load(...) same signature (default ~/.cache/sonata/ckpt); sonata/transform.py default()
  = CenterShift(apply_z) + GridSample(0.02) + NormalizeColor + ToTensor + Collect; demo/0_pca.py PCA q=6.
  Both READMEs: without FlashAttention pass custom_config=dict(enc_patch_size=[1024]*5, enable_flash=False).
- nuscenes-devkit: NuScenes.get_sample_data(sd) returns boxes in the sensor frame (box_vis_level only filters
  cameras); utils.geometry_utils.points_in_box(box, points (3, N)); eval.detection.utils.category_to_detection_name;
  eval.detection.constants.DETECTION_NAMES (10 classes).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------- PointNet (6a)


class TNet(nn.Module):
    """k x k alignment network of PointNet (Qi et al. 2017, Sec. 4.2), initialised to the identity."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, k, N)
        g = self.mlp(x).max(dim=2).values
        eye = torch.eye(self.k, device=x.device, dtype=x.dtype)
        return self.fc(g).view(-1, self.k, self.k) + eye


class PointNetCls(nn.Module):
    """PointNet classifier: (optional T-Net) -> shared MLP per point -> max-pool -> MLP head."""

    def __init__(self, num_classes: int, in_dim: int = 3, input_transform: bool = True,
                 feature_transform: bool = False, dropout: float = 0.3):
        super().__init__()
        self.tnet = TNet(in_dim) if input_transform else None
        self.mlp1 = nn.Sequential(nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.ftnet = TNet(64) if feature_transform else None
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor):  # x (B, N, C)
        x = x.transpose(1, 2)
        if self.tnet is not None:
            x = torch.bmm(self.tnet(x), x)
        x = self.mlp1(x)
        t_feat = None
        if self.ftnet is not None:
            t_feat = self.ftnet(x)
            x = torch.bmm(t_feat, x)
        x = self.mlp2(x)
        g = x.max(dim=2).values  # symmetric function -> permutation invariance
        return self.head(g), t_feat


def orthogonality_loss(t: torch.Tensor) -> torch.Tensor:
    """||I - T T^T||_F^2 averaged over the batch (PointNet feature-transform regulariser)."""
    eye = torch.eye(t.shape[1], device=t.device, dtype=t.dtype)
    return ((eye - torch.bmm(t, t.transpose(1, 2))) ** 2).sum(dim=(1, 2)).mean()


def _augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Box-frame augmentation: random scale 0.9-1.1, left/right mirror (y -> -y), jitter sigma 1 cm clipped 5 cm."""
    b = x.shape[0]
    scale = torch.empty(b, 1, 1, device=x.device).uniform_(0.9, 1.1, generator=gen)
    mirror = (torch.rand(b, 1, device=x.device, generator=gen) < 0.5).float() * -2 + 1
    x = x * scale
    x = torch.cat([x[..., :1], x[..., 1:2] * mirror[:, None], x[..., 2:]], dim=-1)
    noise = (torch.randn(x.shape, device=x.device, generator=gen) * 0.01).clamp(-0.05, 0.05)
    return x + noise


def fit_pointnet(model: PointNetCls, X: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int = 32,
                 lr: float = 1e-3, weight_decay: float = 1e-4, reg_weight: float = 1e-3, device: str = "cuda",
                 seed: int = 0, augment: bool = True, class_weight: np.ndarray | None = None) -> list[dict]:
    """Adam + cosine LR. Returns per-epoch history. BatchNorm needs >1 sample per batch -> drop_last when possible."""
    model.to(device).train()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    w = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    n = len(Xt)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        n_full = n - n % batch_size if n > batch_size else n
        tot, correct, loss_sum = 0, 0, 0.0
        for s in range(0, n_full, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            xb = _augment(Xt[idx], gen) if augment else Xt[idx]
            logits, t_feat = model(xb)
            loss = F.cross_entropy(logits, yt[idx], weight=w)
            if t_feat is not None:
                loss = loss + reg_weight * orthogonality_loss(t_feat)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(idx)
            correct += int((logits.argmax(1) == yt[idx]).sum())
            tot += len(idx)
        sched.step()
        hist.append({"epoch": ep + 1, "loss": loss_sum / max(tot, 1), "train_acc": correct / max(tot, 1)})
    return hist


@torch.no_grad()
def predict_pointnet(model: PointNetCls, X: np.ndarray, batch_size: int = 256, device: str = "cuda"):
    """Returns (pred (M,), prob (M, C), seconds)."""
    model.to(device).eval()
    probs = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[s:s + batch_size], dtype=torch.float32, device=device)
        probs.append(F.softmax(model(xb)[0], dim=1).cpu())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    prob = torch.cat(probs).numpy() if probs else np.zeros((0, 0), np.float32)
    return prob.argmax(1), prob, sec


# --------------------------------------------------------------------------------------------- object crops (6a)


def detection_classes() -> list[str]:
    from nuscenes.eval.detection.constants import DETECTION_NAMES

    return list(DETECTION_NAMES)


def _resample(n_have: int, n_points: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_have, n_points, replace=n_have < n_points)


def object_crops(nusc, sample_tokens: list[str], n_points: int, min_points: int = 5, seed: int = 0,
                 classes: list[str] | None = None) -> dict:
    """LiDAR points inside each GT box of the given keyframes, in the box frame (x forward, y left, z up).

    Box frame = (p - center) rotated by the inverse box orientation, so the crop is translation/heading invariant
    but keeps metric size (the classifier can use object extent). Objects with < min_points LiDAR points are dropped.
    Each crop is resampled to n_points (with replacement if fewer).
    Category -> class: nuscenes.eval.detection.utils.category_to_detection_name (None -> skipped).
    """
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.geometry_utils import points_in_box

    from ..nusc import load_lidar

    classes = classes or detection_classes()
    cls_id = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)
    pts_out, labels, n_in, dist, ann, smp = [], [], [], [], [], []
    skipped = Counter()
    for st in sample_tokens:
        sd = nusc.get("sample", st)["data"]["LIDAR_TOP"]
        _, boxes, _ = nusc.get_sample_data(sd)  # boxes in the LIDAR_TOP frame
        pts = load_lidar(nusc, sd)[:, :3]
        for box in boxes:
            name = category_to_detection_name(box.name)
            if name is None or name not in cls_id:
                skipped["not_a_detection_class"] += 1
                continue
            m = points_in_box(box, pts.T)
            k = int(m.sum())
            if k < min_points:
                skipped[f"lt_{min_points}_points"] += 1
                continue
            local = (pts[m] - box.center) @ box.orientation.rotation_matrix  # row-vector form of R^T (p - c)
            pts_out.append(local[_resample(k, n_points, rng)].astype(np.float32))
            labels.append(cls_id[name])
            n_in.append(k)
            dist.append(float(np.linalg.norm(box.center[:2])))
            ann.append(box.token)
            smp.append(st)
    X = np.stack(pts_out) if pts_out else np.zeros((0, n_points, 3), np.float32)
    return {"points": X, "labels": np.asarray(labels, np.int64), "n_in_box": np.asarray(n_in, np.int64),
            "dist": np.asarray(dist, np.float32), "ann_tokens": ann, "sample_tokens": smp, "classes": classes,
            "skipped": dict(skipped)}


# --------------------------------------------------------------------------------------------- lidarseg labels


def lidarseg16_lut(ignore_index: int = -1) -> np.ndarray:
    """Raw nuScenes lidarseg id (0..31) -> Pointcept 16-class id (NuScenesDataset.get_learning_map)."""
    from pointcept.datasets.nuscenes import NuScenesDataset

    lm = NuScenesDataset.get_learning_map(ignore_index)
    lut = np.full(max(lm) + 1, ignore_index, np.int64)
    for k, v in lm.items():
        lut[k] = v
    return lut


def load_lidarseg16(nusc, lidar_sd_token: str, lut: np.ndarray) -> np.ndarray:
    rec = nusc.get("lidarseg", lidar_sd_token)
    raw = np.fromfile(Path(nusc.dataroot) / rec["filename"], dtype=np.uint8)
    return lut[raw]


def transfer_labels_nn(src_xyz: np.ndarray, src_labels: np.ndarray, dst_xyz: np.ndarray, max_dist: float = 0.5,
                       ignore_index: int = -1):
    """Label each dst point with the label of its nearest src point if closer than max_dist, else ignore_index.

    Returns (labels (M,), nn_dist (M,) with inf where no neighbour within max_dist).
    """
    from scipy.spatial import cKDTree

    d, i = cKDTree(np.asarray(src_xyz, np.float64)).query(np.asarray(dst_xyz, np.float64), k=1,
                                                         distance_upper_bound=max_dist)
    ok = np.isfinite(d)
    out = np.full(len(dst_xyz), ignore_index, np.int64)
    out[ok] = np.asarray(src_labels)[i[ok]]
    return out, d


# --------------------------------------------------------------------------------------------- Sonata / Concerto


# Concerto demo/0_pca_outdoor.py TRANSFORM_CONFIG (verbatim)
CONCERTO_OUTDOOR_TRANSFORM = [
    dict(type="RandomScale", scale=[0.2, 0.2]),
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True,
         return_inverse=True),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"), feat_keys=("coord", "color", "normal")),
]
# demo PCA settings: (component-block weights, brightness)
PCA_SONATA = {"weights": (0.6, 0.4), "brightness": 1.2}  # sonata demo/0_pca.py
PCA_CONCERTO_OUTDOOR = {"weights": (0.2, 0.2, 0.1, 0.5), "brightness": 1.0}  # concerto demo/0_pca_outdoor.py


def flash_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def load_ptv3_package(package: str, name: str, repo_id: str, enable_flash: bool, download_root: str | None = None):
    """sonata.load / concerto.load with the README fallback when FlashAttention is off or missing.

    Returns (model, custom_config or None).
    """
    pkg = importlib.import_module(package)
    custom = None
    if not (enable_flash and flash_available()):
        custom = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = pkg.load(name, repo_id=repo_id, download_root=download_root, custom_config=custom)
    return model, custom


def estimate_normals(xyz: np.ndarray, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """open3d normals flipped to face the sensor (as NuScenesImagePointDataset.get_normals in Pointcept)."""
    import open3d as o3d

    xyz = np.asarray(xyz, np.float64)
    view = xyz - np.asarray(center)[None]
    view /= np.linalg.norm(view, axis=1, keepdims=True).clip(1e-9)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    n = np.asarray(pcd.normals)
    flip = (n * view).sum(1) > 0
    n[flip] *= -1
    return (n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)).astype(np.float32)


def point_input(xyz: np.ndarray, normal: str = "zeros") -> dict:
    """Sonata/Concerto input dict for a LiDAR-like cloud without RGB: color = 0, normal = zeros | estimate."""
    xyz = np.ascontiguousarray(xyz[:, :3], dtype=np.float32)
    nrm = estimate_normals(xyz) if normal == "estimate" else np.zeros_like(xyz)
    return {"coord": xyz, "color": np.zeros_like(xyz), "normal": nrm}


@torch.no_grad()
def ptv3_features(model, point: dict, transform, device: str = "cuda"):
    """Encoder forward + the README/demo upcast loop. Returns (feat on grid points, inverse to input points, s)."""
    model.to(device).eval()
    point = transform(dict(point))
    for k, v in point.items():
        if isinstance(v, torch.Tensor):
            point[k] = v.to(device, non_blocking=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    point = model(point)
    for _ in range(2):
        assert "pooling_parent" in point and "pooling_inverse" in point
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point:
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return point.feat, point.inverse, time.perf_counter() - t0


@torch.no_grad()
def pca_colors(feat: torch.Tensor, weights=(0.6, 0.4), brightness: float = 1.2, center: bool = True) -> torch.Tensor:
    """Demo PCA coloring: q = 3 * len(weights) components, RGB = sum_i w_i * block_i, min-max scaled."""
    q = 3 * len(weights)
    _, _, v = torch.pca_lowrank(feat.float(), center=center, q=q, niter=5)
    proj = feat.float() @ v
    rgb = sum(w * proj[:, 3 * i:3 * i + 3] for i, w in enumerate(weights))
    lo, hi = rgb.min(dim=-2, keepdim=True)[0], rgb.max(dim=-2, keepdim=True)[0]
    return ((rgb - lo) / torch.clamp(hi - lo, min=1e-6) * brightness).clamp(0.0, 1.0)


# --------------------------------------------------------------------------------------------- Pointcept runners


def git_head(path: str | Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001 - provenance only
        return f"unknown ({type(e).__name__})"


def _fmt_option(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (tuple, list)):
        inner = ",".join(_fmt_option(x) for x in v)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    s = str(v)
    if " " in s or "," in s:
        raise ValueError(f"Pointcept DictAction strips spaces and splits on commas; bad option value: {s!r}")
    return s


def pointcept_options(options: dict) -> list[str]:
    return [f"{k}={_fmt_option(v)}" for k, v in options.items()]


def run_pointcept(tool: str, config_file: str | Path, options: dict, *, pointcept_root: str | Path,
                  log_path: str | Path, pythonpath: str | Path | None = None, python: str | None = None,
                  timeout: float | None = None) -> dict:
    """Run `tools/<tool>.py --config-file ... --num-gpus 1 --options k=v ...` from `pointcept_root`.

    stdout+stderr go to log_path. `pythonpath` is prepended to PYTHONPATH (used to put Pointcept-v1.5.1 ahead of
    the main checkout that the env registers through a .pth file).
    """
    cmd = [python or sys.executable, f"tools/{tool}.py", "--config-file", str(config_file), "--num-gpus", "1",
           "--options", *pointcept_options(options)]
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            rc = subprocess.run(cmd, cwd=str(pointcept_root), env=env, stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return {"returncode": rc, "seconds": time.perf_counter() - t0, "log": str(log_path), "cmd": cmd}


def tail(path: str | Path, n: int = 25) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return f"(no log at {path})"


_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)"


def parse_semseg_log(text: str, names: list[str]) -> dict | None:
    """Last SemSegTester summary in a Pointcept log (see module docstring for the line formats)."""
    vals = re.findall(rf"Val result: mIoU/mAcc/allAcc {_NUM}/{_NUM}/{_NUM}", text)
    if not vals:
        return None
    miou, macc, allacc = (float(x) for x in vals[-1])
    cls = re.findall(rf"Class_(\d+) - (\S+) Result: iou/accuracy {_NUM}/{_NUM}", text)[-len(names):]
    per_iou = {n: float(i) for _, n, i, _ in cls}
    per_acc = {n: float(a) for _, n, _, a in cls}
    times = re.findall(rf"Batch {_NUM} \({_NUM}\) Accuracy", text)
    return {"miou": miou, "macc": macc, "allacc": allacc, "per_class_iou": per_iou, "per_class_acc": per_acc,
            "scan_time_avg_s": float(times[-1][1]) if times else None}


def parse_missing_keys(text: str) -> list[str] | None:
    """CheckpointLoader's last 'Missing keys: [...]' line (strict=False load)."""
    found = re.findall(r"Missing keys: (\[.*\])", text)
    if not found:
        return None
    try:
        return list(ast.literal_eval(found[-1]))
    except (ValueError, SyntaxError):
        return [found[-1]]


def write_infos_without_labels(src_info_pkl: str | Path, raw_root: str | Path, out_root: str | Path,
                               sweeps: int = 10, split: str = "val") -> dict:
    """Pointcept data_root for clouds that replace the LiDAR files (e.g. pseudo-LiDAR).

    Keeps every info whose `lidar_path` exists under raw_root, drops `gt_segment_path` (NuScenesDataset then
    fills segment with ignore_index), writes <out_root>/info/nuscenes_infos_{sweeps}sweeps_{split}.pkl and links
    <out_root>/raw -> raw_root.
    """
    import pickle

    infos = pickle.loads(Path(src_info_pkl).read_bytes())
    raw_root, out_root = Path(raw_root).resolve(), Path(out_root)
    keep, missing = [], []
    for info in infos:
        if (raw_root / info["lidar_path"]).exists():
            keep.append({k: v for k, v in info.items() if k != "gt_segment_path"})
        else:
            missing.append(info["token"])
    (out_root / "info").mkdir(parents=True, exist_ok=True)
    (out_root / "info" / f"nuscenes_infos_{sweeps}sweeps_{split}.pkl").write_bytes(pickle.dumps(keep))
    raw = out_root / "raw"
    if raw.is_symlink() and raw.resolve() != raw_root:
        raw.unlink()
    if not raw.is_symlink():
        raw.symlink_to(raw_root, target_is_directory=True)
    return {"n_kept": len(keep), "n_missing": len(missing), "missing_sample_tokens": missing[:20]}


def class_palette(n: int) -> np.ndarray:
    """(n, 3) float RGB in [0, 1], tab20-based; index -1 (ignore) should be drawn separately in grey."""
    import matplotlib

    cmap = matplotlib.colormaps["tab20"]
    return np.asarray([cmap(i % 20)[:3] for i in range(n)], np.float32)
