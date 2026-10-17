"""Monocular depth model wrappers shared by outdoor 02 (evaluation) and 04 (pseudo-LiDAR). 3dv env only.

Every wrapper maps one RGB image (H, W, 3 uint8) and its 3x3 intrinsics to a depth map at the ORIGINAL
resolution (float32; metres for metric models, arbitrary scale for relative ones). Heavy imports are lazy.

Upstream facts (pinned sources, paths relative to the repo root of each project):
- Depth Anything V2 metric (third_party/Depth-Anything-V2):
  metric_depth/README.md            model_configs per encoder; max_depth 80 for the VKITTI (outdoor) checkpoints
  metric_depth/depth_anything_v2/dpt.py
      DepthAnythingV2.__init__(..., max_depth=20.0)    default is the INDOOR value -> 80 must be passed
      infer_image(raw_image_bgr, input_size=518)       resizes the prediction back to the input H x W (bilinear)
      image2tensor()                                   picks cuda > mps > cpu by itself -> model must be on that device
- Depth Anything 3 (third_party/Depth-Anything-3/src/depth_anything_3):
  api.py           DepthAnything3.inference(image, extrinsics=None, intrinsics=None, ..., process_res=504,
                   process_res_method="upper_bound_resize"); passing extrinsics/intrinsics switches to the
                   pose-conditioned path (_normalize_extrinsics / _align_to_input_extrinsics_intrinsics) -> images only
  utils/io/input_processor.py   longest side -> process_res, then each side rounded to a multiple of 14
                   (1600x900 -> 504x280, i.e. sx=0.3150, sy=0.3111); numpy input is assumed RGB uint8
  model/dpt.py     head output size = processed H x W (down_ratio=1)
  utils/alignment.py apply_metric_scaling: depth * ((fx + fy) / 2) / 300
  model/da3.py     NestedDepthAnything3Net feeds the SAME processed tensor to the metric branch and scales it with
                   intrinsics predicted for the processed image (model/utils/transform.py pose_encoding_to_extri_intri
                   builds fx, fy from image_size_hw = processed H, W). => focal must be in processed-image pixels.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# metric_depth/README.md "Use our models" (vitg has no metric checkpoint)
DAV2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
# HF repo size suffix <-> encoder (metric_depth/README.md download table)
DAV2_SIZES = {"vits": "Small", "vitb": "Base", "vitl": "Large"}
# README FAQ / utils/alignment.py apply_metric_scaling(scale_factor=300.0)
DA3_METRIC_FOCAL_DIVISOR = 300.0
DA3_PROCESS_RES = 504  # api.py inference() default
DA3_PROCESS_RES_METHOD = "upper_bound_resize"  # api.py inference() default


@dataclass
class DepthPrediction:
    depth: np.ndarray  # (H, W) float32 at the input resolution
    kind: str  # "metric" | "relative"
    sky: np.ndarray | None = None  # (H, W) bool, True = sky (DA3 models with a sky head)
    info: dict = field(default_factory=dict)


def family(model_id: str) -> str:
    if model_id.startswith("dav2"):
        return "dav2"
    if model_id.startswith("da3"):
        return "da3"
    raise ValueError(f"unknown depth model family: {model_id}")


def depth_kind(spec: dict) -> str:
    """'metric' for output in {metric, metric_focal}, 'relative' otherwise (config models.depth.<id>.output)."""
    return "relative" if spec["output"] == "relative" else "metric"


def dav2_files(spec: dict, encoder: str) -> tuple[str, str]:
    """(hf_repo, filename) for the DA V2 metric checkpoint of `encoder` (vits|vitb|vitl)."""
    if encoder not in DAV2_SIZES:
        raise ValueError(f"dav2 encoder must be one of {list(DAV2_SIZES)}, got {encoder}")
    return spec["hf_repo"].format(size=DAV2_SIZES[encoder]), spec["filename"].format(encoder=encoder)


def checkpoint_name(model_id: str, spec: dict, encoder: str | None = None) -> str:
    """Checkpoint id written to summary.json and to the prediction cache (used to detect stale caches)."""
    if family(model_id) == "dav2":
        repo, fn = dav2_files(spec, encoder)
        return f"{repo}/{fn}"
    return spec["hf_repo"]


def _upstream_device() -> str:
    """Same rule as Depth-Anything-V2 metric_depth dpt.py image2tensor()."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DAV2Metric:
    """Depth Anything V2 metric (Virtual KITTI 2 fine-tune). Input BGR internally, output metres."""

    kind = "metric"

    def __init__(self, hf_repo: str, filename: str, encoder: str, max_depth: float, input_size: int = 518):
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2  # metric_depth version (via .pth on sys.path)
        from huggingface_hub import hf_hub_download

        if encoder not in DAV2_CONFIGS:
            raise ValueError(f"no metric config for encoder {encoder}")
        self.checkpoint = f"{hf_repo}/{filename}"
        self.input_size = int(input_size)
        ckpt = hf_hub_download(repo_id=hf_repo, filename=filename)
        model = DepthAnythingV2(**{**DAV2_CONFIGS[encoder], "max_depth": float(max_depth)})
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.device = _upstream_device()
        self.model = model.to(self.device).eval()
        # guard against the indoor default (20) silently clipping outdoor depth
        assert float(self.model.max_depth) == float(max_depth), "max_depth was not applied"

    def predict(self, rgb: np.ndarray, K: np.ndarray | None = None) -> DepthPrediction:
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # infer_image expects a cv2 (BGR) image
        depth = self.model.infer_image(bgr, self.input_size)
        return DepthPrediction(np.asarray(depth, np.float32), "metric",
                               info={"input_size": self.input_size, "max_depth": float(self.model.max_depth)})


class DA3Mono:
    """Depth Anything 3 single-image inference (DA3METRIC / DA3MONO / DA3NESTED), images only (no poses).

    output (config models.depth.<id>.output):
      metric_focal : DA3METRIC, metric = f_proc * raw / 300 with f_proc = mean(fx, fy) of the PROCESSED image
      metric       : DA3NESTED, already metres (scaled internally with its own predicted focal)
      relative     : DA3MONO, arbitrary scale -> median scaling in the notebook
    """

    def __init__(self, hf_repo: str, output: str, process_res: int = DA3_PROCESS_RES,
                 process_res_method: str = DA3_PROCESS_RES_METHOD, device: str | None = None, quiet: bool = True):
        import torch
        from depth_anything_3.api import DepthAnything3

        if output not in ("metric_focal", "metric", "relative"):
            raise ValueError(output)
        if quiet:  # api.py logs at INFO for every call
            from depth_anything_3.utils.logger import LOG_LEVELS, logger

            logger.level = LOG_LEVELS["WARN"]
        self.checkpoint = hf_repo
        self.output = output
        self.kind = "relative" if output == "relative" else "metric"
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(hf_repo).to(self.device)
        self.model.eval()

    def predict(self, rgb: np.ndarray, K: np.ndarray | None = None) -> DepthPrediction:
        import cv2

        H, W = rgb.shape[:2]
        # images only: extrinsics/intrinsics would switch DA3 to pose-conditioned mode
        pred = self.model.inference([np.ascontiguousarray(rgb)], process_res=self.process_res,
                                    process_res_method=self.process_res_method)
        raw = np.asarray(pred.depth[0], np.float32)  # (h, w) at the processed resolution
        h, w = raw.shape
        info = {"proc_hw": [int(h), int(w)], "process_res": self.process_res}
        if self.output == "metric_focal":
            if K is None:
                raise ValueError("DA3METRIC needs the camera intrinsics K for the metric conversion")
            K = np.asarray(K, float)
            f_proc = 0.5 * (K[0, 0] * (w / W) + K[1, 1] * (h / H))
            raw = raw * (f_proc / DA3_METRIC_FOCAL_DIVISOR)
            info.update(focal_proc_px=float(f_proc), focal_orig_px=float(0.5 * (K[0, 0] + K[1, 1])))
        if getattr(pred, "scale_factor", None) is not None:
            info["nested_scale_factor"] = float(pred.scale_factor)
        if pred.intrinsics is not None:  # nested: focal predicted for the processed image
            info["pred_focal_proc_px"] = float(0.5 * (pred.intrinsics[0][0, 0] + pred.intrinsics[0][1, 1]))
        # depth values are metres per pixel -> plain spatial resize back to the camera grid
        depth = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
        sky = None
        if pred.sky is not None:
            sky = cv2.resize(pred.sky[0].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        return DepthPrediction(depth.astype(np.float32), self.kind, sky=sky, info=info)


def load_depth_model(model_id: str, spec: dict, encoder: str | None = None):
    """Build a wrapper from config `models.depth.<model_id>` (`encoder` = profile dav2_encoder for DA V2)."""
    if family(model_id) == "dav2":
        repo, fn = dav2_files(spec, encoder)
        return DAV2Metric(repo, fn, encoder, max_depth=spec["max_depth"], input_size=spec.get("input_size", 518))
    return DA3Mono(spec["hf_repo"], spec["output"], process_res=spec.get("process_res", DA3_PROCESS_RES),
                   process_res_method=spec.get("process_res_method", DA3_PROCESS_RES_METHOD))


# ---------------------------------------------------------------- prediction cache (float16 npz per sample/camera)


def pred_cache_path(model_dir: str | Path, sample_token: str, cam: str) -> Path:
    return Path(model_dir) / "pred" / f"{sample_token}_{cam}.npz"


def save_prediction(path: str | Path, pred: DepthPrediction, meta: dict) -> Path:
    """float16 depth (+ optional bool sky) + json meta. float16 step is 1/32 m at 32-64 m, 1/16 m at 64-128 m."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"depth": np.clip(pred.depth, 0, 6.5e4).astype(np.float16),
              "meta": np.array(json.dumps({**meta, "kind": pred.kind, "info": pred.info}))}
    if pred.sky is not None:
        arrays["sky"] = pred.sky.astype(bool)
    np.savez_compressed(path, **arrays)
    return path


def load_prediction(path: str | Path) -> DepthPrediction:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        sky = z["sky"].astype(bool) if "sky" in z.files else None
        depth = z["depth"].astype(np.float32)
    info = {**meta.pop("info", {}), **meta}
    return DepthPrediction(depth, meta.get("kind", "metric"), sky=sky, info=info)


def cached_prediction_ok(path: str | Path, checkpoint: str) -> bool:
    """True if `path` exists and was written by `checkpoint` (guards against profile/encoder changes)."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return json.loads(str(z["meta"])).get("checkpoint") == checkpoint
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):  # corrupt / partial file
        return False
