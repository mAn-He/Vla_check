"""nerfstudio CLI wrappers for scene-representation notebooks (3dv env, nerfstudio 1.1.5).

Everything here runs nerfstudio in a *subprocess* (ns-train / ns-eval / ns-render / mask script), so the notebook
kernel never holds the model on the GPU. Flags were checked against the installed nerfstudio 1.1.5 `--help`
output and source:

- ``ns-train <method> [trainer opts] nuscenes-data [parser opts]`` (tyro subcommand; trainer options must come
  before ``nuscenes-data``). Parser options: ``--data <scene name> --data-dir --version --cameras --mask-dir
  --train-split-fraction`` (``data/dataparsers/nuscenes_dataparser.py``). The parser has no downscale option;
  resolution is set with ``--pipeline.datamanager.camera-res-scale-factor`` (images, masks and intrinsics are
  rescaled in ``data/datasets/base_dataset.py``).
- Output layout ``<output-dir>/<experiment-name>/<method-name>/<timestamp>/config.yml`` and checkpoints
  ``nerfstudio_models/step-XXXXXXXXX.ckpt`` (``configs/experiment_config.py``, ``engine/trainer.py``).
- ``ns-eval --load-config --output-path [--render-output-path]`` writes
  ``{"experiment_name", "method_name", "checkpoint", "results": {psnr, psnr_std, ssim, ..., fps, fps_std}}``
  (``scripts/eval.py``, ``pipelines/base_pipeline.py:get_average_image_metrics``).
- Mask script ``nerfstudio/scripts/datasets/process_nuscenes_masks.py`` has no console entry point in 1.1.5
  (not in ``entry_points.txt``), so it is run with ``python -m``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MASK_MODULE = "nerfstudio.scripts.datasets.process_nuscenes_masks"
TB_GPU_MEMORY_TAG = "GPU Memory (MB)"  # engine/trainer.py: torch.cuda.max_memory_allocated() every steps_per_log
DEFAULT_TIMESTAMP = "bench"  # fixed --timestamp so the run directory is predictable


# ----------------------------------------------------------------------------------------------- commands
def ns_bin(name: str) -> str:
    """CLI from the same env as the running interpreter (falls back to $PATH)."""
    cand = Path(sys.executable).parent / name
    if cand.exists():
        return str(cand)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found next to {sys.executable} or on PATH")
    return found


def mask_cmd(data_dir, output_dir, version: str = "v1.0-mini", cameras=("FRONT",),
             velocity_thresh: float | None = None) -> list[str]:
    """Dynamic-object masks for *all* scenes of `version` (the script has no scene filter)."""
    cmd = [sys.executable, "-m", MASK_MODULE, "--data-dir", str(data_dir), "--output-dir", str(output_dir),
           "--version", version, "--cameras", *cameras]
    if velocity_thresh is not None:
        cmd += ["--velocity-thresh", str(velocity_thresh)]
    return cmd


def mask_path(mask_dir, channel: str, image_filename: str) -> Path:
    """Same rule as NuScenesDataParser: <mask_dir>/masks/<CAM_X>/<image basename with 'jpg' -> 'png'>."""
    return Path(mask_dir) / "masks" / channel / os.path.split(image_filename)[1].replace("jpg", "png")


def train_cmd(method: str, *, scene: str, data_dir, output_dir, experiment_name: str, max_num_iterations: int,
              seed: int, version: str = "v1.0-mini", cameras=("FRONT",), mask_dir=None,
              train_split_fraction: float = 0.9, scale_factor: float = 1.0, timestamp: str = DEFAULT_TIMESTAMP,
              method_args=()) -> list[str]:
    """ns-train command for the nuScenes parser without viewer and without in-training evaluation.

    In-training eval is disabled (step_check(step, 0) is always False) so wall-clock time is training only.
    `--vis tensorboard` keeps the viewer off and writes the "GPU Memory (MB)" scalar we read back.
    """
    cmd = [ns_bin("ns-train"), method,
           "--output-dir", str(output_dir), "--experiment-name", experiment_name, "--timestamp", timestamp,
           "--max-num-iterations", str(int(max_num_iterations)), "--machine.seed", str(int(seed)),
           "--vis", "tensorboard", "--viewer.quit-on-train-completion", "True",
           "--steps-per-eval-batch", "0", "--steps-per-eval-image", "0", "--steps-per-eval-all-images", "0",
           "--pipeline.datamanager.camera-res-scale-factor", str(float(scale_factor)),
           *[str(a) for a in method_args],
           "nuscenes-data", "--data", scene, "--data-dir", str(data_dir), "--version", version,
           "--cameras", *cameras, "--train-split-fraction", str(float(train_split_fraction))]
    if mask_dir is not None:
        cmd += ["--mask-dir", str(mask_dir)]
    return cmd


def eval_cmd(config_yml, output_json, render_dir=None) -> list[str]:
    """ns-eval; with render_dir it saves eval_<key>_<idx:04d>.png per eval image ("img" = [GT | render])."""
    cmd = [ns_bin("ns-eval"), "--load-config", str(config_yml), "--output-path", str(output_json)]
    if render_dir is not None:
        cmd += ["--render-output-path", str(render_dir)]
    return cmd


def render_cmd(config_yml, output_path, split: str = "test", names=("rgb", "raw-depth"),
               image_format: str = "png") -> list[str]:
    """ns-render dataset -> <output_path>/<split>/<name>/<image stem>.(png|npy.gz). raw-* are gzipped npy."""
    return [ns_bin("ns-render"), "dataset", "--load-config", str(config_yml), "--output-path", str(output_path),
            "--split", split, "--image-format", image_format, "--rendered-output-names", *names]


def fps_cmd(config_yml, output_json, n_warmup: int = 2, repeats: int = 3) -> list[str]:
    """Render-only timing in a separate process (see `_measure_fps`)."""
    return [sys.executable, "-m", "bench3d.models.nerfstudio_tools", "fps", "--load-config", str(config_yml),
            "--output-path", str(output_json), "--n-warmup", str(n_warmup), "--repeats", str(repeats)]


def run(cmd: list[str], log_path, check: bool = True, tail_lines: int = 40) -> dict:
    """Run `cmd`, stream stdout+stderr to `log_path`, return wall-clock seconds and return code."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    t0 = time.perf_counter()
    with open(log_path, "w") as f:
        f.write("$ " + shlex.join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    seconds = time.perf_counter() - t0
    if check and proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-tail_lines:])
        raise RuntimeError(f"command failed ({proc.returncode}), log: {log_path}\n{tail}")
    return {"returncode": proc.returncode, "seconds": seconds, "log": str(log_path), "cmd": shlex.join(cmd)}


# ----------------------------------------------------------------------------------------------- outputs
def find_config(output_dir, experiment_name: str, timestamp: str = DEFAULT_TIMESTAMP) -> Path | None:
    hits = sorted(Path(output_dir).glob(f"{experiment_name}/*/{timestamp}/config.yml"))
    if len(hits) > 1:
        raise RuntimeError(f"several runs match: {hits}")
    return hits[0] if hits else None


def latest_checkpoint(run_dir) -> Path | None:
    ckpts = sorted((Path(run_dir) / "nerfstudio_models").glob("step-*.ckpt"))
    return ckpts[-1] if ckpts else None


def parse_eval_json(path) -> dict:
    """Flat {metric: float} from ns-eval output (psnr, ssim, lpips, num_rays_per_sec, fps and their *_std)."""
    rec = json.loads(Path(path).read_text())
    return {k: float(v) for k, v in rec["results"].items()}


def peak_vram_from_tensorboard(run_dir) -> float | None:
    """Max of the "GPU Memory (MB)" scalar (torch max_memory_allocated, not nvidia-smi). None if unavailable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    if not list(Path(run_dir).glob("events.out.tfevents.*")):
        return None
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if TB_GPU_MEMORY_TAG not in ea.Tags().get("scalars", []):
        return None
    vals = [e.value for e in ea.Scalars(TB_GPU_MEMORY_TAG)]
    return float(max(vals)) if vals else None


def checkpoint_stats(ckpt_path) -> dict:
    """File size, learned parameter count/size (LPIPS backbone excluded) and Gaussian count for splatfacto.

    Checkpoints also hold optimizer state (Adam moments), so ckpt_mb is roughly 3x model_mb.
    Splatfacto keys are "_model.gauss_params.<name>" (VanillaPipeline._model, SplatfactoModel.gauss_params).
    """
    import torch

    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")
    pipe = state["pipeline"]
    learned = {k: v for k, v in pipe.items()
               if torch.is_tensor(v) and k.startswith("_model.") and ".lpips." not in k}
    n_gauss = next((int(v.shape[0]) for k, v in pipe.items() if k.endswith("gauss_params.means")), None)
    return {
        "step": int(state.get("step", -1)),
        "ckpt_mb": ckpt_path.stat().st_size / 2**20,
        "n_params": int(sum(v.numel() for v in learned.values())),
        "model_mb": sum(v.numel() * v.element_size() for v in learned.values()) / 2**20,
        "n_gaussians": n_gauss,
    }


def split_eval_image(path) -> tuple[np.ndarray, np.ndarray]:
    """ns-eval "img" output = torch.cat([gt, pred], dim=1) (width axis) -> (gt, pred) float [0,1] HxWx3."""
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    w = a.shape[1] // 2
    return a[:, :w], a[:, w:]


def load_npy_gz(path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.load(f)


# ----------------------------------------------------------------------------------------------- nuScenes parser
def nuscenes_split(n: int, train_split_fraction: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Exact copy of NuScenesDataParser's split: train = linspace(0, n-1, ceil(n*f)), eval = the rest."""
    n_train = math.ceil(n * train_split_fraction)
    i_all = np.arange(n)
    i_train = np.linspace(0, n - 1, n_train, dtype=int)
    i_eval = np.setdiff1d(i_all, i_train)
    return i_train, i_eval


def nuscenes_frames(nusc, scene: str, channel: str = "CAM_FRONT") -> list[dict]:
    """Keyframes of `scene` in the parser's order (sorted by timestamp) with the camera sample_data."""
    samples = [s for s in nusc.sample if nusc.get("scene", s["scene_token"])["name"] == scene]
    samples.sort(key=lambda s: (s["scene_token"], s["timestamp"]))
    out = []
    for s in samples:
        sd = nusc.get("sample_data", s["data"][channel])
        out.append({"sample_token": s["token"], "sd_token": sd["token"], "lidar_sd_token": s["data"]["LIDAR_TOP"],
                    "filename": sd["filename"], "timestamp": sd["timestamp"]})
    return out


def nuscenes_metres_per_unit(nusc, frames: list[dict]) -> float:
    """Metres per nerfstudio world unit for the nuScenes parser.

    The parser centres camera positions and divides by max|t| over all frames and axes. Its axis changes are signed
    permutations, so max|t| equals the same quantity in the global frame. The parser leaves
    dataparser_scale = 1.0, so ns-render raw-depth is in these normalized units, not metres.
    """
    from bench3d.nusc import sensor_to_global

    t = np.stack([sensor_to_global(nusc, f["sd_token"])[:3, 3] for f in frames])
    t = t - t.mean(0)
    return float(np.abs(t).max())


# ----------------------------------------------------------------------------------------------- fps subprocess
def _measure_fps(config_yml: Path, n_warmup: int = 2, repeats: int = 3) -> dict:
    """Pure render time per eval view (get_outputs_for_camera), CUDA-synchronized; no metric computation."""
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, ckpt, step = eval_setup(Path(config_yml), test_mode="test")
    pipeline.eval()
    items = list(pipeline.datamanager.fixed_indices_eval_dataloader)
    cam0 = items[0][0]
    times = []
    with torch.no_grad():
        for _ in range(n_warmup):
            pipeline.model.get_outputs_for_camera(camera=cam0)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            for camera, _batch in items:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pipeline.model.get_outputs_for_camera(camera=camera)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
    return {
        "render_fps": len(times) / sum(times),
        "render_ms_median": 1000 * float(np.median(times)),
        "height": int(cam0.height.flatten()[0]),
        "width": int(cam0.width.flatten()[0]),
        "n_views": len(items),
        "repeats": repeats,
        "render_peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint": str(ckpt),
        "step": int(step),
    }


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fps", help="measure render-only fps of a trained run on its eval split")
    f.add_argument("--load-config", type=Path, required=True)
    f.add_argument("--output-path", type=Path, required=True)
    f.add_argument("--n-warmup", type=int, default=2)
    f.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "fps":
        res = _measure_fps(args.load_config, args.n_warmup, args.repeats)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
