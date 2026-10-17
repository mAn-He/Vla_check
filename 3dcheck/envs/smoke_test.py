"""Import smoke test for a 3dcheck env.

Usage: python envs/smoke_test.py {3dv|3dv-point} [--json out.json]

Only imports modules and checks versions; no model is built and nothing touches a GPU.
Exit code is the number of failed checks.
"""

import argparse
import importlib
import json
import platform
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")


def _ver(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "?")
    except Exception:  # noqa: BLE001 - version lookup is best effort
        return None


def _nerfstudio_methods():
    from nerfstudio.configs.method_configs import method_configs

    missing = {"nerfacto", "instant-ngp", "splatfacto"} - set(method_configs)
    assert not missing, f"missing methods: {missing}"
    from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig  # noqa: F401


def _numpy_below_2():
    import numpy

    assert int(numpy.__version__.split(".")[0]) < 2, numpy.__version__


# (label, module for version, python statement to exec)
CHECKS = {
    "3dv": [
        ("torch", "torch", "import torch"),
        ("torchvision", "torchvision", "import torchvision"),
        ("xformers", "xformers", "import xformers.ops"),
        ("numpy<2", "numpy", _numpy_below_2),
        ("nuscenes-devkit", None, "from nuscenes.nuscenes import NuScenes; from nuscenes.eval.detection.evaluate import DetectionEval"),
        ("open3d", "open3d", "import open3d"),
        ("plotly", "plotly", "import plotly.graph_objects"),
        ("mmengine", "mmengine", "import mmengine"),
        ("mmcv (+ops)", "mmcv", "from mmcv.ops import Voxelization, nms"),
        ("mmdet", "mmdet", "import mmdet"),
        ("mmdet3d", "mmdet3d", "from mmdet3d.apis import init_model, inference_detector"),
        ("nerfstudio", "nerfstudio", _nerfstudio_methods),
        ("gsplat", "gsplat", "import gsplat"),
        ("pycolmap", "pycolmap", "import pycolmap"),
        ("onnx / onnxruntime", "onnxruntime", "import onnx, onnxruntime"),
        ("DUSt3R", None, "from dust3r.model import AsymmetricCroCo3DStereo; from dust3r.cloud_opt import global_aligner"),
        ("MASt3R", None, "from mast3r.model import AsymmetricMASt3R; from mast3r.cloud_opt.sparse_ga import sparse_global_alignment"),
        ("VGGT", None, "from vggt.models.vggt import VGGT; from vggt.utils.pose_enc import pose_encoding_to_extri_intri"),
        ("Depth Anything 3", None, "from depth_anything_3.api import DepthAnything3"),
        ("Depth Anything V2 (metric)", None, "from depth_anything_v2.dpt import DepthAnythingV2"),
        ("bench3d", "bench3d", "import bench3d"),
    ],
    "3dv-point": [
        ("torch", "torch", "import torch"),
        ("numpy<2", "numpy", _numpy_below_2),
        ("spconv", "spconv", "import spconv.pytorch"),
        ("torch-scatter", "torch_scatter", "import torch_scatter"),
        ("torch-cluster", "torch_cluster", "import torch_cluster"),
        ("Pointcept", None, "import pointcept; from pointcept.models import build_model"),
        ("Sonata", None, "import sonata"),
        ("Concerto", None, "import concerto"),
        ("nuscenes-devkit", None, "from nuscenes.utils.data_classes import LidarSegPointCloud"),
        ("open3d", "open3d", "import open3d"),
        ("bench3d", "bench3d", "import bench3d"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env", choices=sorted(CHECKS))
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args()

    rows = []
    for label, ver_mod, stmt in CHECKS[args.env]:
        t0 = time.time()
        try:
            stmt() if callable(stmt) else exec(stmt, {})  # noqa: S102 - constant import strings
            ok, err = True, ""
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            ok = False
            err = f"{type(e).__name__}: {e}".splitlines()[0][:160]
            if "-v" in sys.argv:
                traceback.print_exc()
        rows.append(
            {
                "check": label,
                "ok": ok,
                "version": _ver(ver_mod) if ver_mod else "",
                "seconds": round(time.time() - t0, 2),
                "error": err,
            }
        )

    try:
        import torch

        cuda = f"torch.cuda.is_available()={torch.cuda.is_available()}, built for CUDA {torch.version.cuda}"
    except Exception:  # noqa: BLE001
        cuda = "torch not importable"

    print(f"\nenv={args.env}  python={platform.python_version()}  {cuda}")
    w = max(len(r["check"]) for r in rows)
    for r in rows:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"  {mark}  {r['check']:<{w}}  {r['version']:<14} {r['error']}")
    n_fail = sum(not r["ok"] for r in rows)
    print(f"{len(rows) - n_fail}/{len(rows)} passed")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"env": args.env, "python": platform.python_version(), "cuda": cuda, "checks": rows}, f, indent=2)
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
