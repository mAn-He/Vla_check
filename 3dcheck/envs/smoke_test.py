"""Import smoke test for a 3dcheck env.

Usage: python envs/smoke_test.py {3dv|3dv-point} [--json out.json]

Only imports modules and checks versions; no model is built and nothing touches a GPU.
Exit code is the number of failed checks.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path(__file__).resolve().parents[1] / "third_party"))


class Skip(Exception):
    """Optional component that was not built; reported as SKIP, not as a failure."""


def _ver(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", None)
        return ver or importlib.metadata.version(mod_name)
    except Exception:  # noqa: BLE001 - version lookup is best effort
        return ""


def _nerfstudio_methods():
    from nerfstudio.configs.method_configs import method_configs

    missing = {"nerfacto", "instant-ngp", "splatfacto"} - set(method_configs)
    assert not missing, f"missing methods: {missing}"
    from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig  # noqa: F401


def _numpy_below_2():
    import numpy

    assert int(numpy.__version__.split(".")[0]) < 2, numpy.__version__


def _bevfusion_ops():
    root = THIRD_PARTY / "mmdetection3d"
    if not list((root / "projects/BEVFusion/bevfusion/ops/bev_pool").glob("bev_pool_ext*.so")):
        raise Skip("not built (BUILD_EXT=1)")
    sys.path.insert(0, str(root))
    import projects.BEVFusion.bevfusion  # noqa: F401  (registers BEVFusion modules)
    from mmdet3d.registry import MODELS
    from projects.BEVFusion.bevfusion.ops.bev_pool import bev_pool_ext  # noqa: F401
    from projects.BEVFusion.bevfusion.ops.voxel import voxel_layer  # noqa: F401

    assert "BEVFusion" in MODELS.module_dict, "BEVFusion not registered"


def _curope():
    croco = THIRD_PARTY / "mast3r/dust3r/croco"
    if not list((croco / "models/curope").glob("curope*.so")):
        raise Skip("not built (BUILD_EXT=1)")
    sys.path.insert(0, str(croco))
    from models.curope import cuRoPE2D  # noqa: F401


def _pointops():
    if importlib.util.find_spec("pointops") is None:
        raise Skip("not built (POINTOPS=1)")
    import pointops  # noqa: F401


def _pointcept_models():
    from pointcept.models.builder import MODELS

    assert "PT-v3m1" in MODELS.module_dict, "PTv3 (PT-v3m1) not registered"


# (label, module for version, python statement to exec)
CHECKS = {
    "3dv": [
        ("torch", "torch", "import torch"),
        ("torchvision", "torchvision", "import torchvision"),
        ("xformers", "xformers", "import xformers.ops"),
        ("numpy<2", "numpy", _numpy_below_2),
        (
            "nuscenes-devkit",
            None,
            "from nuscenes.nuscenes import NuScenes; from nuscenes.eval.detection.evaluate import DetectionEval",
        ),
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
        (
            "DUSt3R",
            None,
            "from dust3r.model import AsymmetricCroCo3DStereo; from dust3r.cloud_opt import global_aligner",
        ),
        (
            "MASt3R",
            None,
            "from mast3r.model import AsymmetricMASt3R; from mast3r.cloud_opt.sparse_ga import sparse_global_alignment",
        ),
        (
            "VGGT",
            None,
            "from vggt.models.vggt import VGGT; from vggt.utils.pose_enc import pose_encoding_to_extri_intri",
        ),
        ("Depth Anything 3", None, "from depth_anything_3.api import DepthAnything3"),
        ("Depth Anything V2 (metric)", None, "from depth_anything_v2.dpt import DepthAnythingV2"),
        ("bench3d", "bench3d", "import bench3d"),
        ("BEVFusion ops (opt.)", None, _bevfusion_ops),
        ("DUSt3R curope (opt.)", None, _curope),
    ],
    "3dv-point": [
        ("torch", "torch", "import torch"),
        ("numpy<2", "numpy", _numpy_below_2),
        ("spconv", "spconv", "import spconv.pytorch"),
        ("torch-scatter", "torch_scatter", "import torch_scatter"),
        ("torch-cluster", "torch_cluster", "import torch_cluster"),
        ("Pointcept models (+PTv3)", None, _pointcept_models),
        ("Pointcept train/test engines", None, "import pointcept.engines.train, pointcept.engines.test"),
        (
            "Pointcept nuScenes preprocessing",
            None,
            (
                "import pointcept.datasets.preprocessing.nuscenes.preprocess_nuscenes_info, "
                "pointcept.datasets.preprocessing.concerto.nuscenes.preprocess_nuscenes_info"
            ),
        ),
        ("pointops (opt.)", None, _pointops),
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
        except Skip as e:
            ok, err = None, str(e)
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
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[r["ok"]]
        print(f"  {mark}  {r['check']:<{w}}  {r['version']:<14} {r['error']}")
    n_fail = sum(r["ok"] is False for r in rows)
    n_skip = sum(r["ok"] is None for r in rows)
    print(f"{len(rows) - n_fail - n_skip}/{len(rows)} passed, {n_skip} skipped (optional, not built)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"env": args.env, "python": platform.python_version(), "cuda": cuda, "checks": rows}, f, indent=2)
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
