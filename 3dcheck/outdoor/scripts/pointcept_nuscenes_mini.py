"""Build Pointcept nuScenes info pkls for v1.0-mini with our splits (train = probe6, val = eval4).

UNOFFICIAL ADAPTATION of Pointcept's nuScenes info preprocessing. Run in the 3dv-point env:

    python outdoor/scripts/pointcept_nuscenes_mini.py \
        [--eval-samples results/outdoor/01_data_exploration/eval_samples.json] [--out data/pointcept_nuscenes_mini]

What is reused (imported, not copied) from the Pointcept main checkout (envs/third_party.lock):
  pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py
      get_available_scenes(nusc), fill_trainval_infos(data_path, nusc, train_scenes, test, max_sweeps, with_camera)
  pointcept/datasets/preprocessing/concerto/nuscenes/preprocess_nuscenes_info.py  (only with --with-camera)
      same signature; additionally stores per-sweep camera infos and `is_key_frame` (needed by
      NuScenesImagePointDataset / the "*-withcolornormal-lin" configs, NOT by the "-5a-lin" config we use).

What differs from the official `__main__` of those scripts:
  1. Version: the official script hard-codes v1.0-trainval + v1.0-test and asserts 850 / 150 scenes.
     Here we load `dataset.version` from outdoor/config.yaml (v1.0-mini, 10 scenes) and assert our split instead.
  2. Split: the official train set is `nuscenes.utils.splits.train`, val = every other trainval scene.
     Here train = `splits.probe6`, val = `splits.eval4` (config). fill_trainval_infos puts every non-train sample
     into val, so val is filtered to eval4 scene tokens afterwards (with 10 mini scenes that is a no-op check).
  3. Optional `--eval-samples`: val is restricted to the eval4 keyframes listed in eval_samples.json (notebook 01),
     so notebooks evaluate exactly the same frames (docs/notebook_conventions.md section 5).
  4. No test pkl (there is no lidarseg-labelled test split in mini). Only *_train.pkl and *_val.pkl are written.
  5. Extra checks: every info has a lidarseg file whose label count equals the LiDAR point count.
  6. `<out>/raw` is created as a symlink to the nuScenes root (the README does this by hand with `ln -s`).
  7. A provenance file `<out>/info/mini_split_meta.json` (scenes, counts, pointcept commit, differences above).

Output layout (what pointcept/datasets/nuscenes.py NuScenesDataset expects):
  <out>/raw  -> data/nuscenes                (lidar_path and gt_segment_path in the infos are relative to it)
  <out>/info/nuscenes_infos_{max_sweeps}sweeps_train.pkl
  <out>/info/nuscenes_infos_{max_sweeps}sweeps_val.pkl
NuScenesDataset(sweeps=10) is the default, so keep --max-sweeps 10 unless the config's `sweeps` is changed too.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def third_party_dir(cfg: dict) -> Path:
    """Same rule as bench3d.config.Config.third_party: $THIRD_PARTY overrides `third_party` in the yaml."""
    if os.environ.get("THIRD_PARTY"):
        return Path(os.environ["THIRD_PARTY"])
    p = Path(cfg.get("third_party", "third_party"))
    return p if p.is_absolute() else ROOT / p


def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001 - provenance only
        return f"unknown ({type(e).__name__})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "outdoor" / "config.yaml"))
    ap.add_argument("--out", default="data/pointcept_nuscenes_mini", help="output root (repo-relative or absolute)")
    ap.add_argument("--eval-samples", default=None, help="eval_samples.json from notebook 01 (restricts val keyframes)")
    ap.add_argument("--max-sweeps", type=int, default=10)
    ap.add_argument("--with-camera", action="store_true", help="use the Concerto variant and store camera infos")
    ap.add_argument("--force", action="store_true", help="overwrite existing pkls / replace the raw symlink")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    version = ds["version"]
    data_root = (ROOT / ds["data_root"]).resolve()
    out = Path(args.out)
    out = (out if out.is_absolute() else ROOT / out).resolve()
    train_names, val_names = list(cfg["splits"]["probe6"]), list(cfg["splits"]["eval4"])
    assert not set(train_names) & set(val_names), "probe6 and eval4 overlap"

    if args.with_camera:
        from pointcept.datasets.preprocessing.concerto.nuscenes import preprocess_nuscenes_info as pre
    else:
        from pointcept.datasets.preprocessing.nuscenes import preprocess_nuscenes_info as pre
    import pointcept
    from nuscenes.nuscenes import NuScenes

    print(f"pointcept: {Path(pointcept.__file__).parent}  module: {pre.__name__}")
    print(f"loading {version} from {data_root}")
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    if not getattr(nusc, "lidarseg", None):
        print("lidarseg tables not loaded: extract the lidarseg-mini archive into the dataset root first.")
        return 1

    # 1. scenes: official script asserts 850/150; we assert our split exists and every scene has its files
    available = pre.get_available_scenes(nusc)
    name2tok = {s["name"]: s["token"] for s in available}
    missing = [n for n in train_names + val_names if n not in name2tok]
    assert not missing, f"scenes missing on disk or not in {version}: {missing}"
    train_tokens = {name2tok[n] for n in train_names}
    val_tokens = {name2tok[n] for n in val_names}

    # 2. reuse the official info builder (data_path must match nusc.dataroot for relative_to)
    train_infos, val_infos = pre.fill_trainval_infos(str(data_root), nusc, train_tokens, test=False,
                                                     max_sweeps=args.max_sweeps, with_camera=args.with_camera)
    scene_of = {s["token"]: s["scene_token"] for s in nusc.sample}
    n_val_all = len(val_infos)
    val_infos = [i for i in val_infos if scene_of[i["token"]] in val_tokens]
    print(f"train infos: {len(train_infos)}  val infos: {len(val_infos)} (of {n_val_all} non-train samples)")

    # 3. optional keyframe restriction from notebook 01
    restricted = False
    if args.eval_samples:
        es = json.loads(Path(args.eval_samples).read_text())
        assert sorted(es["eval4"]) == sorted(val_names), "eval_samples.json eval4 differs from config splits.eval4"
        assert sorted(es["probe6"]) == sorted(train_names), "eval_samples.json probe6 differs from config"
        keep = {t for n in val_names for t in es["keyframes"][n]}
        val_infos = [i for i in val_infos if i["token"] in keep]
        restricted = True
        print(f"val restricted to {len(val_infos)} eval_samples.json keyframes")

    # 4. label / point count check (NuScenesDataset reshapes the .bin to (-1, 5))
    bad = []
    for info in train_infos + val_infos:
        n_pts = (data_root / info["lidar_path"]).stat().st_size // (5 * 4)
        seg = data_root / info["gt_segment_path"]
        n_lab = seg.stat().st_size if seg.exists() else -1  # uint8 per point
        if n_pts != n_lab:
            bad.append((info["token"], n_pts, n_lab))
    assert not bad, f"{len(bad)} samples with missing / mismatched lidarseg labels, e.g. {bad[:3]}"

    # 5. write
    info_dir = out / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, infos in (("train", train_infos), ("val", val_infos)):
        p = info_dir / f"nuscenes_infos_{args.max_sweeps}sweeps_{split}.pkl"
        if p.exists() and not args.force:
            print(f"exists, keeping (use --force): {p}")
        else:
            with open(p, "wb") as f:
                pickle.dump(infos, f)
            print(f"wrote {p} ({len(infos)} samples)")
        files[split] = p.name

    raw = out / "raw"
    if raw.is_symlink():
        if raw.resolve() != data_root:
            if not args.force:
                print(f"{raw} points to {raw.resolve()}, expected {data_root} (use --force)")
                return 1
            raw.unlink()
    elif raw.exists():
        print(f"{raw} exists and is not a symlink; remove it or choose another --out")
        return 1
    if not raw.is_symlink():
        raw.symlink_to(data_root, target_is_directory=True)
        print(f"linked {raw} -> {data_root}")

    meta = {
        "adaptation": "UNOFFICIAL: Pointcept nuScenes info preprocessing adapted to v1.0-mini (see script docstring)",
        "script": "outdoor/scripts/pointcept_nuscenes_mini.py",
        "pointcept_module": pre.__name__,
        "pointcept_commit": git_head(third_party_dir(cfg) / "Pointcept"),
        "version": version,
        "max_sweeps": args.max_sweeps,
        "with_camera": args.with_camera,
        "train_scenes": train_names,
        "val_scenes": val_names,
        "val_restricted_to_eval_samples": restricted,
        "n_train": len(train_infos),
        "n_val": len(val_infos),
        "files": files,
        "n_points_val": int(np.sum([(data_root / i["lidar_path"]).stat().st_size // 20 for i in val_infos])),
    }
    (info_dir / "mini_split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
