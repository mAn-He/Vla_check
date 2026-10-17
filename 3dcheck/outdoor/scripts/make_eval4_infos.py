"""Build the eval4 mmdet3d info file and the custom devkit splits (3dv env).

Usage:  python outdoor/scripts/make_eval4_infos.py [--data-root data/nuscenes] [--extra-tag nuscenes]
(called by prepare_mmdet3d_infos.sh after tools/create_data.py)

Why: official checkpoints are trained on the official train split. eval4 = the 4 mini scenes that belong to the
official val split (2 from mini_train, 2 from mini_val; docs/step0_research.md section 2). mmdet3d only writes
mini_train / mini_val infos, so we merge both and keep the eval4 scenes.

Info format (mmdet3d 1.4.0 tools/dataset_converters/update_infos_to_v2.py::update_nuscenes_infos):
    dict(metainfo=dict(categories, dataset, version, info_version), data_list=[{token, sample_idx, ...}, ...])
`token` is the nuScenes sample token -> scene via the devkit (sample['scene_token']).
`sample_idx` is renumbered 0..N-1: NuScenesMetric.format_results indexes `data_list[sample_idx]` of its ann_file
(and FCOS3D's mv_image_based items use sample_idx * 6 + cam), so it must equal the position in this file.

Outputs (in data root):
    <tag>_infos_eval4.pkl
    v1.0-mini/splits.json  += {eval4, probe6, mini_val_official}  (bench3d.metrics.detection.write_custom_splits)
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    os.environ.setdefault("THREEDCHECK_ROOT", str(ROOT))
    from bench3d.config import load_config

    cfg = load_config("outdoor")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(cfg.path(cfg["dataset"]["data_root"])))
    ap.add_argument("--extra-tag", default="nuscenes")
    args = ap.parse_args()

    import mmengine
    from bench3d.metrics.detection import write_custom_splits
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits as nusplits

    data_root = Path(args.data_root).resolve()
    version = cfg["dataset"]["version"]
    eval4 = list(cfg["splits"]["eval4"])
    probe6 = list(cfg["splits"]["probe6"])
    mini_val = list(cfg["splits"]["mini_val_official"])

    # leakage guard: every eval scene must be in the official val split
    leak = [s for s in eval4 if s not in nusplits.val]
    assert not leak, f"eval4 scenes not in the official val split: {leak}"
    assert sorted(mini_val) == sorted(nusplits.mini_val), "config mini_val_official != devkit mini_val"

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    scene_of_token = {s["token"]: nusc.get("scene", s["scene_token"])["name"] for s in nusc.sample}
    n_keyframes = {sc["name"]: sc["nbr_samples"] for sc in nusc.scene}

    metainfo, entries = None, []
    for split in ("train", "val"):
        pkl = data_root / f"{args.extra_tag}_infos_{split}.pkl"
        data = mmengine.load(str(pkl))
        assert set(data) >= {"metainfo", "data_list"}, f"{pkl.name}: not a v2 info file (keys {list(data)})"
        assert data["metainfo"].get("version") == version, (pkl.name, data["metainfo"])
        if metainfo is None:
            metainfo = copy.deepcopy(data["metainfo"])
        else:  # union of category maps (ignored classes are mapped to -1 per file)
            metainfo["categories"].update(data["metainfo"]["categories"])
        for info in data["data_list"]:
            scene = scene_of_token[info["token"]]
            if scene in eval4:
                entries.append((eval4.index(scene), info["timestamp"], scene, info))

    entries.sort(key=lambda e: (e[0], e[1]))
    tokens = [e[3]["token"] for e in entries]
    assert len(tokens) == len(set(tokens)), "duplicate sample tokens across train/val infos"
    for scene in eval4:
        got = sum(e[2] == scene for e in entries)
        assert got == n_keyframes[scene], f"{scene}: {got} infos vs {n_keyframes[scene]} keyframes"

    data_list = []
    for i, (_, _, _, info) in enumerate(entries):
        info = copy.deepcopy(info)
        info["sample_idx"] = i
        data_list.append(info)
    out = data_root / f"{args.extra_tag}_infos_eval4.pkl"
    mmengine.dump(dict(metainfo=metainfo, data_list=data_list), str(out), "pkl")

    n_sweeps = [len(d.get("lidar_sweeps", [])) for d in data_list]
    print(f"wrote {out.name}: {len(data_list)} samples from {eval4}")
    print(f"  lidar sweeps per keyframe: min {min(n_sweeps)}, max {max(n_sweeps)}")
    sp = write_custom_splits(data_root, version, {"eval4": eval4, "probe6": probe6, "mini_val_official": mini_val})
    print(f"wrote {sp} (eval4, probe6, mini_val_official)")


if __name__ == "__main__":
    main()
