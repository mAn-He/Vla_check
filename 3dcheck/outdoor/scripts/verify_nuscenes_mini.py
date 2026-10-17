"""Verify the nuScenes-mini (+ lidarseg-mini) layout and that the devkit can load it.

Usage (3dv or 3dv-point kernel/env):  python outdoor/scripts/verify_nuscenes_mini.py [--config outdoor/config.yaml]

Checks
  1. folders: samples/ sweeps/ maps/ v1.0-mini/ lidarseg/
  2. devkit load of v1.0-mini (with lidarseg tables)
  3. scene count = 10 and the config splits (eval4, probe6) exist and partition the mini scenes
  4. eval4 scenes are all in the official val split (no train leakage for official checkpoints)
  5. every keyframe LIDAR_TOP has a lidarseg label file and the label count matches the point count
  6. a few sample files (camera jpg, lidar .bin) exist on disk
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


class Report:
    def __init__(self):
        self.n_fail = 0

    def check(self, ok, msg, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}" + (f"  ({detail})" if detail else ""))
        self.n_fail += not ok
        return ok

    def warn(self, msg):
        print(f"  WARN  {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "outdoor" / "config.yaml"))
    ap.add_argument("--n-label-checks", type=int, default=20, help="lidarseg files to open and size-check")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    data_root = (ROOT / ds["data_root"]).resolve()
    version = ds["version"]
    r = Report()

    print(f"[1] folders under {data_root}")
    for sub in ["samples", "sweeps", "maps", version, "lidarseg"]:
        r.check((data_root / sub).is_dir(), f"{sub}/")
    if not (data_root / version).is_dir():
        print("\nData not found. Run outdoor/scripts/place_nuscenes_mini.sh first.")
        sys.exit(1)
    r.check((data_root / version / "lidarseg.json").is_file(), f"{version}/lidarseg.json (from lidarseg archive)")

    print("\n[2] devkit load")
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits as nusplits

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    has_seg = hasattr(nusc, "lidarseg") and len(nusc.lidarseg) > 0
    r.check(True, f"NuScenes({version}) loaded", f"{len(nusc.scene)} scenes, {len(nusc.sample)} samples")
    r.check(has_seg, "nusc.lidarseg table loaded")

    print("\n[3] scenes and config splits")
    names = {s["name"] for s in nusc.scene}
    r.check(len(names) == 10, "10 scenes in v1.0-mini", f"found {len(names)}")
    r.check(names == set(nusplits.mini_train) | set(nusplits.mini_val), "scenes == devkit mini_train + mini_val")
    eval4, probe6 = set(cfg["splits"]["eval4"]), set(cfg["splits"]["probe6"])
    r.check(eval4 <= names and probe6 <= names, "eval4 / probe6 scenes exist")
    r.check(not (eval4 & probe6) and (eval4 | probe6) == names, "eval4 and probe6 partition the mini scenes")

    print("\n[4] leakage guard")
    official_train = set(nusplits.train)
    leaked = sorted(eval4 & official_train)
    r.check(not leaked, "eval4 has no official-train scene", f"leaked={leaked}" if leaked else "all in official val")

    print("\n[5] lidarseg labels")
    if has_seg:
        lidar_tokens = [s["data"]["LIDAR_TOP"] for s in nusc.sample]
        seg_tokens = {rec["sample_data_token"] for rec in nusc.lidarseg}
        missing = [t for t in lidar_tokens if t not in seg_tokens]
        r.check(
            not missing, "every keyframe LIDAR_TOP has a lidarseg record", f"missing {len(missing)}/{len(lidar_tokens)}"
        )
        for rec in nusc.lidarseg[: args.n_label_checks]:
            sd = nusc.get("sample_data", rec["sample_data_token"])
            pts = np.fromfile(data_root / sd["filename"], dtype=np.float32)
            labels = np.fromfile(data_root / rec["filename"], dtype=np.uint8)
            # nuScenes lidar .bin stores 5 float32 per point (devkit LidarPointCloud.from_file reshapes to (-1, 5))
            if not r.check(
                pts.size % 5 == 0 and labels.size == pts.size // 5,
                f"label count == point count ({Path(rec['filename']).name})",
                f"{labels.size} vs {pts.size / 5:g}",
            ):
                break
        else:
            print(f"        ({min(args.n_label_checks, len(nusc.lidarseg))} files size-checked)")

    print("\n[6] sample files on disk")
    s0 = nusc.sample[0]
    for ch in ["CAM_FRONT", "LIDAR_TOP"]:
        sd = nusc.get("sample_data", s0["data"][ch])
        r.check((data_root / sd["filename"]).is_file(), f"{ch} file exists", sd["filename"])

    print(f"\n{'OK' if r.n_fail == 0 else 'FAILED'}: {r.n_fail} failed check(s)")
    sys.exit(r.n_fail)


if __name__ == "__main__":
    main()
