"""nuScenes detection evaluation on custom scene subsets with optional FOV / distance filters.

Uses the devkit's own DetectionEval (mAP, NDS, TP errors). Custom splits come from
`<dataroot>/<version>/splits.json`, which devkit >= 1.2 reads for non-predefined eval_set names.
Extra filters are applied identically to predictions and ground truth after the devkit's own
class-range / zero-point / bike-rack filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..nusc import box_center_in_cameras, sample_cameras


def write_custom_splits(dataroot: str | Path, version: str, splits: dict[str, list[str]]) -> Path:
    """Add/overwrite scene-name splits in <dataroot>/<version>/splits.json (other keys are kept)."""
    path = Path(dataroot) / version / "splits.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update({k: list(v) for k, v in splits.items()})
    path.write_text(json.dumps(data, indent=2))
    return path


def _filtered_eval_cls():
    from nuscenes.eval.detection.evaluate import DetectionEval

    class FilteredDetectionEval(DetectionEval):
        def __init__(self, *args, keep_box=None, **kwargs):
            super().__init__(*args, **kwargs)
            if keep_box is not None:
                for boxes in (self.pred_boxes, self.gt_boxes):
                    for tok in boxes.sample_tokens:
                        boxes.boxes[tok] = [b for b in boxes[tok] if keep_box(tok, b)]
            self.sample_tokens = self.gt_boxes.sample_tokens

    return FilteredDetectionEval


def evaluate_detection(nusc, result_path: str | Path, eval_set: str, output_dir: str | Path,
                       fov_cams: tuple[str, ...] | None = None, dist_range: tuple[float, float] | None = None,
                       config_name: str = "detection_cvpr_2019", verbose: bool = False) -> dict:
    """Run nuScenes detection eval. Returns flat headline numbers plus per-class APs and TP errors.

    fov_cams:   keep only boxes whose center is inside the image of at least one of these cameras
                (used to compare LiDAR with camera-only inputs on the same visible region).
    dist_range: keep only boxes with ego distance in [lo, hi) metres (distance-binned AP).
    """
    from nuscenes.eval.common.config import config_factory

    cam_cache: dict[str, list[dict]] = {}

    def keep(tok, box):
        if dist_range is not None and not (dist_range[0] <= box.ego_dist < dist_range[1]):
            return False
        if fov_cams:
            if tok not in cam_cache:
                cam_cache[tok] = sample_cameras(nusc, tok, fov_cams)
            if not box_center_in_cameras(box.translation, cam_cache[tok]):
                return False
        return True

    use_filter = fov_cams is not None or dist_range is not None
    Eval = _filtered_eval_cls()
    ev = Eval(nusc, config=config_factory(config_name), result_path=str(result_path), eval_set=eval_set,
              output_dir=str(output_dir), verbose=verbose, keep_box=keep if use_filter else None)
    metrics, _ = ev.evaluate()
    s = metrics.serialize()
    n_gt = sum(len(ev.gt_boxes[t]) for t in ev.gt_boxes.sample_tokens)
    return {
        "mAP": s["mean_ap"],
        "NDS": s["nd_score"],
        **s["tp_errors"],
        "per_class_ap": s["mean_dist_aps"],
        "n_gt_boxes": n_gt,
        "fov_cams": list(fov_cams) if fov_cams else None,
        "dist_range": list(dist_range) if dist_range else None,
    }

