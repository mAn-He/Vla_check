"""Semantic segmentation metrics from a confusion matrix."""

from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int | None = None) -> np.ndarray:
    """(num_classes x num_classes) counts, rows = GT, cols = prediction."""
    pred, gt = np.asarray(pred).ravel().astype(np.int64), np.asarray(gt).ravel().astype(np.int64)
    m = (gt >= 0) & (gt < num_classes)
    if ignore_index is not None:
        m &= gt != ignore_index
    m &= (pred >= 0) & (pred < num_classes)
    return np.bincount(num_classes * gt[m] + pred[m], minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_from_confusion(conf: np.ndarray, class_names: list[str] | None = None) -> dict:
    """Per-class IoU and mIoU (mean over classes that appear in GT or prediction)."""
    conf = np.asarray(conf, float)
    tp = np.diag(conf)
    denom = conf.sum(0) + conf.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    names = class_names or [str(i) for i in range(len(iou))]
    out = {
        "miou": float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan"),
        "overall_acc": float(tp.sum() / conf.sum()) if conf.sum() else float("nan"),
    }
    out["per_class_iou"] = {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, iou)}
    return out
