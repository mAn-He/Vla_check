"""Novel-view-synthesis image metrics. Images are float arrays in [0, 1], HxWx3."""

from __future__ import annotations

import numpy as np


def psnr(img: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    img, gt = np.asarray(img, float), np.asarray(gt, float)
    diff = (img - gt) ** 2
    if mask is not None:
        diff = diff[np.asarray(mask, bool)]
    mse = float(diff.mean())
    return float("inf") if mse == 0 else float(-10.0 * np.log10(mse))


def ssim(img: np.ndarray, gt: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(np.asarray(img, float), np.asarray(gt, float), channel_axis=-1, data_range=1.0))


_LPIPS = {}


def lpips(img: np.ndarray, gt: np.ndarray, net: str = "alex", device: str | None = None) -> float:
    """LPIPS via torchmetrics (downloads backbone weights on first use)."""
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if net not in _LPIPS:
        _LPIPS[net] = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(device)
    to_t = lambda a: torch.as_tensor(np.asarray(a, np.float32)).permute(2, 0, 1)[None].to(device)  # noqa: E731
    with torch.no_grad():
        return float(_LPIPS[net](to_t(img), to_t(gt)))
