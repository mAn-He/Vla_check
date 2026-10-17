"""Seeding, version logging, timing and peak-VRAM measurement."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import time
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def package_versions(names: list[str]) -> dict[str, str]:
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def env_report(packages: list[str] | None = None) -> dict:
    """Python / torch / CUDA / GPU info plus versions of `packages`."""
    rep = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["torch_cuda"] = torch.version.cuda
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            rep["gpu"] = prop.name
            rep["gpu_total_mb"] = round(prop.total_memory / 2**20)
            rep["gpu_capability"] = f"{prop.major}.{prop.minor}"
            rep["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        rep["torch"] = "not installed"
    if packages:
        rep["packages"] = package_versions(packages)
    return rep


def suggest_profile(gpu_total_mb: float | None) -> str:
    """low <= 8GB < mid <= 16GB < high (matches CLAUDE.md)."""
    if not gpu_total_mb:
        return "low"
    if gpu_total_mb <= 8 * 1024:
        return "low"
    if gpu_total_mb <= 16 * 1024:
        return "mid"
    return "high"


class Measure:
    """Wall time and peak CUDA memory of a block.

    >>> with Measure() as m:
    ...     run()
    >>> m.seconds, m.peak_vram_mb
    """

    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync
        self.seconds = None
        self.peak_vram_mb = None

    def __enter__(self):
        try:
            import torch

            self._torch = torch if torch.cuda.is_available() else None
        except ImportError:
            self._torch = None
        if self._torch:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._torch and self.cuda_sync:
            self._torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._t0
        if self._torch:
            self.peak_vram_mb = self._torch.cuda.max_memory_allocated() / 2**20
        return False


@contextmanager
def free_cuda():
    """Release cached CUDA memory after a model block (use between models in one notebook)."""
    try:
        yield
    finally:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
