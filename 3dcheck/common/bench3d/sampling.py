"""Deterministic evaluation-sample selection shared by all notebooks (so they compare the same frames)."""

from __future__ import annotations


def subsample(tokens: list[str], stride: int) -> list[str]:
    """Every `stride`-th item starting at 0."""
    return list(tokens[:: max(1, int(stride))])


def select_windows(tokens: list[str], n_frames: int, n_windows: int = 1) -> list[list[str]]:
    """`n_windows` non-overlapping windows of `n_frames` consecutive items, evenly spread over the sequence.

    With one window it is centred in the sequence (avoids the start where the ego car may be standing still).
    """
    n = len(tokens)
    if n < n_frames:
        raise ValueError(f"sequence of {n} < n_frames={n_frames}")
    n_windows = max(1, min(n_windows, n // n_frames))
    if n_windows == 1:
        start = (n - n_frames) // 2
        return [list(tokens[start : start + n_frames])]
    span = n - n_frames
    starts = [round(i * span / (n_windows - 1)) for i in range(n_windows)]
    return [list(tokens[s : s + n_frames]) for s in starts]


def middle(tokens: list[str]) -> str:
    return tokens[len(tokens) // 2]
