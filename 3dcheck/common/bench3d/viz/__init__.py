"""Plot helpers shared by notebooks (matplotlib for static figures, plotly for interactive 3D)."""

from __future__ import annotations

import numpy as np

# Okabe-Ito: colorblind-safe categorical palette used for model series across all notebooks.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def model_colors(models: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def colorize(
    values: np.ndarray, vmin: float | None = None, vmax: float | None = None, cmap: str = "turbo"
) -> np.ndarray:
    """Map scalars to RGB uint8 (N,3) or (H,W,3)."""
    import matplotlib

    v = np.asarray(values, float)
    vmin = np.nanpercentile(v, 2) if vmin is None else vmin
    vmax = np.nanpercentile(v, 98) if vmax is None else vmax
    n = np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgb = matplotlib.colormaps[cmap](n)[..., :3]
    return (rgb * 255).astype(np.uint8)


def overlay_points(
    ax,
    image: np.ndarray,
    uv: np.ndarray,
    values: np.ndarray,
    vmax: float = 60.0,
    s: float = 1.0,
    title: str | None = None,
):
    """Image with projected points colored by `values` (e.g. depth)."""
    ax.imshow(image)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=np.clip(values, 0, vmax), s=s, cmap="turbo", vmin=0, vmax=vmax, linewidths=0)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    return sc


def depth_panel(ax, depth: np.ndarray, vmax: float = 60.0, title: str | None = None):
    im = ax.imshow(np.where(depth > 0, depth, np.nan), cmap="turbo", vmin=0, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    return im


def bev(
    ax,
    points: np.ndarray,
    extent: float = 50.0,
    color=None,
    s: float = 0.2,
    label: str | None = None,
    boxes_xy: list[np.ndarray] | None = None,
):
    """Bird's-eye view scatter in the ego/LiDAR frame (x forward, y left). boxes_xy: list of (4,2) corners."""
    pts = np.asarray(points)
    m = (np.abs(pts[:, 0]) < extent) & (np.abs(pts[:, 1]) < extent)
    if color is not None and not isinstance(color, str) and len(color) == len(pts):
        color = np.asarray(color)[m]  # per-point colors must follow the same mask
    ax.scatter(pts[m, 0], pts[m, 1], s=s, c=color, label=label, linewidths=0)
    for c in boxes_xy or []:
        cc = np.vstack([c, c[:1]])
        ax.plot(cc[:, 0], cc[:, 1], lw=0.8, color="#D55E00")
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plotly_points(
    clouds: dict[str, np.ndarray], max_points: int = 60000, size: float = 1.5, colors: dict | None = None
):
    """Interactive 3D scatter of several named clouds (each subsampled to max_points)."""
    import plotly.graph_objects as go

    rng = np.random.default_rng(0)
    colors = colors or model_colors(list(clouds))
    traces = []
    for name, pts in clouds.items():
        pts = np.asarray(pts)
        if len(pts) > max_points:
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=name,
                marker={"size": size, "color": colors.get(name)},
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(scene={"aspectmode": "data"}, height=650, margin={"l": 0, "r": 0, "t": 30, "b": 0})
    return fig


def error_vs_distance_plot(ax, curves: dict[str, dict], loglog: bool = True):
    """curves: model -> output of metrics.depth.error_vs_distance."""
    cols = model_colors(list(curves))
    for name, c in curves.items():
        if not c["centers"]:
            continue
        lab = f"{name} (k={c['slope_k']:.2f})"
        ax.plot(c["centers"], c["mean_abs_err"], "o-", ms=3, color=cols[name], label=lab)
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("GT distance [m]")
    ax.set_ylabel("mean |error| [m]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
