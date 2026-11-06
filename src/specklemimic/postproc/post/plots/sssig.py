"""SSSIG figures: raw/delta heatmaps and 3D surfaces."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from post.loaders import Grid
from post.plots.common import add_heatmap, symmetric_limit

POS_LABELS = ("X Position (pixels)", "Y Position (pixels)")
SSSIG_CMAP = "inferno"
DELTA_CMAP = "RdBu_r"  # Diverging: white = same as MIG, red = above, blue = below.


def plot_sssig_maps(sssig: Optional[Grid], delta: Optional[Grid]) -> Optional[Figure]:
    """Raw SSSIG heatmap and delta (SSSIG - MIG) heatmap side by side.

    Either grid may be None, in which case only the other is drawn.
    Returns None if both are missing.
    """
    panels = [grid for grid in (sssig, delta) if grid is not None]
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.8),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    slot = 0
    if sssig is not None:
        add_heatmap(axes[slot], sssig, SSSIG_CMAP, "SSSIG Heatmap", *POS_LABELS, "SSSIG", mesh=True)
        slot += 1
    if delta is not None:
        limit = symmetric_limit(delta)
        add_heatmap(axes[slot], delta, DELTA_CMAP, "SSSIG Delta Map ($\\mathrm{SSSIG} - \\mathrm{MIG}$)",
                    *POS_LABELS, "$\\Delta$ SSSIG", mesh=True, vmin=-limit, vmax=limit)
    return fig


def plot_sssig_surface(grid: Grid, is_delta: bool = False) -> Figure:
    """3D surface of the SSSIG map (or the delta map if ``is_delta``)."""
    cmap = DELTA_CMAP if is_delta else SSSIG_CMAP
    label = "$\\Delta$ SSSIG" if is_delta else "SSSIG"
    limits = {}
    if is_delta:
        limit = symmetric_limit(grid)
        limits = {"vmin": -limit, "vmax": limit}

    fig = plt.figure(figsize=(9, 7.5), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    x_mesh, y_mesh = np.meshgrid(grid.x, grid.y)
    surface = ax.plot_surface(x_mesh, y_mesh, grid.values, cmap=cmap, edgecolor="none",
                              alpha=0.9, antialiased=True, **limits)
    ax.set_title("SSSIG Delta Surface" if is_delta else "SSSIG Surface")
    ax.set_xlabel(POS_LABELS[0])
    ax.set_ylabel(POS_LABELS[1])
    ax.set_zlabel(label)
    ax.view_init(elev=30, azim=-60)
    fig.colorbar(surface, ax=ax, shrink=0.55, aspect=12, pad=0.08, label=label)
    return fig