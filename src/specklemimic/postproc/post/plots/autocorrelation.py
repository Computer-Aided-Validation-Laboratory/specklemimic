"""Autocorrelation figures: tiled comparison, 3D landscape, watershed verification."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from post.constants import MAX_PLOTS_PER_ROW
from post.correlators import get_correlator_info
from post.loaders import Grid, WatershedRing
from post.plots.common import add_heatmap, draw_watershed_2d, draw_watershed_3d

LAG_LABELS = ("X Lag (pixels)", "Y Lag (pixels)")
TILE_WIDTH_INCHES = 5.0
TILE_HEIGHT_INCHES = 4.6
RADIAL_RING_COLOUR = "black"  # Reads well on the coolwarm radial gradient.


def _tile_columns(index: int, total: int, per_row: int) -> tuple:
    """Gridspec row and column slice for tile ``index``; each tile spans 2 columns.

    An incomplete last row is centred by shifting it right by half the missing tiles.
    """
    row, position = divmod(index, per_row)
    tiles_in_row = min(per_row, total - row * per_row)
    shift = per_row - tiles_in_row  # 2 columns per tile -> half the gap is 'missing' columns
    start = shift + 2 * position
    return row, slice(start, start + 2)


def plot_correlation_grid(grids: Dict[str, Grid],
                          rings: Optional[Dict[str, WatershedRing]] = None) -> Figure:
    """Tile the autocorrelation heatmaps of several correlators.

    Parameters
    ----------
    grids : dict of str to Grid
        Lowercase correlator name -> autocorrelation map (order is kept).
    rings : dict of str to WatershedRing, optional
        If given, the matching watershed ring is overlaid on each tile.

    Returns
    -------
    Figure
        At most ``MAX_PLOTS_PER_ROW`` tiles per row, last row centred.
    """
    total = len(grids)
    per_row = min(MAX_PLOTS_PER_ROW, total)
    n_rows = int(np.ceil(total / per_row))

    fig = plt.figure(figsize=(TILE_WIDTH_INCHES * per_row, TILE_HEIGHT_INCHES * n_rows),
                     layout="constrained")
    gs = fig.add_gridspec(n_rows, 2 * per_row)

    for index, (name, grid) in enumerate(grids.items()):
        info = get_correlator_info(name)
        row, cols = _tile_columns(index, total, per_row)
        ax = fig.add_subplot(gs[row, cols])
        add_heatmap(ax, grid, info.cmap, info.title, *LAG_LABELS, "Value")
        if rings and name in rings:
            draw_watershed_2d(ax, rings[name], info.ring_colour)
    return fig


def plot_autocorrelation_3d(grid: Grid, name: str,
                            ring: Optional[WatershedRing] = None) -> Figure:
    """3D autocorrelation surface, optionally with the watershed ring on top.

    Parameters
    ----------
    grid : Grid
        Autocorrelation map.
    name : str
        Lowercase correlator name (sets title and colormap).
    ring : WatershedRing, optional
        Watershed to overlay; None draws the bare surface.
    """
    info = get_correlator_info(name)
    fig = plt.figure(figsize=(9, 7.5), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    x_mesh, y_mesh = np.meshgrid(grid.x, grid.y)
    surface = ax.plot_surface(x_mesh, y_mesh, grid.values, cmap=info.cmap,
                              edgecolor="none", alpha=0.9, antialiased=True)

    if ring is not None:
        z_range = float(np.nanmax(grid.values) - np.nanmin(grid.values))
        draw_watershed_3d(ax, ring, info.ring_colour, z_range)
        ax.legend(loc="upper right")

    suffix = " with Watershed" if ring is not None else ""
    ax.set_title(f"{info.short} Autocorrelation Surface{suffix}")
    ax.set_xlabel(LAG_LABELS[0])
    ax.set_ylabel(LAG_LABELS[1])
    ax.set_zlabel(f"{info.short} Value")
    ax.view_init(elev=30, azim=-60)
    fig.colorbar(surface, ax=ax, shrink=0.55, aspect=12, pad=0.08, label=f"{info.short} Value")
    return fig


def plot_correlation_with_radial_gradient(corr: Grid, gradr: Grid, name: str,
                                          ring: Optional[WatershedRing] = None) -> Figure:
    """Verification figure: autocorrelation (left) and radial gradient (right) with watershed.

    Parameters
    ----------
    corr : Grid
        Autocorrelation map.
    gradr : Grid
        Radial gradient dR/dr of that map.
    name : str
        Lowercase correlator name.
    ring : WatershedRing, optional
        Watershed overlaid on both panels.
    """
    info = get_correlator_info(name)
    fig, (ax_corr, ax_grad) = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")

    add_heatmap(ax_corr, corr, info.cmap, f"{info.short} Autocorrelation + Watershed",
                *LAG_LABELS, "Value")
    add_heatmap(ax_grad, gradr, "coolwarm", "Radial Gradient $(\\partial R / \\partial r)$ + Watershed",
                *LAG_LABELS, "$\\nabla_r R$")

    if ring is not None:
        draw_watershed_2d(ax_corr, ring, info.ring_colour, label="Watershed")
        draw_watershed_2d(ax_grad, ring, RADIAL_RING_COLOUR, label="Watershed")
        ax_corr.legend(loc="upper right")
        ax_grad.legend(loc="upper right")
    return fig