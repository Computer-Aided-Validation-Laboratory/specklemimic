"""Shared plotting helpers: styling, heatmaps, watershed overlays, saving."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from post.console import log_error, log_info, log_warning
from post.constants import DEFAULT_DPI
from post.loaders import Grid, WatershedRing

# Fraction of the z-range the 3D watershed ring floats above the surface (stops it being hidden).
RING_Z_OFFSET_FRACTION = 0.02


def apply_plot_style() -> None:
    """Set global matplotlib defaults used by every figure."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "savefig.dpi": DEFAULT_DPI,
    })


def finalize_figure(fig: Figure, filename: str, output_dir: Path,
                    save: bool, show: bool) -> Optional[Path]:
    """Save and/or keep a figure open for ``plt.show()``.

    Parameters
    ----------
    fig : Figure
        Finished figure.
    filename : str
        File name inside ``output_dir``.
    output_dir : Path
        Destination folder, created if needed.
    save, show : bool
        Save to disk and/or leave open for interactive display.
        If ``show`` is False the figure is closed to free memory.

    Returns
    -------
    Path or None
        Saved path, or None if nothing was saved.
    """
    saved_path = None
    if save:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_path = output_dir / filename
            fig.savefig(saved_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            log_info(f"Saved: {saved_path}")
        except OSError as exc:
            log_error(f"Could not save {filename}: {exc}")
            saved_path = None
    if not show:
        plt.close(fig)
    return saved_path


def add_heatmap(ax, grid: Grid, cmap: str, title: str, xlabel: str, ylabel: str,
                cbar_label: str, mesh: bool = False, vmin: Optional[float] = None,
                vmax: Optional[float] = None):
    """Draw a 2D heatmap with a colorbar.

    ``mesh=True`` uses pcolormesh (safe for uneven spacing, e.g. SSSIG with a
    clamped last subset); otherwise imshow, which is faster for big images.
    """
    if mesh:
        image = ax.pcolormesh(grid.x, grid.y, grid.values, cmap=cmap,
                              shading="nearest", vmin=vmin, vmax=vmax)
    else:
        image = ax.imshow(grid.values, extent=grid.extent, origin="lower", cmap=cmap,
                          interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.1)
    colourbar = ax.figure.colorbar(image, cax=cax)
    colourbar.set_label(cbar_label)
    return image


def symmetric_limit(*grids: Grid) -> float:
    """Largest absolute value over the grids, for colour ranges centred on zero."""
    limit = max(float(np.nanmax(np.abs(grid.values))) for grid in grids)
    return limit if limit > 0 else 1.0  # avoid a degenerate colour range


def _outline_for(colour: str) -> str:
    # Dark outline for light rings and vice versa, keeps the ring readable on any colormap.
    luminance = np.dot(to_rgb(colour), (0.299, 0.587, 0.114))
    return "black" if luminance > 0.5 else "white"


def draw_watershed_2d(ax, ring: WatershedRing, colour: str, label: Optional[str] = None) -> None:
    """Overlay the watershed ring on a 2D axis."""
    outline = _outline_for(colour)
    xs, ys, _ = ring.closed()
    ax.plot(xs, ys, color=colour, linewidth=1.8, label=label, zorder=5,
            path_effects=[patheffects.withStroke(linewidth=3.4, foreground=outline)])
    ax.scatter(ring.x, ring.y, color=colour, s=10, edgecolors=outline,
               linewidths=0.5, zorder=6)


def draw_watershed_3d(ax, ring: WatershedRing, colour: str, z_range: float,
                      label: str = "Watershed") -> None:
    """Overlay the watershed ring on a 3D surface (needs the ring's z values)."""
    if ring.z is None:
        log_warning("Watershed file has no value column, skipping the 3D ring")
        return
    xs, ys, zs = ring.closed()
    zs = zs + RING_Z_OFFSET_FRACTION * z_range
    ax.plot(xs, ys, zs, color=_outline_for(colour), linewidth=5, zorder=10)
    ax.plot(xs, ys, zs, color=colour, linewidth=2.5, label=label, zorder=11)