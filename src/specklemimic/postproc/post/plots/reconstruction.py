"""Ground truth against reconstructed pattern, with error maps."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from post.loaders import Grid
from post.plots.common import add_heatmap, symmetric_limit

POS_LABELS = ("X Position (pixels)", "Y Position (pixels)")
PATTERN_CMAP = "gray_r"
DIFF_CMAP = "RdBu_r"   # Diverging: white = no error, red = reconstruction too bright.
ERROR_CMAP = "magma"
COLUMN_SPACING = 0.1   # Fraction of the figure width between the two columns.


def calc_error_maps(truth: Grid, result: Grid) -> tuple:
    """Signed difference (result - truth) and squared error maps.

    Both grids must be on the same points, which the backend guarantees because
    width, height and subpixel are fixed.
    """
    difference = Grid(x=truth.x, y=truth.y, values=result.values - truth.values)
    squared = Grid(x=truth.x, y=truth.y, values=difference.values ** 2)
    return difference, squared


def calc_relative_mse(truth: Grid, result: Grid) -> float:
    """Relative gray level MSE: sum((result - truth)^2) / sum(truth^2), as in the C++ loss."""
    norm = float(np.nansum(truth.values ** 2))
    squared_error = float(np.nansum((result.values - truth.values) ** 2))
    if norm <= 0.0:
        return squared_error / truth.values.size
    return squared_error / norm


def plot_reconstruction_comparison(truth: Grid, result: Grid,
                                   title: Optional[str] = None) -> Figure:
    """Ground truth and reconstruction side by side, with their error maps below.

    Parameters
    ----------
    truth, result : Grid
        Ground truth and reconstructed pattern on the same grid.
    title : str, optional
        Figure title, e.g. the loss used. The MSEs are always appended.

    Returns
    -------
    Figure
        2x2 figure: truth | reconstruction on top, signed difference |
        squared error below. The title carries the MSE and relative MSE.
    """
    difference, squared = calc_error_maps(truth, result)
    mse = float(np.nanmean(squared.values))
    relative_mse = calc_relative_mse(truth, result)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), layout="constrained")
    add_heatmap(axes[0, 0], truth, PATTERN_CMAP, "Ground Truth",
                *POS_LABELS, "Intensity", vmin=0.0, vmax=1.0)
    add_heatmap(axes[0, 1], result, PATTERN_CMAP, "Reconstruction",
                *POS_LABELS, "Intensity", vmin=0.0, vmax=1.0)

    limit = symmetric_limit(difference)
    add_heatmap(axes[1, 0], difference, DIFF_CMAP,
                "Difference (Reconstruction $-$ Ground Truth)", *POS_LABELS,
                "$\\Delta f$", vmin=-limit, vmax=limit)
    add_heatmap(axes[1, 1], squared, ERROR_CMAP, "Squared Error Map",
                *POS_LABELS, "$(\\Delta f)^2$", vmin=0.0)

    # The colorbars are not seen by the layout engine, so give the columns room
    # and drop the right-hand y labels (they would sit under the left colorbars).
    fig.get_layout_engine().set(wspace=COLUMN_SPACING)
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")

    heading = title or "Reconstruction vs Ground Truth"
    fig.suptitle(f"{heading}  |  MSE = {mse:.4g}  |  relative MSE = {relative_mse:.4g}")
    return fig
