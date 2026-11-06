"""Raw speckle pattern figure."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from post.loaders import Grid
from post.plots.common import add_heatmap


def plot_speckle(speckle: Grid) -> Figure:
    """Plot the raw speckle pattern f(x, y).

    Parameters
    ----------
    speckle : Grid
        Speckle intensities, values shape (ny, nx).
    """
    fig, ax = plt.subplots(figsize=(7, 6.5), layout="constrained")
    add_heatmap(ax, speckle, "gray_r", "Speckle Pattern $f(x, y)$",
                "X Position (pixels)", "Y Position (pixels)", "Intensity")
    return fig