"""Directional gradient and gradient magnitude figure."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from post.loaders import Grid
from post.plots.common import add_heatmap, symmetric_limit

POS_LABELS = ("X Position (pixels)", "Y Position (pixels)")


def plot_gradients(grad_x: Grid, grad_y: Grid, grad_mag: Grid) -> Figure:
    """Top row: dF/dx and dF/dy. Bottom row (centred): gradient magnitude.

    The two directional plots share one symmetric colour range so they can be
    compared directly.
    """
    fig = plt.figure(figsize=(11, 9.5), layout="constrained")
    gs = fig.add_gridspec(2, 4)
    ax_x = fig.add_subplot(gs[0, 0:2])
    ax_y = fig.add_subplot(gs[0, 2:4])
    ax_mag = fig.add_subplot(gs[1, 1:3])

    limit = symmetric_limit(grad_x, grad_y)
    add_heatmap(ax_x, grad_x, "coolwarm", "Gradient X $(\\partial f / \\partial x)$",
                *POS_LABELS, "$\\nabla_x f$", vmin=-limit, vmax=limit)
    add_heatmap(ax_y, grad_y, "coolwarm", "Gradient Y $(\\partial f / \\partial y)$",
                *POS_LABELS, "$\\nabla_y f$", vmin=-limit, vmax=limit)
    add_heatmap(ax_mag, grad_mag, "viridis", "Gradient Magnitude",
                *POS_LABELS, "Magnitude")
    return fig