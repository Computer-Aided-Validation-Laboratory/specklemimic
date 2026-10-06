"""Plotting functions, one module per figure family."""

from post.plots.autocorrelation import (
    plot_autocorrelation_3d, plot_correlation_grid, plot_correlation_with_radial_gradient,
)
from post.plots.common import apply_plot_style, finalize_figure
from post.plots.gradients import plot_gradients
from post.plots.speckle import plot_speckle
from post.plots.sssig import plot_sssig_maps, plot_sssig_surface

apply_plot_style()