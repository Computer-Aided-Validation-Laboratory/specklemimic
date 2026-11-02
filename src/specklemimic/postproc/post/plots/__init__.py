"""Plotting functions, one module per figure family."""

from post.plots.autocorrelation import (
    plot_autocorrelation_3d, plot_correlation_grid, plot_correlation_with_radial_gradient,
)
from post.plots.common import apply_plot_style, finalize_animation, finalize_figure
from post.plots.gradients import plot_gradients
from post.plots.optimizer import (
    animate_landscape_3d, animate_swarm, plot_loss_contour, plot_loss_curve,
    plot_particle_paths,
)
from post.plots.reconstruction import (
    calc_error_maps, calc_relative_mse, plot_reconstruction_comparison,
)
from post.plots.speckle import plot_speckle
from post.plots.sssig import plot_sssig_maps, plot_sssig_surface

apply_plot_style()