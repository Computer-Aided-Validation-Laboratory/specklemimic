"""Optimizer figures and animations for 2D runs (particle swarm history).

Static: loss curve, particle trajectories, loss contours from the particles.
Animated: swarm + loss curve, and a 3D landscape that fills in as particles
reach new positions. Animation builders return ``(fig, anim)``, pass both to
``finalize_animation`` to save or show them.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

from post.console import log_warning
from post.constants import ANIMATION_FPS, MAX_ANIMATION_FRAMES
from post.loaders import OptimizerHistory, OptimizerLoss

AxisLabels = Tuple[str, str]
DEFAULT_LABELS: AxisLabels = ("x", "y")

LOSS_LABEL = "Loss $f$"
LANDSCAPE_CMAP = "viridis"
ITERATION_CMAP = "plasma"

CONTOUR_LEVELS = 15
LOG_SCALE_MIN_RATIO = 10.0   # Log loss axis only if the loss spans this ratio.
MIN_TRIANGULATION_POINTS = 3
# Distance (as a fraction of the domain) under which two particles are one point.
DUPLICATE_TOLERANCE = 1e-9
Z_MARGIN_FRACTION = 0.05     # Headroom on the 3D loss axis.
ZOOM_MARGIN_FRACTION = 0.1   # Margin around the particles on contour plots.
MIN_ZOOM_FRACTION = 0.01     # Zoom half-width (of the domain) if all particles coincide.
LANDSCAPE_VIEW = {"elev": 28, "azim": -58}


def select_frames(n_iter: int) -> np.ndarray:
    """Iteration indices to animate, evenly skipped above the frame limit."""
    n_frames = min(n_iter, MAX_ANIMATION_FRAMES)
    frames = np.linspace(0, n_iter - 1, n_frames)
    return np.unique(np.round(frames).astype(int))


def build_triangulation(x: np.ndarray, y: np.ndarray, domain: tuple
                        ) -> Optional[Tuple[Triangulation, np.ndarray]]:
    """Delaunay triangulate scattered particles.

    The triangulation is built in unit domain coordinates (the axes can have
    very different scales, e.g. size in px against a ratio) and then reused
    with the physical coordinates. Repeated positions are dropped.

    Returns
    -------
    (Triangulation, np.ndarray) or None
        The triangulation and the indices of the points it uses, or None if
        there are too few distinct or non-collinear points.
    """
    x_lo, x_hi, y_lo, y_hi = domain
    unit_x = (x - x_lo) / ((x_hi - x_lo) or 1.0)
    unit_y = (y - y_lo) / ((y_hi - y_lo) or 1.0)

    keys = np.round(np.column_stack((unit_x, unit_y)) / DUPLICATE_TOLERANCE)
    _, first_seen = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(first_seen)
    if keep.size < MIN_TRIANGULATION_POINTS:
        return None

    try:
        unit_tri = Triangulation(unit_x[keep], unit_y[keep])
    except (ValueError, RuntimeError):  # collinear points
        return None
    if unit_tri.triangles.shape[0] == 0:
        return None

    tri = Triangulation(x[keep], y[keep], triangles=unit_tri.triangles)
    return tri, keep


def _loss_range(history: OptimizerHistory) -> Tuple[float, float]:
    # Never return an empty range, colour scales and axes need some width.
    f_min = float(history.f.min())
    f_max = float(history.f.max())
    if f_max <= f_min:
        f_max = f_min + 1.0
    return f_min, f_max


def _use_log_scale(values: np.ndarray) -> bool:
    if values.min() <= 0:
        return False
    return float(values.max() / values.min()) >= LOG_SCALE_MIN_RATIO


def _style_domain_axes(ax, history: OptimizerHistory, labels: AxisLabels) -> None:
    x_lo, x_hi, y_lo, y_hi = history.domain
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


def plot_loss_curve(loss: OptimizerLoss) -> Figure:
    """Global best loss against iteration (works for any dimension)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.plot(loss.iterations, loss.best, color="tab:blue", linewidth=1.8)
    if _use_log_scale(loss.best):
        ax.set_yscale("log")
    ax.set_title("Optimizer Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global best loss $f$")
    ax.grid(alpha=0.3)
    return fig


def plot_particle_paths(history: OptimizerHistory,
                        labels: AxisLabels = DEFAULT_LABELS) -> Figure:
    """Every particle position coloured by iteration, with the best-so-far path."""
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    iteration_grid = np.broadcast_to(history.iterations[:, None], history.x.shape)

    points = ax.scatter(history.x.ravel(), history.y.ravel(),
                        c=iteration_grid.ravel(), cmap=ITERATION_CMAP,
                        s=14, alpha=0.8)
    best_x, best_y = history.best_path()
    ax.plot(best_x, best_y, color="black", linewidth=1.2, label="Global best path")
    ax.scatter(history.x[0], history.y[0], marker="s", s=40, facecolors="none",
               edgecolors="black", label="Initial swarm")
    ax.scatter(best_x[-1], best_y[-1], marker="*", s=260, color="gold",
               edgecolors="black", zorder=5, label="Final best")

    _style_domain_axes(ax, history, labels)
    ax.set_title("Swarm Trajectories")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(points, ax=ax, label="Iteration")
    return fig


def plot_loss_contour(history: OptimizerHistory, labels: AxisLabels = DEFAULT_LABELS,
                      cumulative: bool = False) -> Figure:
    """Filled loss contours reconstructed from the particles alone.

    Parameters
    ----------
    history : OptimizerHistory
        Particle history of a 2D run.
    labels : tuple of str
        Axis labels ``(x, y)``.
    cumulative : bool
        False uses only the final swarm. True uses every particle of every
        iteration, which shows everything the swarm explored.
    """
    if cumulative:
        x, y, f = history.x.ravel(), history.y.ravel(), history.f.ravel()
        title = "Explored Loss Landscape (all iterations)"
    else:
        x, y, f = history.x[-1], history.y[-1], history.f[-1]
        title = "Loss Landscape at Final Iteration"

    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    built = build_triangulation(x, y, history.domain)
    if built is None:
        log_warning("Too few distinct particle positions for a contour, "
                    "drawing the particles only.")
    else:
        tri, keep = built
        f_min, f_max = float(f[keep].min()), float(f[keep].max())
        if f_max <= f_min:
            f_max = f_min + 1.0
        levels = np.linspace(f_min, f_max, CONTOUR_LEVELS)
        filled = ax.tricontourf(tri, f[keep], levels=levels, cmap=LANDSCAPE_CMAP)
        ax.tricontour(tri, f[keep], levels=levels, colors="black",
                      linewidths=0.4, alpha=0.5)
        fig.colorbar(filled, ax=ax, label=LOSS_LABEL)

    ax.scatter(x, y, s=22, color="white", edgecolors="black", linewidths=0.6,
               label="Particles")
    best = int(np.argmin(f))
    ax.scatter(x[best], y[best], marker="*", s=260, color="gold",
               edgecolors="black", zorder=5, label="Best")

    # Zoom to the particles, a converged swarm is a speck on the full domain.
    _style_domain_axes(ax, history, labels)
    ax.set_xlim(*_zoom_limits(x, history.domain[0], history.domain[1]))
    ax.set_ylim(*_zoom_limits(y, history.domain[2], history.domain[3]))
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return fig


def _zoom_limits(values: np.ndarray, domain_lo: float, domain_hi: float
                 ) -> Tuple[float, float]:
    # Particle extent plus a margin, never outside the search domain.
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * ZOOM_MARGIN_FRACTION
    if pad == 0.0:
        pad = (domain_hi - domain_lo) * MIN_ZOOM_FRACTION
    return max(low - pad, domain_lo), min(high + pad, domain_hi)


def animate_swarm(history: OptimizerHistory, labels: AxisLabels = DEFAULT_LABELS
                  ) -> Tuple[Figure, FuncAnimation]:
    """Swarm moving over the search domain next to a growing loss curve."""
    fig, (ax_swarm, ax_loss) = plt.subplots(
        1, 2, figsize=(12, 5.5), layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.0]})

    f_min, f_max = _loss_range(history)
    norm = Normalize(vmin=f_min, vmax=f_max)
    best_x, best_y = history.best_path()
    best_loss = history.best_so_far()

    # Swarm panel.
    explored = ax_swarm.scatter([], [], s=6, color="0.75", zorder=1,
                                label="Previously visited")
    swarm = ax_swarm.scatter(history.x[0], history.y[0], c=history.f[0],
                             cmap=LANDSCAPE_CMAP, norm=norm, s=50,
                             edgecolors="black", linewidths=0.5, zorder=3)
    trail, = ax_swarm.plot([], [], color="red", linewidth=1.5, zorder=2,
                           label="Global best path")
    star = ax_swarm.scatter([], [], marker="*", s=240, color="gold",
                            edgecolors="black", zorder=4, label="Global best")
    _style_domain_axes(ax_swarm, history, labels)
    ax_swarm.set_title("Swarm")
    ax_swarm.legend(loc="upper right", fontsize=8)
    fig.colorbar(swarm, ax=ax_swarm, label=LOSS_LABEL)

    # Loss panel.
    line, = ax_loss.plot([], [], color="tab:blue", linewidth=1.8)
    dot, = ax_loss.plot([], [], "o", color="tab:red")
    ax_loss.set_xlim(history.iterations[0], history.iterations[-1])
    if _use_log_scale(best_loss):
        ax_loss.set_yscale("log")
    ax_loss.set_ylim(*_padded_limits(best_loss, ax_loss.get_yscale() == "log"))
    ax_loss.set_title("Convergence")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Global best loss $f$")
    ax_loss.grid(alpha=0.3)

    def update(frame: int):
        swarm.set_offsets(np.column_stack((history.x[frame], history.y[frame])))
        swarm.set_array(history.f[frame])
        visited = np.column_stack((history.x[:frame].ravel(),
                                   history.y[:frame].ravel()))
        explored.set_offsets(visited)
        trail.set_data(best_x[:frame + 1], best_y[:frame + 1])
        star.set_offsets([[best_x[frame], best_y[frame]]])
        line.set_data(history.iterations[:frame + 1], best_loss[:frame + 1])
        dot.set_data([history.iterations[frame]], [best_loss[frame]])
        fig.suptitle(f"Iteration {int(history.iterations[frame])}  |  "
                     f"best $f$ = {best_loss[frame]:.4g}")
        return swarm, explored, trail, star, line, dot

    anim = FuncAnimation(fig, update, frames=select_frames(history.f.shape[0]),
                         interval=1000 / ANIMATION_FPS, blit=False)
    return fig, anim


def _padded_limits(values: np.ndarray, log_scale: bool) -> Tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    if log_scale:
        return low / 1.5, high * 1.5
    pad = (high - low) * Z_MARGIN_FRACTION or 1.0
    return low - pad, high + pad


def animate_landscape_3d(history: OptimizerHistory,
                         labels: AxisLabels = DEFAULT_LABELS
                         ) -> Tuple[Figure, FuncAnimation]:
    """3D loss landscape that fills in as the swarm reaches new positions.

    The axes start empty. Each frame triangulates every particle position seen
    so far into a surface, so the landscape grows out from the initial swarm
    and settles as the swarm converges.
    """
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    f_min, f_max = _loss_range(history)
    norm = Normalize(vmin=f_min, vmax=f_max)
    z_low, z_high = _padded_limits(np.array([f_min, f_max]), log_scale=False)
    x_lo, x_hi, y_lo, y_hi = history.domain
    best_loss = history.best_so_far()

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=LANDSCAPE_CMAP)
    fig.colorbar(scalar_map, ax=ax, shrink=0.6, aspect=14, pad=0.08,
                 label=LOSS_LABEL)

    def update(frame: int):
        # Redraw from scratch, the surface changes shape every frame.
        ax.cla()
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_zlim(z_low, z_high)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(LOSS_LABEL)
        ax.view_init(**LANDSCAPE_VIEW)

        seen_x = history.x[:frame + 1].ravel()
        seen_y = history.y[:frame + 1].ravel()
        seen_f = history.f[:frame + 1].ravel()
        built = build_triangulation(seen_x, seen_y, history.domain)
        if built is not None:
            tri, keep = built
            ax.plot_trisurf(tri, seen_f[keep], cmap=LANDSCAPE_CMAP, norm=norm,
                            linewidth=0, antialiased=False, alpha=0.9)

        ax.scatter(history.x[frame], history.y[frame], history.f[frame],
                   c="red", s=18, depthshade=False)
        ax.set_title(f"Iteration {int(history.iterations[frame])}  |  "
                     f"best $f$ = {best_loss[frame]:.4g}")

    anim = FuncAnimation(fig, update, frames=select_frames(history.f.shape[0]),
                         interval=1000 / ANIMATION_FPS, blit=False)
    return fig, anim
