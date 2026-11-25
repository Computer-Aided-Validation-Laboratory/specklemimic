"""Optimizer post-processing for one run folder.

Reads ``<run>/optimizer/`` and makes the loss curve, static contour and
trajectory figures and the animations. Runs that are not 2D only have a loss
history, so they get the loss curve and a message instead of an error.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

from post.console import log_info, log_warning, print_banner, print_section
from post.loaders import OptimizerHistory, OptimizerLoss, OptimizerReader
from post.plots import (
    animate_landscape_3d, animate_swarm, finalize_animation, finalize_figure,
    plot_loss_contour, plot_loss_curve, plot_particle_paths,
)
from post.plots.optimizer import DEFAULT_LABELS
from post.workflow import _run_step


@dataclass
class OptimizerOptions:
    """Levers for one optimizer post-processing run.

    Attributes
    ----------
    xlabel, ylabel : str
        Axis labels for the two optimized parameters.
    plot_static, plot_animations : bool
        Which families of figures to make.
    animation_format : str
        ``"gif"`` or ``"mp4"`` (MP4 needs ffmpeg, otherwise a GIF is saved).
    save, show : bool
        Save figures to ``<run>_post/optimizer/`` and/or show them interactively.
    """

    xlabel: str = DEFAULT_LABELS[0]
    ylabel: str = DEFAULT_LABELS[1]
    plot_static: bool = True
    plot_animations: bool = True
    animation_format: str = "gif"
    save: bool = True
    show: bool = False


def _step_loss_curve(loss: OptimizerLoss, reader: OptimizerReader,
                     options: OptimizerOptions, saved: List[Path]) -> None:
    log_info("Plotting loss curve...")
    path = finalize_figure(plot_loss_curve(loss), "loss_curve.png",
                           reader.output_dir, options.save, options.show)
    if path is not None:
        saved.append(path)


def _step_static(history: OptimizerHistory, reader: OptimizerReader,
                 options: OptimizerOptions, saved: List[Path]) -> None:
    labels = (options.xlabel, options.ylabel)
    figures = (
        ("particle_paths.png", "particle trajectories",
         lambda: plot_particle_paths(history, labels)),
        ("contour_final.png", "final swarm contour",
         lambda: plot_loss_contour(history, labels)),
        ("contour_explored.png", "explored landscape contour",
         lambda: plot_loss_contour(history, labels, cumulative=True)),
    )
    for filename, description, build in figures:
        log_info(f"Plotting {description}...")
        path = finalize_figure(build(), filename, reader.output_dir,
                               options.save, options.show)
        if path is not None:
            saved.append(path)


def _step_animations(history: OptimizerHistory, reader: OptimizerReader,
                     options: OptimizerOptions, saved: List[Path]) -> None:
    labels = (options.xlabel, options.ylabel)
    suffix = f".{options.animation_format.lstrip('.')}"
    animations = (
        (f"swarm{suffix}", "swarm animation",
         lambda: animate_swarm(history, labels)),
        (f"landscape_3d{suffix}", "3D landscape animation",
         lambda: animate_landscape_3d(history, labels)),
    )
    for filename, description, build in animations:
        log_info(f"Building {description}...")
        fig, anim = build()
        path = finalize_animation(fig, anim, filename, reader.output_dir,
                                  options.save, options.show)
        if path is not None:
            saved.append(path)


def run_optimizer_post_processing(run_dir, options: Optional[OptimizerOptions] = None
                                  ) -> List[Path]:
    """Run the optimizer post-processing for one run folder.

    Parameters
    ----------
    run_dir : str or Path
        Run folder written by the C++ backend (the one holding ``optimizer/``).
    options : OptimizerOptions, optional
        Which figures to make. If neither save nor show is set, figures are saved.

    Returns
    -------
    list of Path
        Paths of the files that were saved. Empty if the folder or its optimizer
        data is missing, or if only shown.
    """
    options = options or OptimizerOptions()
    if not (options.save or options.show):
        options.save = True

    start = time.perf_counter()
    print_banner("OPTIMIZER POST PROCESSOR")
    reader = OptimizerReader(run_dir)
    if not reader.is_valid:
        return []
    if not reader.has_optimizer_dir:
        log_warning(f"No optimizer folder in {reader.run_dir}, nothing to plot. "
                    "Run the optimizer with this run folder first.")
        return []

    if not options.show:
        plt.switch_backend("Agg")  # no window needed when only saving

    log_info(f"Analyzing optimizer run: {reader.run_dir}")
    if options.save:
        log_info(f"Saving outputs to: {reader.output_dir}")

    history = reader.read_history()
    loss = reader.read_loss()
    if loss is None and history is not None:
        loss = OptimizerLoss(iterations=history.iterations,
                             best=history.best_so_far())

    saved: List[Path] = []
    if loss is not None:
        _run_step("Loss curve", lambda: _step_loss_curve(loss, reader, options, saved))
    else:
        log_warning("No loss history found, skipping the loss curve.")

    if history is None:
        log_info("No 2D particle history found (it is only saved when the "
                 "optimizer has exactly 2 dimensions), so the swarm contours "
                 "and animations are skipped.")
    else:
        if options.plot_static:
            _run_step("Static figures", lambda: _step_static(history, reader, options, saved))
        if options.plot_animations:
            _run_step("Animations", lambda: _step_animations(history, reader, options, saved))

    elapsed = time.perf_counter() - start
    print_section("OPTIMIZER POST PROCESSING FINISHED.")
    log_info(f"Created {len(saved)} file(s) in: {elapsed:.2f}s")

    if options.show and plt.get_fignums():
        plt.show()
    return saved
