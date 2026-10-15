"""Ties the loaders and plots together into a single-pattern workflow.

Each step is wrapped so one failure is logged and the rest still run.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt

from post.console import log_error, log_info, log_warning, print_banner, print_section
from post.correlators import get_correlator_info, normalise_correlator_names
from post.loaders import PatternReader
from post.plots import (
    finalize_figure, plot_autocorrelation_3d, plot_correlation_grid,
    plot_correlation_with_radial_gradient, plot_gradients, plot_speckle,
    plot_sssig_maps, plot_sssig_surface,
)


@dataclass
class PostOptions:
    """Levers for one post-processing run.

    Attributes
    ----------
    autocorrs : list of str
        Correlator names (any case), e.g. ``["znssd", "zncc"]``.
    plot_3d, plot_grad, plot_speckle, plot_sssig : bool
        Default-workflow plots (3D landscape, gradients, raw speckle, SSSIG maps).
    plot_corr : bool
        Tiled 2D comparison of all ``autocorrs``.
    plot_verify : bool
        Autocorrelation + radial gradient with watershed, per correlator.
    show_watershed : bool
        Draw the watershed ring on the 3D surfaces.
    grid_watershed : bool
        Draw the watershed ring on the comparison tiles.
    save, show : bool
        Save figures to ``<pattern>_post/`` and/or show them interactively.
    """

    autocorrs: List[str] = field(default_factory=lambda: ["znssd"])
    plot_3d: bool = False
    plot_grad: bool = False
    plot_speckle: bool = False
    plot_sssig: bool = False
    plot_corr: bool = False
    plot_verify: bool = False
    show_watershed: bool = True
    grid_watershed: bool = False
    save: bool = True
    show: bool = False

    @property
    def any_plot(self) -> bool:
        """True if at least one plot has been switched on."""
        return any((self.plot_3d, self.plot_grad, self.plot_speckle, self.plot_sssig,
                    self.plot_corr, self.plot_verify))

    def apply_default_workflow(self) -> None:
        """Switch on the quick-look plots: 3D + watershed, gradients, speckle, SSSIG."""
        self.plot_3d = self.plot_grad = self.plot_speckle = self.plot_sssig = True


def _run_step(label: str, step: Callable[[], None]) -> None:
    # One broken plot should never take the whole run down.
    try:
        step()
    except Exception as exc:  # noqa: BLE001
        log_error(f"{label} failed: {type(exc).__name__}: {exc}")


def _finish(fig, filename: str, reader: PatternReader, options: PostOptions,
            saved: List[Path]) -> None:
    path = finalize_figure(fig, filename, reader.output_dir, options.save, options.show)
    if path is not None:
        saved.append(path)


def _step_3d(reader: PatternReader, names: List[str], options: PostOptions, saved: List[Path]) -> None:
    for name in names:
        tag = get_correlator_info(name).short
        grid = reader.read_autocorr(name)
        if grid is None:
            log_warning("Autocorrelation map unavailable, skipping 3D surface.", tag)
            continue
        ring = reader.read_watershed(name) if options.show_watershed else None
        log_info("Plotting 3D autocorrelation surface...", tag)
        fig = plot_autocorrelation_3d(grid, name, ring)
        _finish(fig, f"{name}_autocorrelation_3d.png", reader, options, saved)


def _step_gradients(reader: PatternReader, options: PostOptions, saved: List[Path]) -> None:
    grad_x, grad_y, grad_mag = reader.read_gradients()
    if None in (grad_x, grad_y, grad_mag):
        log_warning("Gradient files incomplete, skipping gradient figure.")
        return
    log_info("Plotting gradients...")
    _finish(plot_gradients(grad_x, grad_y, grad_mag), "gradients.png", reader, options, saved)


def _step_speckle(reader: PatternReader, options: PostOptions, saved: List[Path]) -> None:
    speckle = reader.read_speckle()
    if speckle is None:
        log_warning("Speckle pattern unavailable, skipping speckle figure.")
        return
    log_info("Plotting speckle pattern...")
    _finish(plot_speckle(speckle), "speckle_pattern.png", reader, options, saved)


def _step_correlation_grid(reader: PatternReader, names: List[str], options: PostOptions,
                           saved: List[Path]) -> None:
    grids = {}
    for name in names:
        grid = reader.read_autocorr(name)
        if grid is not None:
            grids[name] = grid
    if not grids:
        log_warning("No autocorrelation maps available, skipping comparison grid.")
        return
    rings = None
    if options.grid_watershed:
        rings = {name: ring for name in grids if (ring := reader.read_watershed(name)) is not None}
    log_info("Plotting autocorrelation comparison grid...")
    _finish(plot_correlation_grid(grids, rings), "correlations_comparison.png", reader, options, saved)


def _step_verify(reader: PatternReader, names: List[str], options: PostOptions,
                 saved: List[Path]) -> None:
    for name in names:
        tag = get_correlator_info(name).short
        corr = reader.read_autocorr(name)
        gradr = reader.read_radial_gradient(name)
        if corr is None or gradr is None:
            log_warning("Autocorrelation or radial gradient unavailable, skipping verification plot.", tag)
            continue
        ring = reader.read_watershed(name)
        log_info("Plotting watershed verification...", tag)
        fig = plot_correlation_with_radial_gradient(corr, gradr, name, ring)
        _finish(fig, f"{name}_vs_radial_gradient.png", reader, options, saved)


def _step_sssig(reader: PatternReader, options: PostOptions, saved: List[Path]) -> None:
    sssig = reader.read_sssig()
    delta = reader.read_sssig_delta()
    log_info("Plotting SSSIG heatmaps...")
    fig = plot_sssig_maps(sssig, delta)
    if fig is None:
        log_warning("No SSSIG files available, skipping SSSIG figures.")
        return
    _finish(fig, "sssig_heatmaps_2d.png", reader, options, saved)
    if sssig is not None:
        _finish(plot_sssig_surface(sssig), "sssig_surface_3d.png", reader, options, saved)
    if delta is not None:
        _finish(plot_sssig_surface(delta, is_delta=True), "sssig_delta_surface_3d.png",
                reader, options, saved)


def run_post_processing(pattern_dir, options: Optional[PostOptions] = None) -> List[Path]:
    """Run the post-processing workflow for one pattern folder.

    Parameters
    ----------
    pattern_dir : str or Path
        Pattern folder written by the C++ backend.
    options : PostOptions, optional
        Which plots to make. If no plot is switched on, the default quick-look
        workflow runs. If neither save nor show is set, figures are saved.

    Returns
    -------
    list of Path
        Paths of the figures that were saved (empty if only shown or nothing worked).
    """
    options = options or PostOptions()
    if not options.any_plot:
        options.apply_default_workflow()
    if not (options.save or options.show):
        options.save = True

    start = time.perf_counter()
    print_banner("POST PROCESSOR")
    reader = PatternReader(pattern_dir)
    if not reader.is_valid:
        return []

    if not options.show:
        plt.switch_backend("Agg")  # no window needed when only saving

    names = normalise_correlator_names(options.autocorrs)
    log_info(f"Analyzing pattern: {reader.pattern_dir}")
    if options.save:
        log_info(f"Saving outputs to: {reader.output_dir}")
    log_info(f"Autocorrelation functions: {', '.join(n.upper() for n in names) or 'none'}")

    saved: List[Path] = []
    needs_names = options.plot_3d or options.plot_corr or options.plot_verify
    if needs_names and not names:
        log_warning("No valid autocorrelation functions given, correlation plots will be skipped.")

    if options.plot_3d:
        _run_step("3D surfaces", lambda: _step_3d(reader, names, options, saved))
    if options.plot_grad:
        _run_step("Gradient plot", lambda: _step_gradients(reader, options, saved))
    if options.plot_speckle:
        _run_step("Speckle plot", lambda: _step_speckle(reader, options, saved))
    if options.plot_corr:
        _run_step("Comparison grid", lambda: _step_correlation_grid(reader, names, options, saved))
    if options.plot_verify:
        _run_step("Verification plots", lambda: _step_verify(reader, names, options, saved))
    if options.plot_sssig:
        _run_step("SSSIG plots", lambda: _step_sssig(reader, options, saved))

    elapsed = time.perf_counter() - start
    print_section("POST PROCESSING FINISHED.")
    log_info(f"Created {len(saved)} figure(s) in: {elapsed:.2f}s")

    if options.show and plt.get_fignums():
        plt.show()
    return saved