"""Compare a reconstructed pattern with its ground truth for one run folder.

Saves the comparison figure and a one-row metrics CSV, so results from
different loss functions can be collected and compared quickly.
"""

import time
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from post.console import log_error, log_info, log_warning, print_banner, print_section
from post.constants import (
    FINALRESULT_FILENAME, GROUNDTRUTH_FILENAME, POST_SUFFIX, RECON_FIGURE_FILENAME,
    RECON_METRICS_FILENAME,
)
from post.loaders import load_grid
from post.plots import (
    calc_error_maps, calc_relative_mse, finalize_figure, plot_reconstruction_comparison,
)


def run_reconstruction_post_processing(run_dir, save: bool = True,
                                       show: bool = False,
                                       title: Optional[str] = None) -> List[Path]:
    """Plot ground truth against the reconstruction for one run folder.

    Parameters
    ----------
    run_dir : str or Path
        Run folder holding ``groundtruth.csv`` and ``finalresult.csv``.
    save, show : bool
        Save to ``<run>_post/`` and/or show interactively. If neither is set,
        the figure is saved.
    title : str, optional
        Figure title, e.g. the loss that produced the reconstruction.

    Returns
    -------
    list of Path
        Saved files (figure and metrics). Empty if an input is missing or the
        two grids do not match.
    """
    if not (save or show):
        save = True

    start = time.perf_counter()
    print_banner("RECONSTRUCTION POST PROCESSOR")
    run_path = Path(run_dir)
    if not run_path.is_dir():
        log_error(f"Run folder not found: {run_path}")
        return []

    truth = load_grid(run_path / GROUNDTRUTH_FILENAME)
    result = load_grid(run_path / FINALRESULT_FILENAME)
    if truth is None or result is None:
        log_warning("Need both the ground truth and the final result, nothing to compare.")
        return []

    same_grid = (truth.values.shape == result.values.shape
                 and np.allclose(truth.x, result.x) and np.allclose(truth.y, result.y))
    if not same_grid:
        log_error("Ground truth and result are on different grids, cannot compare them.")
        return []

    if not show:
        plt.switch_backend("Agg")  # no window needed when only saving

    resolved = run_path.resolve()
    output_dir = resolved.parent / f"{resolved.name}{POST_SUFFIX}"
    difference, squared = calc_error_maps(truth, result)
    mse = float(np.nanmean(squared.values))
    relative_mse = calc_relative_mse(truth, result)
    max_error = float(np.nanmax(np.abs(difference.values)))
    log_info(f"Gray level MSE = {mse:.6g} | Relative MSE = {relative_mse:.6g} | "
             f"Max absolute error = {max_error:.4g}")

    saved: List[Path] = []
    log_info("Plotting reconstruction comparison...")
    path = finalize_figure(plot_reconstruction_comparison(truth, result, title),
                           RECON_FIGURE_FILENAME, output_dir, save, show)
    if path is not None:
        saved.append(path)

    if save:
        metrics_path = output_dir / RECON_METRICS_FILENAME
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                "run,mse,relative_mse,max_abs_error,n_points\n"
                f"{resolved.name},{mse:.10g},{relative_mse:.10g},{max_error:.10g},"
                f"{truth.values.size}\n")
            log_info(f"Saved: {metrics_path}")
            saved.append(metrics_path)
        except OSError as exc:
            log_error(f"Could not save {metrics_path.name}: {exc}")

    print_section("RECONSTRUCTION POST PROCESSING FINISHED.")
    log_info(f"Created {len(saved)} file(s) in: {time.perf_counter() - start:.2f}s")

    if show and plt.get_fignums():
        plt.show()
    return saved
