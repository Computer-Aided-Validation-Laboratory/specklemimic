"""TEMPORARY: figures for the loss function study (delete/move after the study).

Reads the runs written by the C++ study runner (study.cpp) from STUDY_DIR:

    STUDY_DIR/<config>/<loss>/   groundtruth.csv, finalresult.csv, run_summary.csv, optimizer/

and makes:

* per run (next to it, in <loss>_post/): every optimizer figure and animation,
  and the ground truth vs reconstruction comparison titled with the loss;
* per configuration (in STUDY_DIR/figures/): overlaid loss histories, a gallery
  of all reconstructions, the explored loss landscapes side by side, and the
  convergence of the recovered parameters;
* across configurations: parameter recovery errors and compute time;
* STUDY_DIR/figures/study_summary.csv with every run.

Run with:  python study_loss_comparison.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post import (
    OptimizerOptions, OptimizerReader, run_optimizer_post_processing,
    run_reconstruction_post_processing,
)
from post.console import log_info, log_warning, print_banner
from post.loaders import Grid, OptimizerHistory, OptimizerLoss, load_grid
from post.plots import calc_relative_mse
from post.plots.optimizer import build_triangulation

# USER SETTINGS (hardcoded for this study)
STUDY_DIR = Path(r"D:\Projects\Placement\Codes\specklemimic_study\full_study\loss_study")
for position, argument in enumerate(sys.argv):
    if argument == "--study-dir" and position + 1 < len(sys.argv):
        STUDY_DIR = Path(sys.argv[position + 1])
FIGURE_DIR = STUDY_DIR / "figures"
MAKE_ANIMATIONS = "--no-animations" not in sys.argv   # The slow part (about a minute per run).
MAKE_RUN_FIGURES = True  # Per-run optimizer and reconstruction figures.

CONFIGS = {
    "baseline_20px": "Baseline (20 px, ratio 0.7)",
    "large_sparse_60px": "Large sparse (60 px, ratio 0.3)",
}
LOSSES = {
    "graylevel": r"Gray level $\mathcal{G}$",
    "sssig_mse": r"SSSIG $\mathcal{S}$",
    "sssig_var": r"SSSIG var$(\mathcal{S})$",
    "autocorr_mse": r"ZNSSD $\mathcal{A}$",
    "autocorr_var": r"ZNSSD var$(\mathcal{A})$",
    "autocorr_scalars": r"ZNSSD scalars $\lambda$",
}
LOSS_COLOURS = dict(zip(LOSSES, plt.get_cmap("tab10").colors))
PARAM_LABELS = ("Speckle size (px)", "Ratio")

LOSS_FLOOR = 1e-12  # Exact zeros are drawn here on log axes.
FIGURE_DPI = 200


class Run:
    """Everything saved for one (configuration, loss) run."""

    def __init__(self, config: str, loss: str, run_dir: Path) -> None:
        self.config = config
        self.loss = loss
        self.run_dir = run_dir
        self.summary = pd.read_csv(run_dir / "run_summary.csv").iloc[0]
        reader = OptimizerReader(run_dir)
        self.history: Optional[OptimizerHistory] = reader.read_history()
        self.loss_curve: Optional[OptimizerLoss] = reader.read_loss()
        self.truth: Optional[Grid] = load_grid(run_dir / "groundtruth.csv")
        self.result: Optional[Grid] = load_grid(run_dir / "finalresult.csv")

    def label(self) -> str:
        return LOSSES.get(self.loss, self.loss)

    def colour(self):
        return LOSS_COLOURS.get(self.loss, "black")

    def elapsed(self) -> np.ndarray:
        """Wall time at the end of each iteration, assuming iterations cost the same."""
        n_iter = len(self.loss_curve.iterations)
        return float(self.summary["runtime_s"]) * (np.arange(n_iter) + 1) / n_iter


def find_runs() -> Dict[str, List[Run]]:
    """All finished runs per configuration, in the LOSSES order."""
    runs: Dict[str, List[Run]] = {}
    for config in CONFIGS:
        found = []
        for loss in LOSSES:
            run_dir = STUDY_DIR / config / loss
            if (run_dir / "run_summary.csv").is_file():
                found.append(Run(config, loss, run_dir))
            else:
                log_warning(f"No finished run for {config}/{loss}, skipping it.")
        if found:
            runs[config] = found
    return runs


def save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved: {path}")


def make_run_figures(run: Run) -> None:
    """Every optimizer figure/animation and the titled reconstruction comparison."""
    title = f"{CONFIGS[run.config]}  |  {run.label()}"
    options = OptimizerOptions(xlabel=PARAM_LABELS[0], ylabel=PARAM_LABELS[1],
                               plot_animations=MAKE_ANIMATIONS)
    run_optimizer_post_processing(run.run_dir, options)
    run_reconstruction_post_processing(run.run_dir, title=title)


def plot_loss_histories(config: str, runs: List[Run]) -> None:
    """Best loss per iteration for every loss: raw, normalised and against wall time."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    for run in runs:
        if run.loss_curve is None:
            continue
        best = np.maximum(run.loss_curve.best, LOSS_FLOOR)
        start = best[0] if best[0] > 0 else 1.0
        style = {"color": run.colour(), "label": run.label(), "linewidth": 1.8}
        axes[0].plot(run.loss_curve.iterations, best, **style)
        axes[1].plot(run.loss_curve.iterations, best / start, **style)
        axes[2].plot(run.elapsed(), best, **style)

    titles = ("Global best loss", "Normalised by the first iteration", "Against wall time")
    xlabels = ("Iteration", "Iteration", "Elapsed optimizer time (s)")
    for ax, title, xlabel in zip(axes, titles, xlabels):
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Relative loss (each loss in its own units)")
    axes[1].set_ylabel(r"$f_{best} / f_{best,0}$")
    axes[2].set_xscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Loss histories: {CONFIGS[config]}  "
                 f"(exact zeros drawn at {LOSS_FLOOR:g})")
    save(fig, f"loss_histories_{config}.png")


def plot_pattern_gallery(config: str, runs: List[Run]) -> None:
    """Ground truth next to every reconstruction."""
    panels = [run for run in runs if run.result is not None]
    truth = runs[0].truth
    n_cols = min(4, len(panels) + 1)
    n_rows = int(np.ceil((len(panels) + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.4 * n_rows),
                             layout="constrained", squeeze=False)
    flat = axes.ravel()

    true_size = float(runs[0].summary["true_size"])
    true_ratio = float(runs[0].summary["true_ratio"])
    flat[0].imshow(truth.values, extent=truth.extent, origin="lower", cmap="gray_r",
                   vmin=0, vmax=1, interpolation="nearest")
    flat[0].set_title(f"Ground truth\nsize = {true_size:g} px, ratio = {true_ratio:g}",
                      fontsize=9)

    for ax, run in zip(flat[1:], panels):
        ax.imshow(run.result.values, extent=run.result.extent, origin="lower",
                  cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        rel_mse = calc_relative_mse(run.truth, run.result)
        ax.set_title(f"{run.label()}\nsize = {run.summary['best_size']:.3g} px, "
                     f"ratio = {run.summary['best_ratio']:.3g}, "
                     f"$\\mathcal{{G}}$ = {rel_mse:.3g}", fontsize=9)
    for ax in flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flat[len(panels) + 1:]:
        ax.set_visible(False)
    fig.suptitle(f"Reconstructions: {CONFIGS[config]}")
    save(fig, f"patterns_{config}.png")


def plot_landscapes(config: str, runs: List[Run]) -> None:
    """Explored loss landscape (all particles, all iterations) of every loss."""
    panels = [run for run in runs if run.history is not None]
    if not panels:
        return
    n_cols = min(3, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.6 * n_rows),
                             layout="constrained", squeeze=False)
    true_size = float(runs[0].summary["true_size"])
    true_ratio = float(runs[0].summary["true_ratio"])

    for ax, run in zip(axes.ravel(), panels):
        history = run.history
        x, y, f = history.x.ravel(), history.y.ravel(), history.f.ravel()
        built = build_triangulation(x, y, history.domain)
        if built is not None:
            tri, keep = built
            # log10 so the basin and the plateaus both show, whatever the loss scale.
            levels_input = np.log10(np.maximum(f[keep], LOSS_FLOOR))
            filled = ax.tricontourf(tri, levels_input, levels=15, cmap="viridis")
            fig.colorbar(filled, ax=ax, label=r"$\log_{10}$ loss")
        ax.scatter(x, y, s=3, color="white", edgecolors="none", alpha=0.5)
        ax.scatter(true_size, true_ratio, marker="x", s=80, color="red", label="Truth")
        ax.scatter(run.summary["best_size"], run.summary["best_ratio"], marker="*",
                   s=150, color="gold", edgecolors="black", label="Best")
        x_lo, x_hi, y_lo, y_hi = history.domain
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(PARAM_LABELS[0])
        ax.set_ylabel(PARAM_LABELS[1])
        n_evals = int(run.summary["evaluations"])
        ax.set_title(f"{run.label()}  ({n_evals} evaluations)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"Explored loss landscapes: {CONFIGS[config]}")
    save(fig, f"landscapes_{config}.png")


def plot_parameter_convergence(config: str, runs: List[Run]) -> None:
    """Best-so-far speckle size and ratio per iteration, against the truth."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    truths = (float(runs[0].summary["true_size"]), float(runs[0].summary["true_ratio"]))
    for run in runs:
        if run.history is None:
            continue
        best_x, best_y = run.history.best_path()
        for ax, values in zip(axes, (best_x, best_y)):
            ax.plot(run.history.iterations, values, color=run.colour(),
                    label=run.label(), linewidth=1.6)
    for ax, truth, label in zip(axes, truths, PARAM_LABELS):
        ax.axhline(truth, color="black", linestyle="--", linewidth=1, label="Truth")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Best {label.lower()}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Parameter convergence: {CONFIGS[config]}")
    save(fig, f"parameter_convergence_{config}.png")


def plot_recovery_and_cost(summary: pd.DataFrame) -> None:
    """Parameter errors, cross-loss gray level error and compute cost for every run."""
    configs = list(summary["config"].unique())
    losses = [loss for loss in LOSSES if loss in set(summary["loss"])]
    width = 0.8 / max(len(configs), 1)
    positions = np.arange(len(losses))

    metrics = (
        ("size_error", "Relative speckle size error", True),
        ("ratio_error", "Relative ratio error", True),
        ("graylevel_relmse", r"Gray level $\mathcal{G}$ of the reconstruction", True),
        ("time_per_eval", "Time per evaluation (s)", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, (column, title, log_scale) in zip(axes.ravel(), metrics):
        for k, config in enumerate(configs):
            rows = summary[summary["config"] == config].set_index("loss")
            values = [rows[column].get(loss, np.nan) for loss in losses]
            values = np.maximum(np.asarray(values, dtype=float), LOSS_FLOOR)
            ax.bar(positions + k * width, values, width, label=CONFIGS.get(config, config))
        ax.set_xticks(positions + width * (len(configs) - 1) / 2)
        ax.set_xticklabels([LOSSES[loss] for loss in losses], rotation=20, ha="right")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Recovery and compute cost (zeros drawn at {LOSS_FLOOR:g})")
    save(fig, "recovery_and_cost.png")


def build_summary(runs: Dict[str, List[Run]]) -> pd.DataFrame:
    rows = []
    for config_runs in runs.values():
        for run in config_runs:
            row = run.summary.to_dict()
            row["size_error"] = abs(row["best_size"] - row["true_size"]) / row["true_size"]
            row["ratio_error"] = abs(row["best_ratio"] - row["true_ratio"]) / row["true_ratio"]
            row["time_per_eval"] = row["runtime_s"] / max(row["evaluations"], 1)
            rows.append(row)
    summary = pd.DataFrame(rows)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FIGURE_DIR / "study_summary.csv", index=False)
    log_info(f"Saved: {FIGURE_DIR / 'study_summary.csv'}")
    return summary


def main() -> int:
    print_banner("LOSS STUDY FIGURES")
    if not STUDY_DIR.is_dir():
        log_warning(f"Study folder not found: {STUDY_DIR}")
        return 1
    runs = find_runs()
    if not runs:
        log_warning("No finished runs found.")
        return 1

    if MAKE_RUN_FIGURES:
        for config_runs in runs.values():
            for run in config_runs:
                make_run_figures(run)

    print_banner("STUDY COMPARISON FIGURES")
    for config, config_runs in runs.items():
        plot_loss_histories(config, config_runs)
        plot_pattern_gallery(config, config_runs)
        plot_landscapes(config, config_runs)
        plot_parameter_convergence(config, config_runs)
    plot_recovery_and_cost(build_summary(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
