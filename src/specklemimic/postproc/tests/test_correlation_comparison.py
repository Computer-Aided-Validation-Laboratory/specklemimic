"""Compare correlation functions on one pattern using the library functions directly."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post import PatternReader, normalise_correlator_names
from post.console import print_banner
from post.plots import (
    finalize_figure, plot_autocorrelation_3d, plot_correlation_grid,
    plot_correlation_with_radial_gradient,
)

# USER SETTINGS
PATTERN = "pattern1"
CORRELATIONS = ["ssd", "scc", "nssd", "zssd", "znssd", "zncc"]  # Tiled 2D comparison
LANDSCAPES_3D = ["znssd", "zncc"]                              # 3D surfaces
WATERSHED_VERIFY = ["znssd", "zncc"]                           # Correlation + radial gradient
WATERSHED_ON_GRID = True
SAVE = True
SHOW = False

if __name__ == "__main__":
    print_banner("CORRELATION COMPARISON TEST")
    reader = PatternReader(PATTERN)
    if reader.is_valid:
        out = reader.output_dir

        # Tiled comparison, missing correlators are skipped by the reader.
        grids = {}
        for name in normalise_correlator_names(CORRELATIONS):
            grid = reader.read_autocorr(name)
            if grid is not None:
                grids[name] = grid
        if grids:
            rings = {n: reader.read_watershed(n) for n in grids} if WATERSHED_ON_GRID else None
            rings = {n: r for n, r in rings.items() if r is not None} if rings else None
            fig = plot_correlation_grid(grids, rings)
            finalize_figure(fig, "correlations_comparison.png", out, SAVE, SHOW)

        for name in normalise_correlator_names(LANDSCAPES_3D):
            grid = reader.read_autocorr(name)
            if grid is not None:
                fig = plot_autocorrelation_3d(grid, name, reader.read_watershed(name))
                finalize_figure(fig, f"{name}_autocorrelation_3d.png", out, SAVE, SHOW)

        for name in normalise_correlator_names(WATERSHED_VERIFY):
            corr = reader.read_autocorr(name)
            gradr = reader.read_radial_gradient(name)
            if corr is not None and gradr is not None:
                fig = plot_correlation_with_radial_gradient(corr, gradr, name,
                                                            reader.read_watershed(name))
                finalize_figure(fig, f"{name}_vs_radial_gradient.png", out, SAVE, SHOW)

        if SHOW:
            plt.show()