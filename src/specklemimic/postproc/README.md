# postproc

Python post-processing for the files the C++ backend writes. It has two parts:

1. **Pattern post-processing**: figures for one generated speckle pattern (speckle, gradients, autocorrelation, SSSIG).
2. **Optimizer post-processing**: figures and animations for an optimizer run (loss curve, and for 2D runs the swarm, contours and a 3D landscape).

Everything lives in the `post` package. Every reader returns `None` and logs a warning instead of raising, and every plotting step is wrapped, so a missing file or one broken plot only skips that figure.

## Requirements

`numpy`, `pandas` and `matplotlib` (which brings Pillow, used for GIFs). `ffmpeg` is optional and only needed for MP4 animations.

## Folder layout

```
postproc/
  cmdline.py                   Command line for pattern post-processing
  post/
    constants.py               File/folder names mirroring the C++ backend, plot settings
    console.py                 Banner and log helpers styled like the C++ output
    correlators.py             Registry of the autocorrelation functions
    loaders.py                 CSV readers and the data classes they return
    workflow.py                Pattern workflow (run_post_processing)
    optimizer_workflow.py      Optimizer workflow (run_optimizer_post_processing)
    plots/                     One module per figure family
  tests/                       Runnable example scripts
```

If the backend's file names change, `post/constants.py` is the only place to edit.

## Pattern post-processing

Input is a pattern folder written by the speckle generator, for example `pattern1/`:

```
pattern1/
  speckles.csv
  gradients/       grad_x.csv, grad_y.csv, grad_mag.csv
  autocorrelation/ <name>_autocorrelator.csv, <name>_autogradr.csv, <name>_wshedgrad.csv
  sssig/           sssig_heatmap.csv, sssig_deltamap.csv
```

Figures are saved to `pattern1_post/` next to the pattern folder.

| Flag | Figure |
|---|---|
| `--plot-speckle` | Raw speckle pattern |
| `--plot-grad` | Directional gradients and magnitude |
| `--plot-3d` | 3D autocorrelation surface with the watershed ring |
| `--plot-sssig` | SSSIG and delta-map heatmaps, plus 3D surfaces |
| `--plot-corr` | Tiled comparison of the chosen autocorrelation functions |
| `--plot-verify` | Autocorrelation next to the radial gradient, with the watershed |

With no plot flag the default quick-look runs (3D, gradients, speckle, SSSIG). Other flags:

- `-p, --pattern` pattern folder (required).
- `-a, --autocorr` one or more of `ssd scc nssd zssd znssd zncc` (default `znssd`; `cc` is accepted for `scc`).
- `--no-watershed`, `--grid-watershed` control the watershed overlay.
- `--save` (default) / `--show`.

```
python cmdline.py -p pattern1 -a znssd zncc --plot-corr --plot-3d
```

From Python:

```python
from post import PostOptions, run_post_processing

run_post_processing("pattern1", PostOptions(autocorrs=["znssd"], plot_sssig=True))
```

`tests/test_basic_workflow.py`, `test_correlation_comparison.py` and `test_error_handling.py` are example scripts for this part.

## Reconstruction comparison

Compares the reconstructed pattern with the ground truth for one run folder (`groundtruth.csv` and `finalresult.csv`, both written by `main.cpp`):

```
python tests/test_reconstruction_compare.py -r 50px
```

Saves to `50px_post/`:

- `reconstruction_comparison.png`: ground truth, reconstruction, signed difference (reconstruction minus ground truth) and squared error map. The title carries the gray level MSE and the relative MSE, `sum(diff^2) / sum(truth^2)`, which is the quantity the C++ gray level loss minimizes.
- `reconstruction_metrics.csv`: one row (`run, mse, relative_mse, max_abs_error, n_points`), so runs with different loss functions can be concatenated into one table.

If either file is missing, or the two are on different grids, it warns and exits without a figure. From Python: `run_reconstruction_post_processing("50px", title="Gray level loss")`. The optional `title` replaces the default figure heading.

## Optimizer post-processing

The C++ optimizer saves everything into an `optimizer` subfolder of the run folder (the `runfolder` set in `main.cpp`):

```
50px/
  optimizer/
    loss.csv       iteration, f           Global best loss per iteration (any dimension)
    history.csv    iteration, x, y, f     Every particle at every iteration (2D runs only)
    domain.csv     lo_x, lo_y, hi_x, hi_y Search bounds (2D runs only)
```

For a 2D run, `x` and `y` are the two optimized parameters in physical units. For the parameter loss that is speckle size in px and the ratio.

Run it from the command line with just the run folder:

```
python tests/test_optimizer_visuals.py -r 50px --xlabel "Speckle size (px)" --ylabel "Ratio"
```

Options:

- `-r, --runfolder` run folder (required).
- `--xlabel`, `--ylabel` names of the two parameters.
- `--static-only` skips the animations, which are the slow part (about a minute for 100 iterations).
- `--animations-only` skips the static figures.
- `--format gif|mp4` animation format. MP4 needs ffmpeg and falls back to GIF with a warning.
- `--show` also opens the figures interactively.

Output goes to `50px_post/optimizer/`:

| File | Kind | Content |
|---|---|---|
| `loss_curve.png` | static | Global best loss against iteration (log axis when the loss spans a wide range) |
| `particle_paths.png` | static | Every particle position coloured by iteration, with the global-best path |
| `contour_final.png` | static | Filled loss contours from the final swarm only, zoomed to the particles |
| `contour_explored.png` | static | Same, using every particle of every iteration |
| `swarm.gif` | animation | Swarm moving over the domain next to a growing loss curve |
| `landscape_3d.gif` | animation | 3D loss landscape that starts empty and fills in as particles reach new positions |

The contours and the 3D surface are Delaunay triangulations of the particle positions only. They show what the swarm sampled, not the true landscape, so regions the swarm never visited stay blank.

Graceful exits:

- Run folder missing: error message, exit code 1.
- No `optimizer` folder: warning, nothing plotted, exit code 0.
- `optimizer` folder without `history.csv` (the optimizer had more than 2 dimensions): the loss curve is made and the swarm figures are skipped with a message.

From Python:

```python
from post import OptimizerOptions, run_optimizer_post_processing

run_optimizer_post_processing(
    "50px",
    OptimizerOptions(xlabel="Speckle size (px)", ylabel="Ratio", plot_animations=False),
)
```

`OptimizerReader("50px")` gives direct access to the data (`read_history()`, `read_loss()`). `OptimizerHistory` has `best_so_far()` and `best_path()` helpers. The plot functions in `post.plots` (`plot_loss_curve`, `plot_particle_paths`, `plot_loss_contour`, `animate_swarm`, `animate_landscape_3d`) can be used on their own. Animation builders return `(fig, anim)`, which you pass to `finalize_animation`.

Animation settings (frame rate, resolution, maximum number of frames) are constants in `post/constants.py`. Above `MAX_ANIMATION_FRAMES` iterations are skipped evenly.
