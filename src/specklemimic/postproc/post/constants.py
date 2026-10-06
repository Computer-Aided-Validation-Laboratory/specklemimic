"""Shared constants for the post-processing library.

File and folder names here mirror what the C++ backend (main.cpp) writes.
If the backend naming changes, this is the only place that needs editing.
"""

# Console formatting (matches the C++ banners).
BANNER_WIDTH = 99

# Output settings.
POST_SUFFIX = "_post"
DEFAULT_DPI = 300

# Tiled correlation figures never put more than this many tiles in a row.
MAX_PLOTS_PER_ROW = 3

# Pattern folder layout.
SPECKLE_FILENAME = "speckles.csv"
GRAD_SUBDIR = "gradients"
AUTOC_SUBDIR = "autocorrelation"
SSSIG_SUBDIR = "sssig"

GRAD_X_FILENAME = "grad_x.csv"
GRAD_Y_FILENAME = "grad_y.csv"
GRAD_MAG_FILENAME = "grad_mag.csv"
SSSIG_FILENAME = "sssig_heatmap.csv"
SSSIG_DELTA_FILENAME = "sssig_deltamap.csv"

# Per-autocorrelator files, {name} is the lowercase correlator name.
AUTOCORR_FILE_PATTERN = "{name}_autocorrelator.csv"
GRADR_FILE_PATTERN = "{name}_autogradr.csv"
WSHED_FILE_PATTERN = "{name}_wshedgrad.csv"

# Old backend names (shared between correlators, so they may belong to the wrong one).
LEGACY_GRADR_FILENAME = "autogradr.csv"
LEGACY_WSHED_FILENAME = "wshedgrad.csv"

# Column names used in the CSVs.
COL_X = "x"
COL_Y = "y"
COL_VALUE = "f"
COL_VALUE_ALT = "r"  # Some watershed files store the value in "r".