"""Safe CSV readers for the files written by the C++ backend.

Every reader returns ``None`` (and logs a warning) instead of raising, so a
missing or broken file only skips the plots that need it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from post.console import log_error, log_warning
from post.constants import (
    AUTOC_SUBDIR, AUTOCORR_FILE_PATTERN, COL_VALUE, COL_VALUE_ALT, COL_X, COL_Y,
    GRAD_MAG_FILENAME, GRAD_SUBDIR, GRAD_X_FILENAME, GRAD_Y_FILENAME,
    GRADR_FILE_PATTERN, LEGACY_GRADR_FILENAME, LEGACY_WSHED_FILENAME, POST_SUFFIX,
    SPECKLE_FILENAME, SSSIG_DELTA_FILENAME, SSSIG_FILENAME, SSSIG_SUBDIR,
    WSHED_FILE_PATTERN,
)


@dataclass
class Grid:
    """A scalar field on a rectilinear grid.

    Attributes
    ----------
    x : np.ndarray
        Sorted unique x coordinates, shape (nx,).
    y : np.ndarray
        Sorted unique y coordinates, shape (ny,).
    values : np.ndarray
        Field values, shape (ny, nx), i.e. rows follow y and columns follow x.
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray

    @property
    def extent(self) -> list:
        """Pixel-centred ``[xmin, xmax, ymin, ymax]`` for ``imshow``."""
        dx = float(np.median(np.diff(self.x))) if self.x.size > 1 else 1.0
        dy = float(np.median(np.diff(self.y))) if self.y.size > 1 else 1.0
        return [self.x[0] - dx / 2, self.x[-1] + dx / 2,
                self.y[0] - dy / 2, self.y[-1] + dy / 2]


@dataclass
class WatershedRing:
    """Watershed points sorted by angle around the origin.

    ``z`` is the value stored with each point (None if the file had none).
    """

    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray]

    def closed(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return the ring with the first point repeated so the line closes."""
        xs = np.append(self.x, self.x[0])
        ys = np.append(self.y, self.y[0])
        zs = None if self.z is None else np.append(self.z, self.z[0])
        return xs, ys, zs


def read_csv_table(path: Path, required_cols: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    """Read a CSV and check it has data and the required columns.

    Returns
    -------
    pandas.DataFrame or None
        None if the file is missing, unreadable, empty or lacks columns.
    """
    path = Path(path)
    if not path.is_file():
        log_warning(f"Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log_warning(f"Could not read {path}: {exc}")
        return None

    df.columns = df.columns.str.strip()  # C++ writer can leave whitespace in headers
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        log_warning(f"{path.name} is missing column(s) {missing}, found {list(df.columns)}")
        return None
    if df.empty:
        log_warning(f"{path.name} contains no data")
        return None
    return df


def load_grid(path: Path, value_col: str = COL_VALUE) -> Optional[Grid]:
    """Load an ``x, y, f`` CSV into a :class:`Grid`.

    Parameters
    ----------
    path : Path
        CSV file written by the backend.
    value_col : str
        Column holding the field values.

    Returns
    -------
    Grid or None
        None on any read/shape problem (a warning is logged).
    """
    df = read_csv_table(path, (COL_X, COL_Y, value_col))
    if df is None:
        return None

    cols = [COL_X, COL_Y, value_col]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[COL_X, COL_Y])

    try:
        table = df.pivot(index=COL_Y, columns=COL_X, values=value_col)
    except ValueError as exc:  # duplicate (x, y) pairs
        log_warning(f"{Path(path).name} could not be arranged into a grid: {exc}")
        return None

    values = table.to_numpy(dtype=float)
    if np.isnan(values).all():
        log_warning(f"{Path(path).name} has no valid numeric values")
        return None
    n_nan = int(np.isnan(values).sum())
    if n_nan:
        log_warning(f"{Path(path).name} has {n_nan} missing/invalid cell(s), plots will show gaps")

    return Grid(x=table.columns.to_numpy(dtype=float),
                y=table.index.to_numpy(dtype=float),
                values=values)


def load_watershed(path: Path) -> Optional[WatershedRing]:
    """Load a watershed CSV (``x, y`` plus ``f`` or ``r``) sorted by angle."""
    df = read_csv_table(path, (COL_X, COL_Y))
    if df is None:
        return None

    z_col = COL_VALUE if COL_VALUE in df.columns else (COL_VALUE_ALT if COL_VALUE_ALT in df.columns else None)
    use_cols = [COL_X, COL_Y] + ([z_col] if z_col else [])
    df[use_cols] = df[use_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[COL_X, COL_Y])
    if df.empty:
        log_warning(f"{Path(path).name} has no valid points")
        return None

    # Sort by angle so the ring line connects cleanly around the origin.
    theta = np.arctan2(df[COL_Y].to_numpy(), df[COL_X].to_numpy())
    order = np.argsort(theta)
    z = df[z_col].to_numpy(dtype=float)[order] if z_col else None
    return WatershedRing(x=df[COL_X].to_numpy(dtype=float)[order],
                         y=df[COL_Y].to_numpy(dtype=float)[order],
                         z=z)


class PatternReader:
    """Reads (and caches) everything the backend saved for one pattern folder.

    Parameters
    ----------
    pattern_dir : str or Path
        Folder such as ``pattern1`` containing ``speckles.csv`` etc.

    Attributes
    ----------
    is_valid : bool
        False if the folder does not exist.
    output_dir : Path
        ``<pattern>_post`` next to the pattern folder.
    """

    def __init__(self, pattern_dir) -> None:
        self.pattern_dir = Path(pattern_dir)
        self.is_valid = self.pattern_dir.is_dir()
        resolved = self.pattern_dir.resolve()
        self.name = resolved.name
        self.output_dir = resolved.parent / f"{resolved.name}{POST_SUFFIX}"
        self._cache: Dict[str, object] = {}
        if not self.is_valid:
            log_error(f"Pattern folder not found: {self.pattern_dir}")

    def _cached(self, key: str, loader: Callable[[], object]) -> object:
        # Cache failures too so the same warning is not repeated.
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def _grid(self, subdir: str, filename: str) -> Optional[Grid]:
        path = self.pattern_dir / subdir / filename if subdir else self.pattern_dir / filename
        return self._cached(str(path), lambda: load_grid(path))

    def read_speckle(self) -> Optional[Grid]:
        """Load the raw speckle pattern."""
        return self._grid("", SPECKLE_FILENAME)

    def read_gradients(self) -> Tuple[Optional[Grid], Optional[Grid], Optional[Grid]]:
        """Load ``(grad_x, grad_y, grad_mag)``; any entry may be None."""
        return (self._grid(GRAD_SUBDIR, GRAD_X_FILENAME),
                self._grid(GRAD_SUBDIR, GRAD_Y_FILENAME),
                self._grid(GRAD_SUBDIR, GRAD_MAG_FILENAME))

    def read_sssig(self) -> Optional[Grid]:
        """Load the SSSIG heatmap."""
        return self._grid(SSSIG_SUBDIR, SSSIG_FILENAME)

    def read_sssig_delta(self) -> Optional[Grid]:
        """Load the SSSIG delta map (SSSIG - MIG)."""
        return self._grid(SSSIG_SUBDIR, SSSIG_DELTA_FILENAME)

    def read_autocorr(self, name: str) -> Optional[Grid]:
        """Load the autocorrelation map for a lowercase correlator name."""
        return self._grid(AUTOC_SUBDIR, AUTOCORR_FILE_PATTERN.format(name=name))

    def read_radial_gradient(self, name: str) -> Optional[Grid]:
        """Load the radial gradient for a correlator (falls back to the old shared file)."""
        folder = self.pattern_dir / AUTOC_SUBDIR
        specific = folder / GRADR_FILE_PATTERN.format(name=name)
        legacy = folder / LEGACY_GRADR_FILENAME

        def loader() -> Optional[Grid]:
            if specific.is_file():
                return load_grid(specific)
            if legacy.is_file():
                log_warning(f"{specific.name} not found, using shared {legacy.name} "
                            "(may belong to a different correlator)", name)
                return load_grid(legacy)
            log_warning(f"Missing file: {specific}", name)
            return None

        return self._cached(f"gradr:{name}", loader)

    def read_watershed(self, name: str) -> Optional[WatershedRing]:
        """Load the watershed ring for a correlator (falls back to the old shared file)."""
        folder = self.pattern_dir / AUTOC_SUBDIR
        specific = folder / WSHED_FILE_PATTERN.format(name=name)
        legacy = folder / LEGACY_WSHED_FILENAME

        def loader() -> Optional[WatershedRing]:
            if specific.is_file():
                return load_watershed(specific)
            if legacy.is_file():
                log_warning(f"{specific.name} not found, using shared {legacy.name} "
                            "(may belong to a different correlator)", name)
                return load_watershed(legacy)
            log_warning(f"Missing file: {specific}", name)
            return None

        return self._cached(f"wshed:{name}", loader)