"""Registry of supported autocorrelation functions and their plot styling."""

from typing import Iterable, List, NamedTuple

from post.console import log_warning


class CorrelatorInfo(NamedTuple):
    """Display settings for one autocorrelation function.

    Attributes
    ----------
    title : str
        Long title used on heatmap tiles.
    short : str
        Short label (e.g. "ZNSSD") used in console output and 3D titles.
    cmap : str
        Matplotlib colormap name.
    ring_colour : str
        Watershed ring colour that stays visible on top of ``cmap``.
    """

    title: str
    short: str
    cmap: str
    ring_colour: str


# Colormaps are taken from the original comparison figure, do not change them.
CORRELATORS = {
    "ssd": CorrelatorInfo("SSD (Sum of Squared Differences)", "SSD", "viridis", "red"),
    "scc": CorrelatorInfo("SCC (Standard Cross-Correlation)", "SCC", "plasma", "cyan"),
    "nssd": CorrelatorInfo("NSSD (Normalized SSD)", "NSSD", "cividis", "red"),
    "zssd": CorrelatorInfo("ZSSD (Zero-Mean SSD)", "ZSSD", "magma", "cyan"),
    "znssd": CorrelatorInfo("ZNSSD (Zero-Mean Normalized SSD)", "ZNSSD", "inferno", "cyan"),
    "zncc": CorrelatorInfo("ZNCC (Zero-Mean Normalized CC)", "ZNCC", "turbo", "white"),
}

# Alternative spellings mapped onto the backend names.
CORRELATOR_ALIASES = {"cc": "scc"}


def get_correlator_info(name: str) -> CorrelatorInfo:
    """Return styling for a correlator, with a neutral fallback for unknown names."""
    key = name.strip().lower()
    key = CORRELATOR_ALIASES.get(key, key)
    if key in CORRELATORS:
        return CORRELATORS[key]
    return CorrelatorInfo(key.upper(), key.upper(), "viridis", "red")


def normalise_correlator_names(names: Iterable[str]) -> List[str]:
    """Lowercase, de-alias and de-duplicate correlator names, skipping unknown ones.

    Parameters
    ----------
    names : iterable of str
        Names as typed by the user (any case).

    Returns
    -------
    list of str
        Valid lowercase names in the order given.
    """
    valid: List[str] = []
    for raw in names:
        key = str(raw).strip().lower()
        key = CORRELATOR_ALIASES.get(key, key)
        if key not in CORRELATORS:
            log_warning(f"Unknown autocorrelation function '{raw}', skipping. "
                        f"Choose from {sorted(CORRELATORS)}")
            continue
        if key not in valid:
            valid.append(key)
    return valid