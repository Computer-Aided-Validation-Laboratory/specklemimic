"""Console output helpers, styled to match the C++ backend."""

import sys
from typing import Optional

from post.constants import BANNER_WIDTH


def _emit(message: str) -> None:
    print(message, flush=True)


def print_line() -> None:
    """Print a full-width dashed separator."""
    _emit("-" * BANNER_WIDTH)


def print_banner(title: str) -> None:
    """Print a title between two separators."""
    print_line()
    _emit(title)
    print_line()


def print_section(message: str) -> None:
    """Print a one-line section message between two separators."""
    print_line()
    _emit(message)
    print_line()


def _format(message: str, tag: Optional[str], level: str = "") -> str:
    prefix = f"({tag.upper()}) " if tag else ""
    level_text = f"{level}: " if level else ""
    return f"{prefix}{level_text}{message}"


def log_info(message: str, tag: Optional[str] = None) -> None:
    """Log a normal message, e.g. ``(ZNSSD) Plotting...``."""
    _emit(_format(message, tag))


def log_warning(message: str, tag: Optional[str] = None) -> None:
    """Log a recoverable problem (something gets skipped)."""
    _emit(_format(message, tag, "WARNING"))


def log_error(message: str, tag: Optional[str] = None) -> None:
    """Log a failure of a step (the run continues with the next step)."""
    _emit(_format(message, tag, "ERROR"))
    sys.stdout.flush()