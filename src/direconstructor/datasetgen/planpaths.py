"""
Where each image of a run plan lives.

    truth:   <root>/<directory>/<name>.bmp            (+ <name>.json)
    variant: <root>/<directory>/<name>_<suffix>.bmp   (+ .json), same directory as its truth

directory is relative, uses "/" separators and never leaves root. Names, suffixes and
directory parts only use [A-Za-z0-9_.-], so paths behave the same on Windows and Linux.

Standard library only, so ML code can import it without numpy or riley.
"""

import re
from pathlib import Path

SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{kind}{n}" for kind in ("COM", "LPT") for n in range(1, 10)}


def check_safe_name(value):
    """
    Returns an error message, or None if value is usable as a file name part.
    """
    if not SAFE_NAME.fullmatch(value):
        return f"must be non-empty and only use A-Z a-z 0-9 _ . - (got {value!r})"
    if value.endswith("."):
        return f"must not end with '.', Windows drops it (got {value!r})"
    if value.split(".")[0].upper() in WINDOWS_RESERVED:
        return f"is a reserved device name on Windows (got {value!r})"
    return None


def check_relative_dir(value):
    """
    Returns an error message, or None if value is a usable relative directory.
    """
    if value == "":
        return "must not be empty"
    if "\\" in value:
        return f"must use '/' separators (got {value!r})"
    if value.startswith("/") or ":" in value:
        return f"must be relative to the dataset root (got {value!r})"
    for part in value.split("/"):
        if part in ("", ".", ".."):
            return f"must not contain empty, '.' or '..' parts (got {value!r})"
        problem = check_safe_name(part)
        if problem:
            return f"bad part {part!r}: {problem}"
    return None


def variant_name(truth_name, suffix):
    return f"{truth_name}_{suffix}"


def default_directory(pattern, truth_id, split=None):
    """
    Suggested layout for plan generators: [<split>/]<pattern>/<truth_id>.
    Use truth_id as the truth's name too, so every file name is unique across the dataset.
    """
    return "/".join(part for part in (split, pattern, truth_id) if part)


def relative_image(directory, name):
    """
    Image path relative to the dataset root, with "/" separators (as written to the manifest).
    """
    return f"{directory}/{name}.bmp"


def relative_json(directory, name):
    return f"{directory}/{name}.json"


def image_path(root, directory, name):
    return Path(root) / directory / f"{name}.bmp"


def json_path(root, directory, name):
    return Path(root) / directory / f"{name}.json"
