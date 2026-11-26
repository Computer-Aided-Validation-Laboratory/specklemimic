"""
Column layout of a run plan and conversion between CSV cells and RileyRunner args.

Both plan files (truths.csv and variants.csv) share the parameter columns defined here.
A blank cell means "not set" (the words nan/none/null count as blank too, so CSVs written
by pandas work). A variant inherits every value it does not set from its truth.

Standard library only, so ML code can import it without numpy or riley.
"""

import math
from types import SimpleNamespace

#Scalar parameters and their types. The types are fixed on purpose: RileyRunner seeds its
#distortion RNG with an f-string of these values, so reading 1 instead of 1.0 would change images.
SCALARS = {
    "pattern": str,
    "realization": int,
    "coverage": float,
    "scale": float,
    "side": float,
    "width": int,
    "height": int,
    "focal_length": float,
    "fov_scale": float,
    "sub_sample": int,
    "effects_seed": int,
}
#Tuple parameters, stored as one float column per component.
VECTORS = {
    "world_rot": ("x", "y", "z"),
    "roi_cent_world": ("x", "y", "z"),
}
#(MIN, MAX) parameters RileyRunner samples from uniformly. MIN == MAX pins an exact value.
RANGES = ("k1", "k2", "k3", "p1", "p2", "blur_sigma")

#Parameter groups, in column order.
MESH_FIELDS = ("pattern", "realization", "coverage", "scale", "side")
BASE_CAMERA_FIELDS = ("width", "height", "focal_length", "fov_scale", "world_rot", "roi_cent_world", "sub_sample")
EFFECT_FIELDS = ("effects_seed",) + RANGES
FIELDS = MESH_FIELDS + BASE_CAMERA_FIELDS + EFFECT_FIELDS

TRUTH_ID_COLUMNS = ("truth_id", "directory", "name")
VARIANT_ID_COLUMNS = ("truth_id", "suffix")

BLANK_WORDS = {"", "nan", "none", "null"}
U32_MAX = 0xFFFFFFFF


def field_columns(field):
    """
    CSV columns that hold one args field.
    """
    if field in VECTORS:
        return tuple(f"{field}_{component}" for component in VECTORS[field])
    if field in RANGES:
        return (f"{field}_min", f"{field}_max")
    return (field,)


def column_type(column):
    if column in SCALARS:
        return SCALARS[column]
    return float #Vector components and range bounds.


PARAM_COLUMNS = tuple(column for field in FIELDS for column in field_columns(field))
MESH_COLUMNS = tuple(column for field in MESH_FIELDS for column in field_columns(field))
TRUTH_COLUMNS = TRUTH_ID_COLUMNS + PARAM_COLUMNS
VARIANT_COLUMNS = VARIANT_ID_COLUMNS + PARAM_COLUMNS


def is_blank(cell):
    return cell is None or cell.strip().lower() in BLANK_WORDS


def parse_cell(column, cell):
    """
    Parses one non-blank cell into its column's type. Raises ValueError with a readable message.
    """
    kind = column_type(column)
    text = cell.strip()
    if kind is str:
        return text

    if kind is int:
        try:
            return int(text)
        except ValueError:
            pass
        #Accept integral floats like "42.0": pandas writes int columns that contain NaN this way.
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"expected an integer, got {cell!r}") from None
        if not (math.isfinite(number) and number.is_integer()):
            raise ValueError(f"expected an integer, got {cell!r}")
        return int(number)

    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"expected a number, got {cell!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {cell!r}")
    return number


def parse_cells(row, where, errors):
    """
    Typed values of the non-blank parameter cells in a CSV row, keyed by column.
    Blank cells are left out, so inheritance is a plain dict.update.
    Problems are appended to errors as "<where>: <column>: <message>".
    """
    values = {}
    for column in PARAM_COLUMNS:
        cell = row.get(column)
        if is_blank(cell):
            continue
        try:
            values[column] = parse_cell(column, cell)
        except ValueError as error:
            errors.append(f"{where}: {column}: {error}")
    return values


def format_value(column, value):
    """
    Formats one value for a cell. Floats use repr, the shortest form that reads back exactly.
    """
    if value is None:
        return ""
    kind = column_type(column)
    if kind is str:
        return str(value)
    if kind is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{column}: expected an integer, got {value!r}")
        return str(int(value))
    return repr(float(value)) #float() first, so numpy scalars don't print as np.float64(...).


def flatten(values):
    """
    Turns args-style values (a dict or namespace with field names such as "k1" -> (MIN, MAX))
    into CSV cells for every parameter column. Missing or None fields become blank cells.
    """
    if not isinstance(values, dict):
        values = vars(values)
    unknown = sorted(set(values) - set(FIELDS))
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")

    cells = {}
    for field in FIELDS:
        columns = field_columns(field)
        value = values.get(field)
        if len(columns) == 1:
            cells[columns[0]] = format_value(columns[0], value)
            continue
        parts = (None,) * len(columns) if value is None else tuple(value)
        if len(parts) != len(columns):
            raise ValueError(f"{field}: expected {len(columns)} values, got {len(parts)}")
        for column, part in zip(columns, parts):
            cells[column] = format_value(column, part)
    return cells


def to_args(values, where, errors, required=FIELDS):
    """
    Builds the args namespace RileyRunner expects from column values (as from parse_cells).
    Fields in `required` must be complete; any other incomplete field is set to None.
    Returns None if anything is wrong; the problems are appended to errors.
    """
    start = len(errors)
    args = {}
    for field in FIELDS:
        columns = field_columns(field)
        missing = [column for column in columns if column not in values]
        if missing:
            if field in required or len(missing) < len(columns):
                #Required, or half filled (e.g. k1_min without k1_max), which is always a mistake.
                errors.extend(f"{where}: {column}: required but blank" for column in missing)
            args[field] = None
            continue
        parts = tuple(values[column] for column in columns)
        args[field] = parts[0] if len(columns) == 1 else parts

    def check(field, ok, message):
        if args[field] is not None and not ok(args[field]):
            errors.append(f"{where}: {field}: {message}, got {args[field]!r}")

    check("pattern", lambda value: value != "", "must not be empty")
    check("realization", lambda value: 0 <= value <= U32_MAX, "must fit an unsigned 32-bit seed")
    check("effects_seed", lambda value: 0 <= value <= U32_MAX, "must fit an unsigned 32-bit seed")
    check("coverage", lambda value: 0 < value < 1, "must be in (0, 1)")
    for field in ("scale", "side", "focal_length", "fov_scale"):
        check(field, lambda value: value > 0, "must be positive")
    for field in ("width", "height"):
        check(field, lambda value: value >= 1, "must be at least 1")
    check("sub_sample", lambda value: 1 <= value <= 8, "must be in 1..8")
    for field in RANGES:
        check(field, lambda pair: pair[0] <= pair[1], "needs MIN <= MAX")
    check("blur_sigma", lambda pair: pair[0] >= 0, "MIN must be nonnegative")

    if len(errors) > start:
        return None
    return SimpleNamespace(**args)
