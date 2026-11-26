"""
Reads and writes run plans.

A plan is two CSV files with the parameter columns from planschema:
    truths.csv:   truth_id, directory, name, <parameters>    one row per clean image (one mesh)
    variants.csv: truth_id, suffix, <parameters>             one row per distorted image

Every truth row must set the mesh and base camera columns. Its effects columns are optional
defaults for its variants (the clean render does not use them). A variant row is its truth row
with the variant's non-blank cells written over it, so it only needs the values it mutates.
Variants may change camera columns but not mesh columns (that would be another pattern).

read_plan checks everything and reports all problems at once as a PlanError.

Standard library only, so ML code can import it without numpy or riley.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import planpaths
import planschema

ENCODING = "utf-8"
MAX_LISTED_ERRORS = 100


class PlanError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        listed = self.errors[:MAX_LISTED_ERRORS]
        more = len(self.errors) - len(listed)
        message = f"{len(self.errors)} problem(s) in the plan:\n  " + "\n  ".join(listed)
        if more:
            message += f"\n  ... and {more} more"
        super().__init__(message)


@dataclass
class VariantJob:
    suffix: str
    name: str
    args: SimpleNamespace


@dataclass
class TruthJob:
    truth_id: str
    directory: str
    name: str
    args: SimpleNamespace
    variants: list[VariantJob] = field(default_factory=list)

    def images(self):
        """
        Every image of this truth as (kind, suffix, name, args); the truth itself has suffix "".
        """
        yield "truth", "", self.name, self.args
        for variant in self.variants:
            yield "variant", variant.suffix, variant.name, variant.args


def _read_rows(path, columns, errors):
    """
    Yields (where, row) for every data row of a plan CSV after checking its header.
    """
    path = Path(path)
    #utf-8-sig also accepts the byte order mark Excel adds.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        duplicated = sorted({column for column in header if header.count(column) > 1})
        unknown = [column for column in header if column not in columns]
        missing = [column for column in columns if column not in header]
        if duplicated:
            errors.append(f"{path.name}: duplicated column(s): {', '.join(duplicated)}")
        if unknown:
            errors.append(f"{path.name}: unknown column(s): {', '.join(unknown)}")
        if missing:
            errors.append(f"{path.name}: missing column(s): {', '.join(missing)}")
        if duplicated or unknown or missing:
            return

        for row in reader:
            where = f"{path.name}:{reader.line_num}"
            if None in row:
                errors.append(f"{where}: more cells than columns")
                continue
            if any(value is None for value in row.values()):
                errors.append(f"{where}: fewer cells than columns")
                continue
            yield where, row


def _check_name(where, column, value, check, errors):
    problem = check(value)
    if problem:
        errors.append(f"{where}: {column}: {problem}")
        return False
    return True


def read_plan(truths_csv, variants_csv=None):
    """
    Reads and validates a plan. Returns a list of TruthJob in truths.csv order, each holding
    its VariantJobs in variants.csv order. Raises PlanError listing every problem found.
    """
    errors = []
    jobs = {}           #truth_id -> TruthJob
    truth_values = {}   #truth_id -> column values, for inheritance; None if the row was bad
    image_owner = {}    #lower-cased image path -> where; Windows paths are case-insensitive

    def claim_image(where, directory, name):
        key = planpaths.relative_image(directory, name).lower()
        if key in image_owner:
            errors.append(f"{where}: image {planpaths.relative_image(directory, name)} is also written by {image_owner[key]}")
        else:
            image_owner[key] = where

    for where, row in _read_rows(truths_csv, planschema.TRUTH_COLUMNS, errors):
        truth_id = row["truth_id"].strip()
        directory = row["directory"].strip()
        name = row["name"].strip()
        ok = _check_name(where, "truth_id", truth_id, planpaths.check_safe_name, errors)
        ok &= _check_name(where, "directory", directory, planpaths.check_relative_dir, errors)
        ok &= _check_name(where, "name", name, planpaths.check_safe_name, errors)
        if ok and truth_id in jobs:
            errors.append(f"{where}: truth_id {truth_id!r} is used more than once")
            continue

        start = len(errors)
        values = planschema.parse_cells(row, where, errors)
        required = planschema.MESH_FIELDS + planschema.BASE_CAMERA_FIELDS
        args = planschema.to_args(values, where, errors, required=required) if len(errors) == start else None
        if not ok:
            continue
        claim_image(where, directory, name)
        jobs[truth_id] = TruthJob(truth_id, directory, name, args)
        truth_values[truth_id] = values if args is not None else None

    if variants_csv is not None:
        seen = set()
        for where, row in _read_rows(variants_csv, planschema.VARIANT_COLUMNS, errors):
            truth_id = row["truth_id"].strip()
            suffix = row["suffix"].strip()
            if not _check_name(where, "suffix", suffix, planpaths.check_safe_name, errors):
                continue
            job = jobs.get(truth_id)
            if job is None:
                errors.append(f"{where}: truth_id {truth_id!r} is not in the truths file")
                continue
            if (truth_id, suffix) in seen:
                errors.append(f"{where}: suffix {suffix!r} is used more than once for truth {truth_id!r}")
                continue
            seen.add((truth_id, suffix))

            name = planpaths.variant_name(job.name, suffix)
            claim_image(where, job.directory, name)

            start = len(errors)
            values = planschema.parse_cells(row, where, errors)
            inherited = truth_values[truth_id]
            if len(errors) > start or inherited is None:
                continue #The truth's own problems are already reported.

            #Restating the truth's mesh value is fine; changing it is not.
            for column in planschema.MESH_COLUMNS:
                if column in values and values[column] != inherited.get(column):
                    errors.append(
                        f"{where}: {column}: variants cannot change the pattern "
                        f"(truth has {inherited.get(column)!r}, variant has {values[column]!r})"
                    )
            if len(errors) > start:
                continue

            merged = dict(inherited)
            merged.update(values)
            args = planschema.to_args(merged, where, errors)
            if args is not None:
                job.variants.append(VariantJob(suffix, name, args))

    if errors:
        raise PlanError(errors)
    return list(jobs.values())


def _write(path, columns, id_columns, rows):
    path = Path(path)
    with path.open("w", encoding=ENCODING, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            values = dict(row) if isinstance(row, dict) else dict(vars(row))
            try:
                cells = {column: str(values.pop(column)) for column in id_columns}
            except KeyError as error:
                raise ValueError(f"row {index}: missing {error.args[0]!r}") from None
            try:
                cells.update(planschema.flatten(values))
            except ValueError as error:
                raise ValueError(f"row {index}: {error}") from None
            writer.writerow(cells)


def write_truths(path, rows):
    """
    Writes truths.csv. Each row is a dict or namespace with truth_id, directory, name and
    args-style parameters (e.g. world_rot=(0, 0, 0), k1=(-0.2, 0.2)); missing ones stay blank.
    """
    _write(path, planschema.TRUTH_COLUMNS, planschema.TRUTH_ID_COLUMNS, rows)


def write_variants(path, rows):
    """
    Writes variants.csv. Each row is a dict or namespace with truth_id, suffix and only the
    args-style parameters the variant changes; everything else stays blank (inherited).
    """
    _write(path, planschema.VARIANT_COLUMNS, planschema.VARIANT_ID_COLUMNS, rows)
