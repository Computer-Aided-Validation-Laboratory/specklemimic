"""
Run records kept in the dataset root:

    manifest.csv     one row per finished image: paths, sha256 and the camera effects that
                     were actually sampled (the ML labels). It is also the list of what is done,
                     which is how an interrupted run resumes.
    failures.csv     one row per image that failed, with the error and traceback.
    run_info.jsonl   one line per run: versions, plan hashes, and what the run was asked to do.

Only the main process writes these files, so no locking is needed.
Standard library only.
"""

import csv
import hashlib
import json
import os
import time
from pathlib import Path

ENCODING = "utf-8"

#Camera fields copied into the manifest. Values come from the camera actually used.
EFFECT_COLUMNS = ("distort_k1", "distort_k2", "distort_k3", "distort_p1", "distort_p2", "psf_sigma_x")
MANIFEST_COLUMNS = (
    "kind", "truth_id", "suffix", "image", "json", "truth_image", "sha256", "distorted",
) + EFFECT_COLUMNS
#distortion holds the sampled camera effects (JSON) when the failure happened after sampling.
FAILURE_COLUMNS = ("timestamp", "kind", "truth_id", "suffix", "image", "error", "distortion", "traceback")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def make_failure(kind, truth_id, suffix, image, error, trace="", distortion=None):
    return dict(
        timestamp=now(), kind=kind, truth_id=truth_id, suffix=suffix, image=image, error=error,
        distortion="" if distortion is None else json.dumps(distortion), traceback=trace,
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _drop_partial_last_line(path):
    """
    A run killed mid-write can leave a last line without its newline. Cut it off, so the
    next append does not glue a new row onto it. Rows end with their newline, so every
    line that has one was written completely.
    """
    with path.open("rb+") as file:
        end = file.seek(0, os.SEEK_END)
        if end == 0:
            return
        file.seek(end - 1)
        if file.read(1) == b"\n":
            return
        position = end
        while position > 0:
            step = min(1 << 16, position)
            position -= step
            file.seek(position)
            chunk = file.read(step)
            newline = chunk.rfind(b"\n")
            if newline != -1:
                file.truncate(position + newline + 1)
                return
        file.truncate(0)


class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.csv"
        self.failures_path = self.root / "failures.csv"
        self.run_info_path = self.root / "run_info.jsonl"
        self._files = []

    @staticmethod
    def _check(path, columns):
        """
        Repairs a cut-off last line of an existing record file, and raises RuntimeError if its
        columns are not this module's (appending would corrupt it). Returns True if it has data.
        """
        if not path.exists():
            return False
        _drop_partial_last_line(path)
        if path.stat().st_size == 0:
            return False
        with path.open("r", encoding=ENCODING, newline="") as file:
            header = next(csv.reader(file), [])
        if tuple(header) != tuple(columns):
            raise RuntimeError(f"{path} has unexpected columns {header}; move it away to start a new one")
        return True

    def check(self):
        """
        Checks both record files before a run; raises RuntimeError if one cannot be appended to.
        """
        self._check(self.manifest_path, MANIFEST_COLUMNS)
        self._check(self.failures_path, FAILURE_COLUMNS)

    def done(self):
        """
        Set of (truth_id, suffix) already in the manifest.
        """
        if not self._check(self.manifest_path, MANIFEST_COLUMNS):
            return set()
        with self.manifest_path.open("r", encoding=ENCODING, newline="") as file:
            return {(row["truth_id"], row["suffix"]) for row in csv.DictReader(file)}

    def last_run_info(self):
        if not self.run_info_path.exists():
            return None
        lines = [line for line in self.run_info_path.read_text(encoding=ENCODING).splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None

    def write_run_info(self, info):
        with self.run_info_path.open("a", encoding=ENCODING) as file:
            file.write(json.dumps(info, default=str) + "\n")

    def _open(self, path, columns):
        has_data = self._check(path, columns)
        file = path.open("a", encoding=ENCODING, newline="")
        writer = csv.DictWriter(file, fieldnames=columns)
        if not has_data:
            writer.writeheader()
        self._files.append(file)
        return file, writer

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_file, self._manifest = self._open(self.manifest_path, MANIFEST_COLUMNS)
        self._failures_file, self._failures = self._open(self.failures_path, FAILURE_COLUMNS)
        return self

    def __exit__(self, *exc):
        for file in self._files:
            file.close()
        self._files = []

    @staticmethod
    def _sync(file):
        file.flush()
        os.fsync(file.fileno())

    def append(self, records, failures):
        """
        Appends one truth's results and forces them to disk, so a crash loses at most the
        truth being rendered.
        """
        if records:
            self._manifest.writerows(records)
            self._sync(self._manifest_file)
        if failures:
            self._failures.writerows(failures)
            self._sync(self._failures_file)

    def compact(self):
        """
        Rewrites the manifest keeping only the last row per (truth_id, suffix). Needed after
        an image listed as done was missing on disk and got rendered again.
        """
        with self.manifest_path.open("r", encoding=ENCODING, newline="") as file:
            rows = list(csv.DictReader(file))
        latest = {}
        for row in rows:
            key = (row["truth_id"], row["suffix"])
            latest.pop(key, None) #Re-insert, so the row moves to where it was last written.
            latest[key] = row
        if len(latest) == len(rows):
            return 0
        temporary = self.manifest_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding=ENCODING, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(latest.values())
            self._sync(file)
        os.replace(temporary, self.manifest_path)
        return len(rows) - len(latest)
