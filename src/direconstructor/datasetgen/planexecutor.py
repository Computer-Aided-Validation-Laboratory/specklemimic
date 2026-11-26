"""
Renders every image of a run plan (see planio for the CSV format).

Example (one line):
  python datasetgen/planexecutor.py plans/truths.csv plans/variants.csv --root D:/datasets/run1 --workers 8

Images go to <root>/<directory>/<name>.bmp (root defaults to the folder holding truths.csv),
and every finished image is listed in <root>/manifest.csv. Running the same command again
resumes: images already in the manifest (and on disk) are skipped. Failed images are logged
in <root>/failures.csv and the run carries on; rerun to retry them. Exit code: 0 if everything
rendered, 1 if some images failed, 2 if the plan or the root's record files are invalid,
130 if interrupted.
"""

import rileyrunner #First: rileyrunner pins thread pools before numpy loads.

import argparse
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]
import riley  # type: ignore[reportMissingImports]

import planio
import planpaths
from planledger import Ledger, file_sha256, make_failure, now
from planworker import render_truth


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("truths", type=Path, help="truths.csv")
    parser.add_argument("variants", type=Path, nargs="?", help="variants.csv (optional)")
    parser.add_argument("--root", type=Path, help="dataset root (default: the folder holding truths.csv)")
    parser.add_argument("--workers", type=int, default=1, help="render processes (default 1: no pool)")
    parser.add_argument("--limit", type=int, help="only render the first K truths that have work left")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def check_patterns(jobs):
    allowed = set(riley.SpecklePattern.__members__) | {"perlin"}
    bad = sorted({job.args.pattern for job in jobs} - allowed)
    if bad:
        raise planio.PlanError([f"unknown pattern(s) {bad}; riley supports {sorted(allowed)}"])


def find_work(jobs, root, done):
    """
    (job, pending suffixes) for every truth with images left, plus the done keys whose
    image went missing (they get rendered again and the manifest is compacted afterwards).
    """
    work, missing = [], set()
    for job in jobs:
        pending = set()
        for _, suffix, name, _ in job.images():
            key = (job.truth_id, suffix)
            if key not in done:
                pending.add(suffix)
            elif not planpaths.image_path(root, job.directory, name).exists():
                pending.add(suffix)
                missing.add(key)
        if pending:
            work.append((job, pending))
    return work, missing


def crashed(job, pending, reason):
    return [
        make_failure(kind, job.truth_id, suffix, planpaths.relative_image(job.directory, name), reason)
        for kind, suffix, name, _ in job.images() if suffix in pending
    ]


def run_inline(work, root, on_result):
    for job, pending in work:
        on_result(job, *render_truth(job, root, pending))


def collect(future, job, pending, on_result):
    """
    Hands a finished future's results to on_result. Returns True if the pool broke.
    render_truth catches render errors itself, so anything raised here is infrastructure.
    """
    try:
        records, failures = future.result()
    except BrokenProcessPool:
        on_result(job, [], crashed(job, pending, "worker process died; rerun to retry"))
        return True
    except Exception as error:
        on_result(job, [], crashed(job, pending, f"{type(error).__name__}: {error}"))
        return False
    on_result(job, records, failures)
    return False


def run_pool(work, root, workers, on_result):
    """
    Renders truths in worker processes, keeping at most 2 per worker queued so huge plans
    don't create millions of futures. If a worker process dies (e.g. a native crash), the
    truths in flight are logged as failures and a fresh pool carries on.
    """
    queue = iter(work)
    in_flight = {}
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        while True:
            while len(in_flight) < 2 * workers:
                item = next(queue, None)
                if item is None:
                    break
                in_flight[pool.submit(render_truth, item[0], str(root), item[1])] = item
            if not in_flight:
                return

            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            broken = False
            for future in finished:
                broken |= collect(future, *in_flight.pop(future), on_result)

            if broken:
                #Every other future of a broken pool fails too; keep any that finished first.
                for future in wait(in_flight)[0]:
                    collect(future, *in_flight[future], on_result)
                in_flight.clear()
                pool.shutdown(wait=True)
                pool = ProcessPoolExecutor(max_workers=workers)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        jobs = planio.read_plan(args.truths, args.variants)
        check_patterns(jobs)
    except planio.PlanError as error:
        print(error, file=sys.stderr)
        return 2

    root = (args.root or args.truths.parent).resolve()
    ledger = Ledger(root)
    try:
        ledger.check()
        done = ledger.done()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    work, missing = find_work(jobs, root, done)
    if args.limit is not None:
        work = work[:args.limit]

    total_images = sum(1 for job in jobs for _ in job.images())
    pending_images = sum(len(pending) for _, pending in work)
    plans = {str(path.resolve()): file_sha256(path) for path in (args.truths, args.variants) if path}

    previous = ledger.last_run_info()
    if previous and sorted(previous.get("plans", {}).values()) != sorted(plans.values()) and done:
        print("WARNING: the plan files changed since the last run in this root. Images already in "
              "the manifest are NOT re-rendered, even if their rows changed.", file=sys.stderr)
    if missing:
        print(f"WARNING: {len(missing)} image(s) listed in the manifest are missing on disk; rendering them again.",
              file=sys.stderr)

    root.mkdir(parents=True, exist_ok=True)
    ledger.write_run_info(dict(
        started=now(),
        plans=plans,
        root=str(root),
        workers=args.workers,
        limit=args.limit,
        truths=len(jobs),
        images=total_images,
        already_done=len(done),
        pending=pending_images,
        riley_module=str(riley.__file__),
        numpy_version=np.__version__,
        python_version=platform.python_version(),
    ))
    print(f"Plan: {len(jobs)} truths, {total_images} images. Rendering {pending_images} image(s) "
          f"from {len(work)} truth(s) into {root} with {args.workers} worker(s).", flush=True)

    totals = dict(rendered=0, failed=0, truths=0)
    failures_seen = []
    started = time.perf_counter()

    def on_result(job, records, failures):
        ledger.append(records, failures)
        totals["rendered"] += len(records)
        totals["failed"] += len(failures)
        totals["truths"] += 1
        failures_seen.extend(failures[:max(0, 5 - len(failures_seen))])
        elapsed = time.perf_counter() - started
        remaining = elapsed / totals["truths"] * (len(work) - totals["truths"])
        print(f"[{totals['truths']}/{len(work)}] {job.truth_id}: {len(records)} rendered, "
              f"{len(failures)} failed | {elapsed:.0f}s elapsed, ~{remaining:.0f}s left", flush=True)

    interrupted = False
    try:
        with ledger:
            if args.workers == 1:
                run_inline(work, root, on_result)
            else:
                run_pool(work, root, args.workers, on_result)
    except KeyboardInterrupt:
        interrupted = True

    if missing:
        ledger.compact()

    print(f"Done in {time.perf_counter() - started:.0f}s: {totals['rendered']} rendered, "
          f"{totals['failed']} failed.", flush=True)
    for failure in failures_seen:
        print(f"  FAILED {failure['image']}: {failure['error']}", file=sys.stderr)
    if totals["failed"]:
        print(f"  Details in {ledger.failures_path}; rerun the same command to retry.", file=sys.stderr)
    if interrupted:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
