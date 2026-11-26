"""
Renders one TruthJob (its clean image and its variants) with RileyRunner.

Runs in the main process (--workers 1) or in a worker process. It only returns results;
the main process writes them to the ledger.
"""

from rileyrunner import RileyRunner #First: rileyrunner pins thread pools before numpy loads.

import traceback
from pathlib import Path

import planpaths
from planledger import EFFECT_COLUMNS, make_failure


def base_camera_key(args):
    """
    The args RileyRunner reads when it builds its cached base camera.
    """
    return (
        args.width, args.height, args.focal_length, args.side, args.fov_scale,
        args.world_rot, args.roi_cent_world, args.sub_sample,
    )


def manifest_record(job, kind, suffix, name, record):
    camera = record["camera"]
    row = dict(
        kind=kind,
        truth_id=job.truth_id,
        suffix=suffix,
        image=planpaths.relative_image(job.directory, name),
        json=planpaths.relative_json(job.directory, name),
        truth_image=planpaths.relative_image(job.directory, job.name),
        sha256=record["sha256"],
        distorted=int(record["distorted"]),
    )
    row.update({column: float(camera[column]) for column in EFFECT_COLUMNS})
    return row


def failure_for(job, kind, suffix, name, error, distortion=None):
    return make_failure(
        kind, job.truth_id, suffix, planpaths.relative_image(job.directory, name),
        f"{type(error).__name__}: {error}", traceback.format_exc(), distortion,
    )


def render_truth(job, root, pending):
    """
    Renders the images of job whose suffix is in pending ("" is the clean image).
    Returns (manifest records, failures). One failing image does not stop the others;
    a failing mesh or rasterizer setup fails every pending image of the truth.
    """
    images = [image for image in job.images() if image[1] in pending]
    records, failures = [], []
    if not images:
        return records, failures

    runner = RileyRunner(job.args, Path(root) / job.directory, job.name)
    try:
        try:
            runner.set_rasterizer()
            runner.generate_pattern()
        except Exception as error:
            failures.extend(failure_for(job, kind, suffix, name, error) for kind, suffix, name, _ in images)
            return records, failures

        camera_key = None #RileyRunner builds its base camera on the first set_camera.
        for kind, suffix, name, args in images:
            try:
                #A variant may change camera fields: rebuild the cached base camera when it does.
                key = base_camera_key(args)
                if key != camera_key:
                    runner.base_camera = None
                    camera_key = key
                runner.args = args
                runner.name = name #Before set_camera: the name seeds the distortion RNG.
                runner.distortion_parameters = None #So a failure never reports the previous image's.
                runner.set_camera(distortion=(kind == "variant"))
                record = runner.run() #Renders <name>.bmp, writes <name>.json, returns its metadata.
                records.append(manifest_record(job, kind, suffix, name, record))
            except Exception as error:
                failures.append(failure_for(job, kind, suffix, name, error, runner.distortion_parameters))
                runner.base_camera = None
                camera_key = None
    finally:
        runner.release()
    return records, failures
