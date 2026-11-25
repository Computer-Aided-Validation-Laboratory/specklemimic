"""Generate paired clean/distorted speckle images using Riley's Python API.

Example (Windows PowerShell, Command Prompt, or Linux; run as one line):
  python scripts/sweep_procedural_plane.py --output out/pairs --seeds 1 2 3 --coverages 0.3 0.5 --variants-per-pattern 3

Lens coefficients and Gaussian blur sigma are sampled uniformly from the given
MIN MAX ranges. Set all coefficient ranges to 0 0 to disable distortion, or
--blur-sigma 0 0 to disable optical blur. Blur sigma is in output pixels.
Each numbered folder contains clean.bmp and observed_000.bmp, etc. Metadata
records the exact cameras and clean target for every observation. Clean targets
are ideal-camera renders, not mesh-free UV images; distorted pixels are not
spatially aligned with the target. Distortion can move content outside the frame.
Keep every image sharing a pattern seed in the same train/validation/test split.

Requires the checkout's Python package on Windows or Linux. Images are
rendered sequentially, with one render thread and fixed [0, 1] intensity scaling.
This uses a generated quad mesh, not a mesh-free pattern evaluator.
"""

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]
import riley  # type: ignore[reportMissingImports]


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def seed(value):
    number = int(value, 16 if value.lower().startswith("0x") else 10)
    if not 0 <= number <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("seed must fit u32")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--pattern", choices=("disk", "gaussian", "perlin"), default="disk"
    )
    parser.add_argument("--seeds", type=seed, nargs="+", default=list(range(10)))
    parser.add_argument("--coverages", type=positive, nargs="+")
    parser.add_argument("--radii", type=positive, nargs="+")
    parser.add_argument("--cell-sizes", type=positive, nargs="+")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--variants-per-pattern", type=int, default=3)
    parser.add_argument(
        "--effects-seed",
        type=seed,
        default=42,
        help="camera-effect RNG seed, independent of pattern seeds",
    )
    parser.add_argument(
        "--k1", type=float, nargs=2, default=(-0.2, 0.2), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--k2", type=float, nargs=2, default=(-0.03, 0.03), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--p1", type=float, nargs=2, default=(-0.001, 0.001), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--p2", type=float, nargs=2, default=(-0.001, 0.001), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        nargs=2,
        default=(0.4, 1.2),
        metavar=("MIN", "MAX"),
        help="Gaussian PSF sigma in output pixels; 0 disables",
    )
    parser.add_argument(
        "--sub-sample",
        type=int,
        default=4,
        help="subpixel samples per axis for both clean and affected images",
    )
    parser.add_argument(
        "--fov-scale",
        type=positive,
        default=1.2,
        help="view margin around the plane; 1 fills the sensor",
    )
    args = parser.parse_args(argv)
    if args.variants_per_pattern < 0:
        parser.error(
            "--variants-per-pattern must be nonnegative (0 renders clean images only)"
        )
    if not 1 <= args.sub_sample <= 8:
        parser.error("--sub-sample must be in 1..8")
    if args.fov_scale < 1:
        parser.error("--fov-scale must be at least 1")
    for name in ("k1", "k2", "p1", "p2", "blur_sigma"):
        low, high = getattr(args, name)
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            parser.error(f"--{name.replace('_', '-')} requires finite MIN <= MAX")
    if args.blur_sigma[0] < 0:
        parser.error("--blur-sigma must be nonnegative")
    if not (1 <= args.width <= 2048 and 1 <= args.height <= 2048):
        parser.error("sweep dimensions must be in 1..2048")
    if args.pattern == "perlin":
        if args.coverages is not None or args.radii is not None:
            parser.error("Perlin accepts --cell-sizes, not --coverages/--radii")
        args.cell_sizes = args.cell_sizes or [0.25]
    else:
        if args.cell_sizes is not None:
            parser.error("--cell-sizes requires Perlin")
        args.coverages = args.coverages or [0.3, 0.5, 0.7]
        args.radii = args.radii or [0.25]
        if any(value >= 1 for value in args.coverages):
            parser.error("coverage must be below 1")
    return args


def cases(args):
    if args.pattern == "perlin":
        return itertools.product(args.seeds, [None], args.cell_sizes)
    return itertools.product(args.seeds, args.coverages, args.radii)


def effect_parameters(args, rng):
    coefficients = {
        f"distort_{name}": rng.uniform(*getattr(args, name))
        for name in ("k1", "k2", "p1", "p2")
    }
    sigma = rng.uniform(*args.blur_sigma)
    return dict(
        coefficients,
        distort_model=int(any(coefficients.values())),
        psf_type=int(sigma > 0),
        psf_sigma_x=sigma,
        psf_support_rad=3 * sigma,
        psf_separable=1,
    )


def main(argv=None):
    args = parse_args(argv)
    # Prevent numerical libraries reserving large thread pools/address ranges.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    #Create the output folder directory.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    #Write the settings to a JSON file in the output directory.
    settings = vars(args).copy()
    settings.update(
        output=str(output),
        plane_side=40.0,
        render_threads=1,
        riley_module=str(riley.__file__),
        numpy_version=np.__version__,
    )

    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    #Setup a big fat square of a mesh.
    side = 40.0
    h = side / 2
    coords = np.array(((-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)))
    connect = np.array(((0, 1, 2, 3),), dtype=np.int64) #Connectivity map for the mesh.
    uvs = np.array(((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
    convention = riley.ConnectConvention(
        riley.EElemType.QUAD4,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )

    #Setup the riley camera for effects.
    focal_length = 50e-3
    camera = riley.Camera(
        pixels_num=(args.width, args.height),
        pixels_size=(focal_length / args.width, focal_length / args.height),
        pos_world=(0.0, 0.0, side * args.fov_scale),
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=(0.0, 0.0, 0.0),
        focal_length=focal_length,
        sub_sample=args.sub_sample,
    )

    #Setup the rasterizer.
    config = riley.RasterConfig(
        parallel=1,
        save_strategy=riley.SaveStrategy.disk,
        image_save_mode=riley.ImageSaveMode.multifield,
        save_format=riley.ImageFormat.bmp,
        save_bits=8,
        save_scaling=riley.ScaleStrategy.fixed,
        save_scaling_min=0.0,
        save_scaling_max=1.0,
        report=riley.ReportMode.off,
        output_name_format="clean",
        background_value=1.0,
    )
    with (output / "manifest.jsonl").open("x") as manifest:
        for index, (realization, coverage, scale) in enumerate(cases(args)):
            if args.pattern == "perlin":
                params = riley.Perlin2DParams(seed=realization, cell_size=scale)
            else:
                params = riley.Dot2DParams(
                    seed=realization,
                    coverage=coverage,
                    radius=scale,
                    pattern=riley.SpecklePattern[args.pattern],
                )
            shader = riley.SpeckleShader(
                uvs=uvs,
                params=params,
                bits=-1,
                scaling_type=riley.ScaleStrategy.none,
            )
            mesh = riley.create_mesh(
                convention, riley.MeshType.quad4, coords, connect, shader
            )
            directory = output / f"{index:06d}"

            #Project the generated speckle pattern onto the mesh assuming a perfect camera.
            riley.raster(mesh, camera, config, out_dir=str(directory))
            clean = directory / "clean.bmp"
            pattern_info = {
                "pattern": args.pattern,
                "seed": realization,
                "coverage": coverage,
                "radius": scale if coverage is not None else None,
                "cell_size": scale if coverage is None else None,
            }
            metadata = {
                "pattern": pattern_info,
                "clean": str(clean.relative_to(output)),
                "clean_camera": asdict(camera),
                "clean_sha256": hashlib.sha256(clean.read_bytes()).hexdigest(),
                "observations": [],
            }
            # Stable per-pattern RNG: changing sweep order does not change its effects.
            rng = random.Random(
                f"{args.effects_seed}:{args.pattern}:{realization}:{coverage}:{scale}"
            )
            metadata_path = directory / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )

            #Create camera distorted variants.
            for variant in range(args.variants_per_pattern):
                affected_camera = replace(camera, **effect_parameters(args, rng))
                name = f"observed_{variant:03d}"
                affected_config = replace(config, output_name_format=name)
                riley.raster(
                    mesh, affected_camera, affected_config, out_dir=str(directory)
                )
                image = directory / f"{name}.bmp"
                record = dict(
                    pattern_info,
                    image=str(image.relative_to(output)),
                    clean=metadata["clean"],
                    camera=asdict(affected_camera),
                    width=args.width,
                    height=args.height,
                    sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                )
                metadata["observations"].append(record)
                metadata_path.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                print(image, flush=True)
            print(clean, flush=True)
            del mesh, shader, params
            gc.collect()


if __name__ == "__main__":
    main()
