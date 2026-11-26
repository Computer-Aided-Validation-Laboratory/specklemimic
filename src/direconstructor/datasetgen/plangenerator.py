"""
Run plan generator, rung 1: radial lens distortion on disk speckles.

Writes <out>/train/ and <out>/test/ (truths.csv + variants.csv each), plus <out>/plan_info.json.
Render each with planexecutor.py.

Truths (ground truths), sampled per truth:
    coverage        in --coverage LO HI
    radius (scale)  in --radius LO HI

Mutations, sampled per mutation, the same set applied to every truth:
    k1, k2, k3      symmetric ranges (-K, K), K in --k1/--k2/--k3 LO HI
                    RileyRunner draws the actual coefficient inside (-K, K) per image

Fixed (simplifications for this rung):
    pattern = disk only (no gaussian / perlin)
    no tangential distortion (p1 = p2 = 0)
    no blur (blur_sigma = 0)
    no camera pose change (world_rot = 0, camera straight above the plane)
    512 x 512 px, focal_length 50 mm, fov_scale 0.8, sub_sample 4, side 40

Sampling:
    scrambled Halton, seeded by --seed
    prefix-stable: n = 100 starts with the same 10 points as n = 10 (truths and mutations)

Splits:
    train: n_train truths x mutation set A
    test:  new truths x m_test mutations, split in halves (odd: the extra one is seen)
           seen = first half of A (train's own), unseen = first half of B (held out)
    test speckle seeds never overlap train's
"""

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]
import scipy  # type: ignore[reportMissingImports]
from scipy.stats import qmc  # type: ignore[reportMissingImports]

import planio
import planpaths
from planledger import file_sha256, now

#Default sampling domains (LO, HI). Tune these, or override them on the command line.
COVERAGE = (0.2, 0.6)
RADIUS = (0.2, 0.5)     #World units; about 4-11 px speckle diameter at 512 px, fov_scale 1.2.
K1 = (0.0, 0.2)         #Domain of K, the half width of the symmetric range (-K, K).
K2 = (0.0, 0.03)
K3 = (0.0, 0.005)

#Fixed for this rung.
PATTERN = "disk"
CAMERA = dict(
    side=40.0, width=512, height=512, focal_length=50e-3, fov_scale=0.8,
    world_rot=(0.0, 0.0, 0.0), roi_cent_world=(0.0, 0.0, 0.0), sub_sample=4,
)
EFFECTS = dict(
    effects_seed=42, k1=(0.0, 0.0), k2=(0.0, 0.0), k3=(0.0, 0.0),
    p1=(0.0, 0.0), p2=(0.0, 0.0), blur_sigma=(0.0, 0.0),
)

#Independent random streams derived from --seed. Never renumber: it changes every dataset.
STREAM_TRAIN_TRUTHS, STREAM_TEST_TRUTHS, STREAM_MUTATIONS_A, STREAM_MUTATIONS_B, STREAM_REALIZATION = range(5)
TEST_REALIZATION_SHIFT = 2**31 #Test speckle seeds sit half the u32 range away from train's.


def halton(seed, stream, count, dimensions):
    """
    The first `count` points of a scrambled Halton sequence in [0, 1)^dimensions.
    The points do not depend on count, so a longer run starts with a shorter run's points.
    """
    if count == 0:
        return np.empty((0, dimensions))
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    return qmc.Halton(d=dimensions, scramble=True, rng=rng).random(count)


def scale(unit, domain):
    low, high = domain
    return low + unit * (high - low)


def realization_base(seed):
    return int(np.random.SeedSequence([seed, STREAM_REALIZATION]).generate_state(1)[0])


def make_truths(seed, stream, count, realization_start, args):
    rows = []
    for i, (u_coverage, u_radius) in enumerate(halton(seed, stream, count, 2)):
        truth_id = f"{i:07d}"
        rows.append(dict(
            truth_id=truth_id,
            directory=planpaths.default_directory(PATTERN, truth_id),
            name=truth_id,
            pattern=PATTERN,
            realization=(realization_start + i) % 2**32,
            coverage=float(scale(u_coverage, args.coverage)),
            scale=float(scale(u_radius, args.radius)),
            **CAMERA,
            **EFFECTS,
        ))
    return rows


def make_mutations(seed, stream, label, count, args):
    """
    Mutation set `label`: suffix -> symmetric k ranges. Only the mutated columns are set,
    everything else is inherited from the truth row.
    """
    mutations = []
    for j, unit in enumerate(halton(seed, stream, count, 3)):
        k1, k2, k3 = (float(scale(u, domain)) for u, domain in zip(unit, (args.k1, args.k2, args.k3)))
        mutations.append(dict(suffix=f"{label}{j:05d}", k1=(-k1, k1), k2=(-k2, k2), k3=(-k3, k3)))
    return mutations


def apply(truths, mutations):
    """
    Every truth gets every mutation: the mutation rows repeated with each truth_id.
    """
    return [dict(mutation, truth_id=truth["truth_id"]) for truth in truths for mutation in mutations]


def write_split(folder, truths, variants):
    folder.mkdir(parents=True, exist_ok=True)
    planio.write_truths(folder / "truths.csv", truths)
    planio.write_variants(folder / "variants.csv", variants)
    #Read back with the executor's own reader, so an invalid plan is never left behind.
    jobs = planio.read_plan(folder / "truths.csv", folder / "variants.csv")
    return dict(
        truths=len(jobs),
        images=sum(1 for job in jobs for _ in job.images()),
        sha256={name: file_sha256(folder / name) for name in ("truths.csv", "variants.csv")},
    )


def domain(text):
    return float(text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="dataset folder; train/ and test/ go inside")
    parser.add_argument("--seed", type=int, default=0, help="one seed for everything (default 0)")
    parser.add_argument("--train-truths", type=int, required=True)
    parser.add_argument("--test-truths", type=int, default=0)
    parser.add_argument("--mutations", type=int, required=True, help="size of mutation set A (train)")
    parser.add_argument("--test-mutations", type=int,
                        help="mutations per test truth in total, half seen and half unseen (default: --mutations)")
    for name, default, help_text in (
        ("coverage", COVERAGE, "truth speckle coverage"),
        ("radius", RADIUS, "truth speckle radius (world units)"),
        ("k1", K1, "domain of K for k1 in (-K, K)"),
        ("k2", K2, "domain of K for k2 in (-K, K)"),
        ("k3", K3, "domain of K for k3 in (-K, K)"),
    ):
        parser.add_argument(f"--{name}", type=domain, nargs=2, default=default, metavar=("LO", "HI"),
                            help=f"{help_text} (default {default[0]} {default[1]})")
    parser.add_argument("--force", action="store_true", help="replace existing plans (never ones already rendered)")
    args = parser.parse_args(argv)

    if args.test_mutations is None:
        args.test_mutations = args.mutations
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be in 0..2^32-1")
    if args.train_truths < 1 or args.test_truths < 0 or args.mutations < 0 or args.test_mutations < 0:
        parser.error("--train-truths must be >= 1; the other counts >= 0")
    args.test_seen = (args.test_mutations + 1) // 2
    args.test_unseen = args.test_mutations // 2
    if args.test_seen > args.mutations:
        parser.error("half of --test-mutations cannot exceed --mutations (seen mutations are the first ones of set A)")
    if max(args.train_truths, args.test_truths) > TEST_REALIZATION_SHIFT:
        parser.error(f"at most {TEST_REALIZATION_SHIFT} truths per split")
    for name in ("coverage", "radius", "k1", "k2", "k3"):
        low, high = getattr(args, name)
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            parser.error(f"--{name} needs finite LO <= HI")
    if not (0 < args.coverage[0] and args.coverage[1] < 1):
        parser.error("--coverage must be inside (0, 1)")
    if args.radius[0] <= 0:
        parser.error("--radius must be positive")
    if min(args.k1[0], args.k2[0], args.k3[0]) < 0:
        parser.error("--k1/--k2/--k3 are half widths K and must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out.resolve()
    splits = ["train"] + (["test"] if args.test_truths else [])

    for split in splits:
        folder = out / split
        if (folder / "manifest.csv").exists():
            print(f"{folder} has already been rendered; refusing to replace its plan. Use a new --out.", file=sys.stderr)
            return 2
        if (folder / "truths.csv").exists() and not args.force:
            print(f"{folder} already has a plan; pass --force to replace it.", file=sys.stderr)
            return 2

    base = realization_base(args.seed)
    set_a = make_mutations(args.seed, STREAM_MUTATIONS_A, "A", args.mutations, args)
    set_b = make_mutations(args.seed, STREAM_MUTATIONS_B, "B", args.test_unseen, args)

    summary = {}
    train = make_truths(args.seed, STREAM_TRAIN_TRUTHS, args.train_truths, base, args)
    summary["train"] = write_split(out / "train", train, apply(train, set_a))
    if args.test_truths:
        test = make_truths(args.seed, STREAM_TEST_TRUTHS, args.test_truths, base + TEST_REALIZATION_SHIFT, args)
        seen = set_a[:args.test_seen] #Prefix-stable sampler: exactly train's first mutations.
        summary["test"] = write_split(out / "test", test, apply(test, seen + set_b))

    info = dict(
        created=now(),
        generator="plangenerator.py (rung 1: radial k1, k2, k3 on disk speckles)",
        arguments={key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        fixed=dict(pattern=PATTERN, **CAMERA, **EFFECTS),
        realization_base=base,
        test_realization_shift=TEST_REALIZATION_SHIFT,
        splits=summary,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        python_version=platform.python_version(),
    )
    (out / "plan_info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    print(f"Wrote plans to {out}")
    print(f"  train: {args.train_truths} truths x {args.mutations} mutations (set A) = {summary['train']['images']} images")
    if args.test_truths:
        print(f"  test:  {args.test_truths} truths x ({args.test_seen} seen A + {args.test_unseen} unseen B) "
              f"= {summary['test']['images']} images")
    for split in splits:
        print(f"Render with: python datasetgen/planexecutor.py {out / split / 'truths.csv'} {out / split / 'variants.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
