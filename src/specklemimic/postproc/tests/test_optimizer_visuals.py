"""Make the optimizer figures and animations for one run folder.

Example
-------
    python test_optimizer_visuals.py -r 50px

Looks for ``50px/optimizer/`` (written by the C++ optimizer) and saves the
figures to ``50px_post/optimizer/``. Runs with more than 2 parameters only get
the loss curve. A run folder with no optimizer data exits with a message.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post import OptimizerOptions, run_optimizer_post_processing


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot and animate the optimizer data of a run folder.")
    parser.add_argument("-r", "--runfolder", required=True,
                        help="Run folder that contains the optimizer subfolder.")
    parser.add_argument("--xlabel", default="x", help="Label of the first parameter.")
    parser.add_argument("--ylabel", default="y", help="Label of the second parameter.")
    parser.add_argument("--static-only", action="store_true",
                        help="Skip the animations (they are the slow part).")
    parser.add_argument("--animations-only", action="store_true",
                        help="Skip the static figures.")
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif",
                        help="Animation format (mp4 needs ffmpeg, default: gif).")
    parser.add_argument("--show", action="store_true",
                        help="Show figures interactively as well as saving.")
    return parser


def main(argv=None) -> int:
    """Parse arguments and run the workflow. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    options = OptimizerOptions(
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        plot_static=not args.animations_only,
        plot_animations=not args.static_only,
        animation_format=args.format,
        save=True,
        show=args.show,
    )
    run_optimizer_post_processing(args.runfolder, options)
    return 0 if Path(args.runfolder).is_dir() else 1


if __name__ == "__main__":
    sys.exit(main())
