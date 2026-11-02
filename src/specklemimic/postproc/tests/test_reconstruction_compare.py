"""Compare the reconstructed pattern with the ground truth for one run folder.

Example
-------
    python test_reconstruction_compare.py -r 50px

Reads ``50px/groundtruth.csv`` and ``50px/finalresult.csv`` and saves a 2x2
figure (ground truth, reconstruction, difference, squared error) plus a
metrics CSV to ``50px_post/``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post import run_reconstruction_post_processing


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare a reconstruction with its ground truth.")
    parser.add_argument("-r", "--runfolder", required=True,
                        help="Run folder with groundtruth.csv and finalresult.csv.")
    parser.add_argument("--show", action="store_true",
                        help="Show the figure interactively as well as saving.")
    return parser


def main(argv=None) -> int:
    """Parse arguments and run the comparison. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    run_reconstruction_post_processing(args.runfolder, save=True, show=args.show)
    return 0 if Path(args.runfolder).is_dir() else 1


if __name__ == "__main__":
    sys.exit(main())
