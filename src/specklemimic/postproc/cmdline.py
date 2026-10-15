"""Command line entry point for quick single-pattern post-processing.

Example
-------
    python cmdline.py -p pattern1 -a znssd --save

With no plot flags the default workflow runs: 3D surface + watershed, gradients,
speckle pattern and SSSIG heatmaps. Give any plot flag to run only those plots.
"""

import argparse
import sys
from pathlib import Path

from post.workflow import PostOptions, run_post_processing


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="Post-process a speckle pattern folder.")
    parser.add_argument("-p", "--pattern", required=True, help="Path to the pattern directory.")
    parser.add_argument("-a", "--autocorr", nargs="+", default=["znssd"],
                        help="One or more autocorrelation functions (default: znssd).")
    parser.add_argument("--plot-3d", action="store_true",
                        help="3D autocorrelation landscape (with watershed).")
    parser.add_argument("--plot-grad", action="store_true", help="Directional gradients and magnitude.")
    parser.add_argument("--plot-speckle", action="store_true", help="Raw speckle pattern.")
    parser.add_argument("--plot-sssig", action="store_true", help="SSSIG 2D heatmaps and 3D surfaces.")
    parser.add_argument("--plot-corr", action="store_true",
                        help="Tiled 2D comparison of all autocorrelation functions.")
    parser.add_argument("--plot-verify", action="store_true",
                        help="Autocorrelation vs radial gradient with watershed.")
    parser.add_argument("--no-watershed", action="store_true", help="Leave the watershed off the 3D surfaces.")
    parser.add_argument("--grid-watershed", action="store_true",
                        help="Overlay the watershed on the comparison tiles.")
    parser.add_argument("--save", action="store_true", help="Save figures to <pattern>_post/ (default).")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    return parser


def main(argv=None) -> int:
    """Parse arguments and run the workflow. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    options = PostOptions(
        autocorrs=args.autocorr,
        plot_3d=args.plot_3d,
        plot_grad=args.plot_grad,
        plot_speckle=args.plot_speckle,
        plot_sssig=args.plot_sssig,
        plot_corr=args.plot_corr,
        plot_verify=args.plot_verify,
        show_watershed=not args.no_watershed,
        grid_watershed=args.grid_watershed,
        save=args.save,
        show=args.show,
    )
    run_post_processing(args.pattern, options)
    return 0 if Path(args.pattern).is_dir() else 1


if __name__ == "__main__":
    sys.exit(main())