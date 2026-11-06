"""Run the default workflow on one pattern from Python (same as the command line)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post import PostOptions, run_post_processing

# USER SETTINGS
PATTERN = "pattern1"
AUTOCORRS = ["znssd"]
SHOW = False

if __name__ == "__main__":
    options = PostOptions(
        autocorrs=AUTOCORRS,
        plot_3d=True,
        plot_grad=True,
        plot_speckle=True,
        plot_sssig=True,
        save=True,
        show=SHOW,
    )
    run_post_processing(PATTERN, options)