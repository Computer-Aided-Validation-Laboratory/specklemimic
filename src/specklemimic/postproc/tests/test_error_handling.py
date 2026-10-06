"""Check that missing folders, files and unknown names are skipped instead of crashing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post import PostOptions, run_post_processing

# USER SETTINGS
PATTERN = "pattern1"

if __name__ == "__main__":
    # Missing pattern folder.
    run_post_processing("does_not_exist", PostOptions())

    # Unknown correlator, plus one (zssd) that may have no file.
    run_post_processing(PATTERN, PostOptions(autocorrs=["ZSSD", "bogus", "ZNSSD"],
                                             plot_3d=True, plot_corr=True, plot_verify=True))