"""Script entry point for the spline/base ratio histogram analysis."""

from __future__ import annotations

import sys
from pathlib import Path

# This repo is not packaged, so the script adds src manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis_layer.evaluate_spline_base_ratio_histograms_4 import main


if __name__ == "__main__":
    main()
