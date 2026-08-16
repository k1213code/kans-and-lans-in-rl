"""Script entry point for the size-experiment depth aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

# This repo is not packaged, so the script adds src manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis_layer.aggregate_training_plots_1_by_depth import main


if __name__ == "__main__":
    main()
