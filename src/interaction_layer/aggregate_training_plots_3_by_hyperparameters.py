"""Script entry point for the hyperparameter-experiment aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

# This repo is not packaged, so the script adds src manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis_layer.aggregate_training_plots_3_by_hyperparameters import main


if __name__ == "__main__":
    main()
