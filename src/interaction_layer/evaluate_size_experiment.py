"""Script entry point for the size-experiment evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

# This repo is not packaged, so the script adds src manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis_layer.evaluate_size_experiment_1 import main


if __name__ == "__main__":
    main()
