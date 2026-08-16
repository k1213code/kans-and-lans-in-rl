"""Script entry point for trained-model inspection."""

from __future__ import annotations

import sys
from pathlib import Path

# This repo is not packaged, so the script adds src manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis_layer.inspect_trained_model import main


if __name__ == "__main__":
    main()
