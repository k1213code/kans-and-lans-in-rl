"""Shared config-file IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load one YAML config file as a mapping."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return data
