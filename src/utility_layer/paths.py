"""Central project paths used across layers."""

from __future__ import annotations

from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
INTERACTION_ROOT = SRC_ROOT / "interaction_layer"

CONFIG_ROOT = SRC_ROOT / "configuration_layer" / "configs"
OUTPUT_ROOT = SRC_ROOT / "output_layer" / "outputs"
DEFAULT_OUTPUT_ROOT_TEXT = OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix()

TRAIN_SCRIPT = INTERACTION_ROOT / "train_sb3_ppo.py"


def resolve_project_path(path_text: str | Path) -> Path:
    """Resolve a path against the project root unless it is already absolute."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_config_path(path_text: str | Path) -> Path:
    """
    Resolve config paths relative to the project root first, then config root.
    """
    path = Path(path_text)
    if path.is_absolute():
        return path

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    if path.parts and path.parts[0] == "configs":
        return CONFIG_ROOT.joinpath(*path.parts[1:])

    return CONFIG_ROOT / path


def resolve_output_path(path_text: str | Path = "") -> Path:
    """Resolve output paths relative to the project root, defaulting to OUTPUT_ROOT."""
    if not path_text:
        return OUTPUT_ROOT

    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return OUTPUT_ROOT.joinpath(*path.parts[1:])
    return PROJECT_ROOT / path
