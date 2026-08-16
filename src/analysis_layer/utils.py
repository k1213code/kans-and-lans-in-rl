"""Shared helpers for analysis scripts."""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from utility_layer.paths import resolve_project_path
from utility_layer.plotting.common import sanitize_filename

DEFAULT_PLOT_FORMATS = ("png", "svg")
DEFAULT_PLOT_METRICS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/approx_kl",
    "train/clip_fraction",
    "train/entropy_loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/explained_variance",
    "time/fps",
    "memory/main_rss_mb",
    "memory/children_rss_mb",
    "memory/total_rss_mb",
    "memory/peak_total_rss_mb",
    "custom_time/elapsed_s",
    "train/learning_rate",
    "train/actor_backbone/out_of_grid_ratio",
    "train/actor_backbone/base_vs_spline_ratio",
    "train/critic_backbone/out_of_grid_ratio",
    "train/critic_backbone/base_vs_spline_ratio",
]
DIAGNOSTIC_LAYER_PREFIXES = (
    "train/actor_backbone/layer_",
    "train/critic_backbone/layer_",
)
TIME_SECONDS_COLUMNS = ("custom_time/elapsed_s", "time/time_elapsed")
TIME_MINUTE_COLUMNS = ("custom_time/elapsed_min",)
MEMORY_MB_COLUMNS = (
    "memory/peak_total_rss_mb",
    "memory/total_rss_mb",
    "memory/main_rss_mb",
    "memory/children_rss_mb",
)
AxisLimits = tuple[float, float]


def resolve_analysis_path(path: str | Path) -> Path:
    """Resolve relative CLI paths from the project root."""
    return resolve_project_path(path)


def timestamp() -> str:
    """Return a compact timestamp for generated result rows."""
    return datetime.now().isoformat(timespec="seconds")


def parse_int(value: Any, fallback: int | None = None) -> int | None:
    """Convert a value to int and return a fallback when conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def read_simple_yaml(path: Path) -> dict[str, str]:
    """Read simple top-level key-value pairs from a YAML file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def backbone_from_parts(actor: str, critic: str) -> str:
    """Build one readable backbone label from actor and critic backbone names."""
    actor = actor or "unknown"
    critic = critic or "unknown"
    return actor if actor == critic else f"actor-{actor}_critic-{critic}"


def display_label(value: str, labels: Mapping[str, str]) -> str:
    """Return a configured display label, falling back to the raw value."""
    return labels.get(value, value)


def numeric_values(df: Any, column: str) -> Any:
    """Return one numeric DataFrame column without missing or invalid values."""
    import pandas as pd

    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def max_numeric_column(df: Any, candidates: Sequence[str]) -> tuple[str, float]:
    """Return the maximum value from the first usable candidate column."""
    for column in candidates:
        values = numeric_values(df, column)
        if not values.empty:
            return column, float(values.max())
    return "", float("nan")


def final_training_hours(
    df: Any,
    seconds_columns: Sequence[str] = TIME_SECONDS_COLUMNS,
    minute_columns: Sequence[str] = TIME_MINUTE_COLUMNS,
) -> float | None:
    """Read the final logged training time and convert it to hours."""
    for column in seconds_columns:
        values = numeric_values(df, column)
        if not values.empty:
            return float(values.iloc[-1]) / 3600.0

    for column in minute_columns:
        values = numeric_values(df, column)
        if not values.empty:
            return float(values.iloc[-1]) / 60.0
    return None


def peak_memory_mb(df: Any, columns: Sequence[str] = MEMORY_MB_COLUMNS) -> float | None:
    """Return the highest available memory value from one run."""
    for column in columns:
        values = numeric_values(df, column)
        if not values.empty:
            return float(values.max())
    return None


def find_layer_metrics(columns: Sequence[str]) -> list[str]:
    """Find dynamically logged per-layer diagnostic columns."""
    return sorted(str(column) for column in columns if str(column).startswith(DIAGNOSTIC_LAYER_PREFIXES))


def write_csv(df: Any, path: Path, label: str) -> Path:
    """Write one CSV file and print where it was saved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[ok] wrote {label}: {path}")
    return path


def save_plot_variants(
    plot_func: Callable[..., Path | None],
    outdir: Path,
    stem: str,
    label: str,
    *,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
    **kwargs: Any,
) -> list[Path]:
    """Save one plot through all configured file formats."""
    paths = []
    for extension in plot_formats:
        path = plot_func(out_path=outdir / f"{stem}.{extension.lstrip('.')}", **kwargs)
        if path is not None:
            paths.append(path)
            print(f"[ok] saved {label}: {path}")
    return paths


def register_legacy_module_aliases() -> None:
    """Keep older SB3 model zips loadable after package layout changes."""
    import model_layer.policies as custom_policies
    from model_layer.backbones import (
        debug_constant,
        kan,
        kan_no_base,
        kan_no_spline,
        lan,
        mlp,
        registry,
    )

    legacy_sb3 = sys.modules.setdefault("sb3", types.ModuleType("sb3"))
    legacy_sb3.__path__ = []
    legacy_sb3.policies = custom_policies
    sys.modules.setdefault("sb3.policies", custom_policies)

    legacy_models = sys.modules.setdefault("models", types.ModuleType("models"))
    legacy_models.__path__ = []
    legacy_backbones = sys.modules.setdefault("models.backbones", types.ModuleType("models.backbones"))
    legacy_backbones.__path__ = []
    legacy_models.backbones = legacy_backbones

    aliases = {
        "debug_constant": debug_constant,
        "kan": kan,
        "kan_no_base": kan_no_base,
        "kan_no_spline": kan_no_spline,
        "lan": lan,
        "mlp": mlp,
        "registry": registry,
    }
    for name, module in aliases.items():
        setattr(legacy_backbones, name, module)
        sys.modules.setdefault(f"models.backbones.{name}", module)
