"""Shared matplotlib setup and styling helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utility_layer.graph_config import GraphConfig, load_graph_config


def sanitize_filename(value: str) -> str:
    """Convert labels such as metric names or env ids into safe filename parts."""
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def get_pyplot() -> Any:
    """Import pyplot with a non-interactive backend for scripts and cluster jobs."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def load_style(dpi: int | None = None, graph_config: GraphConfig | None = None) -> GraphConfig:
    """Load the central graph config and apply optional plot-local overrides."""
    cfg = load_graph_config()
    if graph_config is not None:
        for section, values in deepcopy(graph_config).items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                # Merge section by section so a plot can override one value without replacing the whole section.
                cfg[section].update(values)
    if dpi is not None:
        cfg["figure"]["dpi"] = int(dpi)
    return cfg


def save_figure(fig: Any, out_path: str | Path, cfg: GraphConfig) -> Path:
    """Create the target directory, save the figure, and close it."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=cfg["figure"]["dpi"])
    get_pyplot().close(fig)
    return target


def font_weight(cfg: GraphConfig) -> str:
    """Translate the boolean config flag into matplotlib's font-weight string."""
    return "bold" if cfg["text"]["bold"] else "normal"


def _format_3d_axis(ax: Any, cfg: GraphConfig, weight: str) -> None:
    """Apply 2D axis styling equivalents to matplotlib 3D axes."""
    if not hasattr(ax, "zaxis"):
        return

    lines = cfg["lines"]
    text = cfg["text"]
    line_width = lines["axis_spine_width"]

    ax.tick_params(axis="z", which="major", width=lines["tick_width"], labelsize=text["tick_label_size"])
    for tick_label in ax.get_zticklabels():
        tick_label.set_fontweight(weight)

    ax.zaxis.label.set_fontsize(text["axis_label_size"])
    ax.zaxis.label.set_fontweight(weight)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "line"):
            axis.line.set_linewidth(line_width)
        if hasattr(axis, "_axinfo"):
            axis._axinfo.setdefault("axisline", {})["linewidth"] = line_width


def format_axis(
    ax: Any,
    cfg: GraphConfig,
    *,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    title: str = "",
    grid_axis: str | None = "both",
) -> None:
    """Apply shared labels, tick, spine, title, and grid styling to one axis."""
    text = cfg["text"]
    lines = cfg["lines"]
    weight = font_weight(cfg)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=text["axis_label_size"], fontweight=weight)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=text["axis_label_size"], fontweight=weight)
    if zlabel and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(zlabel, fontsize=text["axis_label_size"], fontweight=weight)
    if title:
        ax.set_title(title, fontsize=text["title_size"], fontweight=weight)

    ax.tick_params(axis="both", which="major", width=lines["tick_width"], labelsize=text["tick_label_size"])
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight(weight)

    for spine in getattr(ax, "spines", {}).values():
        spine.set_linewidth(lines["axis_spine_width"])
    _format_3d_axis(ax, cfg, weight)

    if grid_axis:
        try:
            ax.grid(True, axis=grid_axis, alpha=cfg["alpha"]["grid"])
        except ValueError:
            ax.grid(True, alpha=cfg["alpha"]["grid"])


def format_legend(legend: Any, cfg: GraphConfig) -> None:
    """Apply the shared font sizing and weight to an existing legend."""
    if legend is None:
        return
    text = cfg["text"]
    weight = font_weight(cfg)
    for item in legend.get_texts():
        item.set_fontsize(text["legend_text_size"])
        item.set_fontweight(weight)
    legend.get_title().set_fontsize(text["legend_title_size"])
    legend.get_title().set_fontweight(weight)


def color_map(labels: Sequence[str]) -> dict[str, Any]:
    """Assign stable tab10 colors to labels in their provided order."""
    cmap = get_pyplot().get_cmap("tab10")
    return {label: cmap(index % cmap.N) for index, label in enumerate(labels)}


def scaled_x(values: Any, scale: str) -> np.ndarray:
    """Apply the supported x-axis transformation used by scaling plots."""
    values = np.asarray(values, dtype=float)
    return np.log2(values) if scale == "log2" else values


def bar_width(values: Any, fallback: float = 0.55) -> float:
    """Choose a bar width from the spacing of finite x-values."""
    finite = np.unique(np.asarray(values, dtype=float)[np.isfinite(values)])
    if finite.size < 2:
        return fallback
    spacing = float(np.min(np.diff(np.sort(finite))))
    return spacing * 0.42 if np.isfinite(spacing) and spacing > 0 else fallback


def value_limits(df: Any) -> tuple[float, float] | None:
    """Return y-limits that include means, optional standard deviations, zero, and margin."""
    if df.empty:
        return None
    lower = df["metric_value"] - df.get("metric_std", 0.0)
    upper = df["metric_value"] + df.get("metric_std", 0.0)
    y_min = min(0.0, float(lower.min()))
    y_max = max(0.0, float(upper.max()))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    margin = (y_max - y_min) * 0.08
    return y_min - margin, y_max + margin


def evaluate_function_grid(function: Any, x_values: Sequence[float], y_values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a two-input function on a mesh grid for heatmaps or 3D surfaces."""
    x_grid, y_grid = np.meshgrid(np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float))
    z_grid = np.vectorize(function, otypes=[float])(x_grid, y_grid)
    return x_grid, y_grid, z_grid
