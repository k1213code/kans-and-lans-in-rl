"""Ridgeline plots for grouped scalar distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from utility_layer.graph_config import GraphConfig
from utility_layer.plotting.common import color_map, format_axis, get_pyplot, load_style, save_figure


def plot_ridgeline(
    groups: Mapping[str, Sequence[float]],
    out_path: str | Path,
    *,
    title: str,
    x_label: str,
    y_label: str = "Hidden size",
    bins: int = 80,
    x_limits: tuple[float, float] | None = None,
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
) -> Path | None:
    """Plot vertically offset density histograms for several groups."""
    labels = []
    values = []
    for label, raw_values in groups.items():
        clean_values = np.asarray(raw_values, dtype=float)
        clean_values = clean_values[np.isfinite(clean_values)]
        if clean_values.size:
            labels.append(str(label))
            values.append(clean_values)

    if not values:
        return None

    if x_limits is None:
        combined = np.concatenate(values)
        lower = float(np.min(combined))
        upper = float(np.max(combined))
    else:
        lower, upper = x_limits
    if lower == upper:
        margin = max(abs(lower) * 0.05, 1.0)
        lower -= margin
        upper += margin

    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    colors = color_map(labels)
    fig, ax = plt.subplots(figsize=cfg["figure"]["default_size"])

    edges = np.linspace(lower, upper, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    offset_step = 1.0
    ridge_height = 0.85

    for offset, (label, group_values) in enumerate(zip(labels, values, strict=True)):
        density, _ = np.histogram(group_values, bins=edges, density=True)
        if not np.any(np.isfinite(density)) or np.nanmax(density) <= 0:
            continue
        density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
        y_base = offset * offset_step
        y_values = y_base + density / density.max() * ridge_height
        ax.fill_between(centers, y_base, y_values, color=colors[label], alpha=0.34)
        ax.plot(centers, y_values, color=colors[label], linewidth=cfg["lines"]["line_width"])

    ax.set_yticks(np.arange(len(labels)) * offset_step)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.2, (len(labels) - 1) * offset_step + ridge_height + 0.2)
    ax.set_xlim(lower, upper)
    format_axis(ax, cfg, xlabel=x_label, ylabel=y_label, title=title, grid_axis="x")
    fig.tight_layout()
    return save_figure(fig, out_path, cfg)
