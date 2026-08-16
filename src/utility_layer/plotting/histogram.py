"""2D histogram plots for grouped scalar distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from utility_layer.graph_config import GraphConfig
from utility_layer.plotting.common import color_map, format_axis, format_legend, get_pyplot, load_style, save_figure


def plot_grouped_histogram(
    groups: Mapping[str, Sequence[float]],
    out_path: str | Path,
    *,
    title: str,
    x_label: str,
    y_label: str = "Count",
    bins: int = 40,
    density: bool = False,
    x_limits: tuple[float, float] | None = None,
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
) -> Path | None:
    """Plot one outlined histogram per group."""
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

    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    colors = color_map(labels)
    fig, ax = plt.subplots(figsize=cfg["figure"]["default_size"])
    hist_bins: int | np.ndarray = bins
    if isinstance(bins, int):
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
        hist_bins = np.linspace(lower, upper, bins + 1)

    for label, group_values in zip(labels, values, strict=True):
        ax.hist(
            group_values,
            bins=hist_bins,
            histtype="step",
            linewidth=cfg["lines"]["line_width"],
            color=colors[label],
            label=label,
            density=density,
        )

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    format_axis(ax, cfg, xlabel=x_label, ylabel=y_label, title=title, grid_axis="y")
    format_legend(ax.legend(title="Hidden size", loc="best"), cfg)
    fig.tight_layout()
    return save_figure(fig, out_path, cfg)
