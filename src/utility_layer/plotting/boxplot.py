"""2D box plots for grouped scalar summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from utility_layer.graph_config import GraphConfig
from utility_layer.plotting.common import color_map, format_axis, get_pyplot, load_style, save_figure


def plot_boxplot(
    groups: Mapping[str, Sequence[float]],
    out_path: str | Path,
    *,
    title: str,
    y_label: str,
    x_label: str = "",
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
    show_points: bool = True,
    y_limits: tuple[float, float] | None = None,
) -> Path | None:
    """Plot one box per label from scalar values such as final runtime or peak memory."""
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
    box = ax.boxplot(values, patch_artist=True, widths=0.62, showmeans=False, showfliers=False)

    box_line_width = cfg["lines"]["line_width"]

    for patch, label in zip(box["boxes"], labels, strict=True):
        patch.set_facecolor("none")
        patch.set_edgecolor("black")
        patch.set_linewidth(box_line_width)

    for key in ["whiskers", "caps", "medians"]:
        for item in box[key]:
            item.set_color("red" if key == "medians" else "black")
            item.set_linewidth(box_line_width)

    if show_points:
        point_size = cfg["markers"]["scatter_size"] * 1.1
        for index, (label, group_values) in enumerate(zip(labels, values, strict=True), start=1):
            offsets = np.zeros(len(group_values)) if len(group_values) == 1 else np.linspace(-0.11, 0.11, len(group_values))
            ax.scatter(
                index + offsets,
                group_values,
                s=point_size,
                marker="o",
                color=colors[label],
                edgecolors="none",
                linewidth=0,
                alpha=cfg["alpha"]["mean_line"],
                zorder=3,
            )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    format_axis(ax, cfg, xlabel=x_label, ylabel=y_label, title=title, grid_axis="y")
    fig.tight_layout()
    return save_figure(fig, out_path, cfg)
