"""2D grouped bar plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utility_layer.graph_config import GraphConfig
from utility_layer.plotting.common import (
    bar_width,
    font_weight,
    format_axis,
    format_legend,
    get_pyplot,
    load_style,
    save_figure,
    scaled_x,
    value_limits,
)


def plot_size_depth_bars(
    value_df: Any,
    out_path: str | Path,
    *,
    value_label: str,
    title: str,
    categories: Sequence[str],
    colors: Mapping[str, Any],
    labels: Mapping[str, str],
    x_scale: str = "linear",
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
) -> Path | None:
    """Plot grouped hidden-size bars in one panel per depth."""
    import pandas as pd
    from matplotlib.patches import Patch

    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    df = value_df.copy()
    for column in ["hidden_size", "depth", "metric_value", "metric_std"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["hidden_size", "depth", "metric_value"])
    if df.empty:
        return None

    depths = sorted(df["depth"].astype(int).unique())
    height = max(cfg["figure"]["width_scaling_min_height"], cfg["figure"]["width_scaling_row_height"] * len(depths))
    fig, axes = plt.subplots(len(depths), 1, figsize=(cfg["figure"]["width_scaling_width"], height), sharey=True, squeeze=False)
    y_limits = value_limits(df)

    for row, depth in enumerate(depths):
        ax = axes[row][0]
        part = df[df["depth"].astype(int) == depth].copy()
        widths = sorted(part["hidden_size"].astype(int).unique())
        ticks = scaled_x(widths, x_scale)
        positions = dict(zip(widths, ticks, strict=True))
        slot_width = bar_width(ticks, fallback=0.75) * 0.92
        width = slot_width / (len(categories) + 0.2)
        offsets = np.asarray([0.0]) if len(categories) == 1 else np.linspace(
            -slot_width / 2 + width / 2,
            slot_width / 2 - width / 2,
            len(categories),
        )

        for category, offset in zip(categories, offsets, strict=True):
            bars = part[part["backbone"].astype(str) == str(category)].sort_values("hidden_size")
            if bars.empty:
                continue
            ax.bar(
                [positions[int(value)] + offset for value in bars["hidden_size"]],
                bars["metric_value"].to_numpy(dtype=float),
                yerr=bars["metric_std"].fillna(0.0).to_numpy(dtype=float),
                width=width,
                color=colors[category],
                alpha=cfg["alpha"]["bar"],
                edgecolor="black",
                linewidth=cfg["lines"]["bar_edge_width"],
                error_kw={
                    "elinewidth": cfg["lines"]["error_line_width"],
                    "capsize": cfg["markers"]["bar_error_capsize"],
                    "capthick": cfg["markers"]["bar_error_capthick"],
                    "ecolor": "black",
                },
            )

        ax.set_title(f"d={depth}", loc="left", fontsize=max(10, cfg["text"]["title_size"] - 2))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(value) for value in widths])
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        format_axis(ax, cfg, xlabel="Hidden size" if row == len(depths) - 1 else "", ylabel=value_label, grid_axis="y")

    handles = [Patch(facecolor=colors[item], edgecolor="black", label=labels[item]) for item in categories]
    format_legend(fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985)), cfg)
    fig.suptitle(title, y=0.995, fontsize=cfg["text"]["title_size"], fontweight=font_weight(cfg))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return save_figure(fig, out_path, cfg)
