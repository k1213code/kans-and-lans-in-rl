"""3D graph plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from utility_layer.graph_config import GraphConfig
from utility_layer.plotting.common import (
    evaluate_function_grid,
    format_axis,
    format_legend,
    get_pyplot,
    load_style,
    save_figure,
    scaled_x,
)


def plot_size_depth_points_3d(
    data: Any,
    out_path: str | Path,
    *,
    value_col: str,
    value_label: str,
    title: str,
    categories: Sequence[str],
    colors: Mapping[str, Any],
    markers: Mapping[str, str],
    labels: Mapping[str, str],
    category_col: str = "actor_backbone",
    x_scale: str = "linear",
    view_elev: float = 28.0,
    view_azim: float = -135.0,
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
) -> Path | None:
    """Plot hidden-size/depth/value points in 3D."""
    import pandas as pd

    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    df = data.copy()
    for column in ["hidden_size", "depth", value_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["hidden_size", "depth", value_col])
    if df.empty:
        return None

    z_floor = min(0.0, float(df[value_col].min()))
    fig = plt.figure(figsize=cfg["figure"]["size_3d"])
    ax = fig.add_subplot(111, projection="3d")
    plotted = False

    for category in categories:
        part = df[df[category_col].astype(str) == str(category)].copy()
        if part.empty:
            continue
        for _, group in part.groupby("depth", sort=True):
            if len(group) > 1:
                group = group.sort_values("hidden_size")
                ax.plot(
                    scaled_x(group["hidden_size"], x_scale),
                    group["depth"],
                    group[value_col],
                    color=colors[category],
                    alpha=cfg["alpha"]["support_line"],
                    linewidth=cfg["lines"]["support_line_width"],
                )
        for _, group in part.groupby("hidden_size", sort=True):
            if len(group) > 1:
                group = group.sort_values("depth")
                ax.plot(
                    scaled_x(group["hidden_size"], x_scale),
                    group["depth"],
                    group[value_col],
                    color=colors[category],
                    alpha=cfg["alpha"]["support_line"],
                    linewidth=cfg["lines"]["support_line_width"],
                )

        x_values = scaled_x(part["hidden_size"], x_scale)
        y_values = part["depth"].to_numpy(dtype=float)
        z_values = part[value_col].to_numpy(dtype=float)
        for xv, yv, zv in zip(x_values, y_values, z_values, strict=True):
            ax.plot(
                [xv, xv],
                [yv, yv],
                [z_floor, zv],
                color=colors[category],
                alpha=cfg["alpha"]["stem_line"],
                linewidth=cfg["lines"]["stem_line_width"],
            )
        ax.scatter(
            x_values,
            y_values,
            np.full_like(z_values, z_floor),
            color=colors[category],
            marker=markers[category],
            s=cfg["markers"]["floor_scatter_size"],
            alpha=cfg["alpha"]["floor_scatter"],
            edgecolor="none",
        )
        ax.scatter(
            x_values,
            y_values,
            z_values,
            color=colors[category],
            marker=markers[category],
            s=cfg["markers"]["scatter_size"],
            edgecolor="black",
            linewidth=cfg["lines"]["marker_edge_width"],
            label=labels[category],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    hidden_sizes = sorted(df["hidden_size"].astype(int).unique())
    ax.set_xticks(scaled_x(hidden_sizes, x_scale))
    ax.set_xticklabels([str(value) for value in hidden_sizes])
    ax.set_yticks(sorted(df["depth"].astype(int).unique()))
    ax.view_init(elev=view_elev, azim=view_azim)
    format_axis(ax, cfg, xlabel="Hidden size", ylabel="Depth", zlabel=value_label, title=title)
    format_legend(ax.legend(loc="upper left"), cfg)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.92)
    return save_figure(fig, out_path, cfg)


def plot_function_surface_3d(
    function: Callable[[float, float], float],
    x_values: Sequence[float],
    y_values: Sequence[float],
    out_path: str | Path,
    *,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "value",
    title: str = "",
    colormap: str = "viridis",
    view_elev: float = 28.0,
    view_azim: float = -135.0,
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
) -> Path | None:
    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    x_grid, y_grid, z_grid = evaluate_function_grid(function, x_values, y_values)
    fig = plt.figure(figsize=cfg["figure"]["size_3d"])
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, z_grid, cmap=colormap, linewidth=0, antialiased=True)
    ax.view_init(elev=view_elev, azim=view_azim)
    format_axis(ax, cfg, xlabel=x_label, ylabel=y_label, zlabel=z_label, title=title)
    fig.tight_layout()
    return save_figure(fig, out_path, cfg)
