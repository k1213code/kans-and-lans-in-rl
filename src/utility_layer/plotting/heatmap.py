"""Heatmap plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from utility_layer.graph_config import GraphConfig
from utility_layer.plotting.common import (
    evaluate_function_grid,
    font_weight,
    format_axis,
    format_legend,
    get_pyplot,
    load_style,
    save_figure,
)


def _cell_value(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_best_category_heatmap(
    value_df: Any,
    out_path: str | Path,
    *,
    title: str,
    categories: Sequence[str],
    colors: Mapping[str, Any],
    labels: Mapping[str, str],
    higher_is_better: bool = True,
    value_formatter: Callable[[float], str] = _cell_value,
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
) -> Path | None:
    """Color each hidden-size/depth cell by the best category."""
    import pandas as pd
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch, Patch

    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    df = value_df.copy()
    for column in ["hidden_size", "depth", "metric_value"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["hidden_size", "depth", "metric_value"])
    df = df[df["backbone"].isin(categories)].copy()
    if df.empty:
        return None

    widths = sorted(df["hidden_size"].astype(int).unique())
    depths = sorted(df["depth"].astype(int).unique())
    cat_order = {category: index for index, category in enumerate(categories)}
    df["category_order"] = df["backbone"].map(cat_order)

    ascending = not higher_is_better
    winners = np.full((len(depths), len(widths)), np.nan)
    cell_values: list[list[dict[str, float]]] = [[{} for _ in widths] for _ in depths]
    best_per_category = set()

    for category in categories:
        part = df[df["backbone"] == category]
        if not part.empty:
            row = part.sort_values(["metric_value", "hidden_size", "depth"], ascending=[ascending, True, True]).iloc[0]
            best_per_category.add((int(row["hidden_size"]), int(row["depth"]), category))

    for (width, depth), group in df.groupby(["hidden_size", "depth"], sort=True):
        group = group.sort_values(["metric_value", "category_order"], ascending=[ascending, True])
        winner = str(group.iloc[0]["backbone"])
        row = depths.index(int(depth))
        col = widths.index(int(width))
        winners[row, col] = cat_order[winner]
        cell_values[row][col] = {str(item["backbone"]): float(item["metric_value"]) for item in group.to_dict(orient="records")}

    min_size = cfg["figure"]["heatmap_min_size"]
    cell_size = cfg["figure"]["heatmap_cell_size"]
    padding = cfg["figure"]["heatmap_padding"]
    fig_size = (
        max(min_size[0], cell_size[0] * len(widths) + padding[0]),
        max(min_size[1], cell_size[1] * len(depths) + padding[1]),
    )
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor("#f2f2f2")

    for row in range(len(depths)):
        for col in range(len(widths)):
            if not np.isfinite(winners[row, col]):
                continue
            category = categories[int(winners[row, col])]
            ax.add_patch(
                FancyBboxPatch(
                    (col - 0.49, row - 0.49),
                    0.98,
                    0.98,
                    boxstyle="square,pad=0.015",
                    facecolor=to_rgba(colors[category], cfg["alpha"]["heatmap_cell"]),
                    edgecolor="white",
                    linewidth=cfg["lines"]["heatmap_cell_edge_width"],
                    zorder=1,
                )
            )

    ax.set_xlim(-0.64, len(widths) - 0.36)
    ax.set_ylim(len(depths) - 0.36, -0.64)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(value) for value in widths])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(value) for value in depths])
    format_axis(ax, cfg, xlabel="Hidden size", ylabel="Depth", title=title, grid_axis=None)

    offsets = np.asarray([0.0]) if len(categories) == 1 else np.linspace(-0.2, 0.2, len(categories))
    for row in range(len(depths)):
        for col in range(len(widths)):
            values = cell_values[row][col]
            if not values:
                continue
            ax.add_patch(
                FancyBboxPatch(
                    (col - 0.42, row - 0.36),
                    0.84,
                    0.72,
                    boxstyle="round,pad=0.04,rounding_size=0.12",
                    facecolor="white",
                    edgecolor="none",
                    alpha=cfg["alpha"]["heatmap_value_box"],
                    zorder=2,
                )
            )
            for offset, category in zip(offsets, categories, strict=True):
                color = colors[category] if (widths[col], depths[row], category) in best_per_category else "black"
                ax.text(
                    col - 0.35,
                    row + offset,
                    f"{labels[category]}:",
                    ha="left",
                    va="center",
                    fontsize=cfg["text"]["cell_text_size"],
                    fontweight=font_weight(cfg),
                    fontfamily="monospace",
                    color="black",
                    zorder=3,
                )
                ax.text(
                    col + 0.04,
                    row + offset,
                    value_formatter(values.get(category, float("nan"))),
                    ha="left",
                    va="center",
                    fontsize=cfg["text"]["cell_text_size"],
                    fontweight=font_weight(cfg),
                    fontfamily="monospace",
                    color=color,
                    zorder=3,
                )

    handles = [Patch(facecolor=colors[item], edgecolor="black", label=labels[item]) for item in categories]
    format_legend(fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.98)), cfg)
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    return save_figure(fig, out_path, cfg)


def plot_function_heatmap(
    function: Callable[[float, float], float],
    x_values: Sequence[float],
    y_values: Sequence[float],
    out_path: str | Path,
    *,
    x_label: str = "x",
    y_label: str = "y",
    value_label: str = "value",
    title: str = "",
    colormap: str = "viridis",
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
) -> Path | None:
    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    _x_grid, _y_grid, z_grid = evaluate_function_grid(function, x_values, y_values)
    fig, ax = plt.subplots(figsize=cfg["figure"]["default_size"])
    image = ax.imshow(z_grid, origin="lower", aspect="auto", cmap=colormap)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(value) for value in y_values])
    format_axis(ax, cfg, xlabel=x_label, ylabel=y_label, title=title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_label, fontsize=cfg["text"]["axis_label_size"], fontweight=font_weight(cfg))
    fig.tight_layout()
    return save_figure(fig, out_path, cfg)
