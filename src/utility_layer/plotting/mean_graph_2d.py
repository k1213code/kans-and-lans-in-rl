"""2D seed overlay and mean-curve plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utility_layer.graph_config import GraphConfig
from utility_layer.plotting.common import color_map, format_axis, format_legend, get_pyplot, load_style, save_figure


def plot_seed_mean_curves(
    curves: Mapping[str, Mapping[str, Any]],
    out_path: str | Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    legend_title: str = "Run",
    dpi: int | None = None,
    graph_config: GraphConfig | None = None,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> Path | None:
    """Plot faint seed curves and one stronger mean curve for each label."""
    cfg = load_style(dpi, graph_config)
    plt = get_pyplot()
    colors = color_map(list(curves))
    fig, ax = plt.subplots(figsize=cfg["figure"]["default_size"])
    plotted = False

    for name, curve in curves.items():
        for x_values, y_values in curve.get("seeds", []):
            ax.plot(
                x_values,
                y_values,
                color=colors[name],
                alpha=cfg["alpha"]["seed_line"],
                linewidth=cfg["lines"]["seed_line_width"],
            )
        if "mean" not in curve:
            continue
        x_values, y_values, count = curve["mean"]
        ax.plot(
            x_values,
            y_values,
            color=colors[name],
            alpha=cfg["alpha"]["mean_line"],
            linewidth=cfg["lines"]["line_width"],
            label=curve.get("label", f"{name} mean (n={count})"),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    format_axis(ax, cfg, xlabel=x_label, ylabel=y_label, title=title)
    format_legend(ax.legend(title=legend_title, loc="best"), cfg)
    fig.tight_layout()
    return save_figure(fig, out_path, cfg)
