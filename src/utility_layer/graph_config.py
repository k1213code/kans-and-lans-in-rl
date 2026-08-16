"""Central graph styling configuration for matplotlib plots."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

GraphConfig = dict[str, dict[str, Any]]


GRAPH_CONFIG: GraphConfig = {
    "figure": {
        # Standard size for 2D line plots and generic heatmaps, in inches.
        "default_size": (10.0, 6.0),
        # Size for 3D plots, in inches.
        "size_3d": (10.5, 7.3),
        # Width for stacked size/depth bar plots.
        "width_scaling_width": 9.8,
        # Minimum height for stacked size/depth bar plots.
        "width_scaling_min_height": 4.6,
        # Extra height per depth row in stacked size/depth bar plots.
        "width_scaling_row_height": 2.45,
        # Minimum size for best-category heatmaps, in inches.
        "heatmap_min_size": (9.0, 5.8),
        # Extra heatmap size per hidden-size/depth cell.
        "heatmap_cell_size": (1.35, 1.15),
        # Extra heatmap space for labels, title, and legend.
        "heatmap_padding": (3.4, 3.0),
        # Default raster output resolution, for example for png files.
        "dpi": 150,
    },
    "text": {
        # Axis-label font size for all plots.
        "axis_label_size": 20,
        # Tick-label font size for all plots.
        "tick_label_size": 17,
        # Title font size for all plots.
        "title_size": 21,
        # Legend-entry font size for all plots.
        "legend_text_size": 16,
        # Legend-title font size for all plots.
        "legend_title_size": 16,
        # Text size for values printed inside heatmap cells.
        "cell_text_size": 14,
        # Whether plot text should be bold.
        "bold": True,
    },
    "lines": {
        # Main line width, for example mean curves.
        "line_width": 2.0,
        # Line width for faint individual seed curves.
        "seed_line_width": 1.2,
        # Line width for 3D support lines connecting grid points.
        "support_line_width": 1.0,
        # Line width for vertical 3D stems.
        "stem_line_width": 0.8,
        # Edge width around 3D scatter markers.
        "marker_edge_width": 0.75,
        # Spine width for all plots.
        "axis_spine_width": 1.8,
        # White border width around heatmap cells.
        "heatmap_cell_edge_width": 2.4,
        # Edge width around bars.
        "bar_edge_width": 0.8,
        # Error-bar line width.
        "error_line_width": 1.0,
        # Tick mark width for all plots.
        "tick_width": 1.6,
    },
    "markers": {
        # Marker size for actual 3D data points.
        "scatter_size": 72,
        # Marker size for faint floor projections in 3D plots.
        "floor_scatter_size": 28,
        # Cap size for bar-plot error bars.
        "bar_error_capsize": 3.0,
        # Cap thickness for bar-plot error bars.
        "bar_error_capthick": 1.0,
    },
    "alpha": {
        # Transparency for individual seed curves.
        "seed_line": 0.4,
        # Transparency for mean curves.
        "mean_line": 1.0,
        # Transparency for 3D support lines.
        "support_line": 0.45,
        # Transparency for vertical 3D stems.
        "stem_line": 0.3,
        # Transparency for projected floor markers in 3D plots.
        "floor_scatter": 0.22,
        # Transparency for bars.
        "bar": 0.82,
        # Transparency for grid lines.
        "grid": 0.28,
        # Transparency for colored winner cells in heatmaps.
        "heatmap_cell": 0.86,
        # Transparency for white value boxes in heatmap cells.
        "heatmap_value_box": 0.76,
    },
}


def load_graph_config(dpi: int | None = None) -> GraphConfig:
    """Return a copy of the central graph config, optionally overriding dpi."""
    config = deepcopy(GRAPH_CONFIG)
    if dpi is not None:
        config["figure"]["dpi"] = int(dpi)
    return config
