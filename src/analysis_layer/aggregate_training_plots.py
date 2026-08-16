"""Shared helpers for aggregate training plots.

One training run writes one ``progress.csv`` file. This module combines many of
those files into plots that compare several runs at once.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.utils import (
    AxisLimits,
    DEFAULT_PLOT_FORMATS,
    DEFAULT_PLOT_METRICS,
    DIAGNOSTIC_LAYER_PREFIXES,
    sanitize_filename,
)
from utility_layer.paths import OUTPUT_ROOT
from utility_layer.plotting.mean_graph_2d import plot_seed_mean_curves

# These aliases describe small helper functions passed in from experiment-specific scripts.
# For example, the hyperparameter script passes its own run-name parser because its folders
# encode "learning rate", "clip range", etc. differently from the generic runs.
RunNameParser = Callable[[str], tuple[str, str, int] | None]
LabelFormatter = Callable[[str], str]
CurveLabelFormatter = Callable[[str, int], str]


def discover_progress_files(runs_root: str | Path) -> list[Path]:
    """
    Find all SB3 progress.csv files below one experiment output folder.
    """
    runs_root = Path(runs_root)
    return sorted(runs_root.rglob("progress.csv")) if runs_root.exists() else []


def available_plot_metrics_from_files(
    progress_files: Sequence[Path],
    requested: Sequence[str] | None = None,
) -> list[str]:
    """
    Decide which metrics can be plotted from the available CSV headers.
    """
    # A caller-provided metric list is treated as intentional, even if some runs miss columns.
    if requested is not None:
        return list(requested)

    columns: set[str] = set()
    layer_metrics: set[str] = set()
    for csv_path in progress_files:
        try:
            # Reading only the header is enough here and avoids loading large progress files.
            header = pd.read_csv(csv_path, nrows=0)
        except Exception as exc:
            print(f"[skip] Failed to read header from {csv_path}: {exc}")
            continue
        for column in map(str, header.columns):
            columns.add(column)
            # Per-layer diagnostics are dynamic, so discover them from the data.
            if column.startswith(DIAGNOSTIC_LAYER_PREFIXES):
                layer_metrics.add(column)

    metrics = [metric for metric in DEFAULT_PLOT_METRICS if metric in columns]
    metrics.extend(metric for metric in sorted(layer_metrics) if metric not in metrics)
    return metrics


def available_plot_metrics(runs_root: str | Path, requested: Sequence[str] | None = None) -> list[str]:
    """
    Convenience wrapper for metric discovery from an experiment folder.
    """
    return available_plot_metrics_from_files(discover_progress_files(runs_root), requested)


def parse_run_name(run_dir_name: str) -> tuple[str, str, int] | None:
    """
    Parse the standard run directory name into env, group label, and seed.

    The generic training run names look roughly like:

    ``Walker2d-v5__actor-kan__critic-kan__seed2025__run-...``
    """
    parts = run_dir_name.split("__")
    if len(parts) < 4:
        return None

    env_id, actor_part, critic_part, seed_part = parts[:4]
    if not actor_part.startswith("actor-") or not critic_part.startswith("critic-") or not seed_part.startswith("seed"):
        return None

    try:
        actor = actor_part.replace("actor-", "")
        critic = critic_part.replace("critic-", "")
        seed = int(seed_part.replace("seed", ""))
    except ValueError:
        return None

    if actor != "sb3" and critic != "sb3" and actor == critic:
        config_label = f"{actor} (actor+critic)"
    elif actor == critic:
        config_label = f"actor/critic={actor}"
    else:
        config_label = f"actor={actor} | critic={critic}"
    return env_id, config_label, seed


def read_progress_frame(csv_path: Path, metrics: Sequence[str], x_axis: str) -> pd.DataFrame | None:
    """
    Read one progress.csv and keep only the x-axis plus requested metric columns.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[skip] Failed to read {csv_path}: {exc}")
        return None

    if x_axis not in df.columns:
        print(f"[skip] Missing x-axis '{x_axis}' in {csv_path}")
        return None

    # Missing metric columns are normal because not every run logs every diagnostic.
    keep_cols = [x_axis] + [metric for metric in metrics if metric in df.columns]
    return df[keep_cols].copy()


def load_runs_grouped(
    runs_root: str | Path,
    metrics: Sequence[str] | None = None,
    x_axis: str = "time/total_timesteps",
    parse_name: RunNameParser = parse_run_name,
    progress_files: Sequence[Path] | None = None,
) -> dict[str, dict[str, list[pd.DataFrame]]]:
    """
    Load progress CSVs and group them as env -> condition -> seeded runs.

    The returned structure is nested like this:

    ``grouped[environment][condition] = [seed_run_1, seed_run_2, ...]``
    """
    files = list(progress_files) if progress_files is not None else discover_progress_files(runs_root)
    metric_names = list(metrics) if metrics is not None else available_plot_metrics_from_files(files)
    grouped: dict[str, dict[str, list[pd.DataFrame]]] = defaultdict(lambda: defaultdict(list))

    for csv_path in files:
        parsed = parse_name(csv_path.parent.name)
        if parsed is None:
            print(f"[skip] Could not parse run name: {csv_path.parent.name}")
            continue

        env_id, group_id, seed = parsed
        frame = read_progress_frame(csv_path, metric_names, x_axis)
        if frame is None:
            continue
        # Store the seed in the DataFrame so later helpers do not need a side table.
        frame["__seed__"] = seed
        grouped[env_id][group_id].append(frame)

    return grouped


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    """
    Apply the same simple rolling mean used by the per-run training plots.

    A rolling mean replaces each point by the average of nearby recent points.
    This reduces visual noise while keeping the overall training trend visible.
    """
    if window <= 1 or len(y) == 0:
        return y
    return pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()


def prepare_metric_series(
    runs: Sequence[pd.DataFrame],
    metric: str,
    x_axis: str,
    smooth_window: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Convert grouped run DataFrames into clean seed-level x/y arrays.

    Each returned entry contains:

    - the seed number
    - x-values, usually training timesteps
    - y-values, usually reward or another logged metric
    """
    series = []
    for df in runs:
        if metric not in df.columns or x_axis not in df.columns:
            continue

        # Coerce to numeric before plotting because pandas may read some columns as object.
        sub = df[[x_axis, metric]].copy()
        sub[x_axis] = pd.to_numeric(sub[x_axis], errors="coerce")
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            continue

        x = sub[x_axis].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        # Sort by x so interpolation is well-defined.
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # If SB3 logged the same timestep more than once, keep the latest value.
        unique_x = []
        unique_y = []
        for xi, yi in zip(x, y):
            if unique_x and xi == unique_x[-1]:
                unique_y[-1] = yi
            else:
                unique_x.append(xi)
                unique_y.append(yi)

        x = np.asarray(unique_x, dtype=float)
        y = np.asarray(unique_y, dtype=float)
        if len(x) == 0:
            continue
        if smooth_window > 1 and len(y) >= 2:
            y = smooth_series(y, smooth_window)

        # The plotting utility can draw individual seed curves if the seed is known.
        seed = int(df["__seed__"].iloc[0]) if "__seed__" in df.columns else -1
        series.append((seed, x, y))
    return series


def interpolate_series(
    series: Sequence[tuple[int, np.ndarray, np.ndarray]],
    num_points: int = 300,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Interpolate seed curves onto one shared x-grid before averaging.

    Different seeds often do not log at exactly the same timesteps. A pointwise
    mean is only meaningful when all curves are compared at the same x-values.
    Interpolation estimates each seed's value on a common grid of ``num_points``.
    """
    if not series:
        return None

    x_min = min(x[0] for _, x, _ in series)
    x_max = max(x[-1] for _, x, _ in series)
    if x_max < x_min:
        return None

    # The shared grid makes seed-wise means comparable even when log rows differ.
    x_common = np.array([x_min], dtype=float) if x_max == x_min else np.linspace(x_min, x_max, num_points)
    y_interp = []
    for _, x, y in series:
        # Outside a seed's observed x-range we keep NaN so it does not affect the mean.
        yi = np.full_like(x_common, np.nan, dtype=float)
        if len(x) == 1:
            yi[x_common >= x[0]] = y[0]
        else:
            mask = (x_common >= x[0]) & (x_common <= x[-1])
            yi[mask] = np.interp(x_common[mask], x, y)
        y_interp.append(yi)
    return x_common, np.vstack(y_interp)


def build_seed_mean_curves(
    grouped_runs: dict[str, list[pd.DataFrame]],
    metric: str,
    x_axis: str = "time/total_timesteps",
    num_points: int = 300,
    smooth_window: int = 1,
    label_for_group: LabelFormatter | None = None,
    curve_label: CurveLabelFormatter | None = None,
) -> dict[str, dict[str, object]]:
    """
    Build the curve dictionary consumed by the shared 2D mean-graph utility.

    The plotting utility expects both the individual seed curves and the mean
    curve. This function prepares that structure for every plotted condition.
    """
    curves: dict[str, dict[str, object]] = {}
    format_label = label_for_group or str

    for group_id, runs in grouped_runs.items():
        # Each group is one condition, e.g. one architecture or one hyperparameter value.
        series = prepare_metric_series(runs, metric, x_axis, smooth_window)
        interpolated = interpolate_series(series, num_points)
        if interpolated is None:
            continue

        x_common, y_stack = interpolated
        valid_mask = np.sum(~np.isnan(y_stack), axis=0) > 0
        if not np.any(valid_mask):
            continue

        # Average only over x positions where at least one seed has data.
        mean = np.nanmean(y_stack[:, valid_mask], axis=0)
        count = int(y_stack.shape[0])
        curves[str(group_id)] = {
            "seeds": [(x_values, y_values) for _, x_values, y_values in series],
            "mean": (x_common[valid_mask], mean, count),
            "label": (
                curve_label(str(group_id), count)
                if curve_label
                else f"{format_label(str(group_id))} mean (n={count})"
            ),
        }

    return curves


def finite_limits(values: Sequence[np.ndarray]) -> AxisLimits | None:
    """
    Return finite lower and upper limits across several arrays.

    Axis limits should ignore NaN and infinite values.
    """
    finite_values = []
    for value in values:
        array = np.asarray(value, dtype=float)
        array = array[np.isfinite(array)]
        if array.size:
            finite_values.append(array)
    if not finite_values:
        return None

    combined = np.concatenate(finite_values)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if lower == upper:
        margin = max(abs(lower) * 0.05, 1.0)
        return lower - margin, upper + margin
    return lower, upper


def merge_limits(first: AxisLimits | None, second: AxisLimits | None) -> AxisLimits | None:
    """
    Merge two optional axis ranges into one range.

    This is useful when several plots should share the same scale.
    """
    if first is None:
        return second
    if second is None:
        return first
    return min(first[0], second[0]), max(first[1], second[1])


def padded_limits(limits: AxisLimits | None, fraction: float = 0.05) -> AxisLimits | None:
    """
    Add a small visual margin around an axis range for visual clarity.
    """
    if limits is None:
        return None
    lower, upper = limits
    margin = (upper - lower) * fraction
    return lower - margin, upper + margin


def curve_axis_limits(curves: dict[str, dict[str, object]]) -> tuple[AxisLimits | None, AxisLimits | None]:
    """
    Compute x/y limits from both individual seed curves and mean curves.
    """
    x_values = []
    y_values = []
    for curve in curves.values():
        for seed_x, seed_y in curve.get("seeds", []):
            x_values.append(seed_x)
            y_values.append(seed_y)
        if "mean" in curve:
            mean_x, mean_y, _ = curve["mean"]
            x_values.append(mean_x)
            y_values.append(mean_y)
    return finite_limits(x_values), finite_limits(y_values)


def aggregate_plot_path(
    outdir: str | Path,
    metric: str,
    output_stem: str | None = None,
    plot_format: str = "png",
) -> Path:
    """
    Build one output path for an aggregate plot.

    Metric names contain slashes such as ``rollout/ep_rew_mean``. Those slashes
    are replaced before saving so they do not become unintended folders.
    """
    suffix = plot_format.lstrip(".") or "png"
    stem = sanitize_filename(output_stem or metric)
    return Path(outdir) / f"{stem}.{suffix}"


def aggregate_plot_paths(
    outdir: str | Path,
    metric: str,
    output_stem: str | None = None,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """
    Build all output paths for the requested plot formats.
    """
    return [aggregate_plot_path(outdir, metric, output_stem, plot_format) for plot_format in plot_formats]


def plot_aggregated_metric(
    env_id: str,
    grouped_runs: dict[str, list[pd.DataFrame]],
    metric: str,
    outdir: str | Path,
    x_axis: str = "time/total_timesteps",
    num_points: int = 300,
    smooth_window: int = 1,
    dpi: int = 150,
    label_for_group: LabelFormatter | None = None,
    curve_label: CurveLabelFormatter | None = None,
    legend_title: str = "Run",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    output_stem: str | None = None,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
    x_limits: AxisLimits | None = None,
    y_limits: AxisLimits | None = None,
) -> list[Path]:
    """
    Create one aggregate metric plot and save it in all requested formats.

    This is the central plotting function used by the experiment-specific
    aggregation scripts. It prepares the curves, delegates the actual drawing to
    the plotting utility, and returns the paths that were written.
    """
    curves = build_seed_mean_curves(
        grouped_runs=grouped_runs,
        metric=metric,
        x_axis=x_axis,
        num_points=num_points,
        smooth_window=smooth_window,
        label_for_group=label_for_group,
        curve_label=curve_label,
    )
    saved_paths = []
    for out_path in aggregate_plot_paths(outdir, metric, output_stem, plot_formats):
        # All drawing details live in the utility layer so aggregate scripts stay compact.
        saved_path = plot_seed_mean_curves(
            curves=curves,
            out_path=out_path,
            title=title or f"{env_id} - {metric}",
            x_label=x_label or x_axis,
            y_label=y_label or metric,
            legend_title=legend_title,
            dpi=dpi,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        if saved_path is not None:
            saved_paths.append(saved_path)
            print(f"[ok] saved aggregated plot: {saved_path}")
    return saved_paths


def create_aggregated_training_plots(
    runs_root: str | Path,
    outdir: str | Path,
    metrics: Sequence[str] | None = None,
    x_axis: str = "time/total_timesteps",
    num_points: int = 300,
    smooth_window: int = 1,
    dpi: int = 150,
    parse_name: RunNameParser = parse_run_name,
    label_for_group: LabelFormatter | None = None,
    legend_title: str = "Run",
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> None:
    """
    Aggregate all plottable metrics below one runs root.

    This is the high-level helper for simple aggregation tasks. More specialized
    experiment scripts reuse the lower-level functions above when they need
    custom grouping or custom output folders.
    """
    runs_root = Path(runs_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover files once and reuse the list for metric detection and loading.
    progress_files = discover_progress_files(runs_root)
    metric_names = available_plot_metrics_from_files(progress_files, metrics)
    if not metric_names:
        print(f"[warn] No plottable metrics found below: {runs_root}")
        return

    grouped = load_runs_grouped(
        runs_root=runs_root,
        metrics=metric_names,
        x_axis=x_axis,
        parse_name=parse_name,
        progress_files=progress_files,
    )
    if not grouped:
        print(f"[warn] No runs found below: {runs_root}")
        return

    for env_id, env_runs in grouped.items():
        # Keep environment outputs separated so figures are easy to browse.
        env_outdir = outdir / sanitize_filename(env_id)
        for metric in metric_names:
            plot_aggregated_metric(
                env_id=env_id,
                grouped_runs=env_runs,
                metric=metric,
                outdir=env_outdir,
                x_axis=x_axis,
                num_points=num_points,
                smooth_window=smooth_window,
                dpi=dpi,
                label_for_group=label_for_group,
                legend_title=legend_title,
                plot_formats=plot_formats,
            )


def main() -> None:
    """
    Default CLI entry for quick aggregation of the generic runs folder.
    """
    create_aggregated_training_plots(
        runs_root=OUTPUT_ROOT / "runs",
        outdir=OUTPUT_ROOT / "aggregated_plots",
        smooth_window=20,
    )


if __name__ == "__main__":
    main()
