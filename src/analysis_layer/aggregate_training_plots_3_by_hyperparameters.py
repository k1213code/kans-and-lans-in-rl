"""Aggregate hyperparameter-sensitivity training plots.

The hyperparameter experiment produces finished PPO runs where one parameter is
varied at a time. This script turns those runs into comparison figures:

1. find all `progress.csv` files below one experiment folder
2. recover environment, backbone, hyperparameter, value, and seed metadata
3. group seeded runs by environment, backbone, hyperparameter, and value
4. create mean training curves with one line per varied value
5. add training-time boxplots for the same value groups

"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.aggregate_training_plots import (
    AxisLimits,
    DEFAULT_PLOT_FORMATS,
    aggregate_plot_paths,
    available_plot_metrics_from_files,
    build_seed_mean_curves,
    curve_axis_limits,
    merge_limits,
    padded_limits,
    plot_aggregated_metric,
)
from analysis_layer.utils import (
    backbone_from_parts,
    display_label,
    final_training_hours,
    read_simple_yaml,
    resolve_analysis_path as resolve_path,
    sanitize_filename,
)
from utility_layer.paths import OUTPUT_ROOT
from utility_layer.plotting.boxplot import plot_boxplot

DEFAULT_RUNS_ROOT = OUTPUT_ROOT / "2_Hyperparameter_experiment"

# These columns are enough to create training-time boxplots next to the curves.
TIME_COLUMNS = ("custom_time/elapsed_s", "custom_time/elapsed_min", "time/time_elapsed")

# Hyperparameter run folders encode the high-level run metadata in their names.
RUN_NAME_RE = re.compile(
    r"^(?P<env_id>.+?)__actor-(?P<actor>.+?)__critic-(?P<critic>.+?)__seed(?P<seed>\d+)__run-(?P<run_id>.+)$"
)

# The run id stores which hyperparameter changed and which value index it used.
HPARAM_RE = re.compile(
    r"^palma-hparam-(?P<hparam>[^-]+)-.+?-v(?P<value_idx>\d+)-(?P<value_label>.+)-seed(?P<seed>\d+)$"
)

# Short names in Slurm/run ids are mapped back to the resolved config keys.
HPARAM_CONFIG_KEYS = {
    "clip": "clip_coef",
    "ent": "ent_coef",
    "grid": "grid_size",
    "lr": "learning_rate",
    "spline": "spline_order",
}
DISPLAY_METRIC_LABELS = {
    "rollout/ep_rew_mean": "Reward per Episode",
}
DISPLAY_X_LABELS = {
    "time/total_timesteps": "Timesteps",
}

# Stable ordering makes generated folders, legends, and dry-run output predictable.
HPARAM_ORDER = {name: index for index, name in enumerate(["clip", "ent", "lr", "grid", "spline"])}
BACKBONE_ORDER = {"mlp": 1, "kan": 2, "lan": 3, "kan_no_base": 4, "kan_no_spline": 5, "debug_constant": 6}


# Data model
#--------------------------------------


@dataclass(frozen=True)
class RunInfo:
    """Metadata needed to aggregate one hyperparameter-sensitivity run."""

    run_dir: Path
    progress_path: Path
    env_id: str
    backbone: str
    hparam: str
    hparam_key: str
    value_idx: int
    value: Any
    seed: int
    run_id: str


# CLI setup
#--------------------------------


def parse_args() -> argparse.Namespace:
    """
    Read CLI options for hyperparameter aggregation.
    """
    parser = argparse.ArgumentParser(
        description="Create aggregate training plots for the 2_Hyperparameter_experiment outputs."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--envs", nargs="+", default=None, help="Optional env_id filter, e.g. Ant-v5 Walker2d-v5.")
    parser.add_argument("--backbones", nargs="+", default=None, help="Optional backbone filter, e.g. mlp kan lan.")
    parser.add_argument("--hyperparameters", nargs="+", default=None, help="Optional filter, e.g. clip ent lr grid spline.")
    parser.add_argument("--x-col", type=str, default="time/total_timesteps")
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# Generic parsing and formatting helpers
#-----------------------------


def parse_scalar(value: str | None) -> Any:
    """
    Convert simple config strings into Python scalar values.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none"}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        integer = int(text)
    except ValueError:
        pass
    else:
        return integer
    try:
        return float(text)
    except ValueError:
        return text


def value_sort_key(value: Any, value_idx: int) -> tuple[int, float | str, int]:
    """
    Sort hyperparameter values numerically when possible and by value index as fallback.
    """
    if isinstance(value, bool) or value is None:
        return 2, str(value), value_idx
    if isinstance(value, (int, float)):
        return 0, float(value), value_idx
    return 1, str(value), value_idx


def value_label(value: Any) -> str:
    """
    Convert one hyperparameter value into a compact legend label.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# Run discovery
#-------------------------------


def run_info_from_dir(run_dir: Path) -> RunInfo | None:
    """
    Build RunInfo for one run folder if it belongs to the hyperparameter sweep.
    """
    progress_path = run_dir / "progress.csv"
    if not progress_path.exists():
        return None

    # First parse the standard run-folder prefix: env, actor, critic, seed, run id.
    name_match = RUN_NAME_RE.match(run_dir.name)
    if name_match is None:
        print(f"[skip] Could not parse run directory name: {run_dir.name}")
        return None

    # Then parse the experiment-specific run id for hyperparameter metadata.
    run_id = name_match.group("run_id")
    hparam_match = HPARAM_RE.match(run_id)
    if hparam_match is None:
        print(f"[skip] Could not parse hyperparameter run id: {run_id}")
        return None

    # Config values are the source of truth when present; the run name is fallback.
    config = read_simple_yaml(run_dir / "resolved_config.yaml")
    hparam = hparam_match.group("hparam")
    hparam_key = HPARAM_CONFIG_KEYS.get(hparam, hparam)
    value_idx = int(hparam_match.group("value_idx"))
    seed = int(config.get("seed") or name_match.group("seed"))
    value = parse_scalar(config.get(hparam_key))

    if value is None:
        # Some old folders may not have the varied value in resolved_config.yaml.
        value = hparam_match.group("value_label")

    actor = config.get("actor_backbone_type") or name_match.group("actor")
    critic = config.get("critic_backbone_type") or name_match.group("critic")
    env_id = config.get("env_id") or name_match.group("env_id")

    return RunInfo(
        run_dir=run_dir,
        progress_path=progress_path,
        env_id=str(env_id),
        backbone=backbone_from_parts(str(actor), str(critic)),
        hparam=hparam,
        hparam_key=hparam_key,
        value_idx=value_idx,
        value=value,
        seed=seed,
        run_id=run_id,
    )


def discover_runs(runs_root: Path) -> list[RunInfo]:
    """
    Find all valid hyperparameter-sensitivity runs below one output folder.
    """
    runs = [info for path in sorted(runs_root.rglob("progress.csv")) if (info := run_info_from_dir(path.parent))]
    # Stable sorting keeps CSV-free plot generation deterministic.
    return sorted(
        runs,
        key=lambda run: (
            run.env_id,
            BACKBONE_ORDER.get(run.backbone, 99),
            run.backbone,
            HPARAM_ORDER.get(run.hparam, 99),
            value_sort_key(run.value, run.value_idx),
            run.seed,
        ),
    )


# Progress loading and grouping
#----------------------------------


def load_progress_frames(runs: Sequence[RunInfo], metrics: Sequence[str], x_col: str) -> dict[Path, pd.DataFrame]:
    """
    Load progress.csv files and keep only columns used by curves and boxplots.
    """
    frames = {}
    keep = list(dict.fromkeys([x_col, *metrics]))
    for run in runs:
        try:
            df = pd.read_csv(run.progress_path)
        except Exception as exc:
            print(f"[skip] Could not read {run.progress_path}: {exc}")
            continue
        if x_col not in df.columns:
            print(f"[skip] Missing x-axis '{x_col}' in {run.progress_path}")
            continue
        # Copy only the needed columns so later grouping works on compact frames.
        frames[run.progress_path] = df[[column for column in keep if column in df.columns]].copy()
    return frames


def filter_runs(runs: Sequence[RunInfo], args: argparse.Namespace) -> list[RunInfo]:
    """
    Apply optional CLI filters for environment, backbone, and hyperparameter.
    """
    envs = set(args.envs or [])
    backbones = set(args.backbones or [])
    hparams = set(args.hyperparameters or [])
    return [
        run
        for run in runs
        if (not envs or run.env_id in envs)
        and (not backbones or run.backbone in backbones)
        and (not hparams or run.hparam in hparams or run.hparam_key in hparams)
    ]


def grouped_runs(runs: Sequence[RunInfo]) -> dict[tuple[str, str, str], list[RunInfo]]:
    """
    Group runs by the plot family: environment, backbone, and varied hyperparameter.
    """
    groups: dict[tuple[str, str, str], list[RunInfo]] = defaultdict(list)
    for run in runs:
        groups[(run.env_id, run.backbone, run.hparam)].append(run)
    return groups


def group_frames_by_hparam_value(
    runs: Sequence[RunInfo],
    frames: dict[Path, pd.DataFrame],
    metric: str,
    x_col: str,
) -> dict[str, list[pd.DataFrame]]:
    """
    Group progress frames by the varied hyperparameter value for one curve plot.
    """
    grouped: dict[str, list[pd.DataFrame]] = {}
    for run in sorted(runs, key=lambda item: (value_sort_key(item.value, item.value_idx), item.seed)):
        frame = frames.get(run.progress_path)
        if frame is None or x_col not in frame.columns or metric not in frame.columns:
            continue
        # The shared aggregate plot helper expects the training seed in this column.
        run_frame = frame.copy()
        run_frame["__seed__"] = run.seed
        grouped.setdefault(value_label(run.value), []).append(run_frame)
    return grouped


def output_dir_for_group(outdir: Path, env_id: str, backbone: str, hparam: str) -> Path:
    """
    Build the output folder for one environment/backbone/hyperparameter group.
    """
    return outdir / sanitize_filename(env_id) / sanitize_filename(backbone) / sanitize_filename(hparam)


def legend_label_for_value(value: str, count: int) -> str:
    """
    Keep legend labels focused on the varied value.
    """
    return value


# Training-time boxplots
#----------------------------------------


def training_time_groups_by_hparam_value(
    runs: Sequence[RunInfo],
    frames: dict[Path, pd.DataFrame],
) -> dict[str, list[float]]:
    """
    Collect one final training-time value per seed for each hyperparameter value.
    """
    groups: dict[str, list[float]] = {}
    for run in sorted(runs, key=lambda item: (value_sort_key(item.value, item.value_idx), item.seed)):
        frame = frames.get(run.progress_path)
        if frame is None:
            continue
        hours = final_training_hours(frame)
        if hours is not None:
            groups.setdefault(value_label(run.value), []).append(hours)
    return groups


def save_training_time_boxplots(
    runs: Sequence[RunInfo],
    frames: dict[Path, pd.DataFrame],
    outdir: Path,
    dpi: int,
    dry_run: bool,
) -> int:
    """
    Save training-time boxplots for every environment/backbone/hyperparameter group.
    """
    created = 0
    for (env_id, backbone, hparam), group_runs in sorted(
        grouped_runs(runs).items(),
        key=lambda item: (
            item[0][0],
            BACKBONE_ORDER.get(item[0][1], 99),
            item[0][1],
            HPARAM_ORDER.get(item[0][2], 99),
        ),
    ):
        groups = training_time_groups_by_hparam_value(group_runs, frames)
        if not groups:
            continue

        target_dir = output_dir_for_group(outdir, env_id, backbone, hparam)
        hparam_key = group_runs[0].hparam_key
        stem = "training_time_hours_boxplot"
        paths = [target_dir / f"{stem}.{plot_format}" for plot_format in DEFAULT_PLOT_FORMATS]

        if dry_run:
            # Dry-run mode reports exactly which plot files would be written.
            seed_count = sum(len(values) for values in groups.values())
            print(
                f"[plot] {', '.join(str(path) for path in paths)} "
                f"({hparam_key}, {len(groups)} value(s), {seed_count} run(s))"
            )
            created += 1
            continue

        # Save one boxplot per configured output format.
        saved = False
        for path in paths:
            saved_path = plot_boxplot(
                groups,
                path,
                title=env_id,
                x_label=hparam_key,
                y_label="Training time (hours)",
                dpi=dpi,
            )
            if saved_path is not None:
                saved = True
                print(f"[ok] saved training-time boxplot: {saved_path}")
        if saved:
            created += 1
    return created


# Axis scaling
#------------------------


def common_axis_limits_by_env_metric(
    runs: Sequence[RunInfo],
    frames: dict[Path, pd.DataFrame],
    metrics: Sequence[str],
    x_col: str,
    num_points: int,
    smooth_window: int,
) -> dict[tuple[str, str], tuple[AxisLimits | None, AxisLimits | None]]:
    """
    Compute common curve limits for each environment and metric.

    Hyperparameter plots are split by backbone and hyperparameter, so sharing
    axis limits within an environment/metric makes those plots easier to compare.
    """
    limits: dict[tuple[str, str], tuple[AxisLimits | None, AxisLimits | None]] = {}
    for (env_id, _, _), group_runs in grouped_runs(runs).items():
        for metric in metrics:
            # Reuse the same curve preparation as the actual plot path.
            value_runs = group_frames_by_hparam_value(group_runs, frames, metric, x_col)
            curves = build_seed_mean_curves(value_runs, metric, x_col, num_points, smooth_window)
            x_limits, y_limits = curve_axis_limits(curves)
            key = (env_id, metric)
            old_x, old_y = limits.get(key, (None, None))
            limits[key] = (merge_limits(old_x, x_limits), merge_limits(old_y, y_limits))

    return {key: (x_limits, padded_limits(y_limits)) for key, (x_limits, y_limits) in limits.items()}


# Pipeline orchestration
#------------------------------------


def main() -> None:
    """
    CLI entry point for creating all hyperparameter-sensitivity plots.
    """
    args = parse_args()
    runs_root = resolve_path(args.runs_root)
    outdir = resolve_path(args.outdir) if args.outdir else runs_root / "aggregated_by_hyperparameter_plots"

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    # First discover and filter run metadata without loading every CSV.
    runs = filter_runs(discover_runs(runs_root), args)
    if not runs:
        raise FileNotFoundError(f"No matching hyperparameter progress.csv files found below: {runs_root}")

    # Then infer plottable metrics from available CSV headers unless the CLI fixes them.
    metrics = available_plot_metrics_from_files([run.progress_path for run in runs], args.metrics)
    if not metrics:
        raise ValueError("No plottable metrics found.")

    # Load line-plot metrics and time columns once, then reuse the frames for all outputs.
    frames = load_progress_frames(runs, [*metrics, *TIME_COLUMNS], args.x_col)
    axis_limits = (
        {}
        if args.dry_run
        else common_axis_limits_by_env_metric(runs, frames, metrics, args.x_col, args.num_points, args.smooth_window)
    )
    created = 0

    # Each group produces one folder with one curve plot per metric.
    for (env_id, backbone, hparam), group_runs in sorted(
        grouped_runs(runs).items(),
        key=lambda item: (
            item[0][0],
            BACKBONE_ORDER.get(item[0][1], 99),
            item[0][1],
            HPARAM_ORDER.get(item[0][2], 99),
        ),
    ):
        hparam_key = group_runs[0].hparam_key
        target_dir = output_dir_for_group(outdir, env_id, backbone, hparam)
        for metric in metrics:
            # Inside one plot, each colored curve is one varied hyperparameter value.
            value_runs = group_frames_by_hparam_value(group_runs, frames, metric, args.x_col)
            if not value_runs:
                continue

            if args.dry_run:
                # Dry-run output mirrors the paths that plot_aggregated_metric would save.
                seed_count = sum(len(runs_for_value) for runs_for_value in value_runs.values())
                paths = aggregate_plot_paths(target_dir, metric, metric, DEFAULT_PLOT_FORMATS)
                print(
                    f"[plot] {', '.join(str(path) for path in paths)} "
                    f"({len(value_runs)} value(s), {seed_count} run(s), metric={metric})"
                )
                created += 1
                continue

            # The shared helper builds seed curves, means, legends, and file variants.
            x_limits, y_limits = axis_limits.get((env_id, metric), (None, None))
            saved_paths = plot_aggregated_metric(
                env_id=env_id,
                grouped_runs=value_runs,
                metric=metric,
                outdir=target_dir,
                x_axis=args.x_col,
                num_points=args.num_points,
                smooth_window=args.smooth_window,
                dpi=args.dpi,
                curve_label=legend_label_for_value,
                legend_title=hparam_key,
                title=env_id,
                x_label=display_label(args.x_col, DISPLAY_X_LABELS),
                y_label=display_label(metric, DISPLAY_METRIC_LABELS),
                output_stem=metric,
                x_limits=x_limits,
                y_limits=y_limits,
            )
            if saved_paths:
                created += 1

    # Training time is a scalar per run, so it is plotted separately as boxplots.
    created += save_training_time_boxplots(runs, frames, outdir, args.dpi, args.dry_run)
    file_count = created * len(DEFAULT_PLOT_FORMATS)
    print(f"{'Would create' if args.dry_run else 'Created'} {created} plot set(s) ({file_count} file(s)) in: {outdir}")


if __name__ == "__main__":
    main()
