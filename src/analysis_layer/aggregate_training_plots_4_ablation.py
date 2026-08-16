"""Aggregate the KAN ablation experiment.

The ablation experiment compares the full KAN backbone against variants where
one branch is removed. This script turns finished training runs into figures:

1. find all `progress.csv` files below the ablation output folder
2. recover environment, backbone variant, and seed from each run folder
3. create mean training curves with one line per KAN variant
4. add training-time and peak-memory boxplots for the same variants

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.aggregate_training_plots import (
    DEFAULT_PLOT_FORMATS,
    aggregate_plot_paths,
    available_plot_metrics_from_files,
    discover_progress_files,
    load_runs_grouped,
    plot_aggregated_metric,
)
from analysis_layer.utils import (
    display_label,
    final_training_hours,
    peak_memory_mb,
    resolve_analysis_path as resolve_path,
    sanitize_filename,
    save_plot_variants,
)
from utility_layer.paths import OUTPUT_ROOT
from utility_layer.plotting.boxplot import plot_boxplot

DEFAULT_RUNS_ROOT = OUTPUT_ROOT / "4_ablation_experiment"

# These columns are loaded in addition to curve metrics for runtime/memory boxplots.
BOXPLOT_COLUMNS = [
    "custom_time/elapsed_s",
    "custom_time/elapsed_min",
    "time/time_elapsed",
    "memory/peak_total_rss_mb",
    "memory/total_rss_mb",
    "memory/main_rss_mb",
    "memory/children_rss_mb",
]

# The ablation plots should use thesis-facing labels rather than raw backbone ids.
BACKBONE_LABELS = {
    "kan": "KAN",
    "kan_no_base": "KAN no base",
    "kan_no_spline": "KAN no spline",
}

# Stable ordering keeps legends, boxplots, and generated output predictable.
BACKBONE_ORDER = {name: index for index, name in enumerate(BACKBONE_LABELS)}
DISPLAY_METRIC_LABELS = {"rollout/ep_rew_mean": "Reward per Episode"}
DISPLAY_X_LABELS = {"time/total_timesteps": "Timesteps"}


# CLI setup
#--------------------------------


def parse_args() -> argparse.Namespace:
    """
    Read CLI options for ablation aggregation.
    """
    parser = argparse.ArgumentParser(description="Create aggregate plots for the 4_ablation_experiment outputs.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--x-col", type=str, default="time/total_timesteps")
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# Run-name parsing and labels
#-------------------------------------


def parse_ablation_run_name(run_dir_name: str) -> tuple[str, str, int] | None:
    """
    Extract environment, ablation condition, and seed from one run folder.

    The shared aggregation helper expects exactly these three values so it can
    group frames as environment -> condition -> seeded runs.
    """
    parts = run_dir_name.split("__")
    if len(parts) < 4:
        return None

    # Standard run names store actor, critic, and seed as separate "__" parts.
    env_id = parts[0]
    actor_part = next((part for part in parts if part.startswith("actor-")), "")
    critic_part = next((part for part in parts if part.startswith("critic-")), "")
    seed_part = next((part for part in parts if part.startswith("seed")), "")
    if not actor_part or not critic_part or not seed_part:
        return None

    try:
        seed = int(seed_part.replace("seed", ""))
    except ValueError:
        return None

    actor = actor_part.replace("actor-", "", 1)
    critic = critic_part.replace("critic-", "", 1)
    # The normal ablation setup uses the same variant for actor and critic.
    group = actor if actor == critic else f"actor={actor} | critic={critic}"
    return env_id, group, seed


def backbone_sort_key(backbone: str) -> tuple[int, str]:
    """
    Sort known ablation variants before any unexpected labels.
    """
    return BACKBONE_ORDER.get(backbone, 99), backbone


def display_backbone_label(backbone: str) -> str:
    """
    Convert an internal backbone id into a plot label.
    """
    return BACKBONE_LABELS.get(backbone, backbone)


def ordered_env_runs(env_runs: dict[str, list[pd.DataFrame]]) -> dict[str, list[pd.DataFrame]]:
    """
    Return environment runs sorted by ablation variant order.
    """
    return dict(sorted(env_runs.items(), key=lambda item: backbone_sort_key(item[0])))


# Summary boxplots
#------------------------------------


def collect_box_values(
    env_runs: dict[str, list[pd.DataFrame]],
    value_fn,
) -> dict[str, list[float]]:
    """
    Collect one scalar summary value per seed run for each ablation variant.
    """
    groups = {}
    for backbone, runs in ordered_env_runs(env_runs).items():
        values = [value for df in runs if (value := value_fn(df)) is not None]
        if values:
            groups[display_backbone_label(backbone)] = values
    return groups


def save_boxplot_variants(
    groups: dict[str, list[float]],
    outdir: Path,
    stem: str,
    *,
    title: str,
    y_label: str,
    dpi: int,
    dry_run: bool,
) -> int:
    """
    Save one boxplot in all configured formats or report its dry-run paths.
    """
    if not groups:
        return 0

    paths = [outdir / f"{stem}.{plot_format}" for plot_format in DEFAULT_PLOT_FORMATS]
    if dry_run:
        # Dry-run output mirrors the files that save_plot_variants would write.
        value_count = sum(len(values) for values in groups.values())
        print(f"[plot] {', '.join(str(path) for path in paths)} ({value_count} run(s))")
        return 1

    # The shared helper handles directory creation and output formats.
    saved_paths = save_plot_variants(
        plot_boxplot,
        outdir,
        stem,
        "summary boxplot",
        groups=groups,
        title=title,
        y_label=y_label,
        x_label="Network",
        dpi=dpi,
    )
    return int(bool(saved_paths))


def save_ablation_boxplots(
    env_id: str,
    env_runs: dict[str, list[pd.DataFrame]],
    outdir: Path,
    dpi: int,
    dry_run: bool,
) -> int:
    """
    Create the scalar resource-use boxplots for one environment.
    """
    created = save_boxplot_variants(
        collect_box_values(env_runs, final_training_hours),
        outdir,
        "training_time_hours_boxplot",
        title=f"{env_id} - Training time",
        y_label="Training time (hours)",
        dpi=dpi,
        dry_run=dry_run,
    )
    created += save_boxplot_variants(
        collect_box_values(env_runs, peak_memory_mb),
        outdir,
        "peak_memory_mb_boxplot",
        title=f"{env_id} - Peak memory",
        y_label="Peak memory (MB)",
        dpi=dpi,
        dry_run=dry_run,
    )
    return created


def display_title(env_id: str, metric: str) -> str:
    """
    Keep the reward plot title compact and include metric names for diagnostics.
    """
    return env_id if metric in DISPLAY_METRIC_LABELS else f"{env_id} - {metric}"


# Pipeline orchestration
#-------------------------------------------------


def main() -> None:
    """
    CLI entry point for creating all ablation aggregate plots.
    """
    args = parse_args()
    runs_root = resolve_path(args.runs_root)
    outdir = resolve_path(args.outdir) if args.outdir else runs_root / "aggregated_training_plots"

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    # Discover files once, then reuse that list for metric discovery and loading.
    progress_files = discover_progress_files(runs_root)
    metrics = available_plot_metrics_from_files(progress_files, args.metrics)
    read_metrics = list(dict.fromkeys([*metrics, *BOXPLOT_COLUMNS]))
    grouped = load_runs_grouped(
        runs_root=runs_root,
        metrics=read_metrics,
        x_axis=args.x_col,
        parse_name=parse_ablation_run_name,
        progress_files=progress_files,
    )
    if not grouped:
        raise FileNotFoundError(f"No ablation progress.csv files found below: {runs_root}")

    created = 0
    # Each environment gets one output folder with line plots and boxplots.
    for env_id, env_runs in sorted(grouped.items()):
        env_runs = ordered_env_runs(env_runs)
        env_outdir = outdir / sanitize_filename(env_id)

        for metric in metrics:
            if args.dry_run:
                # Dry-run mode prints the paths that plot_aggregated_metric would save.
                paths = aggregate_plot_paths(env_outdir, metric, metric, DEFAULT_PLOT_FORMATS)
                seed_count = sum(len(runs) for runs in env_runs.values())
                print(f"[plot] {', '.join(str(path) for path in paths)} ({seed_count} run(s), metric={metric})")
                created += 1
                continue

            # Inside one plot, each colored curve is one ablation variant.
            saved_paths = plot_aggregated_metric(
                env_id=env_id,
                grouped_runs=env_runs,
                metric=metric,
                outdir=env_outdir,
                x_axis=args.x_col,
                num_points=args.num_points,
                smooth_window=args.smooth_window,
                dpi=args.dpi,
                label_for_group=display_backbone_label,
                legend_title="Network",
                title=display_title(env_id, metric),
                x_label=display_label(args.x_col, DISPLAY_X_LABELS),
                y_label=display_label(metric, DISPLAY_METRIC_LABELS),
                output_stem=metric,
            )
            if saved_paths:
                created += 1

        # Training time and memory are scalar summaries, so they are plotted separately.
        created += save_ablation_boxplots(env_id, env_runs, env_outdir, args.dpi, args.dry_run)

    file_count = created * len(DEFAULT_PLOT_FORMATS)
    print(f"{'Would create' if args.dry_run else 'Created'} {created} plot set(s) ({file_count} file(s)) in: {outdir}")


if __name__ == "__main__":
    main()
