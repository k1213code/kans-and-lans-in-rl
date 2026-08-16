"""Aggregate the 0-pre-experiment runs.

The pre-experiment contains two small control studies:

- ``mlp`` compares the custom MLP extractor with the standard SB3 extractor
- ``update`` compares different KAN grid-update strategies

For both studies, this script creates the usual aggregated learning curves and
adds summary boxplots for training time and peak memory usage.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.aggregate_training_plots import (
    create_aggregated_training_plots,
    discover_progress_files,
    load_runs_grouped,
)
from analysis_layer.utils import (
    final_training_hours,
    peak_memory_mb,
    resolve_analysis_path as resolve_path,
    sanitize_filename,
    save_plot_variants,
)
from utility_layer.paths import OUTPUT_ROOT
from utility_layer.plotting.boxplot import plot_boxplot

DEFAULT_RUNS_ROOT = OUTPUT_ROOT / "0_pre_experiment"
DEFAULT_SUBSETS = ("mlp", "update")
BOXPLOT_COLUMNS = [
    # These columns are enough to summarize runtime and memory for the boxplots.
    "custom_time/elapsed_s",
    "custom_time/elapsed_min",
    "time/time_elapsed",
    "memory/peak_total_rss_mb",
    "memory/total_rss_mb",
    "memory/main_rss_mb",
    "memory/children_rss_mb",
]


def parse_args() -> argparse.Namespace:
    """
    Read the small set of CLI options needed for pre-experiment aggregation.
    """
    parser = argparse.ArgumentParser(description="Create aggregate plots for the 0_pre_experiment outputs.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS), help="Subfolders below runs-root to aggregate.")
    return parser.parse_args()


def display_run_label(run_id: str) -> str:
    """
    Turn internal run ids into shorter labels for legends and boxplot x-axes.

    """
    if "_" in run_id:
        prefix, suffix = run_id.split("_", 1)
        if prefix.startswith("palma-") and suffix:
            return suffix
    return run_id


def parse_pre_run_name(run_dir_name: str) -> tuple[str, str, int] | None:
    """
    Extract environment, pre-experiment condition, and seed from one run folder.

    The generic aggregation helper needs those three values to group the runs.
    The condition is taken from the ``run-...`` part of the folder name because
    this is where the pre-experiment scripts store labels such as the MLP setup
    or grid-update strategy.
    """
    parts = run_dir_name.split("__")
    if len(parts) < 5:
        return None

    # Example pieces: env id, actor, critic, seed, run id.
    env_id = parts[0]
    seed_part = next((part for part in parts if part.startswith("seed")), "")
    run_part = next((part for part in parts if part.startswith("run-")), "")
    if not seed_part or not run_part:
        return None

    try:
        seed = int(seed_part.replace("seed", ""))
    except ValueError:
        return None

    run_id = run_part.replace("run-", "", 1)
    # The seed is already stored separately, so remove duplicate seed text from the label.
    run_id = re.sub(r"-seed\d+$", "", run_id)
    return env_id, run_id, seed


def create_pre_aggregated_training_plots(runs_root: str | Path, outdir: str | Path) -> None:
    """
    Create learning-curve plots and summary boxplots for one pre-experiment subset.
    """
    # Reuse the generic curve aggregation. Only the run-name parser and labels are specific here.
    create_aggregated_training_plots(
        runs_root=runs_root,
        outdir=outdir,
        smooth_window=20,
        parse_name=parse_pre_run_name,
        label_for_group=display_run_label,
        legend_title="Run",
    )
    # The pre-experiment additionally compares resource usage with boxplots.
    create_pre_summary_boxplots(runs_root=runs_root, outdir=outdir)


def collect_box_values(
    env_runs: dict[str, list[pd.DataFrame]],
    value_fn,
) -> dict[str, list[float]]:
    """
    Collect one summary value per seed run for every plotted condition.

    ``value_fn`` is a small function such as ``final_training_hours`` or
    ``peak_memory_mb``. It turns one run DataFrame into one boxplot value.
    """
    groups = {}
    for run_id in sorted(env_runs, key=display_run_label):
        values = []
        for df in env_runs[run_id]:
            value = value_fn(df)
            if value is not None:
                values.append(value)
        if values:
            groups[display_run_label(run_id)] = values
    return groups


def create_pre_summary_boxplots(runs_root: str | Path, outdir: str | Path) -> None:
    """
    Create training-time and peak-memory boxplots for one pre-experiment subset.
    """
    runs_root = Path(runs_root)
    outdir = Path(outdir)

    # Reuse the same grouping logic as the learning-curve aggregation.
    progress_files = discover_progress_files(runs_root)
    grouped = load_runs_grouped(
        runs_root=runs_root,
        metrics=BOXPLOT_COLUMNS,
        parse_name=parse_pre_run_name,
        progress_files=progress_files,
    )

    for env_id, env_runs in grouped.items():
        # Each environment gets its own folder, matching the line-plot output structure.
        env_outdir = outdir / sanitize_filename(env_id)
        save_plot_variants(
            plot_boxplot,
            env_outdir,
            "training_time_hours_boxplot",
            "summary boxplot",
            groups=collect_box_values(env_runs, final_training_hours),
            title=f"{env_id} - Training time",
            y_label="Training time (hours)",
            x_label="Run",
        )
        save_plot_variants(
            plot_boxplot,
            env_outdir,
            "highest_memory_mb_boxplot",
            "summary boxplot",
            groups=collect_box_values(env_runs, peak_memory_mb),
            title=f"{env_id} - Highest memory",
            y_label="Highest memory (MB)",
            x_label="Run",
        )


def main() -> None:
    """
    CLI entry point for aggregating the configured pre-experiment subsets.
    """
    args = parse_args()
    runs_root = resolve_path(args.runs_root)
    base_outdir = resolve_path(args.outdir) if args.outdir else runs_root

    for subset in args.subsets:
        # With several subsets, keep their outputs separated by subset name.
        outdir = (
            base_outdir / f"aggregated_plots_{subset}"
            if len(args.subsets) > 1 or args.outdir is None
            else base_outdir
        )
        create_pre_aggregated_training_plots(
            runs_root=runs_root / subset,
            outdir=outdir,
        )


if __name__ == "__main__":
    main()
