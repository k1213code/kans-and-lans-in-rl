"""Aggregate size-experiment training curves by depth and width.

The size experiment trains many models that differ in:

- backbone type, for example MLP, KAN, or LAN
- hidden layer width, for example 8, 16, 32, ...
- number of hidden layers, called depth here
- random seed

This script creates line plots for one environment folder. For each backbone and
depth, it groups the seeded runs by width. The final plot therefore shows one
mean training curve per width, while the individual seed curves remain visible
in the background.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.aggregate_training_plots import (
    DEFAULT_PLOT_FORMATS,
    aggregate_plot_paths,
    available_plot_metrics_from_files,
    plot_aggregated_metric,
)
from analysis_layer.utils import (
    backbone_from_parts,
    parse_int,
    read_simple_yaml,
    resolve_analysis_path as resolve_path,
)
from utility_layer.paths import OUTPUT_ROOT

DEFAULT_RUNS_ROOT = OUTPUT_ROOT / "1_Size_experiment" / "HalfCheetah-v5"

# Size-experiment run names contain width, depth, and seed in a compact suffix.
# The regular expression extracts those values if they are not available from
# the resolved config file.
SIZE_RE = re.compile(r"-w(?P<width>\d+)-d(?P<depth>\d+)-seed(?P<seed>\d+)")

# Actor and critic backbones are also encoded in the run folder name.
BACKBONE_RE = re.compile(r"__actor-(?P<actor>.+?)__critic-(?P<critic>.+?)(?:__|$)")

# The order is only used for stable output folders and predictable plot order.
BACKBONE_ORDER = {
    "mlp": 1,
    "kan": 2,
    "lan": 3,
    "kan_no_base": 4,
    "kan_no_spline": 5,
    "debug_constant": 6,
}


@dataclass(frozen=True)
class RunInfo:
    """Small container for the metadata needed to aggregate one run."""

    run_dir: Path
    progress_path: Path
    backbone: str
    width: int
    depth: int
    seed: int
    run_name: str


def parse_args() -> argparse.Namespace:
    """
    Read CLI options for this aggregation script.
    """
    parser = argparse.ArgumentParser(
        description="Create aggregate plots grouped by backbone and depth, with one mean curve per width."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--x-col", type=str, default="time/total_timesteps")
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_part(value: str) -> str:
    """
    Convert a label into a safe folder-name part.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def run_info_from_dir(run_dir: Path) -> RunInfo | None:
    """
    Extract all aggregation metadata from one run directory.

    The preferred source is ``resolved_config.yaml`` because it stores the actual
    settings used for the run. The run folder name is used as a fallback.
    """
    progress_path = run_dir / "progress.csv"
    if not progress_path.exists():
        return None

    # The resolved config is written by the training pipeline before the run starts.
    config = read_simple_yaml(run_dir / "resolved_config.yaml")
    run_name = config.get("run_name") or run_dir.name
    size_match = SIZE_RE.search(run_name)
    backbone_match = BACKBONE_RE.search(run_name)

    # Prefer explicit config values, but fill missing actor/critic names from the folder name.
    actor = config.get("actor_backbone_type", "")
    critic = config.get("critic_backbone_type", "")
    if backbone_match:
        actor = actor or backbone_match.group("actor")
        critic = critic or backbone_match.group("critic")

    # Width, depth, and seed define how runs are grouped and averaged.
    width = parse_int(config.get("actor_hidden_size"))
    depth = parse_int(config.get("actor_num_hidden_layers"))
    seed = parse_int(config.get("seed"))
    if size_match:
        # Keep config values when present; otherwise use the values encoded in the run name.
        width = width or int(size_match.group("width"))
        depth = depth or int(size_match.group("depth"))
        seed = seed or int(size_match.group("seed"))

    if width is None or depth is None or seed is None:
        print(f"[skip] Could not infer width/depth/seed for: {run_dir}")
        return None

    return RunInfo(
        run_dir=run_dir,
        progress_path=progress_path,
        backbone=safe_part(backbone_from_parts(actor, critic)),
        width=width,
        depth=depth,
        seed=seed,
        run_name=run_name,
    )


def discover_runs(runs_root: Path) -> list[RunInfo]:
    """
    Find all valid runs below one environment folder.
    """
    runs = []
    for progress_path in sorted(runs_root.rglob("progress.csv")):
        info = run_info_from_dir(progress_path.parent)
        if info is not None:
            runs.append(info)

    # Sorting once here keeps later plots and output folders stable between runs.
    return sorted(runs, key=lambda run: (BACKBONE_ORDER.get(run.backbone, 99), run.backbone, run.depth, run.width, run.seed))


def backbone_subdir_name(backbone: str) -> str:
    """
    Prefix backbone folders with a number so file browsers show them in a useful order.
    """
    return f"{BACKBONE_ORDER.get(backbone, 99):02d}_{backbone}"


def output_dir_for_group(outdir: Path, backbone: str, depth: int) -> Path:
    """
    Build the output folder for one backbone-depth combination.
    """
    return outdir / backbone_subdir_name(backbone) / f"depth_{depth}"


def load_progress_frames(runs: list[RunInfo]) -> dict[Path, pd.DataFrame]:
    """
    Load the progress.csv file for each valid run.
    """
    frames = {}
    for run in runs:
        try:
            frames[run.progress_path] = pd.read_csv(run.progress_path)
        except Exception as exc:
            print(f"[skip] Could not read {run.progress_path}: {exc}")
    return frames


def group_frames_by_width(
    runs: list[RunInfo],
    frames: dict[Path, pd.DataFrame],
    metric: str,
    x_col: str,
) -> dict[str, list[pd.DataFrame]]:
    """
    Group run DataFrames by hidden width for one backbone-depth plot.

    The output becomes the input for the shared mean-curve plotter. Each width
    gets one color in the final plot, and all seeds for that width are averaged.
    """
    grouped: dict[str, list[pd.DataFrame]] = {}
    for run in sorted(runs, key=lambda item: (item.width, item.seed)):
        frame = frames.get(run.progress_path)
        if frame is None or x_col not in frame.columns or metric not in frame.columns:
            continue

        # Copy before adding metadata so the cached original frame stays unchanged.
        run_frame = frame.copy()
        # The shared plotting helper expects the seed to be stored in this column.
        run_frame["__seed__"] = run.seed
        grouped.setdefault(f"w={run.width}", []).append(run_frame)
    return grouped


def grouped_by_depth(runs: list[RunInfo]) -> dict[tuple[str, int], list[RunInfo]]:
    """
    Split runs into separate groups for each backbone and depth.
    """
    groups: dict[tuple[str, int], list[RunInfo]] = defaultdict(list)
    for run in runs:
        groups[(run.backbone, run.depth)].append(run)
    return groups


def main() -> None:
    """
    CLI entry point for creating all depth/width aggregate plots.
    """
    args = parse_args()
    runs_root = resolve_path(args.runs_root)
    outdir = resolve_path(args.outdir) if args.outdir else runs_root / "aggregated_by_depth_plots"

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    runs = discover_runs(runs_root)
    if not runs:
        raise FileNotFoundError(f"No progress.csv files found below: {runs_root}")

    frames = load_progress_frames(runs)
    # If the user did not request specific metrics, infer plottable metrics from CSV headers.
    metrics = available_plot_metrics_from_files([run.progress_path for run in runs], args.metrics)
    if not metrics:
        raise ValueError("No plottable metrics found.")

    created = 0
    for (backbone, depth), group_runs in sorted(
        grouped_by_depth(runs).items(),
        key=lambda item: (BACKBONE_ORDER.get(item[0][0], 99), item[0][0], item[0][1]),
    ):
        target_dir = output_dir_for_group(outdir, backbone, depth)
        for metric in metrics:
            # For this plot, the colored groups are hidden widths.
            width_runs = group_frames_by_width(group_runs, frames, metric, args.x_col)
            if not width_runs:
                continue

            output_stem = f"{metric}__d{depth}"
            if args.dry_run:
                # Dry-run mode is useful on large folders to check what would be created.
                seed_count = sum(len(runs_for_width) for runs_for_width in width_runs.values())
                target_paths = aggregate_plot_paths(target_dir, metric, output_stem, DEFAULT_PLOT_FORMATS)
                print(f"[plot] {', '.join(str(path) for path in target_paths)} ({seed_count} run(s), metric={metric})")
                created += 1
                continue

            # The heavy lifting is done by the shared aggregate plotting helper.
            saved_paths = plot_aggregated_metric(
                env_id=f"{backbone}: depth {depth}",
                grouped_runs=width_runs,
                metric=metric,
                outdir=target_dir,
                x_axis=args.x_col,
                smooth_window=args.smooth_window,
                dpi=args.dpi,
                legend_title="Width",
                title=f"{backbone}: depth {depth} - {metric}",
                output_stem=output_stem,
            )
            if saved_paths:
                created += 1

    file_count = created * len(DEFAULT_PLOT_FORMATS)
    print(f"{'Would create' if args.dry_run else 'Created'} {created} aggregate plot set(s) ({file_count} file(s)) in: {outdir}")


if __name__ == "__main__":
    main()
