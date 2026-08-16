"""Evaluate trained KAN/LAN models and plot spline/base ratio distributions.

This analysis uses trained size-experiment models to inspect how strongly the
spline branch contributes relative to the base branch inside KAN/LAN layers.
It creates the intermediate CSV files and distribution plots used for the
internal-behaviour part of the thesis:

1. find trained size-experiment models for selected depth/backbone settings
2. collect evaluation observations from each model
3. run those observations through actor and critic backbones
4. extract per-layer spline/base ratio values
5. write raw values, summaries, and a layer/seed table
6. create histograms, ridgelines, and boxplots by environment/backbone/network

The script can reuse cached ratio CSVs unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.evaluate_size_experiment_1 import (
    DEFAULT_RUNS_ROOT,
    RunInfo,
    discover_runs,
    ensure_dependencies,
)
from analysis_layer.utils import (
    DEFAULT_PLOT_FORMATS,
    register_legacy_module_aliases,
    resolve_analysis_path as resolve_cli_path,
    sanitize_filename,
    save_plot_variants,
    timestamp,
    write_csv,
)
from utility_layer.paths import OUTPUT_ROOT
from utility_layer.plotting.boxplot import plot_boxplot
from utility_layer.plotting.histogram import plot_grouped_histogram
from utility_layer.plotting.ridgeline import plot_ridgeline

PLOT_FORMATS = DEFAULT_PLOT_FORMATS
DEFAULT_OUTDIR = OUTPUT_ROOT / "3_Spline_to_base_analysis"
VALUES_CSV = "spline_base_ratio_values.csv"
SUMMARY_CSV = "spline_base_ratio_summary.csv"
LAYER_SEED_TABLE_CSV = "spline_base_ratio_layer_seed_table.csv"

# The analysis only applies to backbones that expose spline/base diagnostics.
DEFAULT_BACKBONES = ("kan", "lan")

# These columns identify one ratio distribution in the summary output.
METADATA_COLUMNS = [
    "env_id",
    "backbone",
    "network",
    "hidden_size",
    "depth",
    "seed",
    "run_name",
    "layer",
]

# The layer table is indexed by architecture position and then expanded by env.
LAYER_TABLE_INDEX_COLUMNS = ["backbone", "network", "depth", "hidden_size", "layer"]
ENV_COLUMN_ORDER = ("Ant-v5", "Walker2d-v5", "HalfCheetah-v5")
AxisLimits = tuple[float, float]


# CLI setup
#--------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Read CLI options for spline/base ratio collection and plotting.
    """
    parser = argparse.ArgumentParser(
        description="Measure concrete spline/base ratio distributions for trained size-experiment KAN/LAN models."
    )
    parser.add_argument("--runs-root", type=str, default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--max-observations", type=int, default=512, help="0 means collect all observations.")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--force", action="store_true", help="Recollect ratio values even when cached CSVs exist.")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--density", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--histogram-scale", choices=["linear", "log10"], default="log10", help="Scale used for histograms, ridgelines, and boxplots.")
    parser.add_argument("--log-eps", type=float, default=1e-12)
    parser.add_argument("--per-layer-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


# Run filtering and metadata
#-------------------------------------


def filter_runs(runs: Sequence[RunInfo], args: argparse.Namespace) -> list[RunInfo]:
    """
    Keep only trained runs that match the selected internal-analysis setup.
    """
    envs = set(args.envs or [])
    backbones = set(args.backbones or [])
    hidden_sizes = set(args.hidden_sizes or [])
    filtered = [
        run
        for run in runs
        if run.depth == args.depth
        and run.actor_backbone == run.critic_backbone
        and run.actor_backbone in backbones
        and (not envs or run.env_id in envs)
        and (not hidden_sizes or run.hidden_size in hidden_sizes)
    ]
    return filtered[: args.max_runs] if args.max_runs > 0 else filtered


def run_metadata(run: RunInfo) -> dict[str, Any]:
    """
    Convert RunInfo into metadata columns shared by raw and error rows.
    """
    return {
        "run_name": run.run_name,
        "run_dir": str(run.run_dir.resolve()),
        "model_path": str(run.model_path.resolve()),
        "env_id": run.env_id,
        "backbone": run.actor_backbone,
        "hidden_size": run.hidden_size,
        "depth": run.depth,
        "seed": run.seed,
    }


# Observation collection and model evaluation
#----------------------------------------------------


def collect_observations(model: Any, run: RunInfo, args: argparse.Namespace) -> np.ndarray:
    """
    Run a trained policy and collect observations used for ratio inspection.
    """
    import gymnasium as gym

    observations = []
    env = gym.make(run.env_id)
    try:
        for episode_idx in range(args.n_eval_episodes):
            # Reuse fixed evaluation seeds so models see comparable states.
            obs, _info = env.reset(seed=args.eval_seed + episode_idx)
            env.action_space.seed(args.eval_seed + episode_idx)
            done = False
            episode_length = 0
            while not done:
                # Store the current observation before the policy advances the env.
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                if args.max_observations > 0 and len(observations) >= args.max_observations:
                    return np.asarray(observations, dtype=np.float32)

                action, _state = model.predict(obs, deterministic=args.deterministic)
                obs, _reward, terminated, truncated, _info = env.step(action)
                episode_length += 1
                # max_episode_steps is optional and mainly useful for quick test runs.
                hit_limit = args.max_episode_steps > 0 and episode_length >= args.max_episode_steps
                done = terminated or truncated or hit_limit
    finally:
        env.close()

    return np.asarray(observations, dtype=np.float32)


def policy_features(model: Any, observations: np.ndarray) -> tuple[Any, Any]:
    """
    Convert collected observations into actor and critic feature tensors.
    """
    import torch

    if observations.size == 0:
        raise ValueError("No observations collected.")
    obs_tensor, _ = model.policy.obs_to_tensor(observations)
    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor)
    # SB3 feature extractors may share one feature tensor or return actor/critic tensors.
    if isinstance(features, tuple):
        return features
    return features, features


def ratio_rows_for_network(run: RunInfo, network_name: str, backbone: Any, features: Any, n_observations: int) -> list[dict[str, Any]]:
    """
    Extract all layer-wise spline/base ratio rows from one actor or critic net.
    """
    collect_layer_values = getattr(backbone, "spline_base_ratio_values_by_layer", None)
    if collect_layer_values is None:
        return []

    rows = []
    for layer_values in collect_layer_values(features):
        # Backbones return tensors; CSV rows need CPU scalar values.
        ratio = layer_values["ratio"].detach().cpu().numpy()
        base_abs = layer_values["base_abs"].detach().cpu().numpy()
        spline_abs = layer_values["spline_abs"].detach().cpu().numpy()
        for value_index, (ratio_value, base_value, spline_value) in enumerate(zip(ratio, base_abs, spline_abs, strict=True)):
            rows.append(
                {
                    **run_metadata(run),
                    "status": "ok",
                    "error": "",
                    "timestamp": timestamp(),
                    "network": network_name,
                    "layer": int(layer_values["layer"]),
                    "unit_type": str(layer_values["unit_type"]),
                    "value_index": value_index,
                    "n_observations": n_observations,
                    "ratio_kind": "abs_spline_over_abs_base",
                    "ratio": float(ratio_value),
                    "base_abs": float(base_value),
                    "spline_abs": float(spline_value),
                }
            )
    return rows


def evaluate_run(run: RunInfo, args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    Load one trained PPO model and collect actor/critic ratio rows.
    """
    from stable_baselines3 import PPO

    register_legacy_module_aliases()
    model = PPO.load(str(run.model_path), device=args.device)
    model.policy.eval()
    observations = collect_observations(model, run, args)
    actor_features, critic_features = policy_features(model, observations)
    extractor = model.policy.mlp_extractor

    # Actor and critic are evaluated separately because their backbones can differ.
    rows = []
    rows.extend(ratio_rows_for_network(run, "actor", extractor.actor_net, actor_features, len(observations)))
    rows.extend(ratio_rows_for_network(run, "critic", extractor.critic_net, critic_features, len(observations)))
    return rows


def error_row(run: RunInfo, error: Exception) -> dict[str, Any]:
    """
    Build a CSV row for a run that failed during ratio collection.
    """
    return {
        **run_metadata(run),
        "status": "error",
        "error": repr(error),
        "timestamp": timestamp(),
        "network": "",
        "layer": np.nan,
        "unit_type": "",
        "value_index": np.nan,
        "n_observations": 0,
        "ratio_kind": "abs_spline_over_abs_base",
        "ratio": np.nan,
        "base_abs": np.nan,
        "spline_abs": np.nan,
    }


def collect_ratio_results(runs: Sequence[RunInfo], args: argparse.Namespace) -> pd.DataFrame:
    """
    Evaluate all selected runs and return one raw ratio table.
    """
    rows = []
    for index, run in enumerate(runs, start=1):
        print(f"[eval] {index}/{len(runs)} {run.env_id} {run.actor_backbone} w={run.hidden_size} d={run.depth} seed={run.seed}")
        try:
            # Each successful run contributes many rows, one per sampled unit/value.
            run_rows = evaluate_run(run, args)
            rows.extend(run_rows)
            print(f"[ok] collected {len(run_rows)} ratio value(s)")
        except Exception as exc:
            rows.append(error_row(run, exc))
            print(f"[error] {run.run_name}: {exc}")
    return pd.DataFrame(rows)


# DataFrame and CSV summaries
#------------------------------------------------


def finite_ok_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep successful rows with finite ratio values.
    """
    if df.empty:
        return df.copy()
    ok = df[df["status"].eq("ok")].copy()
    for column in ["ratio", "base_abs", "spline_abs"]:
        ok[column] = pd.to_numeric(ok[column], errors="coerce")
    return ok[np.isfinite(ok["ratio"])]


def summarize_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each layer-level ratio distribution with moments and quantiles.
    """
    ok = finite_ok_values(df)
    if ok.empty:
        return pd.DataFrame()

    # Basic statistics are computed per run, network, and layer.
    grouped = ok.groupby(METADATA_COLUMNS, as_index=False)
    summary = grouped.agg(
        count=("ratio", "count"),
        mean_ratio=("ratio", "mean"),
        median_ratio=("ratio", "median"),
        std_ratio=("ratio", "std"),
        min_ratio=("ratio", "min"),
        max_ratio=("ratio", "max"),
        mean_base_abs=("base_abs", "mean"),
        mean_spline_abs=("spline_abs", "mean"),
        fraction_ratio_gt_1=("ratio", lambda values: float((values > 1.0).mean())),
        fraction_ratio_gt_10=("ratio", lambda values: float((values > 10.0).mean())),
        fraction_ratio_gt_100=("ratio", lambda values: float((values > 100.0).mean())),
    )
    # Tail quantiles are useful because ratio distributions can be highly skewed.
    quantiles = ok.groupby(METADATA_COLUMNS)["ratio"].quantile([0.1, 0.25, 0.75, 0.9, 0.95, 0.99]).unstack()
    quantiles = quantiles.rename(
        columns={
            0.1: "q10_ratio",
            0.25: "q25_ratio",
            0.75: "q75_ratio",
            0.9: "q90_ratio",
            0.95: "q95_ratio",
            0.99: "q99_ratio",
        }
    ).reset_index()
    return summary.merge(quantiles, on=METADATA_COLUMNS, how="left").sort_values(METADATA_COLUMNS).reset_index(drop=True)


def layer_seed_mean_table(values: pd.DataFrame, log_eps: float) -> pd.DataFrame:
    """
    Build a compact table of mean log10 ratios by layer and environment.
    """
    if values.empty:
        return pd.DataFrame()
    if log_eps <= 0:
        raise ValueError("log_eps must be > 0 for the layer seed mean table.")

    required = [*LAYER_TABLE_INDEX_COLUMNS, "env_id", "seed", "ratio"]
    ok = finite_ok_values(values)
    if ok.empty or any(column not in ok.columns for column in required):
        return pd.DataFrame()

    table_data = ok[required].copy()
    table_data["ratio"] = pd.to_numeric(table_data["ratio"], errors="coerce")
    table_data = table_data[np.isfinite(table_data["ratio"])]
    if table_data.empty:
        return pd.DataFrame()

    # Layer numbers are shifted to one-based indexing for report-facing tables.
    table_data["layer"] = table_data["layer"].astype(int) + 1
    table_data["log10_ratio"] = np.log10(table_data["ratio"].clip(lower=0.0) + log_eps)
    aggregated = table_data.groupby([*LAYER_TABLE_INDEX_COLUMNS, "env_id"], as_index=False).agg(
        mean_log10_ratio=("log10_ratio", "mean"),
        std_log10_ratio=("log10_ratio", "std"),
        n_values=("log10_ratio", "count"),
        n_seeds=("seed", "nunique"),
    )

    # Keep the main MuJoCo environments in a stable, readable order.
    envs = list(ENV_COLUMN_ORDER) + sorted(set(aggregated["env_id"]) - set(ENV_COLUMN_ORDER))
    envs = [env for env in envs if env in set(aggregated["env_id"])]
    table = aggregated[LAYER_TABLE_INDEX_COLUMNS].drop_duplicates().set_index(LAYER_TABLE_INDEX_COLUMNS).sort_index()
    for env_id in envs:
        env_data = aggregated[aggregated["env_id"].eq(env_id)].set_index(LAYER_TABLE_INDEX_COLUMNS)
        table[f"{env_id}_mean_log10_ratio"] = env_data["mean_log10_ratio"]
        table[f"{env_id}_std_log10_ratio"] = env_data["std_log10_ratio"]
        table[f"{env_id}_n_values"] = env_data["n_values"]
        table[f"{env_id}_n_seeds"] = env_data["n_seeds"]
    return table.reset_index()


# Plot preparation
#-------------------------------------------


def grouped_ratio_values(df: pd.DataFrame, group_columns: Sequence[str]) -> dict[tuple[Any, ...], dict[str, list[float]]]:
    """
    Group ratio values for distribution plots, using hidden size as plot groups.
    """
    grouped = {}
    for key, part in df.groupby(list(group_columns)):
        key_tuple = key if isinstance(key, tuple) else (key,)
        hidden_groups = {}
        for hidden_size, hidden_part in part.groupby("hidden_size"):
            hidden_groups[str(int(hidden_size))] = hidden_part["ratio"].astype(float).tolist()
        grouped[key_tuple] = hidden_groups
    return grouped


def save_histogram_variants(
    groups: dict[str, list[float]],
    outdir: Path,
    stem: str,
    *,
    title: str,
    x_label: str,
    bins: int,
    density: bool,
    x_limits: AxisLimits | None,
    dpi: int,
) -> list[Path]:
    """
    Save grouped histogram variants for one ratio distribution.
    """
    return save_plot_variants(
        plot_grouped_histogram,
        outdir,
        stem,
        "ratio histogram",
        plot_formats=PLOT_FORMATS,
        groups=groups,
        title=title,
        x_label=x_label,
        y_label="Density" if density else "Count",
        bins=bins,
        density=density,
        x_limits=x_limits,
        dpi=dpi,
    )


def transformed_plot_values(ok: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str, str]:
    """
    Apply the requested plot scale and return filename/axis labels for it.
    """
    plot_df = ok.copy()
    scale_suffix = "linear"
    x_label = "Spline/base ratio"
    if args.histogram_scale == "log10":
        if args.log_eps <= 0:
            raise ValueError("--log-eps must be > 0 for log10 histograms.")
        # Add a tiny epsilon so zero ratios remain plottable on log10 scale.
        plot_df["ratio"] = np.log10(plot_df["ratio"].clip(lower=0.0) + args.log_eps)
        scale_suffix = "log10"
        x_label = f"log10(spline/base ratio + {args.log_eps:g})"
    return plot_df, scale_suffix, x_label


def ratio_axis_limits_by_network(plot_df: pd.DataFrame) -> dict[tuple[str, str], AxisLimits]:
    """
    Compute common ratio-axis limits for each backbone/network pair.
    """
    limits = {}
    for (backbone, network), part in plot_df.groupby(["backbone", "network"]):
        values = part["ratio"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        lower = float(values.min())
        upper = float(values.max())
        # Use a small margin so the extreme values are not drawn against the frame.
        margin = max((upper - lower) * 0.04, 0.05) if lower != upper else max(abs(lower) * 0.05, 1.0)
        limits[(str(backbone), str(network))] = (lower - margin, upper + margin)
    return limits


def save_ridgeline_variants(
    groups: dict[str, list[float]],
    outdir: Path,
    stem: str,
    *,
    title: str,
    x_label: str,
    bins: int,
    x_limits: AxisLimits | None,
    dpi: int,
) -> list[Path]:
    """
    Save ridgeline variants for one ratio distribution.
    """
    return save_plot_variants(
        plot_ridgeline,
        outdir,
        stem,
        "ratio ridgeline",
        plot_formats=PLOT_FORMATS,
        groups=groups,
        title=title,
        x_label=x_label,
        bins=bins,
        x_limits=x_limits,
        dpi=dpi,
    )


def save_boxplot_variants(
    groups: dict[str, list[float]],
    outdir: Path,
    stem: str,
    *,
    title: str,
    y_label: str,
    y_limits: AxisLimits | None,
    dpi: int,
) -> list[Path]:
    """
    Save boxplot variants for one ratio distribution.
    """
    return save_plot_variants(
        plot_boxplot,
        outdir,
        stem,
        "ratio boxplot",
        plot_formats=PLOT_FORMATS,
        groups=groups,
        title=title,
        x_label="Hidden size",
        y_label=y_label,
        dpi=dpi,
        show_points=False,
        y_limits=y_limits,
    )


def save_distribution_plot_variants(
    groups: dict[str, list[float]],
    outdir: Path,
    stem_prefix: str,
    *,
    title: str,
    axis_label: str,
    bins: int,
    density: bool,
    ratio_limits: AxisLimits | None,
    dpi: int,
) -> list[Path]:
    """
    Save every distribution-plot type for one grouped ratio distribution.
    """
    paths = []
    paths.extend(
        save_histogram_variants(
            groups,
            outdir,
            f"{stem_prefix}_histogram",
            title=title,
            x_label=axis_label,
            bins=bins,
            density=density,
            x_limits=ratio_limits,
            dpi=dpi,
        )
    )
    paths.extend(
        save_ridgeline_variants(
            groups,
            outdir,
            f"{stem_prefix}_ridgeline",
            title=title,
            x_label=axis_label,
            bins=bins,
            x_limits=ratio_limits,
            dpi=dpi,
        )
    )
    paths.extend(
        save_boxplot_variants(
            groups,
            outdir,
            f"{stem_prefix}_boxplot",
            title=title,
            y_label=axis_label,
            y_limits=ratio_limits,
            dpi=dpi,
        )
    )
    return paths


def create_distribution_plots(df: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> list[Path]:
    """
    Create all ratio distribution plots from the raw ratio table.
    """
    ok = finite_ok_values(df)
    if ok.empty:
        print("[warn] No finite ratio values available for plotting.")
        return []

    # Plot values may be transformed, but raw CSV values remain unchanged.
    plot_df, scale_suffix, axis_label = transformed_plot_values(ok, args)
    axis_limits = ratio_axis_limits_by_network(plot_df)
    paths = []
    # The main plots aggregate all layers for each environment/backbone/network.
    for (env_id, backbone, network), groups in grouped_ratio_values(plot_df, ["env_id", "backbone", "network"]).items():
        target_dir = outdir / sanitize_filename(env_id) / sanitize_filename(backbone) / sanitize_filename(network)
        ratio_limits = axis_limits.get((str(backbone), str(network)))
        paths.extend(
            save_distribution_plot_variants(
                groups,
                target_dir,
                f"depth{args.depth}_spline_base_ratio_{scale_suffix}",
                title=f"{env_id}: {backbone.upper()} {network}",
                axis_label=axis_label,
                bins=args.bins,
                density=args.density,
                ratio_limits=ratio_limits,
                dpi=args.dpi,
            )
        )

    if args.per_layer_plots:
        # Per-layer plots use the same axis limits as the corresponding full-network plot.
        for (env_id, backbone, network, layer), groups in grouped_ratio_values(plot_df, ["env_id", "backbone", "network", "layer"]).items():
            target_dir = outdir / sanitize_filename(env_id) / sanitize_filename(backbone) / sanitize_filename(network)
            ratio_limits = axis_limits.get((str(backbone), str(network)))
            paths.extend(
                save_distribution_plot_variants(
                    groups,
                    target_dir,
                    f"depth{args.depth}_layer{int(layer)}_spline_base_ratio_{scale_suffix}",
                    title=f"{env_id}: {backbone.upper()} {network} layer {int(layer)}",
                    axis_label=axis_label,
                    bins=args.bins,
                    density=args.density,
                    ratio_limits=ratio_limits,
                    dpi=args.dpi,
                )
            )
    return paths


# Pipeline orchestration
#-------------------------------------------------


def load_or_collect_values(args: argparse.Namespace, runs_root: Path, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cached ratio CSVs or collect fresh values from trained models.
    """
    values_path = outdir / VALUES_CSV
    summary_path = outdir / SUMMARY_CSV

    loaded_values_from_cache = values_path.exists() and not args.force
    if loaded_values_from_cache:
        # Raw values are the expensive artifact, so they control cache reuse.
        values = pd.read_csv(values_path)
        print(f"[cache] loaded spline/base ratio values: {values_path}")
    else:
        # Reuse size-experiment run discovery, then narrow it to this analysis.
        runs = filter_runs(discover_runs(runs_root, max_runs=0), args)
        if not runs:
            raise FileNotFoundError(f"No matching depth-{args.depth} KAN/LAN size-experiment models found below: {runs_root}")

        print(f"Found {len(runs)} matching run(s) below: {runs_root}")
        values = collect_ratio_results(runs, args)
        write_csv(values, values_path, "spline/base ratio values")

    if summary_path.exists() and loaded_values_from_cache:
        # The summary is valid when it was derived from the cached raw values.
        summary = pd.read_csv(summary_path)
        print(f"[cache] using spline/base ratio summary: {summary_path}")
    else:
        summary = summarize_ratios(values)
        write_csv(summary, summary_path, "spline/base ratio summary")

    write_csv(layer_seed_mean_table(values, args.log_eps), outdir / LAYER_SEED_TABLE_CSV, "layer seed mean table")
    return values, summary


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point for the spline/base ratio analysis pipeline.
    """
    args = parse_args(argv)
    if args.n_eval_episodes <= 0:
        raise ValueError("--n-eval-episodes must be > 0")
    if args.bins <= 0:
        raise ValueError("--bins must be > 0")

    ensure_dependencies()
    runs_root = resolve_cli_path(args.runs_root)
    outdir = resolve_cli_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Values and summaries are generated before plots so plotting can be repeated cheaply.
    values, _summary = load_or_collect_values(args, runs_root, outdir)

    if not args.no_plot:
        paths = create_distribution_plots(values, outdir, args)
        print(f"Created {len(paths)} distribution plot file(s) in: {outdir}")


if __name__ == "__main__":
    main()
