"""Evaluate trained size-experiment models and create summary plots.

The size experiment produces many trained PPO models. Each model differs in
environment, backbone type, hidden width, depth, and seed. This script turns
those finished runs into the result files used for the thesis:

1. find all trained `model.zip` files below one experiment folder
2. load every model and evaluate it for a fixed number of episodes
3. read training time and memory usage from `progress.csv`
4. load every model again to count trainable policy parameters
5. aggregate all seed-level values by environment, backbone, width, and depth
6. generate 3D scaling plots, 2D scaling plots, and reward winner heatmaps

The script can reuse existing CSV files unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.utils import (
    max_numeric_column,
    parse_int,
    register_legacy_module_aliases,
    resolve_analysis_path as resolve_cli_path,
    sanitize_filename,
    save_plot_variants,
    timestamp,
    write_csv,
)
from utility_layer.paths import OUTPUT_ROOT
from utility_layer.plotting.bar_graph_2d import plot_size_depth_bars
from utility_layer.plotting.graph_3d import plot_size_depth_points_3d
from utility_layer.plotting.heatmap import plot_best_category_heatmap

DEFAULT_RUNS_ROOT = OUTPUT_ROOT / "1_Size_experiment"

# Size-experiment run names contain width, depth, and seed in a compact suffix.
# This is used as a fallback when the resolved config is missing or incomplete.
SIZE_RUN_RE = re.compile(r"-w(?P<hidden_size>\d+)-d(?P<depth>\d+)-seed(?P<seed>\d+)")

# Only these three standard architecture families are used for the thesis plots.
BACKBONES = ("mlp", "lan", "kan")

# The plotting utility receives explicit colors and markers so all figures use
# the same visual identity for each architecture.
BACKBONE_COLORS = {"mlp": "#1f77b4", "lan": "#2ca02c", "kan": "#ff7f0e"}
BACKBONE_MARKERS = {"mlp": "o", "lan": "s", "kan": "^"}

# These columns identify one experimental condition across all generated CSVs.
GROUP_COLUMNS = ["env_id", "actor_backbone", "critic_backbone", "hidden_size", "depth"]
SORT_COLUMNS = GROUP_COLUMNS + ["seed", "run_name"]

PLOT_SPECS = [
    # Each entry describes one measured quantity and how it should appear in plots.
    {
        "mean_col": "mean_reward",
        "std_col": "std_reward_across_seeds",
        "z_label": "Mean evaluation reward",
        "title": "evaluation reward",
        "suffix": "reward",
    },
    {
        "mean_col": "mean_training_time_hours",
        "std_col": "std_training_time_hours",
        "z_label": "Mean training time (hours)",
        "title": "training time",
        "suffix": "training_time_hours",
    },
    {
        "mean_col": "mean_max_memory_mb",
        "std_col": "std_max_memory_mb",
        "z_label": "Mean max memory usage (MB)",
        "title": "max memory usage",
        "suffix": "max_memory_mb",
    },
    {
        "mean_col": "mean_policy_trainable_parameters",
        "std_col": "std_policy_trainable_parameters",
        "z_label": "Mean trainable policy parameters",
        "title": "trainable parameter count",
        "suffix": "trainable_policy_parameters",
    },
]

np: Any = None
pd: Any = None
yaml: Any = None



# Data model
#--------------------------------------


@dataclass(frozen=True)
class RunInfo:
    """Metadata needed to evaluate and aggregate one trained run."""

    run_dir: Path
    model_path: Path
    progress_path: Path | None
    config_path: Path | None
    run_name: str
    env_id: str
    seed: int
    actor_backbone: str
    critic_backbone: str
    hidden_size: int
    depth: int



# CLI and dependency setup
#--------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Read CLI options for the size-experiment evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate size-experiment PPO models and create MLP/LAN/KAN plots.")
    parser.add_argument("--runs-root", type=str, default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--outdir", type=str, default="", help="Defaults to <runs-root>/evaluation.")
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--x-scale", type=str, default="log2", choices=["linear", "log2"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--view-elev", type=float, default=28.0)
    parser.add_argument("--view-azim", type=float, default=-135.0)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def ensure_dependencies() -> None:
    """
    Import analysis dependencies only when the script actually runs.
    """
    global np, pd, yaml
    if np is not None:
        return
    try:
        import numpy as numpy_module
        import pandas as pandas_module
        import yaml as yaml_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing analysis dependency. Activate the project environment first.") from exc
    np = numpy_module
    pd = pandas_module
    yaml = yaml_module



# Generic parsing and formatting helpers
#-------------------------------------


def normalize_backbone(value: Any) -> str:
    """
    Convert backbone names from configs or CSVs into a consistent lowercase form.
    """
    text = "" if value is None else str(value).strip().lower()
    return "unknown" if not text or text in {"nan", "none", "null"} else text


def display_backbone(value: Any) -> str:
    """
    Convert an internal backbone id into a label suitable for plot legends.
    """
    backbone = normalize_backbone(value)
    return backbone.upper() if backbone in BACKBONES else backbone.replace("_", " ")


def parse_backbones_from_name(run_name: str) -> tuple[str, str]:
    """
    Extract actor and critic backbone names from a standard run name.
    """
    actor = ""
    critic = ""
    for part in run_name.split("__"):
        if part.startswith("actor-"):
            actor = part.removeprefix("actor-")
        elif part.startswith("critic-"):
            critic = part.removeprefix("critic-")
    return actor, critic


def load_yaml_config(path: Path) -> dict[str, Any]:
    """
    Load one resolved_config.yaml file.

    If the file is missing or does not contain a dictionary, return an empty
    dictionary so the caller can fall back to parsing the run folder name.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def base_run_row(run: RunInfo) -> dict[str, Any]:
    """
    Convert RunInfo into the shared metadata columns used by all output CSVs.
    """
    return {
        "run_name": run.run_name,
        "run_dir": str(run.run_dir.resolve()),
        "env_id": run.env_id,
        "seed": run.seed,
        "actor_backbone": run.actor_backbone,
        "critic_backbone": run.critic_backbone,
        "hidden_size": run.hidden_size,
        "depth": run.depth,
    }


# Run discovery
#-------------------------------------


def build_run_info(run_dir: Path) -> RunInfo | None:
    """
    Build RunInfo for one run folder if it contains a trained model.

    The resolved config is the preferred source for metadata because it stores
    what was actually used during training. The run folder name is only a
    fallback.
    """
    model_path = run_dir / "model.zip"
    if not model_path.exists():
        return None

    # Read metadata written by the training pipeline.
    config_path = run_dir / "resolved_config.yaml"
    config = load_yaml_config(config_path)
    run_name = str(config.get("run_name") or run_dir.name)
    size_match = SIZE_RUN_RE.search(run_name)

    # Actor/critic names may be present in config or encoded in the folder name.
    actor = normalize_backbone(config.get("actor_backbone_type"))
    critic = normalize_backbone(config.get("critic_backbone_type"))
    if actor == "unknown" or critic == "unknown":
        parsed_actor, parsed_critic = parse_backbones_from_name(run_name)
        actor = normalize_backbone(parsed_actor or actor)
        critic = normalize_backbone(parsed_critic or critic)

    # Width, depth, and seed are needed to aggregate across the five seeds.
    hidden_size = parse_int(config.get("actor_hidden_size"))
    depth = parse_int(config.get("actor_num_hidden_layers"))
    seed = parse_int(config.get("seed"))
    if size_match is not None:
        # Keep config values when they exist; otherwise use values from the run name.
        hidden_size = hidden_size or int(size_match.group("hidden_size"))
        depth = depth or int(size_match.group("depth"))
        seed = seed or int(size_match.group("seed"))

    env_id = str(config.get("env_id") or run_name.split("__")[0])
    if hidden_size is None or depth is None or seed is None or not env_id:
        print(f"[skip] Could not infer size/depth/seed/env for: {run_dir}")
        return None

    progress_path = run_dir / "progress.csv"
    return RunInfo(
        run_dir=run_dir,
        model_path=model_path,
        progress_path=progress_path if progress_path.exists() else None,
        config_path=config_path if config_path.exists() else None,
        run_name=run_name,
        env_id=env_id,
        seed=seed,
        actor_backbone=actor,
        critic_backbone=critic,
        hidden_size=hidden_size,
        depth=depth,
    )


def discover_runs(runs_root: Path, max_runs: int = 0) -> list[RunInfo]:
    """
    Find all supported trained models below the size-experiment folder.

    Runs are filtered to the standard comparison setup: actor and critic use the
    same backbone, and the backbone is one of MLP, LAN, or KAN.
    """
    runs = []
    for model_path in sorted(runs_root.rglob("model.zip")):
        run = build_run_info(model_path.parent)
        if run is None:
            continue
        if run.actor_backbone != run.critic_backbone:
            continue
        if run.actor_backbone not in BACKBONES:
            continue
        runs.append(run)

    # Stable sorting makes CSV rows and generated plots predictable.
    runs.sort(key=lambda run: (run.env_id, BACKBONES.index(run.actor_backbone), run.hidden_size, run.depth, run.seed))
    return runs[:max_runs] if max_runs > 0 else runs



# DataFrame and CSV helpers
#------------------------------------------------


def normalize_architecture_columns(df: Any) -> Any:
    """
    Ensure backbone columns are present and consistently formatted.
    """
    if df.empty:
        return df.copy()
    result = df.copy()
    result["actor_backbone"] = result.get("actor_backbone", "unknown")
    result["critic_backbone"] = result.get("critic_backbone", "unknown")

    if "run_name" in result.columns:
        # Recover missing backbone labels from the run name when possible.
        missing = (result["actor_backbone"].map(normalize_backbone) == "unknown") | (result["critic_backbone"].map(normalize_backbone) == "unknown")
        for index, row in result[missing].iterrows():
            actor, critic = parse_backbones_from_name(str(row.get("run_name", "")))
            result.at[index, "actor_backbone"] = actor or row.get("actor_backbone", "unknown")
            result.at[index, "critic_backbone"] = critic or row.get("critic_backbone", "unknown")

    result["actor_backbone"] = result["actor_backbone"].map(normalize_backbone)
    result["critic_backbone"] = result["critic_backbone"].map(normalize_backbone)
    return result


def supported_only(df: Any) -> Any:
    """
    Keep only the standard MLP/LAN/KAN runs with matching actor and critic.
    """
    if df.empty:
        return df.copy()
    df = normalize_architecture_columns(df)
    return df[df["actor_backbone"].isin(BACKBONES) & (df["actor_backbone"] == df["critic_backbone"])].copy()


def coerce_numeric(df: Any, columns: list[str]) -> Any:
    """
    Convert selected DataFrame columns to numeric values.
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def sort_rows(df: Any) -> Any:
    """
    Sort result rows in the same order across all output CSV files.
    """
    columns = [column for column in SORT_COLUMNS if column in df.columns]
    return df.sort_values(columns).reset_index(drop=True) if columns else df.reset_index(drop=True)


def read_csv_or_empty(path: Path) -> Any:
    """
    Read a cached CSV if it exists, otherwise return an empty DataFrame.
    """
    return supported_only(pd.read_csv(path)) if path.exists() else pd.DataFrame()


def cached_ok_by_run(path: Path) -> dict[str, dict[str, Any]]:
    """
    Load successful cached rows by run name.
    """
    if not path.exists():
        return {}
    df = supported_only(pd.read_csv(path))
    if "status" not in df.columns:
        return {}
    return {str(row["run_name"]): row for row in df.to_dict(orient="records") if row.get("status") == "ok"}


# Reward evaluation
#----------------------------------------------------


def evaluate_model(run: RunInfo, args: argparse.Namespace) -> tuple[list[float], list[int]]:
    """
    Load one trained PPO model and run deterministic evaluation episodes.

    The returned reward list contains one total episode reward per evaluation
    episode. The length list stores how many environment steps each episode took.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO

    register_legacy_module_aliases()
    model = PPO.load(str(run.model_path), device=args.device)
    model.policy.eval()
    env = gym.make(run.env_id)
    rewards: list[float] = []
    lengths: list[int] = []
    try:
        for episode_idx in range(args.n_eval_episodes):
            # Use fixed evaluation seeds so different trained models see comparable starts.
            episode_seed = args.eval_seed + episode_idx
            obs, _info = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            done = False
            episode_reward = 0.0
            episode_length = 0
            while not done:
                # model.predict returns the policy action for the current observation.
                action, _state = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, _info = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                # max_episode_steps is optional and mainly useful for quick test runs.
                hit_limit = args.max_episode_steps > 0 and episode_length >= args.max_episode_steps
                done = terminated or truncated or hit_limit
            rewards.append(episode_reward)
            lengths.append(episode_length)
    finally:
        env.close()
    return rewards, lengths


def evaluation_row(run: RunInfo, args: argparse.Namespace, rewards: list[float] | None = None, lengths: list[int] | None = None, error: str = "") -> dict[str, Any]:
    """
    Turn one model's evaluation result into one CSV row.
    """
    rewards = rewards or []
    lengths = lengths or []
    reward_array = np.asarray(rewards, dtype=float)
    length_array = np.asarray(lengths, dtype=float)
    return {
        **base_run_row(run),
        "status": "ok" if rewards and not error else "error",
        "error": error,
        "timestamp": timestamp(),
        "model_path": str(run.model_path.resolve()),
        "config_path": str(run.config_path.resolve()) if run.config_path else "",
        "n_episodes": args.n_eval_episodes,
        "eval_seed": args.eval_seed,
        "deterministic": args.deterministic,
        "device": args.device,
        "max_episode_steps": args.max_episode_steps,
        # These summary columns are used for aggregate reward plots.
        "mean_reward": float(np.mean(reward_array)) if reward_array.size else np.nan,
        "std_reward": float(np.std(reward_array, ddof=1)) if reward_array.size > 1 else 0.0,
        "median_reward": float(np.median(reward_array)) if reward_array.size else np.nan,
        "min_reward": float(np.min(reward_array)) if reward_array.size else np.nan,
        "max_reward": float(np.max(reward_array)) if reward_array.size else np.nan,
        "mean_episode_length": float(np.mean(length_array)) if length_array.size else np.nan,
        # Store raw episode values as compact strings so the CSV remains one row per trained model.
        "episode_rewards": ";".join(f"{reward:.10g}" for reward in rewards),
        "episode_lengths": ";".join(str(length) for length in lengths),
    }


def evaluate_runs(args: argparse.Namespace, runs: list[RunInfo], out_path: Path) -> Any:
    """
    Evaluate all discovered runs and write the seed-level evaluation CSV.
    """
    # Existing successful rows are reused unless --force asks for recomputation.
    cache = {} if args.force else cached_ok_by_run(out_path)
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        if run.run_name in cache:
            rows.append(cache[run.run_name])
            print(f"[cache] {index}/{len(runs)} {run.actor_backbone} w={run.hidden_size} d={run.depth} seed={run.seed}")
            continue

        print(f"[eval] {index}/{len(runs)} {run.actor_backbone} w={run.hidden_size} d={run.depth} seed={run.seed}")
        try:
            # One row represents one trained seed model.
            rewards, lengths = evaluate_model(run, args)
            row = evaluation_row(run, args, rewards, lengths)
            print(f"[ok] mean_reward={row['mean_reward']:.2f} std={row['std_reward']:.2f}")
        except Exception as exc:
            row = evaluation_row(run, args, error=repr(exc))
            print(f"[error] {run.run_name}: {exc}")
        rows.append(row)

    df = sort_rows(normalize_architecture_columns(pd.DataFrame(rows)))
    write_csv(df, out_path, "evaluation results")
    return df



# Aggregation helpers
#-----------------------------------


def seed_list(values: Any) -> str:
    """
    Store the seeds that contributed to an aggregate row as one readable string.
    """
    return "|".join(str(seed) for seed in sorted({int(value) for value in values if not pd.isna(value)}))


def aggregate_ok_rows(
    df: Any,
    out_path: Path,
    label: str,
    numeric_columns: list[str],
    required_columns: list[str],
    aggregations: dict[str, tuple[str, str]],
    std_columns: list[str],
) -> Any:
    """
    Generic helper for turning seed-level rows into condition-level rows.

    The grouping columns define one architecture-size condition. The aggregation
    dictionary defines which means, standard deviations, and counts should be
    computed for that condition.
    """
    if df.empty or "status" not in df.columns:
        aggregate = pd.DataFrame()
    else:
        # Failed rows are useful for debugging but should not influence result means.
        ok = supported_only(df)
        ok = coerce_numeric(ok[ok["status"] == "ok"].copy(), numeric_columns)
        ok = ok.dropna(subset=required_columns)
        aggregate = (
            pd.DataFrame()
            if ok.empty
            else ok.groupby(GROUP_COLUMNS, as_index=False)
            .agg(**aggregations)
            .sort_values(GROUP_COLUMNS)
            .reset_index(drop=True)
        )
        for column in std_columns:
            if column in aggregate.columns:
                # A condition with only one seed has NaN std by definition; use 0 for plotting.
                aggregate[column] = aggregate[column].fillna(0.0)
    write_csv(aggregate, out_path, label)
    return aggregate


def aggregate_evaluation_results(df: Any, out_path: Path) -> Any:
    """
    Average final evaluation rewards across seeds for each architecture-size setup.
    """
    return aggregate_ok_rows(
        df,
        out_path,
        "evaluation aggregate",
        ["hidden_size", "depth", "seed", "mean_reward", "std_reward", "mean_episode_length"],
        ["hidden_size", "depth", "mean_reward"],
        {
            "mean_reward": ("mean_reward", "mean"),
            "std_reward_across_seeds": ("mean_reward", "std"),
            "min_seed_reward": ("mean_reward", "min"),
            "max_seed_reward": ("mean_reward", "max"),
            "mean_episode_std": ("std_reward", "mean"),
            "mean_episode_length": ("mean_episode_length", "mean"),
            "n_runs": ("run_name", "count"),
            "seeds": ("seed", seed_list),
        },
        ["std_reward_across_seeds"],
    )


def merge_aggregate(base: Any, extra: Any) -> Any:
    """
    Merge two aggregate tables on the shared architecture-size columns.
    """
    if extra.empty:
        return base
    if base.empty:
        return extra
    # If a column already exists, replace it with the newer value from extra.
    new_columns = [column for column in extra.columns if column not in GROUP_COLUMNS]
    return base.drop(columns=[column for column in new_columns if column in base.columns], errors="ignore").merge(extra, on=GROUP_COLUMNS, how="outer")



# Training resource extraction
#------------------------------------------------


def resource_row(run: RunInfo) -> dict[str, Any]:
    """
    Extract training time and peak memory usage from one run's progress.csv.
    """
    row = {
        **base_run_row(run),
        "status": "error",
        "error": "",
        "timestamp": timestamp(),
        "progress_path": str(run.progress_path.resolve()) if run.progress_path else "",
        "time_source_column": "",
        "memory_source_column": "",
        "training_time_hours": np.nan,
        "max_memory_mb": np.nan,
    }
    if run.progress_path is None:
        row["error"] = "progress.csv not found"
        return row

    df = pd.read_csv(run.progress_path)
    # Prefer seconds; fall back to minutes when older logs only contain minutes.
    time_column, seconds = max_numeric_column(df, ("custom_time/elapsed_s", "time/time_elapsed"))
    if not time_column:
        minute_column, minutes = max_numeric_column(df, ("custom_time/elapsed_min",))
        time_column, seconds = (minute_column, minutes * 60.0) if minute_column else ("", float("nan"))
    # Use the strongest available memory summary column for the run.
    memory_column, max_memory = max_numeric_column(df, ("memory/peak_total_rss_mb", "memory/total_rss_mb", "memory/main_rss_mb", "memory/children_rss_mb"))
    row.update(
        status="ok" if time_column or memory_column else "error",
        error="" if time_column or memory_column else "No usable time or memory columns found in progress.csv",
        time_source_column=time_column,
        memory_source_column=memory_column,
        training_time_hours=seconds / 3600.0 if np.isfinite(seconds) else np.nan,
        max_memory_mb=max_memory,
    )
    return row


def collect_resource_results(runs: list[RunInfo], out_path: Path) -> Any:
    """
    Collect resource rows for all runs and write the seed-level resource CSV.
    """
    rows: list[dict[str, Any]] = []
    for run in runs:
        try:
            rows.append(resource_row(run))
        except Exception as exc:
            rows.append({**base_run_row(run), "status": "error", "error": repr(exc), "timestamp": timestamp()})
    df = sort_rows(normalize_architecture_columns(pd.DataFrame(rows)))
    write_csv(df, out_path, "training resource results")
    return df


def aggregate_resource_results(df: Any, out_path: Path) -> Any:
    """
    Average training time and memory usage across seeds.
    """
    return aggregate_ok_rows(
        df,
        out_path,
        "training resource aggregate",
        ["hidden_size", "depth", "seed", "training_time_hours", "max_memory_mb"],
        ["hidden_size", "depth"],
        {
            "mean_training_time_hours": ("training_time_hours", "mean"),
            "std_training_time_hours": ("training_time_hours", "std"),
            "mean_max_memory_mb": ("max_memory_mb", "mean"),
            "std_max_memory_mb": ("max_memory_mb", "std"),
            "n_resource_runs": ("run_name", "count"),
            "resource_seeds": ("seed", seed_list),
        },
        ["std_training_time_hours", "std_max_memory_mb"],
    )



# Parameter counting
#---------------------------------------


def trainable_parameter_count(run: RunInfo, device: str) -> int:
    """
    Load one saved model and count trainable policy parameters.

    Only parameters with ``requires_grad=True`` are counted because those are the
    values that were actually optimized during training.
    """
    from stable_baselines3 import PPO

    register_legacy_module_aliases()
    policy = PPO.load(str(run.model_path), device=device).policy
    return int(sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad))


def parameter_row(run: RunInfo, device: str) -> dict[str, Any]:
    """
    Build one CSV row with the trainable parameter count of one model.
    """
    trainable = trainable_parameter_count(run, device)
    return {
        **base_run_row(run),
        "status": "ok",
        "error": "",
        "timestamp": timestamp(),
        "model_path": str(run.model_path.resolve()),
        "policy_trainable_parameters": trainable,
    }


def collect_parameter_results(runs: list[RunInfo], out_path: Path, device: str, force: bool) -> Any:
    """
    Count trainable parameters for all runs and write the seed-level CSV.
    """
    cache = {} if force else cached_ok_by_run(out_path)
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        if run.run_name in cache:
            rows.append(cache[run.run_name])
            print(f"[cache] params {index}/{len(runs)} {run.actor_backbone} w={run.hidden_size} d={run.depth}")
            continue
        try:
            # Parameter counting requires loading the model, so cached rows can save time.
            row = parameter_row(run, device)
            print(f"[ok] params {index}/{len(runs)}: {row['policy_trainable_parameters']}")
        except Exception as exc:
            row = {
                **base_run_row(run),
                "status": "error",
                "error": repr(exc),
                "timestamp": timestamp(),
                "model_path": str(run.model_path.resolve()),
                "policy_trainable_parameters": np.nan,
            }
            print(f"[error] parameter count failed for {run.run_name}: {exc}")
        rows.append(row)

    df = sort_rows(normalize_architecture_columns(pd.DataFrame(rows)))
    # Older result files may contain this column; the thesis plots use trainable parameters only.
    df = df.drop(columns=["policy_total_parameters"], errors="ignore")
    write_csv(df, out_path, "parameter count results")
    return df


def aggregate_parameter_results(df: Any, out_path: Path) -> Any:
    """
    Average trainable parameter counts across seeds.
    """
    return aggregate_ok_rows(
        df,
        out_path,
        "parameter count aggregate",
        ["hidden_size", "depth", "seed", "policy_trainable_parameters"],
        ["hidden_size", "depth", "policy_trainable_parameters"],
        {
            "mean_policy_trainable_parameters": ("policy_trainable_parameters", "mean"),
            "std_policy_trainable_parameters": ("policy_trainable_parameters", "std"),
            "n_parameter_runs": ("run_name", "count"),
            "parameter_seeds": ("seed", seed_list),
        },
        ["std_policy_trainable_parameters"],
    )



# Plot preparation and generation
#-------------------------------------------


def build_value_df(aggregate_df: Any, env_id: str, mean_col: str, std_col: str | None) -> Any:
    """
    Prepare a small plotting table for one environment and one measured value.

    The plotting utilities expect generic column names. This function converts
    columns such as ``mean_reward`` or ``mean_training_time_hours`` into
    ``metric_value`` and optional ``metric_std``.
    """
    env_df = supported_only(aggregate_df)
    env_df = env_df[env_df["env_id"].astype(str) == str(env_id)].copy()
    if env_df.empty or mean_col not in env_df.columns:
        return pd.DataFrame()

    numeric_cols = ["hidden_size", "depth", mean_col] + ([std_col] if std_col and std_col in env_df.columns else [])
    env_df = coerce_numeric(env_df, numeric_cols).dropna(subset=["hidden_size", "depth", mean_col])
    if env_df.empty:
        return pd.DataFrame()

    aggregations = {"metric_value": (mean_col, "mean")}
    if std_col and std_col in env_df.columns:
        aggregations["metric_std"] = (std_col, "mean")
    # Group by backbone, width, and depth so every plotted point is one architecture size.
    value_df = env_df.groupby(["actor_backbone", "hidden_size", "depth"], as_index=False).agg(**aggregations)
    value_df = value_df.rename(columns={"actor_backbone": "backbone"})
    value_df["metric_std"] = value_df["metric_std"].fillna(0.0) if "metric_std" in value_df.columns else 0.0
    return value_df


def create_plots(aggregate_df: Any, outdir: Path, args: argparse.Namespace) -> list[Path]:
    """
    Create all size-experiment summary plots from the combined aggregate table.

    For each environment and each measured quantity, this creates 3D scaling
    plots and 2D width-scaling plots. Reward additionally gets the heatmap that
    marks which backbone performed best at every width/depth combination.
    """
    if aggregate_df.empty:
        print("[warn] No aggregate data available for plotting.")
        return []

    aggregate_df = supported_only(aggregate_df)
    labels = {backbone: display_backbone(backbone) for backbone in BACKBONES}
    paths: list[Path] = []

    for env_id in sorted(aggregate_df["env_id"].dropna().unique()):
        env_id = str(env_id)
        env_df = aggregate_df[aggregate_df["env_id"].astype(str) == env_id].copy()
        for spec in PLOT_SPECS:
            # Convert the combined aggregate table into the shape expected by 2D and heatmap plotters.
            value_df = build_value_df(aggregate_df, env_id, spec["mean_col"], spec["std_col"])
            if spec["mean_col"] in env_df.columns:
                # The 3D plot needs the original aggregate table because it plots all backbones together.
                paths.extend(save_plot_variants(
                    plot_size_depth_points_3d,
                    outdir,
                    f"{sanitize_filename(env_id)}_mlp_lan_kan_size_depth_{spec['suffix']}_mean_points_3d",
                    "3D mean plot",
                    data=env_df,
                    value_col=spec["mean_col"],
                    value_label=spec["z_label"],
                    title=f"{env_id}: MLP/LAN/KAN mean {spec['title']}",
                    categories=BACKBONES,
                    colors=BACKBONE_COLORS,
                    markers=BACKBONE_MARKERS,
                    labels=labels,
                    x_scale=args.x_scale,
                    view_elev=args.view_elev,
                    view_azim=args.view_azim,
                    dpi=args.dpi,
                ))

            # The 2D plot shows how the measured value scales with width for each depth.
            paths.extend(save_plot_variants(
                plot_size_depth_bars,
                outdir,
                f"{sanitize_filename(env_id)}_mlp_lan_kan_{spec['suffix']}_width_scaling_stacked_2d",
                "2D width plot",
                value_df=value_df,
                value_label=spec["z_label"],
                title=f"{env_id}: MLP/LAN/KAN width scaling ({spec['title']})",
                categories=BACKBONES,
                colors=BACKBONE_COLORS,
                labels=labels,
                x_scale=args.x_scale,
                dpi=args.dpi,
            ))

            if spec["suffix"] == "reward":
                # Reward is the only metric where "best architecture" is meaningful.
                paths.extend(save_plot_variants(
                    plot_best_category_heatmap,
                    outdir,
                    f"{sanitize_filename(env_id)}_mlp_lan_kan_best_backbone_reward_heatmap",
                    "best-backbone heatmap",
                    value_df=value_df,
                    title=f"{env_id}: best backbone by mean evaluation reward",
                    categories=BACKBONES,
                    colors=BACKBONE_COLORS,
                    labels=labels,
                    dpi=args.dpi,
                ))
    return paths



# Pipeline orchestration
#-------------------------------------------------


def load_or_create_evaluation(args: argparse.Namespace, runs: list[RunInfo], results_path: Path, aggregate_path: Path) -> Any:
    """
    Load cached evaluation results or run model evaluation when needed.
    """
    if args.skip_evaluation:
        # Explicitly requested cache usage should fail loudly if the cache is missing.
        if not results_path.exists():
            raise FileNotFoundError(f"Cached evaluation file not found: {results_path}")
        results = read_csv_or_empty(results_path)
    elif runs:
        print(f"Found {len(runs)} MLP/LAN/KAN run(s) below: {resolve_cli_path(args.runs_root)}")
        results = evaluate_runs(args, runs, results_path)
    elif results_path.exists():
        # This allows plot regeneration from CSVs after moving or deleting model folders.
        print(f"[cache] no matching model directories found; using cached evaluation results: {results_path}")
        results = read_csv_or_empty(results_path)
    else:
        raise FileNotFoundError(f"No matching MLP/LAN/KAN model.zip files found below: {resolve_cli_path(args.runs_root)}")
    return aggregate_evaluation_results(results, aggregate_path)


def load_or_create_auxiliary(runs: list[RunInfo], args: argparse.Namespace, outdir: Path) -> tuple[Any, Any]:
    """
    Create or load the non-reward aggregate tables.

    "Auxiliary" here means values that do not require running episodes:
    training resources from progress.csv and parameter counts from model files.
    """
    resource_path = outdir / "training_resource_results.csv"
    resource_aggregate_path = outdir / "training_resource_aggregate.csv"
    parameter_path = outdir / "parameter_count_results.csv"
    parameter_aggregate_path = outdir / "parameter_count_aggregate.csv"

    if runs:
        # Fresh run folders are available, so derive auxiliary data from them.
        resources = collect_resource_results(runs, resource_path)
        parameters = collect_parameter_results(runs, parameter_path, args.device, args.force)
    else:
        # If models are unavailable, try to reuse existing auxiliary CSV files.
        resources = read_csv_or_empty(resource_path)
        parameters = read_csv_or_empty(parameter_path)

    resource_aggregate = aggregate_resource_results(resources, resource_aggregate_path) if not resources.empty else pd.DataFrame()
    parameter_aggregate = aggregate_parameter_results(parameters, parameter_aggregate_path) if not parameters.empty else pd.DataFrame()
    return resource_aggregate, parameter_aggregate


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point for the complete size-experiment evaluation pipeline.
    """
    args = parse_args(argv)
    if args.n_eval_episodes <= 0:
        raise ValueError("--n-eval-episodes must be > 0")

    ensure_dependencies()
    runs_root = resolve_cli_path(args.runs_root)
    outdir = resolve_cli_path(args.outdir) if args.outdir else runs_root / "evaluation"
    outdir.mkdir(parents=True, exist_ok=True)

    # First discover trained model folders, then build the reward evaluation table.
    runs = discover_runs(runs_root, max_runs=args.max_runs)
    aggregate_path = outdir / "evaluation_aggregate.csv"
    aggregate_df = load_or_create_evaluation(args, runs, outdir / "evaluation_results.csv", aggregate_path)

    # Then add training resources and parameter counts to the same aggregate table.
    resource_aggregate, parameter_aggregate = load_or_create_auxiliary(runs, args, outdir)

    aggregate_df = merge_aggregate(aggregate_df, resource_aggregate)
    aggregate_df = merge_aggregate(aggregate_df, parameter_aggregate)
    write_csv(aggregate_df, aggregate_path, "combined aggregate")

    if not args.no_plot:
        # Plots are generated from the combined aggregate table, not from raw runs.
        create_plots(aggregate_df, outdir, args)


if __name__ == "__main__":
    main()
