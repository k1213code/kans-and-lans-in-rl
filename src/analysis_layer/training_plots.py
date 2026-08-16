"""Create per-run plots from one SB3 progress.csv file."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis_layer.utils import DEFAULT_PLOT_METRICS, find_layer_metrics, sanitize_filename


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outdir: Path,
    smooth_window: int = 20,
    dpi: int = 150,
) -> None:
    """
    Plot one metric from a single progress.csv file and optionally smooth it.
    """
    # If the x-axis is missing, something is wrong with the CSV itself.
    if x_col not in df.columns:
        raise ValueError(f"X-axis column '{x_col}' not found in CSV")

    # Missing columns are normal because not every run logs every optional metric.
    if y_col not in df.columns:
        print(f"[skip] Column not found: {y_col}")
        return

    # Drop NaNs before plotting so Matplotlib receives clean numeric series.
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        print(f"[skip] No valid data for: {y_col}")
        return

    # Draw the raw series first so smoothing does not hide spikes completely.
    plt.figure(figsize=(8, 5))
    plt.plot(sub[x_col], sub[y_col], alpha=0.35, label="raw")

    if smooth_window > 1 and len(sub) >= 2:
        # The smoothed line is usually easier to read than the raw series.
        smooth = sub[y_col].rolling(window=smooth_window, min_periods=1).mean()
        plt.plot(sub[x_col], smooth, label=f"rolling mean ({smooth_window})")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(y_col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save one file per metric so individual plots can be opened quickly later.
    out_path = outdir / f"{sanitize_filename(y_col)}.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    print(f"[ok] saved: {out_path}")


def create_training_plots(
    progress_csv: str | Path,
    outdir: str | Path,
    smooth_window: int = 20,
    metrics: list[str] | None = None,
    x_axis: str = "time/total_timesteps",
    dpi: int = 150,
    include_layer_metrics: bool = True,
) -> None:
    """
    Create a full set of per-run plots from one SB3 progress.csv file.
    """
    # Make the path handling predictable before touching the CSV.
    progress_csv = Path(progress_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not progress_csv.exists():
        print(f"[warn] progress.csv not found: {progress_csv}")
        return

    # Load the CSV once, then reuse the same DataFrame for all metric plots.
    df = pd.read_csv(progress_csv)

    # Custom metric lists are allowed, but the defaults are usually enough.
    metrics_to_plot = list(metrics or DEFAULT_PLOT_METRICS)

    if include_layer_metrics:
        # Layer metrics are discovered from the CSV because the exact set depends on the run.
        layer_metrics = find_layer_metrics(df)
        if layer_metrics:
            print("\nDetected layer-wise metrics:")
            for metric in layer_metrics:
                print(f"  {metric}")
            metrics_to_plot.extend(layer_metrics)

    # Duplicates can happen when layer metrics overlap with a caller-provided list.
    seen = set()
    metrics_to_plot = [
        m for m in metrics_to_plot
        if not (m in seen or seen.add(m))
    ]

    print("\nCreating plots from:")
    print(progress_csv)

    for metric in metrics_to_plot:
        # Each metric is handled independently so one missing column does not stop the rest.
        plot_metric(
            df=df,
            x_col=x_axis,
            y_col=metric,
            outdir=outdir,
            smooth_window=smooth_window,
            dpi=dpi,
        )
