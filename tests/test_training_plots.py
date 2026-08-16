from __future__ import annotations

import pandas as pd

from analysis_layer.training_plots import create_training_plots


def test_create_training_plots_writes_requested_metric_plot(tmp_path) -> None:
    progress_csv = tmp_path / "progress.csv"
    outdir = tmp_path / "plots"
    pd.DataFrame(
        {
            "time/total_timesteps": [0, 8, 16],
            "rollout/ep_rew_mean": [-10.0, -8.0, -6.0],
        }
    ).to_csv(progress_csv, index=False)

    create_training_plots(
        progress_csv=progress_csv,
        outdir=outdir,
        metrics=["rollout/ep_rew_mean"],
        include_layer_metrics=False,
    )

    assert (outdir / "rollout_ep_rew_mean.png").exists()
