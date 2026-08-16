# Interaction Layer Entry Points

The files in this folder are the user-facing start points for training, model
inspection, evaluation, and experiment aggregation. Run them from the project
root after activating the project environment.

```bash
cd /path/to/kans-and-lans-in-rl
python src/interaction_layer/<entry_point>.py --help
```

Most analysis scripts read finished experiment folders from
`src/output_layer/outputs/`. Use `--runs-root` when the experiment folder is in a
different location, and `--outdir` when the generated plots or CSV files should
be written somewhere else.


## Training

Start one PPO training run:

```bash
python src/interaction_layer/train_sb3_ppo.py --config general_experiments/walker2d_kan.yaml
```

Useful options:

- `--config`: YAML config relative to `src/configuration_layer/configs/`
- `--seed`: override the seed from the config
- `--output_root`: write run folders to a different output root
- any field from `SB3PPOConfig`: override the YAML value from the CLI

Config precedence is:

```text
SB3PPOConfig defaults < YAML config < CLI arguments
```


## Model Inspection

Inspect a saved PPO model:

```bash
python src/interaction_layer/inspect_trained_model.py src/output_layer/outputs/.../model.zip --graph-path src\output_layer\outputs\model_graph.svg
```

## Size Experiment Evaluation

Evaluate trained size-experiment models and generate the scaling plots and
winner heatmaps:

```bash
python src/interaction_layer/evaluate_size_experiment.py
```

Default input:

```text
src/output_layer/outputs/1_Size_experiment
```

Default output:

```text
<runs-root>/evaluation
```

Typical custom-path usage:

```bash
python src/interaction_layer/evaluate_size_experiment.py \
  --runs-root src/output_layer/outputs/1_Size_experiment \
  --outdir src/output_layer/outputs/1_Size_experiment/evaluation
```

Useful options:

- `--runs-root`: folder containing the size-experiment run directories
- `--outdir`: folder for evaluation CSVs and plots
- `--skip-evaluation`: reuse existing evaluation data and only regenerate plots
- `--force`: recompute evaluation data even if cached outputs exist
- `--no-plot`: evaluate without generating figures


## Size Experiment Depth Aggregation

Aggregate size-experiment training curves by backbone, depth, and width:

```bash
python src/interaction_layer/aggregate_training_plots_1_by_depth.py
```

Default input:

```text
src/output_layer/outputs/1_Size_experiment/HalfCheetah-v5
```

Default output:

```text
<runs-root>/aggregated_by_depth_plots
```

Typical custom-path usage:

```bash
python src/interaction_layer/aggregate_training_plots_1_by_depth.py \
  --runs-root src/output_layer/outputs/1_Size_experiment/Ant-v5 \
  --outdir src/output_layer/outputs/1_Size_experiment/Ant-v5/aggregated_by_depth_plots
```

Useful options:

- `--runs-root`: folder containing the size-experiment runs for one environment
- `--outdir`: folder for aggregated depth/width training plots
- `--metrics`: selected metrics from `progress.csv`

## Pre-Experiment Aggregation

Aggregate the MLP integration and grid-update pre-experiment plots:

```bash
python src/interaction_layer/aggregate_training_plots_0_pre.py
```

Default input:

```text
src/output_layer/outputs/0_pre_experiment
```

By default, the script aggregates the `mlp` and `update` subfolders and writes:

```text
src/output_layer/outputs/0_pre_experiment/aggregated_plots_mlp
src/output_layer/outputs/0_pre_experiment/aggregated_plots_update
```

Typical custom-path usage:

```bash
python src/interaction_layer/aggregate_training_plots_0_pre.py \
  --runs-root src/output_layer/outputs/0_pre_experiment \
  --subsets mlp update
```

Useful options:

- `--runs-root`: folder containing the pre-experiment subset folders
- `--outdir`: custom output folder
- `--subsets`: subset folders to aggregate, usually `mlp`, `update`, or both


## Hyperparameter Experiment Aggregation

Aggregate reward curves and time boxplots for the hyperparameter experiment:

```bash
python src/interaction_layer/aggregate_training_plots_3_by_hyperparameters.py
```

Default input:

```text
src/output_layer/outputs/2_Hyperparameter_experiment
```

Default output:

```text
<runs-root>/aggregated_by_hyperparameter_plots
```

Typical custom-path usage:

```bash
python src/interaction_layer/aggregate_training_plots_3_by_hyperparameters.py \
  --runs-root src/output_layer/outputs/2_Hyperparameter_experiment \
  --envs Ant-v5 Walker2d-v5 \
  --backbones mlp kan lan \
  --hyperparameters clip lr ent grid spline
```

Useful options:

- `--runs-root`: folder containing the hyperparameter run directories
- `--outdir`: folder for aggregated plots
- `--metrics`: selected metrics from `progress.csv`
- `--envs`: optional environment filter
- `--backbones`: optional backbone filter
- `--hyperparameters`: optional filter, e.g. `clip`, `lr`, `ent`, `grid`, `spline`


## Spline-to-Base Ratio Analysis

Evaluate trained depth-2 KAN/LAN size-experiment models and generate the
spline-to-base ratio CSVs, histograms, ridgeline plots, and boxplots:

```bash
python src/interaction_layer/evaluate_spline_base_ratio_histograms.py
```

Default model input:

```text
src/output_layer/outputs/1_Size_experiment
```

Default output:

```text
src/output_layer/outputs/3_Spline_to_base_analysis
```

Typical custom-path usage:

```bash
python src/interaction_layer/evaluate_spline_base_ratio_histograms.py \
  --runs-root src/output_layer/outputs/1_Size_experiment \
  --outdir src/output_layer/outputs/3_Spline_to_base_analysis \
  --depth 2 \
  --backbones kan lan
```

Useful options:

- `--runs-root`: folder containing the trained size-experiment models
- `--outdir`: folder for generated CSVs and plots
- `--depth`: hidden-layer depth to analyze
- `--backbones`: backbone filter, usually `kan lan`
- `--envs`: optional environment filter
- `--hidden-sizes`: optional hidden-width filter
- `--force`: recollect ratio values even if cached CSV files exist
- `--no-plot`: write CSVs without generating plots


## Ablation Experiment Aggregation

Aggregate reward curves, training-time boxplots, memory boxplots, and available
diagnostics for the KAN ablation experiment:

```bash
python src/interaction_layer/aggregate_training_plots_4_ablation.py
```

Default input:

```text
src/output_layer/outputs/4_ablation_experiment
```

Default output:

```text
<runs-root>/aggregated_training_plots
```

Typical custom-path usage:

```bash
python src/interaction_layer/aggregate_training_plots_4_ablation.py \
  --runs-root src/output_layer/outputs/4_ablation_experiment \
  --outdir src/output_layer/outputs/4_ablation_experiment/aggregated_training_plots
```

Useful options:

- `--runs-root`: folder containing the ablation run directories
- `--outdir`: folder for aggregated plots
- `--metrics`: selected metrics from `progress.csv`


## Path Rules

Relative paths are resolved from the project root. Absolute paths can also be
used. The experiment analysis scripts expect `--runs-root` to point to the
folder that contains the individual run directories or, for the pre-experiment,
the subset folders containing those run directories.
