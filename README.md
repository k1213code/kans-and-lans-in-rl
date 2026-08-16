# KANs and LANs in Continuous Control RL

This repository compares different actor critic backbone architectures inside a
shared Stable-Baselines3 PPO training setup for continuous control reinforcement
learning.

This Project was developed as part of the Thesis "A Comparative Study of MLPs, Kolmogorov Arnold
Networks, and Learnable Activation Networks in
PPO based Continuous Control".

## Project Scope

Implemented capabilities:

- PPO training on Gymnasium MuJoCo environments
- environments: `Walker2d-v5`, `HalfCheetah-v5`, and `Ant-v5`
- configurable actor and critic backbones
- backbone types: `mlp`, `kan`, `lan`, `kan_no_base`, `kan_no_spline`, and `debug_constant`
- YAML experiment configs with CLI overrides
- automatic run folders with resolved configs, logs, models, and plots
- runtime logging for training time, rollout time, memory, and backbone diagnostics
- trained-model inspection with `torchinfo` and `torchview`
- PALMA II Slurm job arrays for pre-experiments, size/depth sweeps, hyperparameter sweeps, and ablations
- post-processing scripts for training-curve aggregation, final model evaluation, parameter counts, resource summaries, spline/base-ratio analysis, and ablation plots


## Experiment Overview

The repository is organized around five experiment groups under
`src/output_layer/outputs/`.

### `0_pre_experiment`

Pre-experiments for checking the training setup and KAN grid-update behavior.

Main comparisons:

- SB3 default MLP vs project custom MLP
- KAN without grid updates
- KAN with early grid updates
- KAN with continuous grid updates

Default setup:

- environment: `Walker2d-v5`
- width: `64`
- depth: `2`
- seeds: `2025` through `2029`

### `1_Size_experiment`

Main scaling experiment for MLP, KAN, and LAN.

Experiment grid:

- environments: `Walker2d-v5`, `HalfCheetah-v5`, `Ant-v5`
- backbones: `mlp`, `kan`, `lan`
- widths: `8`, `16`, `32`, `64`, `128`, `256`
- depths: `1`, `2`, `3`, `4`
- seeds: `2025` through `2029`

The evaluation pipeline loads trained models, evaluates each model with fixed
evaluation seeds, reads training time and memory from `progress.csv`, counts
trainable policy parameters, and creates reward/resource/parameter plots.

### `2_Hyperparameter_experiment`

Sensitivity experiments that vary one parameter at a time while keeping the
same PPO training pipeline.

Implemented sweeps:

- learning rate: `5e-5`, `1e-4`, `3e-4`, `5e-4`, `1e-3`
- PPO clip range: `0.05`, `0.1`, `0.2`, `0.4`, `0.6`
- entropy coefficient: `0.0`, `5e-4`, `1e-3`, `5e-3`, `1e-2`
- KAN/LAN grid size: `1`, `3`, `5`, `10`, `30`
- spline order: `1`, `2`, `3`, `4`, `5`

Each sweep uses five seeds and writes aggregated reward curves and training-time
boxplots.

### `3_Spline_to_base_analysis`

Internal KAN/LAN analysis for trained size-experiment models.

Generated artifacts include:

- `spline_base_ratio_values.csv`
- `spline_base_ratio_summary.csv`
- `spline_base_ratio_layer_seed_table.csv`
- grouped histograms
- ridgeline plots
- boxplots

The analysis collects observations from trained policies, extracts actor and
critic feature tensors, and measures per-layer absolute spline/base ratios.

### `4_ablation_experiment`

KAN ablation experiment for comparing the full KAN backbone with variants that
remove one branch.

Backbone variants:

- `kan`
- `kan_no_base`
- `kan_no_spline`

The aggregation script creates reward curves, training-time boxplots,
peak-memory boxplots, and available diagnostic plots.


## Repository Layout

- `src/interaction_layer/`
  User-facing script entry points for training, inspection, evaluation, and aggregation.
- `src/configuration_layer/`
  Training config dataclass, CLI parsing, and YAML configs under `configs/`.
- `src/utility_layer/`
  Shared path handling, YAML loading, CLI helpers, graph style, and plotting helpers.
- `src/execution_layer/`
  PPO run setup, environment construction, seed/device handling, PPO kwarg construction, and training orchestration.
- `src/model_layer/`
  Stable-Baselines3 policy integration and MLP, KAN, LAN, and KAN-ablation backbones.
- `src/observation_layer/`
  Training callbacks for timing, memory, and backbone diagnostics.
- `src/analysis_layer/`
  Per-run plots, aggregate plots, final evaluation, parameter counts, resource summaries, spline/base-ratio analysis, and model inspection.
- `src/output_layer/outputs/`
  Generated run and analysis artifacts.
- `slurm/`
  PALMA II job-array scripts and cluster setup notes.

For detailed architecture notes, see [structure.md](structure.md).
For command-specific usage, see [src/interaction_layer/README.md](src/interaction_layer/README.md).


## Setup

Python `3.10` is the project environment target.

Create the Conda environment from the provided environment file:

```bash
conda env create -f environment.yml
conda activate kan_rl_env
```

The environment includes:

- PyTorch CPU build
- Stable-Baselines3
- Gymnasium with MuJoCo support
- MuJoCo
- NumPy
- pandas
- matplotlib
- PyYAML
- psutil
- tensorboard
- torchinfo
- torchview
- rich
- tqdm

For graph export with `torchview`, Graphviz also needs to be installed on the
system. For PALMA II, use the venv setup documented in [slurm/README.md](slurm/README.md).


## Running Tests

Run the unit tests and the small training-pipeline smoke test from the project
environment:

```bash
conda activate kan_rl_env
python -m pytest tests
```

## Running Training

Run commands from the repository root.

Train from a YAML config:

```bash
python src/interaction_layer/train_sb3_ppo.py --config general_experiments/walker2d_mlp.yaml
```

KAN and LAN examples:

```bash
python src/interaction_layer/train_sb3_ppo.py --config general_experiments/walker2d_kan.yaml
python src/interaction_layer/train_sb3_ppo.py --config general_experiments/walker2d_lan.yaml
```

Override config values from the CLI:

```bash
python src/interaction_layer/train_sb3_ppo.py --config general_experiments/walker2d_kan.yaml --seed 2029 --device cpu
```

Run directly from CLI arguments:

```bash
python src/interaction_layer/train_sb3_ppo.py --env_id Walker2d-v5 --use_custom_mlp_extractor true --actor_backbone_type kan --critic_backbone_type kan --total_timesteps 100000 --num_envs 4 --output_root src/output_layer/outputs
```

Configuration precedence:

1. `SB3PPOConfig` code defaults
2. values loaded from `--config`
3. explicit CLI arguments

Each run writes artifacts below:

```text
<output_root>/runs/<env_id>__actor-<actor_backbone>__critic-<critic_backbone>__seed<seed>__run-<run_id>/
```

Typical run artifacts:

- `resolved_config.yaml`
- `progress.csv`
- `model.zip`
- `plots/`


## Running Analysis

Generate pre-experiment aggregate plots:

```bash
python src/interaction_layer/aggregate_training_plots_0_pre.py
```

Aggregate size/depth training curves:

```bash
python src/interaction_layer/aggregate_training_plots_1_by_depth.py
```

Evaluate trained size-experiment models and create reward/resource/parameter
plots:

```bash
python src/interaction_layer/evaluate_size_experiment.py
```

Aggregate hyperparameter-sensitivity plots:

```bash
python src/interaction_layer/aggregate_training_plots_3_by_hyperparameters.py
```

Run spline/base-ratio analysis:

```bash
python src/interaction_layer/evaluate_spline_base_ratio_histograms.py
```

Aggregate KAN ablation plots:

```bash
python src/interaction_layer/aggregate_training_plots_4_ablation.py
```

Inspect a trained model:

```bash
python src/interaction_layer/inspect_trained_model.py src/output_layer/outputs/.../model.zip --graph-path src/output_layer/outputs/model_graph.svg
```

Most analysis scripts accept `--runs-root`, `--outdir`, `--metrics`, and
experiment-specific filters. Run an entry point with `--help` for the available
options.


## Slurm Experiments

PALMA II scripts live under `slurm/`.

Main folders:

- `slurm/0_pre_experiment/`
- `slurm/1_size_exp/`
- `slurm/2_hparam_sensitivity/`
- `slurm/4_ablation_experiment/`

Submit from the repository root on PALMA II after creating the venv from
[slurm/README.md](slurm/README.md):

```bash
sbatch slurm/1_size_exp/palma_size_array_smaller.sbatch
sbatch slurm/1_size_exp/palma_size_array_256.sbatch
```

Override the config for a Slurm array:

```bash
sbatch --export=ALL,CONFIG=general_experiments/ant_kan.yaml slurm/1_size_exp/palma_size_array_smaller.sbatch
```

Cluster logs are written to `slurm/logs/`. Training and analysis artifacts are
written below `src/output_layer/outputs/`.
