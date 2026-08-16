# Project Structure

This document explains the software structure of the repository and the
responsibilities of each layer.

The central design rule is:

Keep the PPO training pipeline fixed and make the actor and critic backbones
modular.

The thesis compares function approximators inside the same reinforcement
learning setup. The code therefore keeps Stable-Baselines3 PPO, environment
construction, logging, and output handling shared, while the model layer provides
interchangeable backbone implementations.


## Design Decisions

- Use Stable-Baselines3 PPO
- Keep experiment definitions in YAML configs under `src/configuration_layer/configs/`.
- Use one training pipeline for all compared architectures.
- Treat actor and critic backbones as independent configurable slots.
- Keep environment construction separate from model code.
- Keep runtime observation callbacks separate from training orchestration.
- Keep evaluation, plotting, and aggregation separate from training.
- Save resolved configs and run artifacts so figures can be traced back to concrete runs.
- Use Slurm job arrays for PALMA II experiment sweeps.


## Layer Overview

### `src/interaction_layer/`

This layer contains the user-facing script entry points.

Each file is a thin wrapper that adds `src/` to `sys.path`, imports the relevant
implementation `main` function, and calls it.

Entry points:

- `train_sb3_ppo.py`
- `inspect_trained_model.py`
- `evaluate_size_experiment.py`
- `evaluate_spline_base_ratio_histograms.py`
- `aggregate_training_plots_0_pre.py`
- `aggregate_training_plots_1_by_depth.py`
- `aggregate_training_plots_3_by_hyperparameters.py`
- `aggregate_training_plots_4_ablation.py`

The interaction layer should contain command routing only. Training behavior
belongs in the execution layer; post-processing behavior belongs in the analysis
layer.


### `src/configuration_layer/`

This layer defines experiment configuration.

It is responsible for:

- environment choice
- training seed
- PPO hyperparameters
- actor and critic backbone choices
- backbone hyperparameters
- logging and output settings
- config defaults and CLI override handling

The main config object is `SB3PPOConfig` in `training.py`. YAML configs live
under `src/configuration_layer/configs/`.

Configuration precedence is:

1. code defaults
2. values from the YAML config
3. explicit CLI arguments


### `src/utility_layer/`

This layer contains generic helpers shared across the project.

It is responsible for:

- project path constants
- config and output path resolution
- YAML file loading
- generic CLI parsing helpers
- graph style configuration
- reusable plotting primitives

The plotting subpackage provides reusable figure builders for boxplots, bar
plots, heatmaps, 3D plots, histograms, ridgelines, and mean training curves.


### `src/execution_layer/`

This layer runs one PPO training experiment.

It is responsible for:

- configuring torch thread behavior
- resolving the runtime device
- setting seeds
- creating run names and output folders
- writing `resolved_config.yaml`
- building vectorized Gymnasium environments
- translating config values into PPO and policy kwargs
- attaching timing, memory, and backbone diagnostic callbacks
- launching PPO training
- saving `model.zip`
- triggering per-run plots from `progress.csv`



### `src/model_layer/`

This layer contains policy integration and backbone implementations.

It is responsible for:

- integrating custom actor and critic branches into Stable-Baselines3
- constructing backbones through one registry
- implementing the MLP baseline
- implementing KAN and LAN backbones
- implementing KAN ablation variants
- exposing optional diagnostics used by callbacks and analysis scripts

Main policy classes:

- `FlexibleActorCriticPolicy`
- `CustomBackboneMlpExtractor`

Backbone names:

- `mlp`
- `kan`
- `kan_no_base`
- `kan_no_spline`
- `lan`
- `debug_constant`

Training code should select backbones by name and should not contain
architecture-specific construction logic outside the registry and policy
integration points.


### `src/observation_layer/`

This layer records information during training.

It is responsible for:

- memory logging
- elapsed-time logging
- rollout-time logging
- KAN/LAN backbone diagnostics

Callbacks:

- `MemoryTrackingCallback`
- `TimeTrackingCallback`
- `BackboneDiagnosticsCallback`



### `src/analysis_layer/`

This layer reads saved logs and trained models after training.

It is responsible for:

- per-run plots from `progress.csv`
- shared aggregate training-curve logic
- experiment-specific aggregation scripts
- size/depth evaluation from trained models
- resource and parameter-count summaries
- spline/base-ratio internal analysis
- trained-model inspection
- analysis-specific helper functions

Shared analysis modules:

- `training_plots.py`: per-run training plots
- `aggregate_training_plots.py`: seeded mean-curve aggregation
- `utils.py`: analysis-specific helpers for path resolution, CSV writing, time and memory extraction, plot variant saving, metric constants, and legacy model-loading aliases

Experiment-specific modules:

- `aggregate_training_plots_0_pre.py`
- `aggregate_training_plots_1_by_depth.py`
- `aggregate_training_plots_3_by_hyperparameters.py`
- `aggregate_training_plots_4_ablation.py`
- `evaluate_size_experiment_1.py`
- `evaluate_spline_base_ratio_histograms_4.py`
- `inspect_trained_model.py`



### `src/output_layer/outputs/`

This folder stores generated artifacts.

Run and analysis artifacts include:

- trained model zips
- `progress.csv`
- `resolved_config.yaml`
- TensorBoard logs
- per-run plots
- aggregate plots
- evaluation CSV files
- analysis CSV files


### `slurm/`

This folder contains PALMA II submit scripts.

It is responsible for:

- mapping Slurm array task ids to experiment settings
- requesting cluster resources
- activating the project environment
- launching interaction-layer training commands
- writing cluster logs under `slurm/logs/`


## Dependency Rules

- `interaction_layer` may import execution or analysis entry points, but should remain thin.
- `configuration_layer` defines experiment settings and config merging.
- `utility_layer` provides generic helpers without depending on project-specific runtime layers.
- `execution_layer` may depend on configuration, utility, model, and observation layers.
- `model_layer` implements policy and backbone behavior without depending on execution or analysis.
- `observation_layer` records training diagnostics without depending on analysis.
- `analysis_layer` reads completed run artifacts and may reuse utility plotting helpers.
- `output_layer/outputs` stores generated artifacts only.
- `slurm` scripts call interaction-layer commands and should not duplicate experiment logic.



## Training Flow

When `src/interaction_layer/train_sb3_ppo.py` is called, the training flow is:

1. Parse CLI arguments and optional YAML config.
2. Merge config defaults, YAML values, and CLI overrides.
3. Configure torch threads, seeds, and device.
4. Create the run name and run directory.
5. Write the resolved config into the run folder.
6. Build the vectorized Gymnasium environment.
7. Build PPO kwargs and policy kwargs.
8. Let `FlexibleActorCriticPolicy` construct actor and critic branches.
9. Build each selected backbone through the registry.
10. Attach timing, memory, and backbone-diagnostic callbacks.
11. Run PPO training.
12. Save the final model.
13. Create per-run plots from `progress.csv`.

This flow is the same for MLP, KAN, LAN, and the KAN ablation variants.


## Analysis Flows

### Aggregate Training Curves

Training aggregation scripts read completed `progress.csv` files and compare
seeded runs.

The shared curve flow is:

1. Discover `progress.csv` files.
2. Infer plottable metrics from CSV headers, unless metrics are provided by CLI.
3. Parse metadata from folder names and, where needed, `resolved_config.yaml`.
4. Group frames by environment and comparison condition.
5. Smooth seed curves.
6. Interpolate seed curves onto a shared x-grid.
7. Plot individual seeds and the seed mean.
8. Save figures in all configured formats.


### Size Experiment Evaluation

`evaluate_size_experiment_1.py` evaluates trained size/depth models and creates
summary plots.

The flow is:

1. Discover trained `model.zip` files below the size-experiment folder.
2. Keep MLP/LAN/KAN runs with matching actor and critic backbones.
3. Evaluate each trained model using fixed evaluation seeds.
4. Write seed-level evaluation results.
5. Aggregate evaluation rewards across training seeds.
6. Read training time and memory usage from `progress.csv`.
7. Count trainable policy parameters by loading each model.
8. Merge reward, resource, and parameter aggregates.
9. Create 3D scaling plots and reward winner heatmaps.

Default evaluation uses `--eval-seed 12345` and `--n-eval-episodes 10`, which
means each trained model is evaluated on seeds `12345` through `12354`.


### Spline/Base Ratio Analysis

`evaluate_spline_base_ratio_histograms_4.py` inspects internal KAN/LAN behavior
from trained size-experiment models.

The flow is:

1. Reuse size-experiment run discovery.
2. Filter to selected depth, backbone, environment, and hidden sizes.
3. Collect observations by running trained policies with fixed evaluation seeds.
4. Extract actor and critic feature tensors.
5. Ask KAN/LAN backbones for per-layer spline/base ratio values.
6. Write raw value CSVs and summary CSVs.
7. Build a layer/seed mean table.
8. Create histograms, ridgelines, and boxplots.



## Run Folder Conventions

Training run folders include enough metadata for later analysis.

The standard run-name shape is:

```text
<env_id>__actor-<actor_backbone>__critic-<critic_backbone>__seed<seed>__run-<label>
```

Example:

```text
Walker2d-v5__actor-kan__critic-kan__seed2025__run-palma-size-w64-d2-seed2025
```

Experiment scripts may add compact metadata to the run label, such as width,
depth, hyperparameter value index, or Slurm sweep identifiers. Analysis scripts
prefer `resolved_config.yaml` when it is available and use folder-name parsing
as fallback.


## Boundaries to Preserve

- Training code owns training orchestration, not cross-run reporting.
- Backbones expose diagnostics through their public methods or attributes.
- Analysis scripts read saved artifacts and create derived artifacts.
- Slurm scripts remain launch wrappers around interaction-layer commands.
- Plot styling stays centralized in `utility_layer/graph_config.py` and `utility_layer/plotting/`.
