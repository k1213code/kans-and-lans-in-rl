# PALMA II Slurm

This folder contains PALMA II launch scripts for the repository experiments.
Each script is a Slurm array job that calls the training entry point:

```bash
python -u src/interaction_layer/train_sb3_ppo.py
```

The scripts expect to be submitted from the repository root, or to receive
`PROJECT_DIR=/path/to/repo` through `--export`.


## Create The Environment

Use a regular Python `venv` on PALMA II. Do not install `environment.yml` with
`pip`; that file is for local Conda setup.

```bash
salloc --partition express --time 01:00:00 --cpus-per-task=2 --mem=8G

module purge
module load palma/2022b GCCcore/12.2.0 Python/3.10.8

python -m venv "$HOME/venvs/kan_rl"
source "$HOME/venvs/kan_rl/bin/activate"
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch==2.5.1+cpu stable-baselines3==2.6.0 "gymnasium[mujoco]==1.1.1" mujoco==2.3.3 \
  numpy==2.2.6 pandas==2.2.3 matplotlib==3.10.1 PyYAML==6.0.2 psutil==7.0.0 \
  tensorboard==2.20.0 torchinfo torchview rich tqdm
```


## Use The Environment Interactively

```bash
module purge
module load palma/2022b GCCcore/12.2.0 Python/3.10.8
source "$HOME/venvs/kan_rl/bin/activate"
```

Quick import and MuJoCo check:

```bash
python -c "import torch, gymnasium as gym; env = gym.make('Walker2d-v5'); print(torch.__version__, env.observation_space.shape, env.action_space.shape)"
```


## Output Locations

From the repository root on PALMA II:

```bash
mkdir -p slurm/logs src/output_layer/outputs
```

Slurm stdout/stderr files are written to:

```text
slurm/logs/
```

Training run folders are written below:

```text
src/output_layer/outputs/
```

Every script also accepts an `OUTPUT_ROOT` override:

```bash
sbatch --export=ALL,OUTPUT_ROOT=/path/to/outputs slurm/1_size_exp/palma_size_array_smaller.sbatch
```


## Experiment Scripts

### Pre-Experiments

Folder:

```text
slurm/0_pre_experiment/
```

Scripts:

- `palma_pre_mlp_sb3_64x2.sbatch`
- `palma_pre_mlp_custom_64x2.sbatch`
- `palma_pre_kan_grid_no_update_64x2.sbatch`
- `palma_pre_kan_grid_beginning_64x2.sbatch`
- `palma_pre_kan_grid_continuous_64x2.sbatch`

These scripts run `Walker2d-v5` with width `64`, depth `2`, and seeds
`2025` through `2029`.

Example:

```bash
sbatch slurm/0_pre_experiment/palma_pre_kan_grid_no_update_64x2.sbatch
```

### Size/Depth Sweep

Folder:

```text
slurm/1_size_exp/
```

Scripts:

- `palma_size_array_smaller.sbatch`: widths `8`, `16`, `32`, `64`, `128`; depths `1`, `2`, `3`, `4`; seeds `2025` through `2029`; `normal` partition.
- `palma_size_array_256.sbatch`: width `256`; depths `1`, `2`, `3`, `4`; seeds `2025` through `2029`; `long` partition.

Examples:

```bash
sbatch slurm/1_size_exp/palma_size_array_smaller.sbatch
sbatch slurm/1_size_exp/palma_size_array_256.sbatch
```

The default config is `general_experiments/walker2d_mlp.yaml`. Submit the same
array for another environment/backbone by overriding `CONFIG`:

```bash
sbatch --export=ALL,CONFIG=general_experiments/walker2d_kan.yaml slurm/1_size_exp/palma_size_array_smaller.sbatch
sbatch --export=ALL,CONFIG=general_experiments/halfcheetah_lan.yaml slurm/1_size_exp/palma_size_array_smaller.sbatch
sbatch --export=ALL,CONFIG=general_experiments/ant_mlp.yaml slurm/1_size_exp/palma_size_array_256.sbatch
```

### Hyperparameter Sensitivity

Folder:

```text
slurm/2_hparam_sensitivity/
```

Scripts:

- `palma_sens_learning_rate.sbatch`: `5e-5`, `1e-4`, `3e-4`, `5e-4`, `1e-3`
- `palma_sens_clip_range.sbatch`: `0.05`, `0.1`, `0.2`, `0.4`, `0.6`
- `palma_sens_ent_coef.sbatch`: `0.0`, `5e-4`, `1e-3`, `5e-3`, `1e-2`
- `palma_sens_grid_size.sbatch`: `1`, `3`, `5`, `10`, `30`
- `palma_sens_spline_order.sbatch`: `1`, `2`, `3`, `4`, `5`

All hyperparameter scripts use seeds `2025` through `2029`. PPO hyperparameter
scripts default to `general_experiments/walker2d_mlp.yaml`; grid and spline
scripts default to `general_experiments/walker2d_kan.yaml`.

Examples:

```bash
sbatch slurm/2_hparam_sensitivity/palma_sens_learning_rate.sbatch
sbatch --export=ALL,CONFIG=general_experiments/ant_kan.yaml slurm/2_hparam_sensitivity/palma_sens_grid_size.sbatch
```

The hyperparameter scripts also accept `WIDTH` and `DEPTH` overrides. The
defaults are width `64` and depth `2`.

```bash
sbatch --export=ALL,CONFIG=general_experiments/halfcheetah_mlp.yaml,WIDTH=128,DEPTH=3 slurm/2_hparam_sensitivity/palma_sens_clip_range.sbatch
```

### KAN Ablation

Folder:

```text
slurm/4_ablation_experiment/
```

Script:

- `palma_ablation_array.sbatch`

The ablation array compares:

- `kan`
- `kan_no_base`
- `kan_no_spline`

Defaults:

- config: `general_experiments/walker2d_kan.yaml`
- width: `64`
- depth: `2`
- seeds: `2025` through `2029`

Examples:

```bash
sbatch slurm/4_ablation_experiment/palma_ablation_array.sbatch
sbatch --export=ALL,CONFIG=general_experiments/ant_kan.yaml slurm/4_ablation_experiment/palma_ablation_array.sbatch
```

`WIDTH` and `DEPTH` can be overridden in the same way as for the hyperparameter
scripts.


## Common Overrides

All scripts support:

- `PROJECT_DIR`: repository path; defaults to the Slurm submit directory.
- `VENV`: Python environment path; defaults to `$HOME/venvs/kan_rl`.
- `OUTPUT_ROOT`: training output root; defaults to `<repo>/src/output_layer/outputs`.
- `CONFIG`: YAML config relative to `src/configuration_layer/configs/`.

Some scripts also support:

- `WIDTH`: hidden width override.
- `DEPTH`: hidden depth override.

Example with several overrides:

```bash
sbatch --export=ALL,PROJECT_DIR=$WORK/kans-and-lans-in-rl,VENV=$HOME/venvs/kan_rl,CONFIG=general_experiments/walker2d_lan.yaml,OUTPUT_ROOT=$WORK/kan_outputs slurm/1_size_exp/palma_size_array_smaller.sbatch
```
