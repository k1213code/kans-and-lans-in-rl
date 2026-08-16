#!/bin/bash
#SBATCH --job-name=sens_spline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-24%13
#SBATCH --output=slurm/logs/sens_spline_%A_%a.out
#SBATCH --error=slurm/logs/sens_spline_%A_%a.err

set -euo pipefail


SPLINE_ORDERS=(1 2 3 4 5)
SEEDS=(2025 2026 2027 2028 2029)
CONFIG="${CONFIG:-general_experiments/walker2d_kan.yaml}"

TOTAL_TASKS=$((${#SPLINE_ORDERS[@]} * ${#SEEDS[@]}))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))."
    exit 2
fi

NSEEDS=${#SEEDS[@]}
NVALUES=${#SPLINE_ORDERS[@]}

seed=${SEEDS[$((TASK_ID % NSEEDS))]}
tmp=$((TASK_ID / NSEEDS))
value_idx=$((tmp % NVALUES))
spline_order=${SPLINE_ORDERS[$value_idx]}
CONFIG="${CONFIG//[[:space:]]/}"
width="${WIDTH:-64}"
depth="${DEPTH:-2}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$HOME/venvs/kan_rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/src/output_layer/outputs}"

module purge
module load palma/2022b GCCcore/12.2.0 Python/3.10.8

if [[ ! -f "$PROJECT_DIR/src/interaction_layer/train_sb3_ppo.py" ]]; then
    echo "PROJECT_DIR does not look like this repository: $PROJECT_DIR"
    echo "Submit from the repository root on scratch, or pass --export=ALL,PROJECT_DIR=/path/to/repo."
    exit 1
fi

if [[ ! -d "$VENV" ]]; then
    echo "Missing Python environment: $VENV"
    echo "Create it first, or submit with --export=ALL,VENV=/path/to/env."
    exit 1
fi

source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export KAN_RL_TORCH_THREADS=1
export KAN_RL_TORCH_INTEROP_THREADS=1

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_DIR"

config_label=$(basename "$CONFIG" .yaml)
run_id="palma-hparam-spline-${config_label}-v${value_idx}-${spline_order}-seed${seed}"

echo "Project: $PROJECT_DIR"
echo "Config: $CONFIG"
echo "Output root: $OUTPUT_ROOT"
echo "Task: $TASK_ID / $((TOTAL_TASKS - 1))"
echo "Width: $width"
echo "Depth: $depth"
echo "Spline order: $spline_order"
echo "Seed: $seed"
echo "Run id: $run_id"

srun python -u src/interaction_layer/train_sb3_ppo.py \
    --config "$CONFIG" \
    --device cpu \
    --cuda false \
    --output_root "$OUTPUT_ROOT" \
    --seed "$seed" \
    --exp_name "hparam_spline_${config_label}_v${value_idx}_${spline_order}" \
    --run_id "$run_id" \
    --actor_hidden_size "$width" \
    --critic_hidden_size "$width" \
    --actor_num_hidden_layers "$depth" \
    --critic_num_hidden_layers "$depth" \
    --spline_order "$spline_order"
