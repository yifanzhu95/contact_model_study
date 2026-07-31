#!/bin/bash
#SBATCH --job-name=cont_modeling_num_rollout_eval
#SBATCH --output=logs/cont_modeling_num_rollout_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_num_rollout_eval%A_%a.err
#SBATCH --array=0-19          # must equal (#MODELS * #N_SAMPLES) - 1
                              # default grid: 4 * 5 = 20  ->  0-19
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Num-rollout (n_samples) sweep as a SLURM job array — same pattern as
# param_search.slurm. The grid (MODELS × N_SAMPLES) is defined inline; this
# script maps $SLURM_ARRAY_TASK_ID to one (model, n_samples) cell, then runs
# N_EPISODES episodes for that one cell via run_num_rollout_cell.py, which calls
# the run_eval_episode function. One array task = one GPU = one cell. No videos.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#MODELS * #N_SAMPLES) - 1). The default 4×5 = 20 → 0-19.
#
# Submit (from the repo root, so logs/results land at the project root -
# #SBATCH output paths are resolved relative to the sbatch submission dir):
#     mkdir -p logs results
#     sbatch experiments/hpc/num_rollout_eval.slurm
#
# Results land in results/num_rollout_eval_<arrayjobid>/<model>_n<n>.json.
# Plot them all together with:
#     python analysis/plot_n_samples_sweep_dir.py results/num_rollout_eval_<id>
# ---------------------------------------------------------------------------

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR" #cd "$PROJ_DIR/experiments/hpc"

# ---------------------------------------------------------------------------
# 1. Fixed settings + the grids to sweep.
#    Edit the arrays below and keep #SBATCH --array in sync (the product of
#    their lengths above).
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
N_EPISODES=10                                    # episodes per (model, n_samples) cell

MODELS=("M1" "M2" "M3" "M4")
N_SAMPLES=("16" "64" "256" "1024" "4096")    # the swept rollout counts

# Shared MPPI / eval knobs (single values, not swept). The rollout schedule is
# specified in SECONDS and quantized against the task's rollout timestep; for
# grasp_reorient (rollout_dt = 4 ms) these are 8 substeps × 8 control steps.
TIME_HORIZON=0.256              # planning horizon in seconds
STEP_TIME=0.032                 # control-step duration in seconds
TEMPERATURE=0.01
NOISE_SIGMA=0.02
DELTA=0.1
EVAL_SIM="none"                 # "none" uses the task default; "mujoco" is faster
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first (model), then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NN=${#N_SAMPLES[@]}

T="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$(( T % NM )); T=$(( T / NM ))
NSAMP_IDX=$(( T % NN ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_N_SAMPLES=${N_SAMPLES[$NSAMP_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the plot
# script can merge the whole directory of per-cell JSONs.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/num_rollout_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$SEL_MODEL n_samples=$SEL_N_SAMPLES "\
     "n_episodes=$N_EPISODES delta=$DELTA -> $OUTDIR"

python experiments/hpc/run_num_rollout_cell.py \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --n_samples   "$SEL_N_SAMPLES" \
    --n_episodes  "$N_EPISODES" \
    --time_horizon "$TIME_HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --delta       "$DELTA" \
    --step_time   "$STEP_TIME" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    --debug
