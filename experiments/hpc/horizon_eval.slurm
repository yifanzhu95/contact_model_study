#!/bin/bash
#SBATCH --job-name=cont_modeling_horizon_eval
#SBATCH --output=logs/cont_modeling_horizon_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_horizon_eval%A_%a.err
#SBATCH --array=0-19          # must equal (#MODELS * #TIME_HORIZONS) - 1
                              # default grid: 4 * 5 = 20  ->  0-19
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# MPPI horizon-length sweep as a SLURM job array — same pattern as
# num_rollout_eval.slurm. The grid (MODELS × TIME_HORIZONS) is defined inline;
# this script maps $SLURM_ARRAY_TASK_ID to one (model, time_horizon) cell, then
# runs N_EPISODES episodes for that one cell via run_horizon_cell.py, which
# calls the run_eval_episode function. One array task = one GPU = one cell.
# No videos.
#
# The horizon is swept in SECONDS; run_horizon_cell.py quantizes it against the
# task's rollout timestep and names the output file with the resulting step
# count, so the plotter's <model>_h<int> labels are unchanged. With
# grasp_reorient's rollout_dt = 4 ms and STEP_TIME = 0.064 s (16 substeps ->
# 64 ms control period), the durations below land on 1/2/4/8/16 steps.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#MODELS * #TIME_HORIZONS) - 1). The default 4×5 = 20 → 0-19.
#
# Submit (from the repo root, so logs/results land at the project root -
# #SBATCH output paths are resolved relative to the sbatch submission dir):
#     mkdir -p logs results
#     sbatch experiments/hpc/horizon_eval.slurm
#
# Results land in results/horizon_eval_<arrayjobid>/<model>_h<horizon>.json.
# Plot them all together with:
#     python analysis/plot_horizon_sweep_dir.py results/horizon_eval_<id>
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
N_EPISODES=30                                # episodes per (model, horizon) cell

MODELS=("M1" "M2" "M3" "M4")
# The swept planning horizons, in SECONDS (nominal 0.256 s = 8 control steps).
TIME_HORIZONS=("0.064" "0.128" "0.256" "0.512" "1.024")

# Shared MPPI / eval knobs (single values, not swept), matching run_eval_episode.py's
# CLI defaults.
N_SAMPLES=256
TEMPERATURE=1.0
NOISE_SIGMA=0.025
STEP_TIME=0.064                 # control-step duration in seconds (nominal)
EVAL_SIM="none"                 # "none" uses the task default; "mujoco" is faster
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first (model), then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NH=${#TIME_HORIZONS[@]}

T="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$(( T % NM )); T=$(( T / NM ))
HOR_IDX=$(( T % NH ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_TIME_HORIZON=${TIME_HORIZONS[$HOR_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the plot
# script can merge the whole directory of per-cell JSONs.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/horizon_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$SEL_MODEL time_horizon=${SEL_TIME_HORIZON}s "\
     "step_time=${STEP_TIME}s n_episodes=$N_EPISODES -> $OUTDIR"

python experiments/hpc/run_horizon_cell.py \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --time_horizon "$SEL_TIME_HORIZON" \
    --n_episodes  "$N_EPISODES" \
    --n_samples   "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --step_time   "$STEP_TIME" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    --debug
