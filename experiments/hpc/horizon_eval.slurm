#!/bin/bash
#SBATCH --job-name=cont_modeling_horizon_eval
#SBATCH --output=logs/cont_modeling_horizon_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_horizon_eval%A_%a.err
#SBATCH --array=0-14          # must equal (#MODELS * #HORIZONS) - 1
                              # default grid: 4 * 4 = 16  ->  0-15
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# MPPI horizon-length sweep as a SLURM job array — same pattern as
# num_rollout_eval.slurm. The grid (MODELS × HORIZONS) is defined inline;
# this script maps $SLURM_ARRAY_TASK_ID to one (model, horizon) cell, then
# runs N_EPISODES episodes for that one cell via run_horizon_cell.py, which
# calls the run_eval_episode function. One array task = one GPU = one cell.
# No videos.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#MODELS * #HORIZONS) - 1). The default 4×4 = 16 → 0-15.
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
N_EPISODES=5                                # episodes per (model, horizon) cell

MODELS=("M2" "M3" "M4")
HORIZONS=("6" "12" "24" "48" "96")            # the swept planning horizons

# Shared MPPI / eval knobs (single values, not swept).
N_SAMPLES=256
TEMPERATURE=25.0
NOISE_SIGMA=0.01
DELTA=0.1
SUBSTEPS=16
EVAL_SIM="mujoco"                 # "none" uses the task default; "mujoco" is faster
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first (model), then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NH=${#HORIZONS[@]}

T="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$(( T % NM )); T=$(( T / NM ))
HOR_IDX=$(( T % NH ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_HORIZON=${HORIZONS[$HOR_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the plot
# script can merge the whole directory of per-cell JSONs.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/horizon_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$SEL_MODEL horizon=$SEL_HORIZON "\
     "n_episodes=$N_EPISODES -> $OUTDIR"

python experiments/hpc/run_horizon_cell.py \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --horizon     "$SEL_HORIZON" \
    --n_episodes  "$N_EPISODES" \
    --n_samples   "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --delta       "$DELTA" \
    --substeps    "$SUBSTEPS" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    --debug
