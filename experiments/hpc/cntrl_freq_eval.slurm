#!/bin/bash
#SBATCH --job-name=cont_modeling_cntrl_freq_eval
#SBATCH --output=logs/cont_modeling_cntrl_freq_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_cntrl_freq_eval%A_%a.err
#SBATCH --array=0-31          # must equal (#MODELS * #SUBSTEPS_SWEEP) - 1
                              # default grid: 4 * 8 = 32  ->  0-31
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Control-frequency (MPPI substeps) sweep as a SLURM job array — same pattern
# as num_rollout_eval.slurm. The grid (MODELS × SUBSTEPS_SWEEP) is defined
# inline; this script maps $SLURM_ARRAY_TASK_ID to one (model, substeps) cell,
# then runs N_EPISODES episodes for that one cell via run_cntrl_freq_cell.py,
# which calls the run_eval_episode function. One array task = one GPU = one
# cell. No videos.
#
# Control frequency for a cell is 1 / (eval_dt * eval_substeps_per_rollout *
# substeps) — run_cntrl_freq_cell.py computes it from the task config and
# writes eval_dt / eval_substeps_per_rollout to <outdir>/meta.json so the
# directory plotter can reconstruct exact frequencies.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#MODELS * #SUBSTEPS_SWEEP) - 1). The default 4×8 = 32 → 0-31.
#
# Submit (from the repo root, so logs/results land at the project root -
# #SBATCH output paths are resolved relative to the sbatch submission dir):
#     mkdir -p logs results
#     sbatch experiments/hpc/cntrl_freq_eval.slurm
#
# Results land in results/cntrl_freq_eval_<arrayjobid>/<model>_sub<n>.json.
# Plot them all together with:
#     python analysis/plot_cntrl_freq_sweep_dir.py results/cntrl_freq_eval_<id>
# ---------------------------------------------------------------------------

# Resolve paths from the script's own location, not the submission directory
# (SLURM_SUBMIT_DIR is wherever `sbatch` was invoked from, which may not be
# experiments/hpc), so results always land in <root>/results regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Fixed settings + the grids to sweep.
#    Edit the arrays below and keep #SBATCH --array in sync (the product of
#    their lengths above).
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
N_EPISODES=10                                   # episodes per (model, substeps) cell

MODELS=("M1" "M2" "M3" "M4")
SUBSTEPS_SWEEP=("1" "3" "5" "7" "9" "11" "13" "15")   # the swept substep counts

# Shared MPPI / eval knobs (single values, not swept).
N_SAMPLES=256
HORIZON=48
TEMPERATURE=0.01
NOISE_SIGMA=0.001
DELTA=0.1
EVAL_SIM="none"                 # "none" uses the task default; "mujoco" is faster
SETTLE=10.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first (model), then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NS=${#SUBSTEPS_SWEEP[@]}

T="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$(( T % NM )); T=$(( T / NM ))
SUB_IDX=$(( T % NS ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_SUBSTEPS=${SUBSTEPS_SWEEP[$SUB_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the plot
# script can merge the whole directory of per-cell JSONs.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/cntrl_freq_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$SEL_MODEL substeps=$SEL_SUBSTEPS "\
     "n_episodes=$N_EPISODES -> $OUTDIR"

python experiments/hpc/run_cntrl_freq_cell.py \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --substeps    "$SEL_SUBSTEPS" \
    --n_episodes  "$N_EPISODES" \
    --n_samples   "$N_SAMPLES" \
    --horizon     "$HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --delta       "$DELTA" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED"
