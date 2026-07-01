#!/bin/bash
#SBATCH --job-name=param_search
#SBATCH --output=logs/param_search_%A_%a.out
#SBATCH --error=logs/param_search_%A_%a.err
#SBATCH --array=0-323          # 4 models x 81 weight combos (default grid). The
                               # submit script overrides this with the real count;
                               # if you run `sbatch param_search.slurm` directly,
                               # set this to `num_cells` (see submit_param_search.sh).
#SBATCH --gres=gpu:1           # one GPU per cell — warp rollouts live on CUDA
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# --- TODO: set these for your cluster ---------------------------------------
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
# ----------------------------------------------------------------------------

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HPC_DIR"
source "$HPC_DIR/search_config.sh"

# Where per-cell JSON goes. Exported by the submit script so the array and the
# combine job agree; otherwise default to a per-array-job directory.
OUTDIR="${OUTDIR:-$HPC_DIR/../../results/param_search_${SLURM_ARRAY_JOB_ID:-manual}}"

# --- TODO: activate your environment ----------------------------------------
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate contact_study
# ----------------------------------------------------------------------------

echo "cell ${SLURM_ARRAY_TASK_ID} / $(num_cells)  ->  $OUTDIR"

python run_param_cell.py \
    --combo_index "$SLURM_ARRAY_TASK_ID" \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --n_episodes  "$N_EPISODES" \
    --models      "${MODELS[@]}" \
    --weights     "${WEIGHTS[@]}" \
    --n_samples   "$N_SAMPLES" \
    --horizon     "$HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --delta       "$DELTA" \
    --substeps    "$SUBSTEPS" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED"
