#!/bin/bash
#SBATCH --job-name=cont_modeling_param_combine
#SBATCH --output=logs/cont_modeling_param_combine%j.out
#SBATCH --error=logs/cont_modeling_param_combine%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

# Resolve from the script's own location, not the submission directory
# (SLURM_SUBMIT_DIR is wherever `sbatch` was invoked from, which may not be
# experiments/hpc).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# OUTDIR + TASK are exported by param_search.slurm's auto-submit. If you run this
# by hand, pass them: `OUTDIR=results/... TASK=grasp_reorient sbatch combine.slurm`
OUTDIR="${OUTDIR:?set OUTDIR to the directory holding cell_*.json}"
TASK="${TASK:-run}"

module load miniconda
conda activate contact_modeling

python combine_results.py \
    --indir  "$OUTDIR" \
    --output "$OUTDIR/combined_${TASK}"
