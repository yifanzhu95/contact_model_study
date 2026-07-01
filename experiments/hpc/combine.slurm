#!/bin/bash
#SBATCH --job-name=param_combine
#SBATCH --output=logs/param_combine_%j.out
#SBATCH --error=logs/param_combine_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
# --- TODO: set these for your cluster ---------------------------------------
##SBATCH --partition=cpu
##SBATCH --account=YOUR_ACCOUNT
# ----------------------------------------------------------------------------

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HPC_DIR"
source "$HPC_DIR/search_config.sh"

OUTDIR="${OUTDIR:?OUTDIR must be set (exported by submit_param_search.sh)}"

# --- TODO: activate your environment ----------------------------------------
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate contact_study
# ----------------------------------------------------------------------------

python combine_results.py \
    --indir  "$OUTDIR" \
    --output "$OUTDIR/combined_${TASK}"
