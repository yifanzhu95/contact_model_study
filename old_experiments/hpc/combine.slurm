#!/bin/bash
#SBATCH --job-name=cont_modeling_param_combine
#SBATCH --output=logs/cont_modeling_param_combine%j.out
#SBATCH --error=logs/cont_modeling_param_combine%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

# Hardcoded project root, so this always runs from experiments/hpc regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
cd "$PROJ_DIR" #cd "$PROJ_DIR/experiments/hpc"

# OUTDIR + TASK are exported by param_search.slurm's auto-submit. If you run this
# by hand, pass them: `OUTDIR=results/... TASK=grasp_reorient sbatch combine.slurm`
OUTDIR="${OUTDIR:?set OUTDIR to the directory holding cell_*.json}"
TASK="${TASK:-run}"

module load miniconda
conda activate contact_modeling

python experiments/hpc/combine_results.py \
    --indir  "$OUTDIR" \
    --output "$OUTDIR/combined_${TASK}"
