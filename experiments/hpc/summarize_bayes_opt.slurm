#!/bin/bash
#SBATCH --job-name=bayes_opt_summary
#SBATCH --output=logs/bayes_opt_summary_%j.out
#SBATCH --error=logs/bayes_opt_summary_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Builds summary.csv, summary.json and best.csv from every cell in OUTDIR, and
# refreshes each cell's bo_summary.json. Queued by submit_bayes_opt.sh to run
# after the whole array (afterany, so it also runs when some cells stopped early).
# By hand:
#   CSV=/abs/bo.csv OUTDIR=/abs/results/bayes_opt_x_123 sbatch experiments/hpc/summarize_bayes_opt.slurm
# or directly: python experiments/run_bayes_opt.py bo.csv --summarize --outdir <OUTDIR>

set -euo pipefail
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"
CSV="${CSV:?set CSV}"
OUTDIR="${OUTDIR:?set OUTDIR to the results folder of the run}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"

python experiments/run_bayes_opt.py "$CSV" --summarize --outdir "$OUTDIR"
