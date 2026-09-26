#!/bin/bash
#SBATCH --job-name=episode_batches_summary
#SBATCH --output=logs/episode_batches_summary_%j.out
#SBATCH --error=logs/episode_batches_summary_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Builds summary.csv / summary.json from every cell in OUTDIR. Queued by
# submit_episode_batches.sh to run after the whole array (afterany, so it also
# runs when some cells failed). By hand:
#   CSV=/abs/batches.csv OUTDIR=/abs/results/episode_batches_x_123 sbatch experiments/hpc/summarize_batches.slurm
# or directly: python experiments/run_episode_batches.py batches.csv --summarize --outdir <OUTDIR>

set -euo pipefail
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"
CSV="${CSV:?set CSV}"
OUTDIR="${OUTDIR:?set OUTDIR to the results folder of the batch}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

python experiments/run_episode_batches.py "$CSV" --summarize --outdir "$OUTDIR"
