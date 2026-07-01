#!/bin/bash
# submit_param_search.sh — one-shot submission of the weight grid search.
#
# Computes the grid size from search_config.sh, submits the SLURM array (one job
# per weight set), then submits the combine job to run automatically once every
# array task has finished (afterok). Run this from the hpc/ directory:
#
#     ./submit_param_search.sh
#
# Optional: cap how many array tasks run at once with MAX_CONCURRENT (default 20).

set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HPC_DIR"
source "$HPC_DIR/search_config.sh"

mkdir -p logs

N=$(num_cells)
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"

# Unique output dir shared by the array and the combine job.
OUTDIR="$HPC_DIR/../../results/param_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "grid cells : $N"
echo "output dir : $OUTDIR"
echo "array      : 0-$((N-1))%${MAX_CONCURRENT}"

# Submit the array; --export makes OUTDIR visible to every task. --parsable
# returns just the job id so we can chain the combine job onto it.
ARRAY_JID=$(sbatch --parsable \
    --array="0-$((N-1))%${MAX_CONCURRENT}" \
    --export=ALL,OUTDIR="$OUTDIR" \
    param_search.slurm)
echo "submitted array job $ARRAY_JID"

# Combine runs only if the whole array succeeds.
COMBINE_JID=$(sbatch --parsable \
    --dependency=afterok:"$ARRAY_JID" \
    --export=ALL,OUTDIR="$OUTDIR" \
    combine.slurm)
echo "submitted combine job $COMBINE_JID (afterok:$ARRAY_JID)"
