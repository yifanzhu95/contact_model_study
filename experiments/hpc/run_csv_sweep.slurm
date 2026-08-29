#!/bin/bash
#SBATCH --job-name=cont_modeling_csv_sweep
#SBATCH --output=logs/cont_modeling_csv_sweep%A_%a.out
#SBATCH --error=logs/cont_modeling_csv_sweep%A_%a.err
#SBATCH --array=0-0          # placeholder — always overridden by submit_csv_sweep.sh,
                               # which sizes it to (CSV data rows - 1). #SBATCH options
                               # on the sbatch command line win over this header.
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
# ^ sized for ordinary synchronous rows. Rows using the new `driver=async` or
# `time_constrained` columns are substantially slower: the async driver spends
# each solve's latency as simulated time (so the sim runs *during* planning
# rather than pausing), and the time-constrained path holds every plan() at its
# full budget instead of returning once the horizon is done. A sweep containing
# many of those wants a longer --time — time_constrained_cntrl_freq_eval.slurm
# went from 5 h to 12 h for the same reason. Override per submission with
# `sbatch --time=12:00:00 ...`; the command line wins over this header.

set -euo pipefail

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR"

# CSV is exported by submit_csv_sweep.sh (--export=CSV=...); set it by hand
# for a manual `sbatch --array=0-N run_csv_sweep.slurm` invocation instead.
CSV="${CSV:?set CSV=path/to/params.csv (submit_csv_sweep.sh does this for you)}"
TASK_LABEL="${TASK_LABEL:-csv_sweep}"

# The out-of-range guard lives in run_csv_cell.py's load_row(), which parses the
# CSV properly. Deliberately NOT re-counting rows here with `wc -l`: that
# miscounts a file with no trailing newline (and any quoted field containing a
# newline), which would reject the last valid row.
T=${SLURM_ARRAY_TASK_ID:-0}

# All array tasks share one output dir (keyed by the array job id) so the
# combine job can gather every cell_*.json.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/csv_sweep_${SLURM_ARRAY_JOB_ID:-manual}}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "row ${T} of $CSV -> $OUTDIR"

# Per-control-step trajectory + planner-distribution recording (see
# contact_study/evaluation/trajectory.py). OFF by default for sweeps: at a
# task's max_steps it is megabytes per episode, which this many rows cannot
# carry.
#
# These are only the FALLBACK for rows that leave the recording columns blank.
# A row's record_trajectory / record_planner_dist / planner_dist_every cell
# overrides them, so to record one interesting cell set `true` in that row
# rather than editing here — changing these flags does nothing for any row
# that fills those columns in.
RECORD_ARGS=(--no-record_trajectory --no-record_planner_dist)

python experiments/hpc/run_csv_cell.py \
    "${RECORD_ARGS[@]}" \
    --csv    "$CSV" \
    --row    "$T" \
    --outdir "$OUTDIR" \
    --debug

# Task 0 queues the combine job once the whole array finishes ok, so a single
# submit_csv_sweep.sh call produces the merged results too. `|| true` so a
# scheduler hiccup here cannot fail row 0's (already completed) experiment
# under `set -e` — combine can always be re-run by hand, see the README.
# --export=ALL,... keeps the inherited environment: the bare --export=VAR=val
# form propagates ONLY those vars, which breaks `module load` in the child job.
# The path is relative to $PROJ_DIR (cd'd to above), where combine.slurm is NOT
# — it lives in experiments/hpc.
if [[ "$T" == "0" ]]; then
    # combine.slurm's #SBATCH --output is relative to ITS submission dir, which
    # for this nested call is $PROJ_DIR (cd'd to above), not experiments/hpc.
    mkdir -p "$PROJ_DIR/logs"
    sbatch --dependency=afterok:"${SLURM_ARRAY_JOB_ID}" \
           --export=ALL,OUTDIR="$OUTDIR",TASK="$TASK_LABEL" \
           experiments/hpc/combine.slurm || true
fi
