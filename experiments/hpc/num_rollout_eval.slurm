#!/bin/bash
#SBATCH --job-name=cont_modeling_num_rollout_eval
#SBATCH --output=logs/cont_modeling_num_rollout_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_num_rollout_eval%A_%a.err
#SBATCH --array=0-3            # one GPU per array task; each runs the FULL sweep.
                               # Add tasks (e.g. 0-7) for more replicates -> more
                               # total episodes per (model, n_samples) cell.
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Num-rollout (n_samples) sweep as a SLURM job array.
#
# Each array task grabs ONE GPU and runs the ENTIRE sweep via
# run_num_rollout_eval.py (which calls run_eval_episode): every contact model in
# MODELS, at every n_samples value in that script's N_SAMPLES_SWEEP list, for
# N_EPISODES episodes each. Array tasks differ only by their seed, so they act as
# independent replicates; each writes its own JSON into a shared results dir.
# No videos are rendered.
#
# The n_samples values themselves live in experiments/run_num_rollout_eval.py
# (edit N_SAMPLES_SWEEP there). Everything else is set inline below.
#
# Submit (from experiments/hpc):
#     mkdir -p logs
#     sbatch num_rollout_eval.slurm
#
# Results land in results/num_rollout_eval_<arrayjobid>/sweep_repNNN.json.
# Plot them all together with:
#     python analysis/plot_n_samples_sweep_dir.py results/num_rollout_eval_<id>
# ---------------------------------------------------------------------------

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HPC_DIR"

# Project root (experiments/hpc -> ../..) so results land in <root>/results.
ROOT_DIR="$(cd "$HPC_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Sweep settings (shared by every array task / GPU).
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
MODELS=("M1" "M2" "M3" "M4")   # every array task runs ALL of these
N_EPISODES=5                   # episodes per (model, n_samples) cell, per task

# Shared MPPI / eval knobs. These mirror run_num_rollout_eval.py's own defaults;
# n_samples is NOT here — it is the swept axis (N_SAMPLES_SWEEP in that script).
HORIZON=48
TEMPERATURE=0.125
NOISE_SIGMA=0.01
GEOMETRY="accurate"
EVAL_SIM="none"                # "none" uses the task default; "mujoco" is faster
SETTLE=1.0
SEED_BASE=0                    # per-task seed = SEED_BASE + array task id

# ---------------------------------------------------------------------------
# 2. Per-task (per-GPU) output + seed.
#    All array tasks share one output dir (keyed by the array job id) so the
#    directory holds every replicate's JSON for the plotting script to merge.
# ---------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUTDIR="${OUTDIR:-$ROOT_DIR/results/num_rollout_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/sweep_rep$(printf '%03d' "$TASK_ID").json"
SEED=$(( SEED_BASE + TASK_ID ))

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "num_rollout_eval rep ${TASK_ID}: task=$TASK models=${MODELS[*]} "\
     "n_episodes=$N_EPISODES seed=$SEED -> $OUTFILE"

python ../run_num_rollout_eval.py \
    --task        "$TASK" \
    --models      "${MODELS[@]}" \
    --n_episodes  "$N_EPISODES" \
    --horizon     "$HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --geometry    "$GEOMETRY" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --seed        "$SEED" \
    --output      "$OUTFILE"
