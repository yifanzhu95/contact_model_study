#!/bin/bash
#SBATCH --job-name=cont_modeling_cntrl_freq_eval
#SBATCH --output=logs/cont_modeling_cntrl_freq_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_cntrl_freq_eval%A_%a.err
#SBATCH --array=0-19          # must equal (#MODELS * #STEP_TIME_SWEEP) - 1
                              # default grid: 4 * 5 = 20  ->  0-19
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Control-frequency sweep as a SLURM job array — same pattern as
# num_rollout_eval.slurm. The grid (MODELS × STEP_TIME_SWEEP) is defined
# inline; this script maps $SLURM_ARRAY_TASK_ID to one (model, step_time) cell,
# then runs N_EPISODES episodes for that one cell via run_cntrl_freq_cell.py,
# which calls the run_eval_episode function. One array task = one GPU = one
# cell. No videos.
#
# The control period is swept in SECONDS; run_cntrl_freq_cell.py quantizes it
# against the task's rollout timestep and names the output file with the
# resulting substep count, so the plotter's <model>_sub<int> labels are
# unchanged. With grasp_reorient's rollout_dt = 4 ms the periods below land on
# 2/4/8/16/32 substeps (nominal 0.032 s = 8 substeps = 31.25 Hz).
#
# Control frequency for a cell is 1 / (eval_dt * eval_substeps_per_rollout *
# substeps) — run_cntrl_freq_cell.py computes it from the task config and
# writes eval_dt / eval_substeps_per_rollout to <outdir>/meta.json so the
# directory plotter can reconstruct exact frequencies.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#MODELS * #STEP_TIME_SWEEP) - 1). The default 4×5 = 20 → 0-19.
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

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR" #cd "$PROJ_DIR/experiments/hpc"

# ---------------------------------------------------------------------------
# 1. Fixed settings + the grids to sweep.
#    Edit the arrays below and keep #SBATCH --array in sync (the product of
#    their lengths above).
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
N_EPISODES=30                                   # episodes per (model, substeps) cell

MODELS=("M1" "M2" "M3" "M4")
# The swept control periods, in SECONDS (nominal 0.032 s = 8 substeps).
STEP_TIME_SWEEP=("0.008" "0.016" "0.032" "0.064" "0.128")

# Shared MPPI / eval knobs (single values, not swept), matching run_eval_episode.py's
# CLI defaults.
N_SAMPLES=256
TIME_HORIZON=0.352              # planning horizon in seconds (nominal)
TEMPERATURE=1.0
NOISE_SIGMA=0.025
EVAL_SIM="none"                 # "none" uses the task default; "mujoco" is faster
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first (model), then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NS=${#STEP_TIME_SWEEP[@]}

T="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$(( T % NM )); T=$(( T / NM ))
SUB_IDX=$(( T % NS ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_STEP_TIME=${STEP_TIME_SWEEP[$SUB_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the plot
# script can merge the whole directory of per-cell JSONs.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/cntrl_freq_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$SEL_MODEL step_time=${SEL_STEP_TIME}s "\
     "time_horizon=${TIME_HORIZON}s n_episodes=$N_EPISODES -> $OUTDIR"

# Per-control-step trajectory + planner-distribution recording (see
# contact_study/evaluation/trajectory.py). OFF for sweeps: at grasp_reorient's
# max_steps it is megabytes per episode, which this many cells cannot carry.
# Drop the --no- prefixes (or raise --planner_dist_every) to record a cell.
RECORD_ARGS=(--no-record_trajectory --no-record_planner_dist)

python experiments/hpc/run_cntrl_freq_cell.py \
    "${RECORD_ARGS[@]}" \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --step_time    "$SEL_STEP_TIME" \
    --n_episodes  "$N_EPISODES" \
    --n_samples   "$N_SAMPLES" \
    --time_horizon "$TIME_HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    --debug
