#!/bin/bash
#SBATCH --job-name=cont_modeling_tc_cntrl_freq_eval
#SBATCH --output=logs/cont_modeling_tc_cntrl_freq_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_tc_cntrl_freq_eval%A_%a.err
#SBATCH --array=0-19          # must equal (#MODELS * #STEP_TIME_SWEEP) - 1
                              # default grid: 4 * 5 = 20  ->  0-19
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# TIME-CONSTRAINED control-frequency sweep. Same grid, same worker and same
# knobs as cntrl_freq_eval.slurm — the one difference is --time_constrained,
# which gives the MPPI planner a wall-clock budget per plan() call equal to the
# control period being swept.
#
# Why this exists
# ---------------
# The plain cntrl_freq_eval sweep plans the FULL horizon at every control step
# no matter how long that takes, so its measured step time routinely overruns
# the control period the cell is nominally running at. It answers "success rate
# vs. the control frequency we asked for". This sweep answers "success rate vs.
# a control frequency the planner can actually hit": at 128 Hz the planner gets
# 8 ms and truncates its horizon at the deadline; at 7.8 Hz it gets 128 ms and
# will usually reach the full horizon. The horizon is a CEILING here, not a
# target — expect the low-frequency cells to look like the unconstrained sweep
# and the high-frequency cells to degrade further than it does.
#
# The worker (run_cntrl_freq_cell.py) defaults the budget to --step_time, so
# the budget tracks the swept axis automatically; --plan_budget_ms can pin it.
# --time_constrained requires the step-graph path, so --no-use_full_graph is
# passed explicitly (the worker would force it anyway and print a note).
# With --debug each plan prints "time-constrained: n/H steps in X ms budget",
# which is the direct readout of how much horizon the budget bought.
#
# Grid and label conventions are unchanged from cntrl_freq_eval.slurm: the
# control period is swept in SECONDS and quantized against the task's rollout
# timestep, and cells are named <model>_sub<substeps>. With grasp_reorient's
# rollout_dt = 4 ms the periods below land on 2/4/8/16/32 substeps (nominal
# 0.032 s = 8 substeps = 31.25 Hz).
#
# Control frequency for a cell is 1 / (eval_dt * eval_substeps_per_rollout *
# substeps) — run_cntrl_freq_cell.py computes it from the task config and
# writes eval_dt / eval_substeps_per_rollout to <outdir>/meta.json so the
# directory plotter can reconstruct exact frequencies.
#
# Walltime: the time-constrained path holds each plan() at the full budget
# instead of returning as soon as the horizon is done, and synchronizes once
# per horizon step. The slowest cell is ~max_steps * budget = 4000 * 128 ms
# ~= 8.5 min/episode before fin_ep_on_success shortens it, so 30 episodes can
# approach 4-5 h. Hence 12 h rather than the baseline sweep's 5 h.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#MODELS * #STEP_TIME_SWEEP) - 1). The default 4x5 = 20 -> 0-19.
#
# Submit (from the repo root, so logs/results land at the project root -
# #SBATCH output paths are resolved relative to the sbatch submission dir):
#     mkdir -p logs results
#     sbatch experiments/hpc/time_constrained_cntrl_freq_eval.slurm
#
# Budget overshoot - read the step-time panel with this in mind
# -------------------------------------------------------------
# plan() overshoots its budget by a roughly constant amount, so mean_step_ms
# lands ABOVE the nominal control period, not under it. Two causes, both in
# _plan_time_constrained: the deadline is only tested AFTER a completed rollout
# step (so the last step always overruns), and the weight update plus action
# extraction run after the loop exits. Measured on grasp_reorient / M2 /
# 8 ms budget: mean_step_ms ~= 10.9 ms, i.e. ~2.9 ms of overhead, so that cell
# realizes ~92 Hz rather than 125 Hz. The overhead is roughly additive, so it
# matters most at the high-frequency end and is negligible at 128 ms.
#
# Practical consequence: treat the x-axis as the REQUESTED rate and the bottom
# panel of the plot as the realized one. If you need the realized rate to match
# the label, pass --plan_budget_ms explicitly with the overhead pre-subtracted
# (it is a per-cell flag; the worker otherwise derives the budget from
# --step_time).
#
# Results land in
# results/time_constrained_cntrl_freq_eval_<arrayjobid>/<model>_sub<n>.json.
# Plot them all together with the SAME plotter as the unconstrained sweep
# (labels are identical); pass the directory explicitly, since that script's
# no-arg default only globs cntrl_freq_eval_*:
#     python analysis/plot_cntrl_freq_sweep_dir.py \
#         results/time_constrained_cntrl_freq_eval_<id>
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
# The swept control periods, in SECONDS (nominal 0.032 s = 8 substeps). Each
# cell's planning budget defaults to exactly this value.
STEP_TIME_SWEEP=("0.008" "0.016" "0.032" "0.064" "0.128")

# Shared MPPI / eval knobs (single values, not swept). Identical to
# cntrl_freq_eval.slurm so the two sweeps are comparable cell-for-cell.
N_SAMPLES=256
TIME_HORIZON=0.256              # planning horizon in seconds (a ceiling here)
TEMPERATURE=0.01
NOISE_SIGMA=0.02
DELTA=0.1
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
# script can merge the whole directory of per-cell JSONs. The prefix keeps this
# sweep's results separate from the unconstrained cntrl_freq_eval_* dirs, whose
# cells carry the exact same <model>_sub<n> labels.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/time_constrained_cntrl_freq_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$SEL_MODEL step_time=${SEL_STEP_TIME}s "\
     "(plan budget = one control period) time_horizon=${TIME_HORIZON}s "\
     "n_episodes=$N_EPISODES -> $OUTDIR"

python experiments/hpc/run_cntrl_freq_cell.py \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --step_time    "$SEL_STEP_TIME" \
    --n_episodes  "$N_EPISODES" \
    --n_samples   "$N_SAMPLES" \
    --time_horizon "$TIME_HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --delta       "$DELTA" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    --time_constrained \
    --no-use_full_graph \
    --debug
