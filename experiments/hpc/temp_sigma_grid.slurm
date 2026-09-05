#!/bin/bash
#SBATCH --job-name=cont_modeling_temp_sigma_grid
#SBATCH --output=logs/cont_modeling_temp_sigma_grid_%A_%a.out
#SBATCH --error=logs/cont_modeling_temp_sigma_grid_%A_%a.err
#SBATCH --array=0-11         # (#OBJECTS * #MODELS) - 1; keep in sync with the
                             # arrays below. Each task prints the grid total, and
                             # an id past the end errors out instead of silently
                             # rerunning cell 0. A %N suffix caps how many cells
                             # hold a GPU at once.
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=20   # >= N_WORKERS below: each worker drives one eval
                             # sim on its own core.
#SBATCH --mem=32G            # N_WORKERS SPAWNED interpreters, each re-importing
                             # warp + mujoco + pinocchio/coal and carrying a CUDA
                             # context's host allocations. 16G OOM-kills a worker.
#SBATCH --time=12:00:00      # one cell = #temperatures x #noise_sigmas x
                             # N_EPISODES episodes (45 by default), run
                             # CONCURRENTLY N_WORKERS at a time, so the cost is
                             # about ceil(episodes / N_WORKERS) x the slowest
                             # episode rather than their sum. A timed-out cell
                             # resumes: finished grid points are skipped on the
                             # next run (see OUTDIR below).

# ===========================================================================
# Temperature x noise_sigma grid — one (contact model, object) per GPU.
#
# The third sweep in this directory, and distinct from the other two:
#
#   * param_search.slurm  — the SLURM script owns the grid and each array task
#     evaluates ONE point, serially, with no object axis.
#   * bayes_opt.slurm     — same pooled execution model as this one, but a GP
#     search over cost weights, so it never draws an even surface over the two
#     planner knobs.
#
# Here the array assigns each GPU a contact model and an object, and
# contact_study/drivers/run_temp_sigma_grid.py walks the whole
# temperature x noise_sigma grid inside that cell using the same EpisodePool
# run_bayes_opt.py uses. That split is the point: the useful MPPI temperature
# scales with the magnitude of the task cost, which differs per contact model
# and per object, so one grid per (model, object) is the readable picture.
#
# Every array task writes cell_*.json into ONE shared $OUTDIR, with ids offset
# by the array index, so a single combine job merges the whole submission.
#
# Submit:   sbatch experiments/hpc/temp_sigma_grid.slurm
# Resume:   sbatch --export=ALL,OUTDIR=/abs/path/to/previous experiments/hpc/temp_sigma_grid.slurm
# Results:  results/temp_sigma_grid_<array job id>/cell_*.json
#           + grid_summary_<array index>.json per cell (its ranked table)
#           + combined_temp_sigma_grid_{rich,agg}.json from the combine job
# ===========================================================================

set -euo pipefail

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR"

# ===========================================================================
# 1. THE CELLS: one (object, contact model) per array task.
# ===========================================================================
TASK="grasp_reorient"
OBJECTS=("cube" "duck" "ball")          # keys of _OBJ_PARAMS in tasks/grasp_reorient.py
MODELS=("M1" "M2" "M3" "M4")
HAND_ACC="high"                         # collision-geometry fidelity; joined with
OBJ_ACC="high"                          # the object into <obj>_<hand>_<obj>

# ===========================================================================
# 2. THE GRID inside each cell. Comma-separated, expanded by the driver — bash
#    never needs to know how many points these make.
# ===========================================================================
TEMPERATURES="40,20,10"                 # MPPI lambda: divides cost in the softmax
NOISE_SIGMAS="0.2,0.1,0.05"             # action-noise std dev; ONE value here
                                        # collapses this to a 1-D temperature sweep

# ===========================================================================
# 3. Episode settings — names match run_eval_episode.py, so a winning point
#    replays by copy-pasting the flags the driver prints.
# ===========================================================================
PLANNER="mppi"                          # temperature is an MPPI-only field
N_EPISODES=10                            # episodes per grid point -> success rate
N_WORKERS=16                            # episodes run CONCURRENTLY, one process
                                        # each. THE VRAM KNOB: every worker holds
                                        # its own CUDA context and MJWarp Data at
                                        # N_SAMPLES worlds. Lower it (with
                                        # --cpus-per-task and --mem) on an OOM.
SEED=42                                  # THE constant episode seed: every grid
                                        # point is scored on the same episodes
N_SAMPLES=256
TIME_HORIZON=0.352                      # planning horizon in SECONDS
STEP_TIME=0.064                         # control-step duration in SECONDS
EVAL_SIM="none"                         # none | mujoco | drake | pinocchio
SETTLE=1.0

# Fixed cost weights for the WHOLE grid (weights are not an axis here — that is
# what param_search.slurm and bayes_opt.slurm are for). Leave empty for the
# task's own defaults.
WEIGHTS=()
# e.g. WEIGHTS=("w_quat=20.0" "w_pos_x=7.5" "w_pos_y=15.0" "w_pos_z=5.0")

# ===========================================================================
# 4. Output. ONE directory for the whole array: cell ids are offset by the
#    array index inside the driver, so tasks cannot collide and one
#    combine_results.py run merges every model/object pair.
#    Export OUTDIR to a FIXED path to resume a previous submission wholesale
#    (the array job id changes on every sbatch); finished points are skipped.
# ===========================================================================
OUTDIR="${OUTDIR:-$ROOT_DIR/results/temp_sigma_grid_${SLURM_ARRAY_JOB_ID:-manual}}"
TASK_LABEL="${TASK_LABEL:-temp_sigma_grid}"

# ===========================================================================
# 5. Map this task's ID to one value per axis (mixed-radix decode, lowest-
#    varying axis first -- consecutive task ids walk the objects).
# ===========================================================================
AXES=(OBJECTS MODELS)

TOTAL=1
for AX in "${AXES[@]}"; do
    declare -n ARR="$AX"
    TOTAL=$(( TOTAL * ${#ARR[@]} ))
done

T=${SLURM_ARRAY_TASK_ID:-0}
if (( T >= TOTAL )); then
    echo "cell $T is outside the grid (TOTAL=$TOTAL cells); set #SBATCH --array=0-$(( TOTAL - 1 ))" >&2
    exit 1
fi

declare -A SEL
for AX in "${AXES[@]}"; do
    declare -n ARR="$AX"
    N=${#ARR[@]}
    SEL[$AX]=${ARR[$(( T % N ))]}
    T=$(( T / N ))
done

OBJ="${SEL[OBJECTS]}"
MODEL="${SEL[MODELS]}"
GEOMETRY="${OBJ}_${HAND_ACC}_${OBJ_ACC}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0} of $TOTAL: model=$MODEL object=$OBJ geometry=$GEOMETRY"
echo "  grid: T=[$TEMPERATURES] x sigma=[$NOISE_SIGMAS] x $N_EPISODES episodes -> $OUTDIR"

# Per-control-step trajectory + planner-distribution recording (see
# contact_study/evaluation/trajectory.py). OFF: at a task's max_steps it is
# megabytes per episode, and this sweep keeps every episode of every point.
RECORD_ARGS=(--no-record_trajectory --no-record_planner_dist)

# Only pass --weights when the array is non-empty: `--weights` with nothing
# after it would swallow the next flag.
WEIGHT_ARGS=()
if (( ${#WEIGHTS[@]} )); then
    WEIGHT_ARGS=(--weights "${WEIGHTS[@]}")
fi

python -u -m contact_study.drivers.run_temp_sigma_grid \
    "${RECORD_ARGS[@]}" \
    --array_index  "${SLURM_ARRAY_TASK_ID:-0}" \
    --outdir       "$OUTDIR" \
    --task         "$TASK" \
    --geometry     "$GEOMETRY" \
    --model        "$MODEL" \
    --planner      "$PLANNER" \
    --temperatures "$TEMPERATURES" \
    --noise_sigmas "$NOISE_SIGMAS" \
    --n_episodes   "$N_EPISODES" \
    --n_workers    "$N_WORKERS" \
    --n_samples    "$N_SAMPLES" \
    --time_horizon "$TIME_HORIZON" \
    --step_time    "$STEP_TIME" \
    --eval_sim     "$EVAL_SIM" \
    --settle       "$SETTLE" \
    --seed         "$SEED" \
    "${WEIGHT_ARGS[@]}" \
    --debug

# Task 0 queues the combine job once the whole array finishes ok, so a single
# sbatch produces the merged, ranked table across every model and object too.
# `|| true` so a scheduler hiccup here cannot fail cell 0's (already completed)
# grid under `set -e`; combine can always be re-run by hand, see the README.
# --export=ALL,... keeps the inherited environment: the bare --export=VAR=val
# form propagates ONLY those vars, which breaks `module load` in the child job.
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    # combine.slurm's #SBATCH --output is relative to ITS submission dir, which
    # for this nested call is $PROJ_DIR (cd'd to above), not experiments/hpc.
    mkdir -p "$PROJ_DIR/logs"
    sbatch --dependency=afterok:"${SLURM_ARRAY_JOB_ID}" \
           --export=ALL,OUTDIR="$OUTDIR",TASK="$TASK_LABEL" \
           experiments/hpc/combine.slurm || true
fi
