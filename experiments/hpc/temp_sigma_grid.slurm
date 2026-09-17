#!/bin/bash
#SBATCH --job-name=cont_modeling_temp_sigma_grid
#SBATCH --output=logs/cont_modeling_temp_sigma_grid_%A_%a.out
#SBATCH --error=logs/cont_modeling_temp_sigma_grid_%A_%a.err
#SBATCH --array=0-0          # placeholder — always overridden by
                             # submit_temp_sigma_grid.sh, which sizes it to
                             # (CSV data rows - 1) and adds the %N throttle
                             # (how many cells hold a GPU at once). #SBATCH
                             # options on the sbatch command line win over this
                             # header. A hand `sbatch --array=...` past the end
                             # of the CSV errors out (read_temp_sigma_cell.py)
                             # instead of silently rerunning cell 0.
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=17   # >= N_WORKERS below: each worker drives one eval
                             # sim on its own core.
#SBATCH --mem=32G            # N_WORKERS SPAWNED interpreters, each re-importing
                             # warp + mujoco + pinocchio/coal and carrying a CUDA
                             # context's host allocations. 16G OOM-kills a worker.
                             # This request is UNIFORM across the array while
                             # n_samples varies per row of the cell CSV, so it
                             # (and the GPU) must cover N_WORKERS x the LARGEST
                             # n_samples in the CSV.
#SBATCH --time=16:00:00      # one cell = #temperatures x #noise_sigmas x
                             # N_EPISODES episodes (90 by default), run
                             # CONCURRENTLY N_WORKERS at a time, so the cost is
                             # about ceil(episodes / N_WORKERS) x the slowest
                             # episode rather than their sum. Uniform across the
                             # array, so it must cover the SLOWEST row of the
                             # cell CSV — largest n_iterations x n_samples x
                             # (time_horizon / step_time), i.e. the most rollout
                             # steps per plan() — several times the cheapest. A
                             # timed-out cell resumes: finished grid points are
                             # skipped on the next run (see OUTDIR below).

# ===========================================================================
# Temperature x noise_sigma grid — one (contact model, object, n_iterations,
# n_samples, time_horizon, step_time) cell per GPU, one cell per CSV row.
#
# The third sweep in this directory, and distinct from the other two:
#
#   * param_search.slurm  — the SLURM script owns the grid and each array task
#     evaluates ONE point, serially, with no object axis.
#   * bayes_opt.slurm     — same pooled execution model as this one, but a GP
#     search over cost weights, so it never draws an even surface over the two
#     planner knobs.
#
# Here each array task is ONE ROW of a cell table (CELLS_CSV below, default
# temp_sigma_cells.csv): a contact model, an object, an iteration count, a
# sample count, and the planning horizon + control-step duration THAT cell runs
# with. contact_study/drivers/run_temp_sigma_grid.py walks the whole
# temperature x noise_sigma grid inside that cell using the same EpisodePool
# run_bayes_opt.py uses. That split is the point: the useful MPPI temperature
# scales with the magnitude of the task cost, which differs per contact model,
# per object and per horizon, so one grid per cell is the readable picture.
#
# The cells are a CSV rather than bash arrays because the horizon and step
# time are PER CELL, not a further axis: each (object, model, n_iterations,
# n_samples) gets exactly one (time_horizon, step_time), which a mixed-radix
# decode of independent lists cannot express. n_iterations and n_samples are
# cell axes rather than grid axes because they change what an episode COSTS,
# not only how it scores: one cell per value keeps a cell's wall clock and VRAM
# constant instead of multiplying the inner grid, and lets the cheap cells
# finish while the expensive ones are still running. The same holds for the
# horizon and step time. USE_CONVERGENCE below swaps the fixed iteration count
# for MPPI's convergence-terminated mode, which makes the CSV's n_iterations
# column irrelevant.
#
# Every array task writes cell_*.json into ONE shared $OUTDIR, with ids offset
# by the array index, so a single combine job merges the whole submission.
#
# Submit:   experiments/hpc/submit_temp_sigma_grid.sh [cells.csv] [label] [throttle]
# Resume:   OUTDIR=/abs/path/to/previous experiments/hpc/submit_temp_sigma_grid.sh cells.csv
#           (same CSV! the cell ids are row-offset, so a reordered CSV would
#           resume the wrong cells)
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
# 1. THE CELLS: one CSV row per array task. Columns, all required:
#
#      object,model,n_iterations,n_samples,time_horizon,step_time
#
#    object       key of _OBJ_PARAMS in tasks/grasp_reorient.py (cube/duck/ball)
#    model        M1..M4
#    n_iterations MPPI optimizer iterations per plan(). IGNORED when
#                 USE_CONVERGENCE=true below (rows differing only here then
#                 just repeat the same configuration).
#    n_samples    rollouts per iteration. THE VRAM AXIS across cells: the
#                 --gpus/--mem request is uniform, so the largest value in the
#                 CSV sets what a node must hold.
#    time_horizon planning horizon in SECONDS
#    step_time    control-step duration in SECONDS
#
#    read_temp_sigma_cell.py rejects a typo'd header, a blank value, and two
#    rows naming the same (object, model, n_iterations, n_samples): the cell
#    label and the analysis script's cell key do not include the horizon or
#    step time, so such rows would merge into one result row.
#
#    CELLS_CSV is exported by submit_temp_sigma_grid.sh; override it by hand
#    for a manual `sbatch --array=0-N temp_sigma_grid.slurm`.
# ===========================================================================
TASK="grasp_reorient"
CELLS_CSV="${CELLS_CSV:-$PROJ_DIR/experiments/hpc/temp_sigma_cells.csv}"
HAND_ACC="high"                         # collision-geometry fidelity; joined with
OBJ_ACC="high"                          # the object into <obj>_<hand>_<obj>

# Convergence-terminated MPPI: plan() keeps iterating until the returned action
# settles -- sum_u (u_i - u_i+1)^2 < CONVERGENCE_TOL -- instead of running a
# fixed count, so the CSV's n_iterations column has nothing to say (drop the
# duplicate rows it would otherwise distinguish, or they rerun the same cell).
USE_CONVERGENCE=false                   # true | false
CONVERGENCE_TOL=1e-4                    # only read when USE_CONVERGENCE=true.
                                        # CALIBRATE THIS: the test is on
                                        # sum_u (u_i - u_i+1)^2 over all joints,
                                        # so a tol tighter than the update ever
                                        # reaches degenerates into a fixed
                                        # MAX_ITERATIONS run plus a device->host
                                        # read per iteration. 1e-4 ran to the cap
                                        # on every plan() of a duck/M2 smoke test.
                                        # Run one cell with --debug first: the
                                        # planner prints "converged in k/cap
                                        # iterations" per plan() call.
MAX_ITERATIONS=10                       # cap on that loop; must be >= 2

if [[ "$USE_CONVERGENCE" != "true" && "$USE_CONVERGENCE" != "false" ]]; then
    echo "USE_CONVERGENCE must be true or false, got '$USE_CONVERGENCE'" >&2
    exit 1
fi

# ===========================================================================
# 2. THE GRID inside each cell. Comma-separated, expanded by the driver — bash
#    never needs to know how many points these make.
# ===========================================================================
TEMPERATURES="100,50,25,12.5,6.25"                 # MPPI lambda: divides cost in the softmax
NOISE_SIGMAS="0.1"             # action-noise std dev; ONE value here
                                        # collapses this to a 1-D temperature sweep

# ===========================================================================
# 3. Episode settings — names match run_eval_episode.py, so a winning point
#    replays by copy-pasting the flags the driver prints. The planning horizon
#    and control-step duration are NOT here: they are per cell, in the CSV.
# ===========================================================================
PLANNER="mppi"                          # temperature is an MPPI-only field
N_EPISODES=10                            # episodes per grid point -> success rate
N_WORKERS=16                            # episodes run CONCURRENTLY, one process
                                        # each. THE VRAM KNOB: every worker holds
                                        # its own CUDA context and MJWarp Data at
                                        # this cell's n_samples worlds, so the
                                        # peak is N_WORKERS x max n_samples in
                                        # the CSV. Lower it (with --cpus-per-task
                                        # and --mem) on an OOM.
SEED=42                                  # THE constant episode seed: every grid
                                        # point is scored on the same episodes
                                        # (n_samples and n_iterations are cell
                                        # axes, up in section 1)
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
#    combine_results.py run merges every cell.
#    Export OUTDIR to a FIXED path to resume a previous submission wholesale
#    (the array job id changes on every sbatch); finished points are skipped.
# ===========================================================================
OUTDIR="${OUTDIR:-$ROOT_DIR/results/temp_sigma_grid_${SLURM_ARRAY_JOB_ID:-manual}}"
TASK_LABEL="${TASK_LABEL:-temp_sigma_grid}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

# ===========================================================================
# 5. This task's cell = row $SLURM_ARRAY_TASK_ID of the CSV. The helper does
#    the parsing (bash cannot read a CSV safely) and the range check; any
#    failure exits non-zero, which `set -eo pipefail` turns into a failed task
#    rather than a run of the wrong cell.
# ===========================================================================
T=${SLURM_ARRAY_TASK_ID:-0}
IFS=$'\t' read -r OBJ MODEL N_ITER N_SAMPLES TIME_HORIZON STEP_TIME < <(
    python experiments/hpc/read_temp_sigma_cell.py "$CELLS_CSV" "$T")
GEOMETRY="${OBJ}_${HAND_ACC}_${OBJ_ACC}"

# How this cell decides its iteration count. --convergence_tol makes the driver
# drop n_iterations entirely (and print that it is ignored), so the two are never
# passed together.
if [[ "$USE_CONVERGENCE" == "true" ]]; then
    ITER_ARGS=(--convergence_tol "$CONVERGENCE_TOL" --max_iterations "$MAX_ITERATIONS")
    ITER_DESC="converge<${CONVERGENCE_TOL} cap ${MAX_ITERATIONS} (CSV n_iterations=${N_ITER} ignored)"
else
    ITER_ARGS=(--n_iterations "$N_ITER")
    ITER_DESC="n_iterations=${N_ITER}"
fi

echo "cell $T (row $T of $CELLS_CSV): model=$MODEL object=$OBJ geometry=$GEOMETRY"
echo "  planner: $ITER_DESC  n_samples=$N_SAMPLES  time_horizon=${TIME_HORIZON}s  step_time=${STEP_TIME}s"
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
    --array_index  "$T" \
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
    "${ITER_ARGS[@]}" \
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
if [[ "$T" == "0" ]]; then
    # combine.slurm's #SBATCH --output is relative to ITS submission dir, which
    # for this nested call is $PROJ_DIR (cd'd to above), not experiments/hpc.
    mkdir -p "$PROJ_DIR/logs"
    sbatch --dependency=afterok:"${SLURM_ARRAY_JOB_ID}" \
           --export=ALL,OUTDIR="$OUTDIR",TASK="$TASK_LABEL" \
           experiments/hpc/combine.slurm || true
fi
