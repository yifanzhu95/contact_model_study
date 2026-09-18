#!/bin/bash
#SBATCH --job-name=cont_modeling_kl_divergence_eval
#SBATCH --output=logs/cont_modeling_kl_divergence_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_kl_divergence_eval%A_%a.err
#SBATCH --array=0-39
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00

set -euo pipefail

# Submission template; this file does not submit jobs itself.
# 5 objects x 4 sample counts x 2 iteration counts = 40 real cells. At 30
# episodes/cell this is 1200 episodes. Optional null diagnostics are not part of
# the default array. Each object's real cells produce 8 scatter points. Check
# resources and benchmark high_high cells before explicitly submitting.
#
# Both planners use the same high_high scene/contact model. The smaller planner
# controls the simulator; the higher-compute reference is a shadow. KL compares
# Gaussian moment approximations of weighted FIRST-action candidates, not the
# full sequence distribution or a proven optimal control distribution.
# At each KL measurement, the reference proposal starts independently from
# N(0, sigma), then updates only within that solve. It terminates when the
# squared-L2 first-action change is below 1e-3 or at the 25-iteration cap.
#
# Optional null cells use equal compute, the same pre-solve proposal, and
# separate random sampling. They remain
# independent closed-loop runs, so their visited states need not match real
# cells. Their KL is a diagnostic comparison, not a pure matched-state noise
# floor or a significance test.
#
# The existing ccd_iterations=35 is deliberately unchanged. Warnings originate
# in GPU MuJoCo-Warp convex collision detection used by rollout, not Pinocchio
# ADMM. A finite KL does not establish collision convergence.
#
# Reference timing around 350-370 ms was observed in an earlier cube_low_high
# local pilot. Do not extrapolate that to high_high objects or this cluster.
# The 16-hour resource request requires fresh per-cell timing validation.
#
# Preview only (no modules, activation, directories, GPU work or submission):
#   bash experiments/hpc/kl_divergence_eval.slurm --dry-run 0
#   bash experiments/hpc/kl_divergence_eval.slurm --dry-run-all
#
# Explicit submission, from the repository root:
#   mkdir -p logs results
#   sbatch experiments/hpc/kl_divergence_eval.slurm
# PROJ_DIR may override the repository path, e.g. when submitting elsewhere.
# SLURM output paths above are relative to the submission directory; its logs/
# directory must already exist. The cluster module/conda setup is preserved;
# KL_SKIP_ENV_SETUP=1 permits an already-configured environment.

MODE="run"
REQUESTED_CELL="${SLURM_ARRAY_TASK_ID:-0}"
case "${1:-}" in
    --dry-run)
        [[ $# -le 2 ]] || { echo "Usage: $0 --dry-run [cell_id]" >&2; exit 2; }
        MODE="dry-run"
        REQUESTED_CELL="${2:-$REQUESTED_CELL}"
        ;;
    --dry-run-all)
        [[ $# -eq 1 ]] || { echo "Usage: $0 --dry-run-all" >&2; exit 2; }
        MODE="dry-run-all"
        ;;
    "")
        if [[ -z "${SLURM_ARRAY_TASK_ID+x}" ]]; then
            echo "No array task selected. Preview with --dry-run [id] or --dry-run-all; submit explicitly with sbatch." >&2
            exit 2
        fi
        ;;
    *)
        echo "Usage: $0 [--dry-run [cell_id] | --dry-run-all]" >&2
        exit 2
        ;;
esac

if [[ -n "${PROJ_DIR:-}" ]]; then
    ROOT_DIR="$PROJ_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ ! -f "$ROOT_DIR/experiments/hpc/run_kl_divergence_cell.py" ]]; then
    echo "Project not found at $ROOT_DIR. Set PROJ_DIR to the repository root." >&2
    exit 2
fi
cd -- "$ROOT_DIR"

TASK="grasp_reorient"
MODEL="${MODEL:-M3}"
N_EPISODES="${N_EPISODES:-30}"
MAX_STEPS="${MAX_STEPS:-1000}"
GOAL_DIFFICULTY=1
OBJECT_SWEEP=(cube duck ball spam tomato)
N_SAMPLES_SWEEP=(16 64 256 1024)
N_ITER_SWEEP=(1 2)
NULL_SWEEP=(no)

REF_N_SAMPLES=4096
REF_N_ITERATIONS=25
REF_CONVERGENCE_TOL=1e-3
REF_MAX_ITERATIONS=25
REF_TEMPERATURE=50.0
REFERENCE_INIT="zero"
KL_EVERY=20
KL_SHRINKAGE=0.001
KL_DIRECTION="forward"
EXECUTE="mean"
TIME_HORIZON=0.352
STEP_TIME=0.064
TEMPERATURE=1.0
NOISE_SIGMA=0.025
EVAL_SIM="pinocchio"
SETTLE=1.0
SEED="${SEED:-0}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
OUTDIR="${OUTDIR:-$ROOT_DIR/results/kl_divergence_eval_${SLURM_ARRAY_JOB_ID:-manual}}"

# Keep #SBATCH --array consistent if any grid axis changes.
NS=${#N_SAMPLES_SWEEP[@]}
NI=${#N_ITER_SWEEP[@]}
NN=${#NULL_SWEEP[@]}
NO=${#OBJECT_SWEEP[@]}
TOTAL_CELLS=$(( NS * NI * NN * NO ))

validate_cell() {
    if [[ ! "$1" =~ ^[0-9]+$ ]] || [[ ${#1} -gt 6 ]]; then
        echo "Invalid cell id: $1" >&2
        return 2
    fi
    if (( 10#$1 >= TOTAL_CELLS )); then
        echo "Cell id must be between 0 and $((TOTAL_CELLS - 1))." >&2
        return 2
    fi
}

run_cell() {
    local cell_id=$((10#$1))
    local t=$cell_id
    local sample_idx=$(( t % NS )); t=$(( t / NS ))
    local iter_idx=$(( t % NI )); t=$(( t / NI ))
    local null_idx=$(( t % NN )); t=$(( t / NN ))
    local object_idx=$(( t % NO ))
    local object_name=${OBJECT_SWEEP[$object_idx]}
    local geometry="${object_name}_high_high"
    local samples=${N_SAMPLES_SWEEP[$sample_idx]}
    local iterations=${N_ITER_SWEEP[$iter_idx]}
    local null_control=${NULL_SWEEP[$null_idx]}
    local null_flag="--no-null_control"
    [[ "$null_control" == yes ]] && null_flag="--null_control"

    # Full per-step trajectory/distribution recording is off to limit storage.
    # Per-measurement KL, ESS, validity and timing remain recorded.
    local command=(
        "$PYTHON_EXECUTABLE" experiments/hpc/run_kl_divergence_cell.py
        --no-record_trajectory --no-record_planner_dist
        --outdir "$OUTDIR" --task "$TASK" --model "$MODEL"
        --geometry "$geometry" --n_samples "$samples"
        --n_iterations "$iterations" --ref_n_samples "$REF_N_SAMPLES"
        --ref_n_iterations "$REF_N_ITERATIONS"
        --ref_convergence_tol "$REF_CONVERGENCE_TOL"
        --ref_max_iterations "$REF_MAX_ITERATIONS"
        --ref_temperature "$REF_TEMPERATURE"
        --reference_init "$REFERENCE_INIT" "$null_flag"
        --n_episodes "$N_EPISODES" --max_steps "$MAX_STEPS"
        --goal_difficulty "$GOAL_DIFFICULTY" --kl_every "$KL_EVERY"
        --kl_shrinkage "$KL_SHRINKAGE" --kl_direction "$KL_DIRECTION"
        --execute "$EXECUTE" --time_horizon "$TIME_HORIZON"
        --step_time "$STEP_TIME" --temperature "$TEMPERATURE"
        --noise_sigma "$NOISE_SIGMA" --eval_sim "$EVAL_SIM"
        --settle "$SETTLE" --seed "$SEED" --debug
    )

    echo "cell=$cell_id geometry=$geometry model=$MODEL samples=$samples iterations=$iterations null=$null_control episodes=$N_EPISODES"
    if [[ "$MODE" != run ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

if [[ "$MODE" == dry-run-all ]]; then
    echo "Preview: $TOTAL_CELLS cells, $((TOTAL_CELLS * N_EPISODES)) episodes; nothing will run."
    for ((cell = 0; cell < TOTAL_CELLS; cell++)); do
        run_cell "$cell"
    done
    exit 0
fi

validate_cell "$REQUESTED_CELL"
if [[ "$MODE" == run ]]; then
    mkdir -p -- "$OUTDIR"
    if [[ "${KL_SKIP_ENV_SETUP:-0}" != 1 ]]; then
        module load miniconda
        eval "$(conda shell.bash hook)"
        conda activate contact_modeling
    fi
    # Resolve imports here rather than to a different editable installation.
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi
run_cell "$REQUESTED_CELL"
