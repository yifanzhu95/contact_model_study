# search_config.sh — single source of truth for the weight grid search.
#
# Sourced by param_search.slurm, combine.slurm, and submit_param_search.sh so the
# grid is defined in exactly one place. Edit the lists below to change the sweep.

# Task and how many episodes to run per weight set (to estimate the success rate).
TASK="grasp_reorient"
N_EPISODES=5

# Contact models to sweep — the outer grid axis (one job block per model).
MODELS=(M1 M2 M3 M4)

# The list of each weight the search ranges over: one "name=v1,v2,..." token per
# weight. The Cartesian product of these (times MODELS) is the set of jobs; each
# array task runs one combination. Names must match the task's cost_weights keys.
WEIGHTS=(
  "w_quat=15,20,25"
  "w_pos=15,20,25"
  "w_contact=7.5,10,12.5"
  "w_joint=0.025,0.05,0.1"
)

# MPPI / eval knobs shared by every cell (single values, not swept).
N_SAMPLES=256
HORIZON=48
TEMPERATURE=1.0
NOISE_SIGMA=0.01
DELTA=0.1
SUBSTEPS=16
EVAL_SIM="none"     # none | mujoco | drake | pinocchio
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# Total number of grid cells = |MODELS| * prod(#values per weight).
# Computed in pure bash so it works on a login node without importing warp/CUDA.
num_cells() {
  local n=${#MODELS[@]} w vals commas
  for w in "${WEIGHTS[@]}"; do
    vals="${w#*=}"          # strip "name="
    commas="${vals//[^,]/}" # keep only commas
    n=$(( n * (${#commas} + 1) ))
  done
  echo "$n"
}
