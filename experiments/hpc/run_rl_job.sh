#!/usr/bin/env bash
# Train one RL policy inside an allocated single-GPU job, or locally:
#   RL_PYTHON=/path/to/envs/contact_kamino/bin/python \
#     experiments/hpc/run_rl_job.sh --config configs/rl/cube_m2.json \
#     --timesteps 1000000 --max-seconds 14000 --output results/rl/cube_m2
# exec hands the job's signals straight to Python, so the scheduler's SIGTERM
# stops training after the current PPO update and the run is saved.
set -euo pipefail
: "${RL_PYTHON:?Set RL_PYTHON to the absolute Python executable in the target environment}"
if [[ "$RL_PYTHON" != /* || ! -x "$RL_PYTHON" ]]; then
    echo "RL_PYTHON must be an absolute executable path" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../.."
exec "$RL_PYTHON" -m ContactModelStudy.Drivers.train_rl_policy "$@"
