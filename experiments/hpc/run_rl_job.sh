#!/usr/bin/env bash
# Scheduler-neutral: call inside an allocated single-GPU job, or locally.
# Forward the scheduler's GPU visibility and signals; never submit another job.
set -euo pipefail
: "${RL_PYTHON:?Set RL_PYTHON to the absolute Python executable in the target environment}"
if [[ "$RL_PYTHON" != /* || ! -x "$RL_PYTHON" ]]; then
    echo "RL_PYTHON must be an absolute executable path" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../.."
exec "$RL_PYTHON" -m ContactModelStudy.Drivers.run_rl_job "$@"
