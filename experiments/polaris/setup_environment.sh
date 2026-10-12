#!/bin/bash -l
# Create the explicit Python 3.12 M5 environment from a Polaris compute node.
# This is a one-time preparation step, never part of a production experiment.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <absolute-repo-dir> <absolute-conda-env-dir>" >&2
    exit 2
fi
PROJ_DIR=$(realpath "$1")
ENV_DIR="$2"
[[ "$PROJ_DIR" = /* && "$ENV_DIR" = /* ]] || {
    echo "repo and environment paths must be absolute" >&2
    exit 2
}
[[ -f "$PROJ_DIR/experiments/polaris/requirements-m5.txt" ]] || {
    echo "not a deployment checkout: $PROJ_DIR" >&2
    exit 2
}
[[ -n "${PBS_JOBID:-}" ]] || {
    echo "run this inside an interactive Polaris compute job, not on a login node" >&2
    exit 2
}

set +u
module use /soft/modulefiles
module load conda
conda activate base
set -u

# Polaris compute nodes reach package indexes through ALCF's documented proxy.
# Only the one-time environment build uses network access; the smoke job does not.
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    conda create --yes --prefix "$ENV_DIR" python=3.12 pip
fi
conda activate "$ENV_DIR"
python -m pip install --requirement "$PROJ_DIR/experiments/polaris/requirements-m5.txt"
python -m pip check
python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
print(sys.version)
PY

cd "$PROJ_DIR"
python -m pytest -q \
    tests/test_provenance.py \
    tests/test_validation_summary.py \
    tests/test_kamino_diagnostics.py \
    tests/test_polaris_package.py \
    tests/test_polaris_stages.py \
    tests/test_polaris_pipeline.py \
    tests/test_polaris_direct.py \
    tests/test_polaris_campaign.py \
    tests/test_parallel_drivers.py \
    tests/test_kamino.py::test_config_validation \
    tests/test_episode_batches.py -m "not gpu"

conda list --explicit > "$ENV_DIR/conda-explicit.txt"
python -m pip freeze --all > "$ENV_DIR/pip-freeze.txt"
echo "environment ready: $ENV_DIR"
