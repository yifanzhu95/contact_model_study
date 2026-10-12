#!/bin/bash
# Validate and submit exactly one Polaris G0 M5 smoke job.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <ALCF-project> <absolute-repo-dir> <absolute-conda-env-dir> <absolute-eagle-output-root>" >&2
    exit 2
fi
ALCF_PROJECT="$1"
PROJ_DIR=$(realpath "$2")
ENV_DIR=$(realpath "$3")
EAGLE_ROOT=$(realpath "$4")

command -v qsub >/dev/null || { echo "qsub is unavailable; run this on Polaris" >&2; exit 2; }
[[ -x "$ENV_DIR/bin/python" ]] || { echo "environment has no Python: $ENV_DIR" >&2; exit 2; }
[[ "$EAGLE_ROOT" == /eagle/* || "$EAGLE_ROOT" == /lus/eagle/projects/* ]] || {
    echo "output root must be on Eagle: $EAGLE_ROOT" >&2
    exit 2
}
for value in "$PROJ_DIR" "$ENV_DIR" "$EAGLE_ROOT"; do
    [[ "$value" != *","* && "$value" != *$'\n'* ]] || {
        echo "PBS -v paths may not contain commas or newlines: $value" >&2
        exit 2
    }
done

cd "$PROJ_DIR"
EXPECTED_SHA=$(git rev-parse HEAD)
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "source tree is dirty; commit or remove every change before submission" >&2
    exit 2
}
# A clean local commit is sufficient. The operator may receive this checkout
# in an offline Git bundle; publication to GitHub is not required.

CHECK_DIR=$(mktemp -d /tmp/kamino-g0-check.XXXXXX)
trap 'rm -rf "$CHECK_DIR"' EXIT
"$ENV_DIR/bin/python" experiments/run_episode_batches.py \
    experiments/polaris/m5_smoke.csv --check --outdir "$CHECK_DIR"

RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="$EAGLE_ROOT/kamino_g0_m5_${RUN_TAG}_${EXPECTED_SHA:0:12}"
mkdir -p "$OUTDIR"
VARIABLES="PROJ_DIR=$PROJ_DIR,ENV_DIR=$ENV_DIR,OUTDIR=$OUTDIR,EXPECTED_SHA=$EXPECTED_SHA,ALCF_PROJECT=$ALCF_PROJECT"
JOB_ID=$(qsub \
    -A "$ALCF_PROJECT" \
    -v "$VARIABLES" \
    -o "$OUTDIR/pbs.stdout" \
    -e "$OUTDIR/pbs.stderr" \
    experiments/polaris/run_m5_smoke.pbs)

echo "submitted Polaris G0 job: $JOB_ID"
echo "source:  $EXPECTED_SHA"
echo "results: $OUTDIR"
echo "inspect: qstat -f $JOB_ID"
