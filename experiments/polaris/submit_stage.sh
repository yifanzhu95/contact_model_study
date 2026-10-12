#!/bin/bash
# Submit one stage; the operator explicitly selects a previously validated N.
set -euo pipefail
if [[ $# -ne 7 && $# -ne 8 ]]; then
    echo "usage: $0 <g1|g2|g3> <project> <repo> <env> <eagle-root> <previous-verdict.json> <N or - for g1> [resume-outdir]" >&2
    exit 2
fi
STAGE="$1"
ALCF_PROJECT="$2"
PROJ_DIR=$(realpath "$3")
ENV_DIR=$(realpath "$4")
EAGLE_ROOT=$(realpath "$5")
PREVIOUS_VERDICT=$(realpath "$6")
NSAMPLES="$7"
case "$STAGE:$NSAMPLES" in
    g1:-|g2:32|g2:64|g2:128|g2:256|g3:32|g3:64|g3:128|g3:256) ;;
    *) echo "choose g1 with -, or g2/g3 with N=32,64,128,256" >&2; exit 2 ;;
esac
command -v qsub >/dev/null || { echo "qsub unavailable; submit from Polaris" >&2; exit 2; }
[[ -x "$ENV_DIR/bin/python" && -f "$PREVIOUS_VERDICT" ]] || {
    echo "environment or previous verdict is missing" >&2; exit 2;
}
[[ "$EAGLE_ROOT" == /eagle/* || "$EAGLE_ROOT" == /lus/eagle/projects/* ]] || {
    echo "output root must be on Eagle" >&2; exit 2;
}
cd "$PROJ_DIR"
EXPECTED_SHA=$(git rev-parse HEAD)
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "a clean local deployment commit is required" >&2; exit 2;
}
options=()
if [[ "$STAGE" != g1 ]]; then options+=(--n-samples "$NSAMPLES"); fi
"$ENV_DIR/bin/python" experiments/polaris/run_stage.py "$STAGE" --check \
    --outdir /tmp/polaris-plan-preview --expected-sha "$EXPECTED_SHA" \
    --previous "$PREVIOUS_VERDICT" "${options[@]}" > /dev/null
if [[ $# -eq 8 ]]; then
    OUTDIR=$(realpath "$8")
    [[ -f "$OUTDIR/stage_plan.json" ]] || { echo "resume directory has no stage plan" >&2; exit 2; }
else
    OUTDIR=$(mktemp -d "$EAGLE_ROOT/kamino_${STAGE}_${EXPECTED_SHA:0:12}.XXXXXX")
fi
[[ "$OUTDIR" == /eagle/* || "$OUTDIR" == /lus/eagle/projects/* ]] || {
    echo "resume output directory must be on Eagle" >&2; exit 2;
}
for value in "$ALCF_PROJECT" "$PROJ_DIR" "$ENV_DIR" "$OUTDIR" "$PREVIOUS_VERDICT"; do
    [[ "$value" != *","* && "$value" != *$'\n'* ]] || {
        echo "PBS variables cannot contain commas or newlines" >&2; exit 2;
    }
done
VARIABLES="PROJ_DIR=$PROJ_DIR,ENV_DIR=$ENV_DIR,OUTDIR=$OUTDIR,EXPECTED_SHA=$EXPECTED_SHA,ALCF_PROJECT=$ALCF_PROJECT,STAGE=$STAGE,PREVIOUS_VERDICT=$PREVIOUS_VERDICT,NSAMPLES=$NSAMPLES"
JOB_ID=$(qsub -A "$ALCF_PROJECT" -v "$VARIABLES" \
    -o "$OUTDIR/pbs.stdout" -e "$OUTDIR/pbs.stderr" experiments/polaris/run_stage.pbs)
echo "submitted $STAGE: $JOB_ID"
echo "results: $OUTDIR"
