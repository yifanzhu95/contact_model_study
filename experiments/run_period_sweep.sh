#!/bin/bash
# Run the period sweep through horizon_closed_loop.py, one job at a time.
#
#   experiments/run_period_sweep.sh                       # sync
#   experiments/run_period_sweep.sh --async               # async
#   TAG=_bsig experiments/run_period_sweep.sh --async --brownian-sigma
#   MODELS="M1 M2" PERIODS="16 64" SEEDS=3 experiments/run_period_sweep.sh
#   MODELS=M5 STAGE=chunks experiments/run_period_sweep.sh   # one model's runs (one HPC array task)
#
# The jobs are period_sweep_jobs.py's: one run per model and control period, MODELS and PERIODS (ms)
# narrowing them, each split into chunks of horizons, every chunk running the same seeds (SEEDS of
# them, default 10) from the initial pose toward one goal (horizon_closed_loop.py --seeds
# --goal-face). STAGE picks what runs (default all, in order):
#   chunks  each chunk's horizons, to $OUTDIR/chunks/NAME_cXXofYY.{json,npz,log};
#   merge   each run whose chunks are all done, into $OUTDIR/NAME.{json,npz}, what the plots read.
# Sync by default; --async, first, runs the async sweep (planning time spent as sim time). Other flags
# go to every horizon_closed_loop.py run, and TAG is appended to every NAME, so a variant does not
# overwrite (or skip on) the base runs.
# OUTDIR defaults to results/horizon_kl_sweep. Anything already written is skipped, so a stopped
# sweep resumes. Async chunks measure latency, so keep the GPU otherwise idle while they run. PYTHON
# picks the interpreter (default: python, in the contact_comfree2 env). Exits non-zero if anything
# failed or a run could not be merged. On the HPC, hpc/submit_period_sweep.sh runs each model as its
# own array task.
set -u
MODE=sync
if [[ ${1:-} == --async ]]; then MODE=async; shift; fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
TAG=${TAG:-}
STAGE=${STAGE:-all}
OUT=${OUTDIR:-results/horizon_kl_sweep}
cd "$REPO" && mkdir -p "$OUT/chunks"
ASYNC=""; [[ $MODE == async ]] && ASYNC="--async"
FAILED=0

# MODELS and PERIODS are deliberately unquoted: each holds a list.
jobList() {
  $PY experiments/period_sweep_jobs.py "$MODE" --stage "$1" ${MODELS:+--models $MODELS} \
      ${PERIODS:+--periods $PERIODS} ${SEEDS:+--seeds $SEEDS}
}

runChunks() {
  local JOBS NAME CH M SUB N W G HS R RC
  JOBS=$(jobList chunks) || { FAILED=$((FAILED + 1)); return; }
  while read -r NAME CH M SUB N W G HS; do
    [ -z "$NAME" ] && continue
    R=$OUT/chunks/${NAME}${TAG}_$CH
    if [ -f "$R.npz" ]; then echo "skip $NAME$TAG $CH (done)"; continue; fi
    echo "start $NAME$TAG $CH $(date +%T)"
    # HS is deliberately unquoted: it holds the horizons.
    $PY experiments/horizon_closed_loop.py $ASYNC --model "$M" --substeps "$SUB" --seeds "$N" --window "$W" \
        --goal-face "$G" --horizons $HS --mean-cost-over-horizon --temperature 0.1 "$@" --out "$R.json" \
        < /dev/null > "$R.log" 2>&1
    RC=$?
    echo "done $NAME$TAG $CH exit $RC $(date +%T)"
    if (( RC != 0 )); then
      FAILED=$((FAILED + 1))
      tail -n 20 "$R.log"
    fi
  done <<< "$JOBS"
}

runMerge() {
  local JOBS NAME M SUB N W G C PARTS c P
  JOBS=$(jobList runs) || { FAILED=$((FAILED + 1)); return; }
  while read -r NAME M SUB N W G C; do
    [ -z "$NAME" ] && continue
    if [ -f "$OUT/$NAME$TAG.npz" ]; then echo "skip merge $NAME$TAG (done)"; continue; fi
    PARTS=()
    for (( c = 0; c < C; c++ )); do
      P=$(printf "%s/chunks/%s%s_c%02dof%02d" "$OUT" "$NAME" "$TAG" "$c" "$C")
      [[ -f $P.npz ]] && PARTS+=("$P.json")
    done
    if (( ${#PARTS[@]} < C )); then
      echo "not merged $NAME$TAG: ${#PARTS[@]} of $C chunks done"
      FAILED=$((FAILED + 1))
      continue
    fi
    $PY experiments/horizon_cl_results.py merge "$OUT/$NAME$TAG.json" "${PARTS[@]}" || FAILED=$((FAILED + 1))
  done <<< "$JOBS"
}

echo "sweep start $(date +%T)  $MODE $STAGE -> $OUT${TAG:+  tag $TAG}${*:+  flags $*}"
case $STAGE in
  chunks) runChunks "$@" ;;
  merge)  runMerge ;;
  all)    runChunks "$@"; runMerge ;;
  *)      echo "STAGE must be chunks, merge or all, not $STAGE"; exit 1 ;;
esac
echo "sweep end $(date +%T), $FAILED failed or not merged"
(( FAILED == 0 ))
