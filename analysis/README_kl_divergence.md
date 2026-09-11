# KL Divergence vs. Success Rate

This document describes the **online closed-loop KL workflow** in this
repository. [Recorded-log replay](README_offline_recorded_kl.md) uses a separate
workflow with a different state-sampling protocol. This workflow studies how a
lower-compute MPPI planner's first-action
distribution differs from a fixed higher-compute reference, and how that
diagnostic relates to closed-loop task success. It measures planner behavior,
not one-step physical state prediction error. It is not a causal test that KL
alone determines success.

See [`kl_delivery_validation_20260907.md`](kl_delivery_validation_20260907.md)
for the local delivery checks, shadow audit and fixed-particle sensitivity.
The [compact plotting example](examples/kl_cube_high_high_20260907/README.md)
can be redrawn on CPU from its four included cell JSONs.

## Scope and geometry

The KL worker supports the five current `grasp_reorient` objects: `cube`,
`duck`, `ball`, `spam`, and `tomato`. Every object uses its `high_high` scene:
high-fidelity hand geometry and high-fidelity object geometry as defined by the
project, not a claim of exact real-world geometry. This choice applies to both
planners' rollout scenes. The existing object-specific evaluation scene is
retained; its hand collision geometry need not be identical to the rollout
hand. Do not replace the evaluation scene simply to force geometric equality.

The per-object evaluation paths are Pinocchio or MuJoCo; local compatibility
checks use Pinocchio. The legacy Drake implementation loads a hand-only URDF,
not these per-object scenes, so this KL grasp-reorient workflow rejects
`--eval_sim drake` rather than misreporting its model identity.

For this worker, `--geometry duck` normalizes to `duck_high_high`; an explicit
`--geometry duck_high_high` is also accepted. The same rule applies to all five
objects. Lower-fidelity variants and ambiguous legacy aliases such as
`accurate` are rejected. Other project drivers retain their existing geometry
behavior. The earlier local `cube_low_high` pilot is a debugging record and
does not satisfy the current high-high protocol.

The current difficulty-1 goal samples a canonical orientation rotated about
the z axis, not a fixed rotation relative to each object's settled pose. The
same difficulty value therefore does not establish equal angular difficulty
across objects; object-specific reporting remains necessary.

## What is compared

The degraded and reference planners use the same contact model, task state,
goal, cost settings, horizon, control period, and noise scale. The acting
temperature remains an independent cell setting; the reference defaults to a
fixed temperature of 50. Their sample counts and iteration budgets also differ.
Therefore, when the acting temperature is not 50, KL is not a pure compute-only
comparison. The reference is a higher-compute numerical comparison, not a proven optimum. The evaluation
simulator (Pinocchio by default) is distinct from this reference planner.

At selected control steps, each MPPI planner produces weighted candidate
action sequences. The worker extracts the first action of every sequence,
matches its weighted action cloud to a Gaussian, and computes both

- forward KL: `KL(reference || degraded)`; and
- reverse KL: `KL(degraded || reference)`.

Forward KL is the default headline metric; both directions are stored. This is
a first-action Gaussian approximation, not an exact KL of the full sequence
distribution. Different multimodal clouds can have the same first two moments.
KL is directional, unbounded, and dimensionless (natural-log units, or nats).

The degraded planner drives the evaluation simulator. The reference is a
shadow and never supplies an applied command. At a measurement step both
planners receive the same physical state. By default, every reference solve
restarts its proposal from `N(0, sigma)`. The reference carries its updated
mean only between optimizer iterations inside that one solve, then discards it
before the next measured state. It never loads the degraded planner's mean.
Explicit compatibility modes can reproduce the older same-pre-solve-mean or
persistent-reference protocols; their results carry different protocol
metadata and must not be pooled with the default.

## Files

- `experiments/hpc/run_kl_divergence_cell.py`: one self-describing compute cell.
- `experiments/hpc/kl_divergence_eval.slurm`: an unsubmitted SLURM grid template.
- `analysis/plot_kl_divergence_dir.py`: compatible-cell aggregation and plots.
- `contact_study/evaluation/distributions.py`: Gaussian moments, ESS and KL.
- `analysis/kl_shrinkage_sensitivity.py`: CPU recomputation on optional raw
  first-action moments from the exact weighted particles used in each solve.
- `analysis/audit_kl_shadow.py`: short GPU audit of shared inputs, buffer
  ownership, environment RNG consumption and degraded sampling counters.
- `tests/test_kl_analysis.py`: CPU arithmetic, validation and aggregation tests.
- `tests/test_kl_worker_config.py`: worker scene and result-identity checks.
- `tests/test_kl_plot_grouping.py`: configuration isolation, duplicate protection
  and multi-figure plotting checks.
- `tests/test_kl_scene_assets.py`: CPU parsing and state/control layout checks
  for all five high/high rollout and evaluation scene pairs.
- `tests/test_kl_delivery.py`: worker/grid default agreement, fixed-particle
  moment reproduction and scoring after the last allowed command.

## Recommended settings and episode scoring

The Python worker and SLURM template now share these scientific defaults:

| Setting | Value |
|---|---:|
| Contact model | M3 |
| Default object geometry | cube_high_high |
| Goal difficulty | 1 |
| Requested horizon / control period | 0.352 s / 0.064 s |
| Acting temperature / proposal sigma | 1.0 / 0.025 |
| Reference temperature | 50 |
| Large reference | 4096 samples, convergence tol. 0.001, cap 25 iterations |
| Reference initialization | Zero mean independently at every measured state |
| KL interval / shrinkage | 20 control steps / 0.001 |
| Executed action | Degraded weighted mean |

The worker defaults to 64 samples, one iteration and ten episodes. The grid
explicitly sweeps sample/iteration counts and requests 30 episodes per cell.
Per-object evaluation uses the task default, Pinocchio; documented reproduction
commands also name it explicitly. The recorded realized horizon is 0.320 s:
five whole 0.064 s control intervals fit inside the requested 0.352 s.

Every episode also scores the state after the last allowed control command.
Success or a drop on that command is no longer mislabelled timeout. The
`config.kl_protocol = first_action_v3_reference_zero_final_state_check` tag prevents the plotter
from pooling this terminal-scoring rule with older result files. This changes
only classification at the final-command boundary, not the applied controls.

## Local smoke check

Run from this checkout with the project dependencies installed. If an editable
installation points to another checkout, explicitly select the current one:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The following real-cell run checks wiring only. Two control steps, one episode
and a reduced reference are not a success-rate experiment or a reference
quality validation. The `duck` shorthand may be replaced by any listed object.

```bash
python experiments/hpc/run_kl_divergence_cell.py \
    --outdir /tmp/kl_high_high_smoke --task grasp_reorient --model M3 \
    --geometry duck --n_samples 16 --n_iterations 1 \
    --ref_n_samples 64 --ref_convergence_tol 1e-3 \
    --ref_max_iterations 2 --ref_temperature 50 --reference_init zero \
    --n_episodes 1 --max_steps 2 --kl_every 1 \
    --time_horizon 0.352 --step_time 0.064 \
    --temperature 1.0 --noise_sigma 0.025 \
    --eval_sim pinocchio --settle 1 --goal_difficulty 1 --seed 0 \
    --no-record_trajectory --no-record_planner_dist

python analysis/plot_kl_divergence_dir.py /tmp/kl_high_high_smoke
```

Inspect resolved geometry and paths, KL, ESS, invalid measurements, planning
times, and numerical warnings. An artificially short run ending in timeout is
expected and does not establish task failure under a normal episode budget.

## Pilot and cluster sweep

A longer pilot with the intended reference settings can be started explicitly:

```bash
python experiments/hpc/run_kl_divergence_cell.py \
    --outdir results/kl_high_high_pilot --task grasp_reorient --model M3 \
    --geometry cube_high_high --n_samples 256 --n_iterations 1 \
    --ref_n_samples 4096 --ref_convergence_tol 1e-3 \
    --ref_max_iterations 25 --ref_temperature 50 --reference_init zero \
    --n_episodes 3 --max_steps 1000 --kl_every 20 \
    --time_horizon 0.352 --step_time 0.064 \
    --temperature 1.0 --noise_sigma 0.025 \
    --eval_sim pinocchio --settle 1 --goal_difficulty 1 --seed 0 \
    --no-record_trajectory --no-record_planner_dist --record_kl_moments
```

Three episodes remain a pilot, not a basis for ranking configurations or
establishing a KL-success relationship.

For a second configuration point, repeat the real command with `--n_samples 16`;
retain all other settings, including the seed and reference.
Do not rerun the same completed cell/seed into that directory and pool it as
additional independent evidence.

Plot this directory explicitly:

```bash
python -m analysis.plot_kl_divergence_dir results/kl_high_high_pilot \
    --out results/kl_high_high_pilot/kl_vs_sr.pdf

python -m analysis.plot_kl_divergence_dir results/kl_high_high_pilot \
    --reference_filter converged \
    --out results/kl_high_high_pilot/kl_vs_sr_converged_only.pdf
```

The first figure includes every finite planner-valid KL measurement. The
second uses the same results but retains only measurements whose reference
solve met its convergence tolerance; it does not rerun either planner.

The SLURM template defines 40 real cells (array indices 0-39): five objects,
four sample counts (16, 64, 256, 1024), and two iteration counts (1, 2).
At 30 episodes/cell this is **1200 episodes**. Null diagnostics are intentionally
left out of the default array and can only be run explicitly.
Each object contributes eight real configuration points. This template has
not been launched as part of the smoke-check workflow.

Preview every generated command without running Python, loading cluster
modules, creating output directories, or submitting jobs:

```bash
bash experiments/hpc/kl_divergence_eval.slurm --dry-run 0
bash experiments/hpc/kl_divergence_eval.slurm --dry-run-all
```

Review the cluster partition, GPU request, environment setup, storage and time
budget. `PROJ_DIR` can override the project path; otherwise the script uses
the SLURM submission directory or its own repository root for local previews.
Submit from the repository root so relative SLURM log paths are correct:

```bash
mkdir -p logs results
sbatch experiments/hpc/kl_divergence_eval.slurm
```

Submission is an explicit separate operation. Invoking the script locally
without an array task or a dry-run option refuses to start an experiment.
`N_EPISODES`, `MAX_STEPS`, `MODEL`, `SEED`, `OUTDIR`, `PROJ_DIR`, and
`PYTHON_EXECUTABLE` can be supplied as environment overrides.
`KL_SKIP_ENV_SETUP=1` skips cluster module/conda setup for an already-configured
environment. If any grid axis is edited, update `#SBATCH --array` accordingly.

Both degraded and shadow reference timings are recorded. The older local
cube-low-high reference timing (roughly 350-370 ms/solve) does not validate
high-high timings on other objects or cluster hardware. Benchmark before
treating the existing 16-hour request as adequate. Synchronous simulation waits
for planning; these measurements do not establish asynchronous real-time
control feasibility.

## Result identity and plotting

Each new cell JSON carries its own configuration and resolved scene identity.
Filenames include geometry and run-identifying information so different
objects, settings and repeated runs do not silently overwrite a shared cell
file. A directory-wide metadata file is not the authority for all cells.

For an explicitly requested null run, the shadow uses the degraded planner's
sample count, fixed iteration count, temperature and pre-solve proposal, but a
different noise seed. Because this same-proposal null protocol differs from the
default real cell's zero-start reference protocol, it is a separate diagnostic,
not a matched companion that should automatically be attached to the main plot.

The plotter separates geometry and incompatible experiment settings into
distinct plots. It must not merge a duck point with a cube point merely because
both have the same model/sample/iteration labels. Legacy records without
geometry identity remain unknown; `accurate` is never retroactively interpreted
as `high_high`. Keep legacy debugging data separate from new protocol data.

Compatible different-seed episodes may be pooled. Duplicate result payloads
or overlapping episode seed identities are rejected: copied files and same-seed
reruns must not silently inflate the independent episode count. Same-seed
repeatability studies need a separate repeated-measures analysis.

Each scientific configuration family produces a separate figure. `--geometry
duck_high_high` selects one geometry; for several families, an explicit `--out`
name gains geometry/configuration suffixes automatically. Legacy null records
without the requested reference budget are matched only if there is one
otherwise-compatible real budget; ambiguous nulls remain unpaired.

```bash
python analysis/plot_kl_divergence_dir.py results/kl_divergence_eval_<job_id>
```

Default aggregation gives each valid episode equal KL weight. Horizontal
uncertainty is the standard error across episode KL means; vertical success
uncertainty is a Wilson 95% interval. These are different statistics. With only
one valid episode, between-episode variability cannot be estimated. A broad
success interval at 0/3 successes is expected, not a plotting defect.

The worker's final console line labels its episode-mean headline to match the
default figure and also prints the legacy pooled-step mean explicitly. The
top-level JSON `kl` summaries retain pooled-step statistics for compatibility;
the plotter reconstructs episode-balanced values from `per_step` or episode
summaries. These can differ when episode lengths differ.

The console prints KL SE as `n/a` rather than zero when it cannot be estimated.
For the default episode-mean view, null screening also remains unavailable
unless both the real and null cells contain at least two valid episodes.
Reaching that minimum only enables the descriptive screening rule; it is not
evidence of adequate statistical power or a formal significance test.

`--weighting step --success_error se` selects the older pooled-step display;
correlated measurements and unequal episode lengths limit its interpretation.
`--stat median` is a robustness view, not an automatically available confidence
interval for the median. Episodes without valid KL still contribute to success
rate but cannot contribute to the KL average; inspect their counts and reasons.

## Controls and numerical diagnostics

### Common random numbers and optional independent null runs

Episode `k` uses the same environment/goal seed across compute cells. A real/null
pair also uses the same degraded-planner seed. Environment, degraded-planner
and reference-planner seeds are stored separately. This reduces deliberate
random-condition imbalance but does not guarantee bitwise repeatability or
identical closed-loop states. A short repeated-process audit found small action
differences at the first degraded solve across processes with exactly matching
input states and noise, before the first shadow solve. The checked degraded
buffers and counters were unchanged by shadow execution. This narrows the
earliest observed discrepancy to planning, without establishing its specific
backend cause or making later closed-loop divergence harmless.

Null is disabled in the default SLURM array. When requested explicitly,
it replaces the high-compute reference with an independently sampled planner
of the degraded planner's compute, temperature and pre-solve proposal. Null
remains a **separate closed-loop run**.
Thus a real/null pair creates four planner instances across two processes: a
degraded controller and high-compute shadow in the real cell, then a new copy
of the degraded controller and an equal-compute shadow in the null cell. The
two degraded controllers are configured with matching seeds, but they are not
the same in-memory controller and only the controller in its own cell is
executed. No KL is computed across the real and null processes.
It diagnoses differences that can occur even at equal compute and the same
proposal within its own run, but finite sampling, trajectory differences and
numerical effects may all contribute. Since the default real reference starts
from zero instead, the null does not isolate a single difference from that real
protocol. Do not subtract it as a calibrated noise correction or interpret it
as a significance test.

### Episode-balanced aggregation

The default first summarizes KL within each episode, then weights episodes
equally. A timeout does not outweigh an early success merely because it has
more measured steps. This does not remove state-distribution differences,
time-dependent errors, within-episode correlations, or selection caused by
early termination. Report sampling interval, valid episode count and duration.

### ESS, covariance and shrinkage

ESS describes weight concentration: it is near 1 when one particle dominates
and near `n_samples` for nearly uniform weights. Low ESS can make moment
estimates highly sensitive to sampling. High ESS is not automatically invalid;
it may mean that candidate costs are similar and requires context.

ESS is **not** a hard bound on covariance rank. For `N` candidate first actions
in dimension `d`, the unregularized centered covariance has rank at most
`min(d, N-1)`. In this task, 16 samples in 16 action dimensions therefore cannot
produce a full-rank raw covariance, even with uniform weights. The estimator
uses `(1-alpha) * covariance + alpha * sigma^2 * I` to make KL computable.
Report ESS and shrinkage, and inspect sensitivity before treating large KL
values as precise differences in planner quality. The Gaussian approximation
can also hide multimodality.

With `--record_kl_moments`, each valid KL step additionally stores both raw
weighted means and raw covariance matrices under `per_step[].moments`, keyed
by control-step number. This records the same particles and weights already
used for that measurement, without any extra planner solve or random draw.
The option is off by default in production sweeps to limit storage. It does
not change the estimator or scientific grouping; extra downloads/storage can
affect wall-clock overhead. Raw moments use full JSON floating-point precision.

Recompute forward KL across shrinkage choices on the fixed recorded moments:

```bash
python -m analysis.kl_shrinkage_sensitivity results/kl_high_high_pilot \
    --outdir results/kl_high_high_pilot/sensitivity \
    --alphas 0.0001 0.001 0.01 0.1
```

This CPU-only command first verifies that raw moments reproduce the saved
forward/reverse KL at the original alpha. It reports raw covariance rank,
eigenvalue range, ESS and mean-action distance, and plots a separate sensitivity
figure for each scientific family. Episode averages remain equally weighted.
Curves represent the same candidates and weights; their change across alpha
comes from the Gaussian regularization, not a change in controller performance.
They do not select an optimal alpha or quantify candidate-sampling uncertainty.
Old results without raw moments cannot support this check and are reported as
unavailable, rather than reconstructed by guessing their covariance.

### Short shadow audit

For a focused diagnosis before a longer run:

```bash
python -m analysis.audit_kl_shadow \
    --geometry cube_high_high --model M3 --eval_sim pinocchio \
    --n_samples 16 --n_iterations 1 --ref_n_samples 4096 \
    --ref_convergence_tol 1e-3 --ref_max_iterations 25 --reference_init zero \
    --n_episodes 1 --max_steps 8 --kl_every 1 --seed 20260907 \
    --record_trajectory --no-record_planner_dist --record_kl_moments \
    --audit_report results/kl_shadow_audit/real_a.json
```

Use a new report path for each repeated default-real process. This focused
audit intentionally rejects compatibility modes and `--null_control`; the
optional null protocol uses a same-proposal rather than zero-reference check.
The audit checks exact equality of state/goal inputs, a zero reference
proposal before every measured solve, distinct mutable planner
arrays, unchanged degraded arrays/counters across a shadow solve, and no
consumption of the environment random stream by planner construction or
planning. It saves per-call input,
noise hashes and actions alongside the ordinary cell result. Extra device
downloads make audit planning timings unsuitable for performance comparisons.
Passing these checks does not certify backend-private state or cross-process
determinism. Audit repeats deliberately share seeds and must not be pooled as
independent episodes in a success-rate figure.

### Validity, collision warnings and timing

Planner-failure flags and nonfinite KL measurements are excluded and recorded
under `invalid_steps`; each cell reports `n_invalid_kl`. Zero invalid KL samples
only means these specific checks passed, not that all physics is converged.

For the default reference, convergence compares consecutive optimized first
actions using squared L2 change. A solve stops when that residual is below
`1e-3`, or after 25 iterations. Every measurement records the iteration count,
residual and convergence flag. The ordinary `kl` summary keeps every finite,
planner-valid measurement; `kl_converged_only` additionally reports the subset
whose reference solve met the tolerance. A converged solve is numerically stable
under this update test, not proof of a globally optimal action. Excluding
non-converged states can preferentially remove difficult states, so report both
summaries and the retained fraction.

The `opt.ccd_iterations=35` warning originates in GPU MuJoCo-Warp convex
GJK/EPA collision detection used by the rollout path. It is **not** Pinocchio's
ADMM warning. The iteration limit is intentionally retained at 35 to preserve
the current planner-speed setting; this does not resolve the numerical issue.
Collision inaccuracies can affect predicted contacts, costs, weights, KL and
applied actions, not only success rate. Preserve warning information in logs;
silencing output alone would not improve collision convergence.

## Conditions for interpreting a larger study

1. Confirm every cell records the intended object's `high_high` scene.
2. Keep contact model, reference compute and shared experimental settings fixed
   within each plotted comparison; report object-specific results separately.
3. Inspect KL alongside ESS, shrinkage sensitivity, invalid records and null
   diagnostics without treating independent null runs as matched-state tests.
4. Use enough episodes and independent repeats to quantify uncertainty;
   short smoke runs and one-to-three-episode pilots establish no success ranking.
5. Investigate unresolved repeated-run differences and retain collision-warning
   limitations in experiment reports.
6. Report both planner timings and actual realized control/horizon schedules.
