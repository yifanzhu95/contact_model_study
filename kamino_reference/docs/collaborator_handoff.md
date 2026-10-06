# Kamino collaborator handoff

## What this handoff is for

I prepared this repository so my collaborators can run larger offline
Kamino/full-NCP sampling experiments from the exact planning inputs I used.
The workflow I support is:

1. load the committed frozen planning input;
2. generate exact candidate chunks with the CPU coordinator;
3. execute every chunk through the GPU guard;
4. reject missing, changed, nonfinite, or nonconverged cells;
5. aggregate repeated costs in float64;
6. perform one MPPI soft-min update only after cost aggregation;
7. preserve hashes, guard logs, summaries and resume state.

I do not present this handoff as evidence that Kamino is globally optimal,
real-world ground truth, real-time MPC, or a production Pinocchio replacement.

## Baseline I froze

- Python: 3.12
- Newton: 1.6.0
- Warp: 1.17.0
- MuJoCo / MuJoCo Warp: 3.12.0
- tested solver policy: sparse dynamics, CRF, fixed PADMM penalty,
  `rho0=0.1`, tolerance `5e-4`, maximum 800 iterations;
- planning scene SHA-256:
  `49f2c66e4f19fec8aa1296e72e1be356a44264fe5dda04dc322b37138eec3255`;
- source `contact_model_study` commit recorded in the bundle:
  `05ed5508213e3057f41bf407936afbbdcbd9f88c`.

I record the source commit and planning-scene SHA-256 in every frozen bundle.
The scene may live anywhere on a collaborator's machine, but I require its
local path through `--scene`; initialization rejects a file whose SHA-256 does
not match my frozen bundle.

## Files I intentionally committed

I include source, tests, documentation, the v3 scaling contract, and three
compact frozen input files under `artifacts/reference_inputs/`.  I leave
generated rollouts, guard logs, and historical result matrices ignored.  This
keeps a clean checkout self-contained for tests, dry runs, and new scaling
experiments without treating my historical output as part of the release.

## Install and verify

To reproduce my setup, start from the parent `contact_model_study` repository
root and create the main M1--M4 environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip check
```

Create the separate Kamino environment:

```bash
python3.12 -m venv kamino_reference/.venv
kamino_reference/.venv/bin/python -m pip install --upgrade pip
kamino_reference/.venv/bin/python -m pip install -e 'kamino_reference[dev]'
kamino_reference/.venv/bin/python -m pip install \
  -r kamino_reference/requirements-lock.txt
kamino_reference/.venv/bin/python -m pip check
```

Do not omit my Kamino lock-file step.  I validated the baseline with Warp
1.17.0.  On my validation host, an unconstrained Warp 1.18.0 environment
exceeded the 120-second G0 guard limit, while the locked environment completed
the same G0 in about 24 seconds.

Validate the integrated checkout and run both CPU test sets:

```bash
.venv/bin/python experiments/run_contact_reference_comparison.py validate \
  --check-cuda
.venv/bin/python -m pytest -q tests/test_reference_integration.py
kamino_reference/.venv/bin/python -m pytest -q kamino_reference/tests
```

`validate_delivery.py` checks package availability, contract/index hashes,
bundle file hashes, semantic fingerprints and optional scene/CUDA readiness.
It performs no physics rollout.

## One-command matched G0 smoke

From the parent repository root:

```bash
.venv/bin/python experiments/run_contact_reference_comparison.py smoke \
  --run-directory results/reference_g0 \
  --budget G0_4x1 \
  --models M1 M2 M3 M4 \
  --chunk-size 4 \
  --headless
```

This validates the checkout, evaluates my exact same four frozen candidates
under M1--M4 and Kamino, collects the guarded Kamino result, and writes a
comparison report.  I configured the guard to refuse by default when the
NVIDIA GPU drives the desktop; add `--allow-display-gpu` only during an
explicit local recovery window.

## CPU-only protocol smoke test

This verifies chunking, aggregation and resume without importing Newton or
running CUDA.  The remaining low-level commands in this document assume:

```bash
cd kamino_reference
```

```bash
.venv/bin/python scripts/reference_optimizer_v2_coordinator.py initialize \
  --contract configs/optimizer_v3_20_repeat.json \
  --inputs-index configs/optimizer_v3_inputs.json \
  --budget G0_4x1 \
  --chunk-size 4 \
  --scene /absolute/path/to/env_leap_rollout_cube_high_high.xml \
  --run-directory artifacts/collaborator_g0_cpu

.venv/bin/python scripts/reference_optimizer_v2_coordinator.py dry-run \
  --run-directory artifacts/collaborator_g0_cpu \
  --synthetic-seed 4001
```

Synthetic results are infrastructure tests only and must never be reported as
Kamino physics evidence.

## One guarded real GPU smoke

Create a fresh run directory:

```bash
.venv/bin/python scripts/reference_optimizer_v2_coordinator.py initialize \
  --contract configs/optimizer_v3_20_repeat.json \
  --inputs-index configs/optimizer_v3_inputs.json \
  --budget G0_4x1 \
  --chunk-size 4 \
  --scene /absolute/path/to/env_leap_rollout_cube_high_high.xml \
  --run-directory artifacts/collaborator_g0_gpu
```

Inspect the host first:

```bash
.venv/bin/python scripts/run_gpu_guarded.py status
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

On a headless compute host with no AnyDesk or X11 requirement:

```bash
.venv/bin/python scripts/run_reference_optimizer_workers.py \
  --run-directory artifacts/collaborator_g0_gpu \
  --no-require-anydesk \
  --no-require-x11 \
  --max-new-tasks 1
```

If the NVIDIA GPU drives the desktop, do not add `--allow-display-gpu` unless
the operator has an explicit recovery window.  If a remote-work lock exists,
the runner also requires the deliberate recovery token described in
`docs/gpu_remote_safety_policy.md`.

## First larger experiment I defined: 32 candidates x 20 repeats

My Step 43 result supports a fixed-four-world, 20-repeat cost-first estimator.
I did not validate a 32-candidate reference in advance, so I treat this run as
a scaling experiment until its gates pass.

```bash
.venv/bin/python scripts/reference_optimizer_v2_coordinator.py initialize \
  --contract configs/optimizer_v3_20_repeat.json \
  --inputs-index configs/optimizer_v3_inputs.json \
  --budget B1_32x20 \
  --chunk-size 4 \
  --scene /absolute/path/to/env_leap_rollout_cube_high_high.xml \
  --run-directory artifacts/collaborator_b1_32x20

.venv/bin/python scripts/run_reference_optimizer_workers.py \
  --run-directory artifacts/collaborator_b1_32x20 \
  --no-require-anydesk \
  --no-require-x11 \
  --max-new-tasks 1
```

I intentionally make the first invocation stop after one newly completed
task.  Audit its result, guard log, resource envelope, and convergence summary,
then resume by repeating the same command without `--max-new-tasks`:

```bash
.venv/bin/python scripts/run_reference_optimizer_workers.py \
  --run-directory artifacts/collaborator_b1_32x20 \
  --no-require-anydesk \
  --no-require-x11
```

For 32 candidates, chunk size 4 and 20 repeats, the run contains 160 guarded
worker tasks and 640 candidate-repeat evaluations.  Completed tasks are hash-
checked and skipped on resume.  A result without complete guard provenance is
rejected rather than silently trusted.

## Larger frozen budgets

The same contract also defines:

| Label | candidates | MPPI iterations | repeats |
|---|---:|---:|---:|
| `B1_32x20` | 32 | 1 | 20 |
| `B2_32x20x3` | 32 | 3 | 20 |
| `B3_64x20x3` | 64 | 3 | 20 |
| `B4_128x20x3` | 128 | 3 | 20 |

Do not skip directly to my largest budget.  Run a guarded G0, then B1, and
inspect strict validity, bootstrap width, rank stability, resource use, and
resume integrity before expanding.

## How to interpret the outputs

Each iteration writes:

- `tasks.json`: exact candidate slices and worker commands;
- `chunks/*.npz`: fingerprinted planning bundles;
- `results/repeat_*/`: raw worker results;
- `guard/*.jsonl`: runtime health history;
- `guard/*.audit.json`: guard acceptance decision;
- `iteration_result.npz/json`: repeat-cost aggregation and one MPPI update;
- `run_manifest.json`: immutable provenance and resume state.

`reference_estimator_eligible=true` means that the run used my supported
20-repeat estimator and all cells passed strict collection.  I do not treat
this as the final scientific claim.  My v3 contract deliberately keeps
`reference_claim_eligible=false` for a single budget: successive-budget
convergence, common validation and held-out evaluation remain required.

## Known boundaries

- My committed planning bundles contain one frozen state/goal/noise tape, not
  a general state generator.
- I demonstrated the 20-repeat calibration on four candidates; I leave larger
  candidate counts as collaborator scaling experiments.
- Results are not bitwise deterministic across independent processes.
- I do not include historical raw artifacts in Git.
- I make no claim that a single frozen planning problem generalizes to other
  states, goals, objects, or tasks.
- The repository currently has no declared redistribution license. I will
  follow the repository owner's and laboratory's policy for internal sharing;
  the owner must choose a license before public redistribution.
