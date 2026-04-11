# Contact Model Study

> **What matters about contact models in dexterous manipulation?**
> A systematic empirical study of contact model fidelity for sampling-based MPC.

---

## Overview

This repo implements (in progress) the experimental study that evaluates different contact
models across manipulation tasks, under two experimental conditions:

- **Condition A** — fixed computation budget (models with lower cost get more samples)
- **Condition B** — fixed sample count (isolates approximation error from sample count)

### Study axes (kept orthogonal in code)

| Axis                    | Lives in                                     | How to vary                                 |
|-------------------------|----------------------------------------------|---------------------------------------------|
| Contact model (M1..M4)  | `ContactModelConfig` (`config.py`)           | `ContactModelConfig.M1()` … `.M4()`         |
| Geometry fidelity       | XML files in `scenes/tasks/*.xml`            | `get_task(name, geometry=GeometryVariant.*)`|
| Physics parameter noise | `contact_study.utils.physics_noise`          | `apply_physics_noise(mjm, PhysicsNoiseParams(...))` |

The 4 contact models stay in `ContactModelConfig`. Geometry and physics-parameter
degradations are **not** fields on that config — they are applied at MjModel load
time in the benchmark script, so any of the 4 contact models can be paired with
any geometry and any noise level without touching the core code.

### Contact model variants

| ID  | Description                                                        |
|-----|--------------------------------------------------------------------|
| M1  | Wanted an Anitescu model, for now just use MuJoCo but with hard contact |
| M2  | MuJoCo default soft contact                                        |
| M3  | Jin 2024 complementarity-free model (`comfree_warp`)               |
| M4  | XPBD-style penalty model (`contact_models/xpbd_backend.py`)        |

### Old M5..M10 mapping

The old hardcoded M5..M10 combinations are replaced by CLI flags on the
benchmark scripts:

| Old ID | New invocation                                                        |
|--------|-----------------------------------------------------------------------|
| M5     | `--models M2 --geometry convex_hull`                                  |
| M6     | `--models M4 --geometry convex_hull`                                  |
| M7     | `--models M2 --friction_sigma 0.2 --mass_sigma 0.1`                   |
| M8     | `--models M4 --friction_sigma 0.2 --mass_sigma 0.1`                   |
| M9     | `--models M2 --geometry convex_hull --friction_sigma 0.2 --mass_sigma 0.1` |
| M10    | `--models M4 --geometry convex_hull --friction_sigma 0.2 --mass_sigma 0.1` |

## Repository Structure

```
contact_study/
├── contact_study/
│   ├── contact_models/
│   │   ├── config.py           # ContactModelConfig + M1..M4 factories + GeometryVariant enum
│   │   ├── api.py              # Unified dispatch surface (put_model/step/forward)
│   │   ├── xpbd_backend.py     # M4: XPBD-style contact model
│   │   └── benchmarks.py       # Speed and approximation error measurement
│   ├── planners/
│   │   ├── mppi.py             # MPPI controller
│   │   └── cem.py              # CEM controller
│   ├── tasks/
│   │   ├── base.py             # BaseTask, TaskSpec, task registry
│   │   └── tasks.py            # PushTask, GraspReorientTask, PegInHoleTask
│   ├── evaluation/
│   │   └── metrics.py          # EpisodeResult, AggregatedResult, serialization
│   └── utils/
│       ├── physics_noise.py    # PhysicsNoiseParams + apply_physics_noise
│       └── rollout.py          # batch_rollout, fixed_budget_rollout, fixed_sample_rollout
│
├── scenes/
│   └── tasks/
│       ├── push_{accurate,convex_hull,primitive_union}.xml
│       ├── grasp_reorient_{accurate,convex_hull,primitive_union}.xml
│       └── peg_in_hole_{accurate,convex_hull,primitive_union}.xml
│
├── experiments/
│   ├── run_experiment.py       # Main study runner (Conditions A & B)
│   ├── benchmark_speed.py      # Throughput benchmark vs batch size
│   └── measure_approx_error.py # Approximation error vs horizon
│
├── analysis/
│   └── plot_results.py         # All paper figures
│
└── tests/
    ├── test_allegro.py
    └── test_primitives.py
```


## Installation

### Install ComFree and the dependencies

## What has been implemented and tested for far
1. Contact models M1-M4. Tested for throughput testing on primitives and allegro scenes.

## What neesd to be done next
1. Test the planner for some manipulation task
2. Implement and test geometry fidelity	and Physics parameter noise


---
## Quick Tests
### Test throughtput of different models with and without the viewer in the Allegro Hand Cube Scene

Run tests/test_allegro.py, see file for options

### Test the viewer and throughtput of different models of the primitives scene
Run tests/test_primitives.py, see file for options


## Usage for benchmarks (Not Tested Yet)

### 1. Speed benchmark (clean)

```bash
python experiments/benchmark_speed.py \
    --task push \
    --models M1 M2 M3 M4 \
    --batch_sizes 64 256 1024 4096 \
    --horizon 50
```

### 2. Speed benchmark with degraded geometry + noisy physics (old "M10")

```bash
python experiments/benchmark_speed.py \
    --task grasp_reorient \
    --models M4 \
    --geometry convex_hull \
    --friction_sigma 0.2 --mass_sigma 0.1
```

### 3. Approximation error

```bash
python experiments/measure_approx_error.py \
    --tasks push grasp_reorient peg_in_hole \
    --models M1 M3 M4 \
    --horizons 5 10 20 40 \
    --n_states 50
```

### 4. Full study, clean baseline

```bash
python experiments/run_experiment.py \
    --tasks push grasp_reorient peg_in_hole \
    --models M1 M2 M3 M4 \
    --conditions A B \
    --n_episodes 20 \
    --budget_seconds 0.1 \
    --n_samples_b 1024
```

### 5. Full study cell: convex-hull geometry + friction noise

```bash
python experiments/run_experiment.py \
    --models M1 M2 M3 M4 \
    --geometry convex_hull \
    --friction_sigma 0.2 --mass_sigma 0.1 \
    --output results/cell_convex_hull_noisy.json
```

To sweep over the full old-M1..M10 grid, wrap this invocation in an outer shell
loop over `--geometry` and `--friction_sigma` values.

### 6. Figures

```bash
python analysis/plot_results.py results/experiment_TIMESTAMP.json
```
