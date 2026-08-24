# Contact Model Study

> **What matters about contact models in dexterous manipulation?**
> A systematic empirical study of contact model fidelity for sampling-based MPC.

---

## Overview

This repo implements (in progress) the experimental study that evaluates different contact
models across manipulation tasks at a fixed sample count, which isolates
approximation error from sample count.

### Study axes (kept orthogonal in code)

| Axis                    | Lives in                                     | How to vary                                 |
|-------------------------|----------------------------------------------|---------------------------------------------|
| Contact model (M1..M4)  | `ContactModelConfig` (`config.py`)           | `ContactModelConfig.M1()` … `.M4()`         |
| Geometry fidelity       | XML files in `scenes/leap/*.xml`             | `get_task(name, geometry="duck_low_high")` |
| Physics parameter noise | `contact_study.utils.physics_noise`          | `apply_physics_noise(mjm, PhysicsNoiseParams(...))` |

The 4 contact models stay in `ContactModelConfig`. Geometry and physics-parameter
degradations are **not** fields on that config — they are applied at MjModel load
time in the benchmark script, so any of the 4 contact models can be paired with
any geometry and any noise level without touching the core code.

### Scene variants (`--geometry`)

`--geometry` names a **scene variant**: which object is manipulated, and at what
collision fidelity the planner's hand and the object are modelled.

    --geometry <object>_<hand_acc>_<obj_acc>     e.g. duck_low_high
    --geometry <object>                          object at default fidelities

Scene files are found by convention (see `contact_study.tasks.config.SceneVariant`):

    rollout:  scenes/leap/env_leap_rollout_{obj}_{hand_acc}_{obj_acc}.xml
    eval:     scenes/leap/env_leap_eval_{obj}.xml

The eval scene carries no accuracy suffix — it *is* the reference fidelity. Only
the planner's model is degraded, which is the axis the study varies. Hand rungs
are an `<include>` swap (`low` -> `leap_right_hand.xml`, `high` ->
`leap_right_hand_eval.xml`); all hand XMLs are kinematically identical, so
rollout and eval scenes differ only in collision geometry.

Available today: `cube_low_high` (default), `cube_high_high`, `duck_low_high`,
`duck_low_low`, `duck_high_high`. Adding an object means dropping in the two
XMLs plus an entry in `_OBJ_OVERRIDES` (`contact_study/tasks/grasp_reorient.py`)
for its initial pose and target — no other code changes.

The retired `GeometryVariant` names (`accurate`, `convex_hull`,
`primitive_union`, `linearized`) are still accepted and map to the default
variant, so existing SLURM scripts keep working.

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
| M5     | `--models M2 --geometry cube_high_high`                                  |
| M6     | `--models M4 --geometry cube_high_high`                                  |
| M7     | `--models M2 --friction_sigma 0.2 --mass_sigma 0.1`                   |
| M8     | `--models M4 --friction_sigma 0.2 --mass_sigma 0.1`                   |
| M9     | `--models M2 --geometry cube_high_high --friction_sigma 0.2 --mass_sigma 0.1` |
| M10    | `--models M4 --geometry cube_high_high --friction_sigma 0.2 --mass_sigma 0.1` |

## Repository Structure

```
contact_study/
├── contact_study/
│   ├── contact_models/
│   │   ├── config.py           # ContactModelConfig + M1..M4 factories
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
│   │   ├── metrics.py          # EpisodeResult, AggregatedResult, serialization
│   │   ├── trajectory.py       # per-control-step state / control / planner-belief recording
│   │   ├── distributions.py    # first-action moments of a planner, Gaussian KL
│   │   └── json_io.py          # JSON writer that keeps bulk arrays on one line
│   └── utils/
│       └── physics_noise.py    # PhysicsNoiseParams + apply_physics_noise
│
├── scenes/
│   └── leap/                   # scene variants, named by convention
│       ├── env_leap_eval_{cube,duck}.xml            # reference fidelity
│       ├── env_leap_rollout_{obj}_{hand}_{obj}.xml  # degraded planner scenes
│       └── leap_right_hand{,_eval,_capsules}.xml    # hand fidelity rungs
│
├── experiments/
│   ├── run_experiment.py       # Main study runner (tasks × models)
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
2. Test physics parameter noise (geometry fidelity is now wired — see Scene variants)


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

### 2. Speed benchmark with a higher-fidelity rollout hand + noisy physics (old "M10")

```bash
python experiments/benchmark_speed.py \
    --task grasp_reorient \
    --models M4 \
    --geometry cube_high_high \
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
    --n_episodes 20 \
    --n_samples 1024
```

### 5. Full study cell: high-fidelity rollout hand + friction noise

```bash
python experiments/run_experiment.py \
    --models M1 M2 M3 M4 \
    --geometry cube_high_high \
    --friction_sigma 0.2 --mass_sigma 0.1 \
    --output results/cell_cube_high_high_noisy.json
```

To sweep over the full old-M1..M10 grid, wrap this invocation in an outer shell
loop over `--geometry` and `--friction_sigma` values.

### 6. Figures

```bash
python analysis/plot_results.py results/experiment_TIMESTAMP.json
```
