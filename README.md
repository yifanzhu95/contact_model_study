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
are an `<include>` swap (`low` / `med` / `high` -> the corresponding
`leap_right_hand_{accuracy}.xml`); all hand XMLs are kinematically identical, so
rollout and eval scenes differ only in collision geometry.

Available convex-hull Duck baselines are `duck_low_high`, `duck_med_high`, and
`duck_high_high`, where the middle token selects hand accuracy and `high` means
the existing eight-hull Duck collision model. The FOAM study adds four object
accuracy labels for every hand accuracy:

| Object label | Duck rollout collision model |
|--------------|------------------------------|
| `foam4`      | calibrated 4-sphere Low      |
| `foam16a`    | calibrated 16-sphere Medium-A |
| `foam16b`    | calibrated 16-sphere Medium-B |
| `foam64`     | calibrated 64-sphere High    |

For the first object-only comparison, keep the hand fixed and run
`duck_low_foam4`, `duck_low_foam16a`, `duck_low_foam16b`, and
`duck_low_foam64`. All four selectors still resolve eval to the same
`env_leap_eval_duck.xml` eight-hull reference. The generated scene manifest,
source geometry metrics, and regeneration instructions live in
`analysis/duck_foam/README.md`.

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

The GPU contact backends are sensitive to the MuJoCo/Warp combination.  The
versions below are pinned in `pyproject.toml` because this repository was
verified with MuJoCo 3.6.0 and Warp 1.12.0; Warp 1.16 does not compile the
current ComFree/MJWarp sensor kernels.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The editable install fetches the official `comfree_warp` revision recorded in
`pyproject.toml`.  To inspect or edit that dependency as a separate checkout,
install the checkout after the command above:

```bash
git clone https://github.com/asu-iris/comfree_warp.git /path/to/comfree_warp
python -m pip install -e /path/to/comfree_warp
```

Verify the environment before a long experiment:

```bash
python -c "import mujoco, warp, comfree_warp; print(mujoco.__version__, warp.__version__)"
```

The expected version line is `3.6.0 1.12.0`.

## What has been implemented and tested so far

1. Contact models M1-M4, with throughput checks on primitives and Allegro scenes.
2. Duck `grasp_reorient` MPPI closed-loop smoke tests on the four FOAM sphere
   scenes, using the fixed eight-hull Duck as the eval model.

## What needs to be done next

1. Tune the Duck MPPI/controller on development seeds until it has nonzero
   success, then freeze the parameters and run the predeclared multi-seed test.
2. Repeat a subset of seeds to quantify non-bitwise-deterministic contact
   variation, then test physics parameter noise (geometry fidelity is wired).


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
