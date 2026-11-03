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

The [KL-vs-success workflow](analysis/README_kl_divergence.md) intentionally
uses a stricter selector: all five current objects (`cube`, `duck`, `ball`,
`spam`, `tomato`) use `high_high` rollout geometry with their fixed evaluation
scenes. In that worker only, object shorthand selects `high_high`, lower
fidelities and legacy aliases are rejected, and plots separate object/config
families. Other drivers retain the general scene-selection behavior above.

### Contact model variants

| ID  | Description                                                        |
|-----|--------------------------------------------------------------------|
| M1  | Wanted an Anitescu model, for now just use MuJoCo but with hard contact |
| M2  | MuJoCo default soft contact                                        |
| M3  | Jin 2024 complementarity-free model (`comfree_warp_v1`)            |
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
│   ├── plot_results.py         # All paper figures
│   └── README_kl_divergence.md # KL-vs-success workflow and diagnostics
│
└── tests/
    ├── test_allegro.py
    └── test_primitives.py
```


## Installation and system setup

### What the machine needs

- **Linux with an NVIDIA GPU and its driver.**
  - The rollouts (M1–M6) all run on the GPU through Warp. Warp ships its own
    CUDA runtime, so only the driver is needed, not the CUDA toolkit.
  - Used on an RTX 4090 (driver 580, CUDA 13.0) and on the HPC's RTX 5000 Ada.
  - Without a GPU, only the CPU parts run, and the tests skip the rest.
- **conda** (Miniconda or Anaconda). Pinocchio and coal come from conda-forge.
- **ffmpeg** on the `PATH` for videos (mediapy calls it). The recipes below
  install it into the env. Without it, pass `--no-video`.
- **On a headless machine**, MuJoCo renders through EGL, which comes with the
  NVIDIA driver.
  - The episode drivers set `MUJOCO_GL=egl` themselves when there is no
    `DISPLAY`.
  - Set it yourself for anything else that renders: `export MUJOCO_GL=egl`.

### Which environment

On this branch every model, M1–M6, runs in one env, `contact_comfree2`:

| Env | Python | Warp | MuJoCo | Runs |
| --- | --- | --- | --- | --- |
| `contact_comfree2` | 3.12 | 1.17 | 3.14 | M1–M6, CPU MuJoCo, Pinocchio, Drake |

- **Each model's package:** M1, M2 and M4 run on the standalone `mujoco-warp`
  3.14, M3 on `comfree_warp_v1` (M3's ComFree 0.1, renamed), M5 on Newton 1.6,
  and M6 on `comfree_warp` 0.3 (ComFree-Sim).
- **Why MuJoCo 3.14:** M6 needs it. Python 3.12 and Warp 1.17 are for M5:
  Newton's Kamino code does not import on Python 3.10, and Newton needs
  Warp ≥ 1.17.
- **main's envs don't run this branch.** `contact_kamino` and
  `contact_modeling` (MuJoCo 3.6) have neither `mujoco_warp` nor
  `comfree_warp_v1`. `results/bo_check` came from `contact_kamino`;
  `results/bo_check_one_env` is the same runs in `contact_comfree2`.

### Building `contact_comfree2`

From the repo root:

```bash
conda create -n contact_comfree2 -c conda-forge python=3.12 pinocchio=4.0.0 coal=3.0.2 numpy=2.2 ffmpeg
conda activate contact_comfree2
pip install drake==1.51.1 mediapy pytest
pip install -e ".[kamino]"     # MuJoCo 3.14, MJWarp 3.14, Warp 1.17, Newton 1.6, comfree_warp_v1 (pinned), skopt, ...
gh repo clone asu-iris/comfree-warp-simbenchmark /path/to/comfree-warp-simbenchmark
pip install --no-deps /path/to/comfree-warp-simbenchmark   # comfree_warp 0.3.0, for M6
```

- **Warp stays at 1.17.** ComFree-Sim pins 1.15, but Newton (M5) needs 1.17,
  and ComFree-Sim runs on 1.17 unchanged. `--no-deps` keeps pip from
  downgrading it, and `pip check` reports the mismatch.
- **The env on the lab's 4090 was upgraded in place** from a clone of
  `contact_kamino`, not built this way: `pip uninstall comfree_warp`, then
  `pip install mujoco==3.14.0 absl-py "etils[epath]"`,
  `pip install --no-deps mujoco-warp==3.14.0`, and the two ComFree packages.
- Pinocchio and Drake are needed only for those eval simulators. Without
  them, the tests that use them skip.

### What MuJoCo 3.14 changes

- **M5 is unchanged.** Kamino's trajectories on all three eval scenes under
  hold, shake and random-walk controls are bit-identical to `contact_kamino`'s.
- **M1–M4 are not.** MuJoCo 3.14 compiles the mesh scenes slightly
  differently: geom orientations move by up to 0.25°, sizes by about 0.04 mm,
  and the duck's hull is rebuilt. M1, M2 and M4 also move to MJWarp 3.14's
  collision and solver.
  - Over 200 steps of random hand motion, the cube and the duck end
    0.1–4 mm (mean) from where `contact_kamino` puts them, against
    run-to-run noise of 0.06–0.26 mm.
  - The ball compiles identically, and M3 and M4 are bit-identical on it.
  - So compare M1–M4 results only within one env.
- **M4 (XPBD) was ported to MJWarp 3.14.** 3.14 stores M as CSR and factors
  it per block, packed, so XPBD densifies M̂ and factors it itself for the
  per-row effective mass. It still uses MJWarp's factor for the per-world
  solves.

### The ComFree dependencies

- **`comfree_warp_v1` (M3)** is installed from this repository's
  `comfree-warp-v1` branch. That is upstream `asu-iris/comfree_warp@ba8b996`
  with:
  - a fix to its vendored MJWarp `sensor.py`, without which Warp ≥ 1.16 does
    not compile it (the `comfree-warp1.17` branch, which main pins);
  - the package renamed to `comfree_warp_v1`;
  - `EnableBit.MULTICCD` falling back to 0, since MuJoCo 3.14 dropped
    `mjENBL_MULTICCD`.
- **`comfree_warp` 0.3 (M6)** is the private
  `asu-iris/comfree-warp-simbenchmark`, tested at `0a8b385`.

The `comfree-warp*` branches stand apart from the study code: they share only
the early comfree history. **Never merge them into the study branches.**

To edit M3's comfree, check that branch out somewhere else and install it over
the pinned one:

```bash
git clone -b comfree-warp-v1 https://github.com/yifanzhu95/contact_model_study.git /path/to/comfree_warp_v1
pip install -e /path/to/comfree_warp_v1 --no-deps
```

### Checking the install

```bash
python -c "import mujoco, mujoco_warp, warp, comfree_warp_v1, comfree_warp; print(mujoco.__version__, mujoco_warp.__version__, warp.__version__, comfree_warp.__version__)"   # 3.14.0 3.14.0 1.17.0 0.3.0
python -c "import warp; warp.init()"         # lists the CUDA devices Warp can see
python -m pytest tests -m "not slow"         # a few minutes; the full suite is longer
```

- On an RTX 4090, `python -m pytest tests` took about 4.5 minutes in
  `contact_comfree2`, with the Warp kernels already compiled.
- The first run of each GPU model also spends a while compiling Warp kernels.
  They are cached afterwards.

### On the HPC

- Build the env once on the login node, the same way, after
  `module load miniconda`.
- The submit scripts in `experiments/hpc/` and their `.slurm` jobs activate
  `contact_kamino`, main's env. For this branch, build `contact_comfree2` on
  the cluster and set `CONDA_ENV` when submitting:
  `CONDA_ENV=contact_comfree2 bash experiments/hpc/submit_episode_batches.sh ...`.
  The submit script exports it, so the jobs activate the same env.
- See `experiments/README.md` for the submission workflow.

## What has been implemented and tested so far

1. Contact models M1-M4, with throughput checks on primitives and Allegro scenes.
2. Duck `grasp_reorient` MPPI closed-loop smoke tests on the four FOAM sphere
   scenes, using the fixed eight-hull Duck as the eval model.
3. An isolated Newton 1.6 / SolverKamino offline-reference subproject under
   `kamino_reference/`.  It consumes the same fingerprinted state, goal, cost,
   timing and candidate tape as M1--M4 without mixing the incompatible Warp and
   MuJoCo dependency stacks.  Start with
   `kamino_reference/docs/collaborator_handoff.md`.

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
