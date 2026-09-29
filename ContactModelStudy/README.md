# ContactModelStudy

A package for studying how the choice of contact model affects sampling-based
control. A planner (MPPI) chooses actions by rolling out many sampled futures
in a fast GPU simulator, the *rollout model*. Those actions are then applied
in a separate, more accurate simulator standing in for reality, the *eval
simulator*. The study measures how well a task is performed as the rollout
model's contact physics (M1–M4) and geometry fidelity change.

The main task is in-hand reorientation with a 16-DOF LEAP hand: turn a cube,
duck or ball to a goal orientation.

## Layout

```
ContactModelStudy/
├── Simulators/            one class per physics engine, all behind one interface
│   ├── Simulator.py            Simulator + SimulatorConfig (base, one world)
│   ├── VectorizedSimulator.py  VectorizedSimulator + config (base, N worlds on the GPU)
│   ├── Mujoco.py               CPU MuJoCo
│   ├── Pinocchio.py            Pinocchio + ADMM contact solver (CPU)
│   ├── Drake.py                Drake MultibodyPlant (CPU)
│   ├── VectorizedMujoco.py     MuJoCo Warp (MJWarp), N worlds
│   ├── ComFree.py              complementarity-free contact on MJWarp
│   └── XPBD.py                 XPBD contact on MJWarp
├── Tasks/                 what to do, what it costs, when it succeeds
│   ├── TaskBase.py             TaskBase, TaskBaseConfig, TaskRole
│   ├── LeapReorient.py         shared LEAP-hand reorientation machinery + cost
│   └── CubeReorient.py, DuckReorient.py, BallReorient.py   per-object numbers
├── SamplingBasedPlanners/
│   ├── SamplingBasedPlannerBase.py   rollout loop, control modes, graph capture
│   └── MPPI.py
├── Renderers/
│   ├── RendererBase.py         RendererBase / VideoRendererBase + configs
│   └── MujocoVideoRenderer.py  draws any simulator's state with MuJoCo
├── Drivers/
│   ├── run_episodes.py            closed-loop episodes, one at a time
│   ├── run_episodes_interwoven.py two episodes at once: one plans on the GPU while the other steps on the CPU
│   ├── run_episodes_pooled.py     many at once, across the GPUs and CPU cores of a machine
│   └── EpisodePool.py             the engine behind the two parallel drivers
└── Utils/
    ├── ContactModelPresets.py  M1–M4 as ready-made rollout simulators
    ├── EvalSimulators.py       eval simulator by name (mujoco/pinocchio/drake)
    ├── EpisodeRecorder.py      record, save, summarize and replay episodes
    ├── MjcfModelInfo.py        MJCF parameters as MuJoCo resolves them
    └── Quaternions.py          wxyz quaternion helpers
```

Outside the package:

| Path | What |
| --- | --- |
| `scenes/leap/` | Scene MJCFs, one eval scene and many rollout scenes per object |
| `experiments/` | Batches of episodes from a CSV, locally or as an HPC job array (see its README) |
| `tests/` | pytest suite |
| `test_scripts/` | Ad-hoc scripts, including `profile_run_episodes.py` |
| `contact_study/` | The old codebase, kept only for comparison tests |

## How an episode runs

```
                ┌──────────── eval task ────────────┐
                │ scene, start state, goal, success │
                └────────────────┬──────────────────┘
      q, q_dot                   │ isSuccess / isFailure
 ┌──────────────────┐     ┌──────┴───────┐     ┌─────────────────────┐
 │  eval simulator  │────▶│    driver    │────▶│  EpisodeRecorder    │
 │  (Mujoco /       │◀────│ run_episodes │     │  per-step data +    │
 │  Pinocchio/Drake)│  u  └──────┬───────┘     │  summaries          │
 └──────────────────┘            │ Plan(q, q_dot)└─────────────────────┘
                          ┌──────┴───────┐     calcCosts (on GPU)
                          │    MPPI      │◀──── rollout task
                          └──────┬───────┘
                                 │ N sampled control sequences
                          ┌──────┴────────────────────┐
                          │ rollout simulator (GPU)   │
                          │ M1/M2 MJWarp, M3 ComFree, │
                          │ M4 XPBD                   │
                          └───────────────────────────┘
```

Each control step:

1. The driver reads the eval simulator's state and checks the eval task's
   `isFailure` and `isSuccess`.
2. `MPPI.Plan(q, q_dot)` does the following:
   - puts that state into all N rollout worlds;
   - rolls out N perturbed control sequences over the horizon, as one captured
     CUDA graph;
   - scores them with the rollout task's GPU cost;
   - returns the softmax-weighted first action, and optionally its
     uncertainty.
3. The eval simulator holds that action for one control step.
4. The recorder stores the state, action, uncertainty and planning time.

Before the first step, the object is left to settle in the hand for `--settle`
seconds.

The two tasks are the same task (for example `CubeReorient`) in two *roles*:

- **`TaskRole.EVAL`:** the accurate scene, at the fine eval timestep.
- **`TaskRole.ROLLOUT`:** the planner's scene. It can have lower-fidelity
  hand or object meshes (`hand_acc`, `obj_acc`), and it runs at
  `timestep × eval_steps_per_rollout_step`.

Goals are sampled once, on the eval task, and set on both.

## Core interfaces

### Simulators

Every simulator is a `Simulator`: `SetState(q, q_dot)`, `GetState()`,
`SetControl(u)`, `GetControl()`, `Step(n)`, `nq`/`nv`/`nu`, `timestep`, `time`.

- **One state layout.** State is always in **MuJoCo's `qpos`/`qvel` layout**,
  whatever the engine stores internally. So Pinocchio or Drake can replace
  MuJoCo without changing anything else, and the MuJoCo renderer can draw any
  of them.
- **`VectorizedSimulator`** runs N worlds on the GPU. It adds:
  - `SetControlSequence(U)` with an `(N, H, nu)` array, and `Step_GPU(n)`;
  - `BroadcastState`, to seed every world from device arrays;
  - `DeviceState()`, the live device arrays that GPU cost functions read.

  Getters and setters take and return `(N, ·)` arrays. `Step_GPU` never copies
  back to the host, so it can be recorded into a CUDA graph.

| Simulator | Kind | Parameters from |
| --- | --- | --- |
| `Mujoco` | CPU, reference | The MJCF; `MujocoConfig` fields override it (`None` keeps the scene's value) |
| `Pinocchio` | CPU, ADMM contact | The MJCF, read through `MjcfModelInfo`; `PinocchioConfig` holds only solver settings |
| `Drake` | CPU, SAP contact | The MJCF, parsed by Drake and corrected from `MjcfModelInfo`; `DrakeConfig` holds contact-solver settings |
| `VectorizedMujoco` | GPU | As `Mujoco`; pyramidal friction cones only |
| `ComFree` | GPU | As `VectorizedMujoco`, plus `stiffness` and `damping` |
| `XPBD` | GPU | As `VectorizedMujoco`, plus `xpbd_substeps`, `xpbd_iterations`, `relaxation`, `vmax_depenetration` |

**Pinocchio and Drake inherit physics from the MJCF.** Damping, armature,
joint limits, friction and servo gains are all read from the scene. So are the
collision pairs, using MuJoCo's own contype/conaffinity/exclude rules. They
come through `Utils/MjcfModelInfo.py`, which compiles the scene with MuJoCo,
because neither engine's own parser reads MJCF defaults correctly. Changing a
value in the XML changes it in all three eval simulators.

### Configs

Each class is built from a dataclass config: `Simulator(xml, SimulatorConfig)`,
`CubeReorient(LeapReorientConfig)`, `MPPI(sim, task, MPPI_Config)`, and so on.

- **Validation:** configs check their values in `__post_init__`, so a bad
  value fails at construction rather than mid-run.
- **Pairs:** two settings can be given either as a step count or as a
  duration. Give at most one of each pair; giving both raises an error.

  | Setting | Step count | Duration | Resolved value |
  | --- | --- | --- | --- |
  | Control step | `substeps` | `ctrl_time_step` | `resolved_substeps`, `control_timestep` |
  | Planning horizon | `horizon` | `time_horizon` | `resolved_horizon`, `horizon_duration` |

  Durations are rounded down to whole steps. The config keeps what you asked
  for, and the `resolved_*` properties give what is actually used.

### Tasks

A task owns its scene path, initial state, goal and cost. It owns no
simulator.

| Method | Purpose |
| --- | --- |
| `getModelPath()` | The MJCF for this role and fidelity |
| `getInitialState()`, `setSimToInitialState(sim)` | Where an episode starts (the second can be captured into a graph on a vectorized sim) |
| `calcCosts(vec_sim, terminal)` | Per-world cost, **GPU only**, a Warp kernel |
| `isSuccess(sim)`, `isFailure(sim)` | On one simulator: a bool. On a vectorized one: a device array. Never both true. |
| `sampleNewGoal()`, `setGoal(g)`, `setRendererToGoal(r)` | Goals, each a rotation of the current goal; ten difficulty levels (`GOAL_DIFFICULTIES`) |
| `alignRendererConfigWithTask(cfg)` | Puts the task's camera into a renderer config |

`LeapReorientConfig` holds what may change between instances:

- `role`, `timestep`, `seed`;
- `hand_acc` and `obj_acc`, which select the rollout scene;
- `eval_steps_per_rollout_step`;
- `goal_difficulty`;
- `cost_weights` overrides.

What is fixed per object (start pose, tuned weights, target) lives in that
object's `objectParams()`. `CubeReorient`, `DuckReorient` and `BallReorient`
contain nothing else.

**Porting notes:**

- The cost reproduces the old study's kernel bit for bit, including two
  quirks: `w_velo` scales the *position* error, and control effort isn't
  penalized. Both are documented in `LeapReorient.py`.
- The duck and ball numbers are copied verbatim from the old code, so they
  still carry the old hand-applied offsets. The cube's thumb (`th_axl`) offset
  has been changed deliberately.

### Planner

`MPPI` (in `SamplingBasedPlanners/`) draws its N (worlds) and H (horizon) from
the simulator it's given.

- **Control modes** (`control_mode`) set how a sampled value U becomes a
  command:
  - `absolute`: `ctrl = U`;
  - `ctrl_relative`: `ctrl_t = ctrl_{t-1} + U_t`;
  - `pos_relative`: `ctrl_t = q_t + U_t`, re-reading joint positions every
    step.

  In every mode, `Plan` returns the absolute command to apply.
- **Graph capture** (`use_graph`) records the whole rollout as one CUDA graph
  and replays it. This is what makes planning fast. The first `Plan` of a run
  pays for kernel compilation and the capture.
- **Uncertainty** (`return_uncertainty=True`) makes `Plan` return
  `(action, sigma)`. `sigma` is the weighted spread of the sampled first
  actions:
  - near `noise_sigma` when the costs couldn't tell the samples apart;
  - near 0 when one sample dominated.
- **Adding a planner:** subclass `SamplingBasedPlannerBase`, implement
  `_buildSamples` and `_updateParams`, and optionally `_actionUncertainty`.

### Rendering and recording

- **`MujocoVideoRenderer(task, VideoRendererBaseConfig)`** draws a state with
  `RenderState(q)`. The driver captures frames on the *simulation* clock, so
  videos play in real time. Rendering is kept out of the simulators.
- **`EpisodeRecorder(eval_task, eval_sim, planner)`** records a batch of
  episodes:
  - `recordStateAndAction`, `recordGoal` and `recordSuccess` as the episode
    runs, then `episodeFinished`;
  - `GenerateSummary()` for per-episode or batch summaries;
  - `Save(path)` writes one JSON, holding every config of both tasks and both
    simulators plus the summaries, and one `.npy` of per-step arrays per
    episode.

  `EpisodeReplayer(path)` reads a saved batch back, by episode or state by
  state.

## Contact models (rollout)

`Utils/ContactModelPresets.py` builds each model with the old study's exact
parameters, for example
`GetContactModelSim("M3", xml, N=256, timestep=..., substeps=4, horizon=8)`:

| | Simulator | What defines it |
| --- | --- | --- |
| M1 | `VectorizedMujoco` | Stiff limit of MuJoCo's soft contact: impedance about 1, solref time constant 2·dt, Newton to 200 iterations and 1e-10 |
| M2 | `VectorizedMujoco` | MuJoCo's default soft contact (pyramidal cone), Newton, 25 iterations, 1e-6 |
| M3 | `ComFree` | Complementarity-free contact (Jin 2024) |
| M4 | `XPBD` | XPBD relaxation over MJWarp's constraint rows |

Any config field can be overridden for a sweep, for example `stiffness=0.5` on
M3.

## Running

Use the `contact_modeling` conda environment. It needs MuJoCo, Warp,
`comfree_warp`, and optionally Pinocchio and Drake. A CUDA GPU is needed for
the rollouts. For offscreen video on a headless machine, set `MUJOCO_GL=egl`;
the driver sets this itself when there's no display.

```bash
# one or more closed-loop episodes (every option: --help)
python ContactModelStudy/Drivers/run_episodes.py --task cube_reorient --n-episodes 5
python ContactModelStudy/Drivers/run_episodes.py --rollout-model M3 --eval-sim pinocchio --no-video
python ContactModelStudy/Drivers/run_episodes.py --hand-acc low --ctrl-time-step 0.032 --time-horizon 0.256

# the same episodes, run concurrently (same options, plus the pool's size)
python ContactModelStudy/Drivers/run_episodes_interwoven.py --n-episodes 10
python ContactModelStudy/Drivers/run_episodes_pooled.py --n-episodes 64 --gpus 0,1 --workers 12

# batches from a CSV, locally or on the HPC: see experiments/README.md
python experiments/run_episode_batches.py experiments/example_batches.csv

# where the time goes, per eval simulator
python test_scripts/profile_run_episodes.py -- --rollout-model M2

# tests (skips what the machine lacks: GPU, Pinocchio, Drake, old package)
python -m pytest tests                 # all, about a minute on an RTX 4090
python -m pytest tests -m "not slow"   # without the end-to-end driver runs
```

Useful driver options:

| Option | Effect |
| --- | --- |
| `--task` | Which object: `cube_reorient`, `duck_reorient`, `ball_reorient` |
| `--rollout-model` | M1–M4 |
| `--eval-sim` | `mujoco`, `pinocchio`, `drake` |
| `--hand-acc` / `--obj-acc` | Rollout scene fidelity; for the duck, `obj_acc` can also be a `foam*` variant |
| `--cost-weight NAME=VALUE` | Override one cost weight; repeatable |
| `--stop-on-success` / `--no-stop-on-success` | Multi-goal episodes |
| `--results` / `--no-results` | Where to save; saved by default to `results/` |
| `--save-steps` | Also save the per-step `.npy` data |
| `--uncertainty` | Record the planner's uncertainty |
| `--video` / `--no-video` | Record a video |

## Running episodes in parallel

`run_episodes.py` alternates a GPU plan with a CPU eval-simulator step, so one
of the two is always idle. The parallel drivers overlap them.

**How the pool works** (`Drivers/EpisodePool.py`):

- **Planner processes**, one or more per GPU, each hold a rollout simulator
  and MPPI. They serve plan requests from any episode.
  - Each request carries that episode's planner state (`SaveState()`: the mean
    sequence, the noise stream, the adaptive temperature) and its goal, and
    the reply carries the updated state back.
  - So no episode is tied to a planner.
- **Worker processes** each own an eval simulator.
  - They run the same episode loop as `run_episodes.py`, but with a
    `RemotePlanner` whose `Plan` sends a request and waits.
- **Scheduling** is two shared queues:
  - a free worker takes the next episode;
  - a free planner takes the next plan request.
- **The main process** collects the finished episodes into one results file,
  as `run_episodes.py` does.

**Seeds.** Each episode's goals and planner noise come from
`(--seed, episode index)`. So results don't depend on the pool's size (1 or 3
workers give the same goals, and first actions agree to MJWarp's noise). They
are not episode-for-episode the same as a sequential run, whose episodes share
one goal stream.

**Planning time.** The recorded planning time is the solve time in the planner
process. Time spent waiting for a free GPU is excluded.

**Stopping.** Ctrl-C, or SIGTERM from `scancel`, stops at the next control
step. Finished episodes are kept, and the ones in progress are saved as
`interrupted`.

**Which driver to use:**

- **`run_episodes_interwoven.py`** is the pool at its smallest: one planner
  and two workers.
- **`run_episodes_pooled.py`** takes `--gpus`, `--planners-per-gpu` and
  `--workers`.
- **The gain depends on the balance.** If planning dominates (MuJoCo eval, a
  long horizon), the GPU is the limit and a second planner on the same GPU
  doesn't help. If the eval simulator dominates (Pinocchio, Drake), more
  workers pay off. The end-of-run line reports how busy the planners and
  workers were.
- **Measured** on 6 Pinocchio episodes:

  | Setup | Sequential | Interwoven | Pooled (6 workers) |
  | --- | --- | --- | --- |
  | GPU-bound | 18.6 s | 12.0 s | 12.1 s |
  | CPU-heavy | 18.5 s | 12.0 s | 9.6 s |

  The totals include startup: each planner process spends about 2.5 s
  compiling kernels.

## Extending

- **A new object** for the LEAP hand:
  1. Add its scenes in `scenes/leap/`, following `env_leap_eval_<obj>.xml` and
     `env_leap_rollout_<obj>_<hand_acc>_<obj_acc>.xml`.
  2. Subclass `LeapReorient`, set `OBJECT`, and return its numbers from
     `objectParams()`.
  3. Register it in `TASKS` in `run_episodes.py`.
- **A new task:**
  1. Subclass `TaskBase`, with its own config if it needs parameters (set
     `CONFIG_CLASS`).
  2. Implement the abstract methods. The cost must be a GPU kernel that reads
     `sim.DeviceState()`.
- **A new eval simulator:**
  1. Subclass `Simulator`, returning MuJoCo-layout state.
  2. Read physical parameters through `MjcfModelInfo` rather than hand-kept
     constants.
  3. Add it to `Utils/EvalSimulators.py`.
- **A new rollout model:**
  1. Subclass `VectorizedMujoco` and override its engine hooks, `_putModel`,
     `_makeData`, `_stepPhysics` and `_forwardPhysics`, as ComFree and XPBD
     do.
  2. Add a preset to `ContactModelPresets.py`.

## Things to know

- **MJWarp isn't bit-deterministic.** GPU atomics make repeated runs differ by
  about 1e-3 after a few hundred steps, and chaotic contact amplifies that.
  Tests compare against the old code within its own run-to-run spread, not
  exactly.
- **GPU precision.** GPU state is float32, and `GetState` on a vectorized
  simulator returns float32 on purpose.
- **Pinocchio's ADMM** sometimes hits its iteration limit, which makes long
  runs chaotic as well. `Pinocchio.Diagnostics()` counts how often.
- **The MJWarp warning `opt.ccd_iterations … needs to be increased`** comes
  from a few collision pairs that never converge. Raising the limit doesn't
  remove it; it only forces a slow kernel recompile.
- **Buffer sizes.** `nconmax` and `njmax` are per world. If MJWarp prints
  `nefc overflow`, raise `njmax`.
- **The old code.** `refactor_progress.md` at the repo root records how each
  piece was ported from `contact_study/`, and why. `codebase_refactor.md` is
  the design spec.
