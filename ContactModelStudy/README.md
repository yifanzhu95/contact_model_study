# ContactModelStudy

A package for studying how the choice of contact model affects sampling-based
control. A planner (MPPI) chooses actions by rolling out many sampled futures
in a fast GPU simulator, the *rollout model*. Those actions are then applied
in a separate, more accurate simulator standing in for reality, the *eval
simulator*. The study measures how well a task is performed as the rollout
model's contact physics (M1–M5) and geometry fidelity change.

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
│   ├── XPBD.py                 XPBD contact on MJWarp
│   ├── Kamino.py               Newton's SolverKamino: full-NCP contact by PADMM (M5)
│   └── SingleWorld.py          a one-world GPU simulator as a plain Simulator (GPU eval sims)
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
│   └── EpisodePool.py             the engine behind the parallel drivers and the BO search
└── Utils/
    ├── ContactModelPresets.py  M1–M5 as ready-made rollout simulators
    ├── EvalSimulators.py       eval simulator by name (mujoco/pinocchio/drake, or M1-M4 on the GPU)
    ├── EpisodeRecorder.py      record, save, summarize and replay episodes
    ├── MjcfModelInfo.py        MJCF parameters as MuJoCo resolves them
    ├── PlannerKLDiv.py         KL between two planners' (or Gaussian) actions at a state
    └── Quaternions.py          wxyz quaternion helpers
```

Outside the package:

| Path | What |
| --- | --- |
| `scenes/leap/` | Scene MJCFs, one eval scene and many rollout scenes per object |
| `experiments/` | Batches of episodes, temperature/noise grids and Bayesian optimization, from a CSV, locally or as an HPC job array (see its README) |
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
| `XPBD` | GPU | As `VectorizedMujoco`, plus `xpbd_substeps`, `xpbd_iterations`, `relaxation`, `contact_compliance`, `joint_compliance`; dense models, Euler or implicitfast only |
| `Kamino` | GPU, Newton | The MJCF, imported by Newton and mapped to MuJoCo's layout by name; `KaminoConfig` holds PADMM and linear-solver settings, and how two geoms' friction combine (`friction_mix_mode`, MuJoCo's max by default) |

**Pinocchio and Drake inherit physics from the MJCF.** Damping, armature,
joint limits, friction and servo gains are all read from the scene. So are the
collision pairs, using MuJoCo's own contype/conaffinity/exclude rules. They
come through `Utils/MjcfModelInfo.py`, which compiles the scene with MuJoCo,
because neither engine's own parser reads MJCF defaults correctly. Changing a
value in the XML changes it in all three eval simulators.

**The GPU contact models can be the eval simulator too.** `--eval-sim M1`–`M5`
builds exactly that rollout preset (`ContactModelPresets`), with one world, on
the eval scene at the eval timestep. It comes wrapped in `SingleWorld`, which
strips the world axis, so the episode loop, the tasks (which judge it on the
host), the recorder and the renderer treat it like CPU MuJoCo. Planning with
one model and judging in another gives a rollout × eval matrix.

- **Steps are replayed from CUDA graphs.** `Step(n)` is split into power-of-two
  blocks, and each block size is captured once, so only a handful of graphs
  ever exist. That is about 4× faster than launching each step.
- **It's still slower than CPU MuJoCo.** A one-world MJWarp step is about
  1.3 ms of back-to-back kernels at the 2 ms eval step: about 41 ms per 64 ms
  control step, against 5 ms for CPU MuJoCo.
- **In the pool, GPU eval workers share the GPU with the planners.** More
  workers on one GPU don't help (8 episodes: 59 s with 1 worker, 61 s with
  4); more GPUs do.
- **M1 isn't bit-reproducible on MJWarp.** Its stiff contact under 200 solver
  iterations ends ~1e-3 apart between two identical runs.

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
| `setCostWeights(w)` | Change cost weights by name on a built task, in place on the device, so a captured rollout graph uses them; the config is updated to match |
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
- **Planning on the eval scene** (`plan_on_eval_scene=True`, driver flag
  `--plan-on-eval`): the planner rolls out on the eval MJCF at the eval
  timestep instead of the rollout scene. The planner checks that its task's
  role agrees with the flag, so the recorded config can't misdescribe the
  rollout. With `--eval-sim Mk` and `--rollout-model Mk`, the planner's model
  is the judged world: an oracle baseline.
  - **Cost:** the control step and horizon resolve at the eval timestep. At
    0.5 ms eval and 4 ms rollout steps, that is 8× the rollout steps per plan:
    about 420 ms per plan instead of about 95 ms (N = 256).
- **Changing settings on a built planner:** `UpdateConfig(temperature=..., noise_sigma=...)`
  changes the fields in `RUNTIME_FIELDS` without a rebuild (they are read on
  every plan). Anything else shaped the buffers or the captured graph and
  needs a new planner.
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
parameters (M1-M3; M4 was rebuilt as exact XPBD), for example
`GetContactModelSim("M3", xml, N=256, timestep=..., substeps=4, horizon=8)`:

| | Simulator | What defines it |
| --- | --- | --- |
| M1 | `VectorizedMujoco` | Stiff limit of MuJoCo's soft contact: impedance about 1, solref time constant 2·dt, Newton to 200 iterations and 1e-10 |
| M2 | `VectorizedMujoco` | MuJoCo's default soft contact (pyramidal cone), Newton, 25 iterations, 1e-6 |
| M3 | `ComFree` | Complementarity-free contact (Jin 2024) |
| M4 | `XPBD` | Exact XPBD over MJWarp's constraint rows: rigid contact (compliance 0), 2 Jacobi sweeps averaged per body at relaxation 2.5, one substep; see [XPBD (M4) tuning](#xpbd-m4-tuning) |
| M5 | `Kamino` | Full NCP solved by PADMM (Newton's `SolverKamino`): ρ₀ = 0.1, full warm start, tolerance 5e-4, up to 100 iterations, contact and joint stabilization 0.05 and 0.1, dense dynamics and a 128-contact buffer per world (tuned against sticking; see [Why M5 objects stuck](#why-m5-objects-stuck)). A contact's friction is the larger of its two geoms', as in MuJoCo and M1–M4 (Newton would average). The accuracy reference, and the slowest model |

Each model is a `ContactModel` in `CONTACT_MODELS`; `getContactModel("M3")`
returns it, and its `makeConfig` / `makeSim` build the config or simulator. A
model's config is layered, each layer overriding the one before:

1. the model's preset (the table above);
2. the task's own tuning for that model, `task.contactParams("M3")`, which by
   default reads the task class's `CONTACT_PARAMS`, merged down the class
   hierarchy:

   ```python
   class CubeReorient(LeapReorient):
       CONTACT_PARAMS = {"M3": {"stiffness": 0.75, "damping": 3e-4}}
   ```

   This is how a model that works best fitted to each task (ComFree, notably)
   gets different values per task. Pass the task to have it applied:
   `GetContactModelSim("M3", xml, N=256, task=task, ...)`. The drivers do, for
   the rollout model and for a GPU eval model (`makeEvalSim(..., task=eval_task)`),
   so the recorded `sim_config` is what actually ran;
3. explicit keywords, which always win, for example `stiffness=0.5` on M3 for a
   sweep.

### ComFree (M3) per-task tuning

ComFree is the one model tuned per task. The tuning ranked three goals: first,
M3 stays stable as an eval sim; second, it gives good SR as a rollout model with
M5 evaluating; third, the three tasks share values. All runs used dt 4 ms,
64 ms control and a 384 ms horizon. The values are set in each task's
`CONTACT_PARAMS`, and the drivers apply them to M3 whether it rolls out or
evaluates. The preset (stiffness 0.75, damping 0.01) still applies when no task
is passed.

| Task | Stiffness | Damping | Rollout SR, M5 eval (tuned / preset) |
| --- | --- | --- | --- |
| Cube | 0.75 | 3e-4 | 0.75 / 0.10 |
| Duck | 0.3 | 3e-4 | 0.75 / 0.70 |
| Ball | 0.75 | 2e-3 | 0.35 / 0.55 |

SR was measured on 20 episodes (seed 2), a seed the searches never saw, and is
good to about ±0.15. The candidates came from small Bayesian optimizations
(13–14 trials of 10 episodes each). Before that, a screen ran batched M3 worlds
under hold, shake and planner-like finger motion and compared them against M5.

Takeaways:

- **Damping is what breaks ComFree, not stiffness.**
  - At damping ≥ 0.03 every task blows up.
  - The preset's 0.01 is at the edge: it launches the cube out of a held
    grasp, NaNs the duck, and makes the held ball vibrate at ~80 mm/s.
  - Stability holds at 3e-4.
- **The duck is the stability limit.** Under planner-like finger motion it goes
  NaN in ~20% of worlds at stiffness 0.75, and in none at ≤ 0.3 (with damping
  ≤ 1e-3).
  - As the eval sim, the preset NaN'd in 8 of 10 closed-loop episodes and the
    tuned values in none.
  - Its rollout SR barely drops at 0.3: 0.80 at stiffness 0.75, 0.75 at 0.3.
- **The ball and cube need stiffness ≈ 0.75 for SR.** Below about 0.6 the ball
  drops in M5 (SR 0 at ≤ 0.35). Below about 0.35 the cube sinks and slips out.
- **The ball and cube want different damping.**
  - Above ~1e-3 the cube starts getting launched.
  - The ball never blew up at any damping ≤ 5e-3, and its SR rose with damping:
    0.15 at 3e-4, 0.35 at 2e-3, 0.40 at 5e-3.
  - The ball's tuned SR is below the preset's. That SR was given up for
    stability, the first priority.
- **So no single set satisfies all three tasks.** The cube and duck share
  damping; the cube and ball share stiffness. If the cube and ball must be
  identical, (0.75, 3e-4) is stable for both but drops the ball's SR to 0.15.
- **A held ball creeps in rotation** at ~10°/s whatever the parameters, because
  ComFree's friction is velocity-proportional with no static part.
- **M3 as an eval sim still fails an M1-planned policy** for every setting
  tried, including the preset.
  - The failures are drops, not blow-ups: M1's grasps lean on friction M3
    doesn't provide.
  - Tuning removes the instability but cannot fix this model mismatch.
  - With M3 planning for itself, the tuned values beat the preset: SR 0.40
    against 0.00 on the cube, 0.20 against 0.10 on the duck (with no NaNs), and
    a tie at 0.20 on the ball.

### XPBD (M4) tuning

M4 is exact XPBD. It solves each row with the exact effective mass
`J M̂⁻¹ Jᵀ`, where `M̂` is the matrix the integrator advances velocity with
(`M + dt·diag(damping)` under Euler). Its compliance is XPBD's own
(`contact_compliance`, `joint_compliance`); MuJoCo's solref/solimp
regularizer is not used. The sweeps stay parallel (Jacobi). As in Newton's
XPBD, each contact row's step is averaged over the contact rows on its bodies
(`relaxation / max(n_a, n_b)`, bodies welded to the world not counted), and
other rows take `relaxation`. There is no depenetration velocity cap. M4 is
held to at most 2 sweeps, to stay close to M3's cost.

The settings were picked from batched worlds (64 per run). Those worlds ran
three motions (hold, sinusoidal shake, planner-like random finger walk) on all
three tasks, in both the rollout and the eval scene. From M5's state at five
points in each run, M4 predicted one planning horizon (384 ms), and the error
is M4's object pose against M5's at the end.

| Setting | Pose error vs M5 (mm / °) | Notes |
| --- | --- | --- |
| M2 (reference) | 2.1 / 3.4 | |
| M3 (preset) | 9.1 / 14.7 | |
| relaxation 1, 1 sweep | 10.4 / 15.1 | |
| relaxation 3, 1 sweep | 6.7 / 9.6 | |
| relaxation 1, 2 sweeps | 7.5 / 10.9 | |
| relaxation 2, 2 sweeps | 6.0 / 8.2 | |
| **relaxation 2.5, 2 sweeps** (preset) | **5.7 / 7.8** | max penetration 11 mm |
| relaxation 3, 2 sweeps | 5.6 / 7.5 | max penetration 14 mm |
| relaxation 4, 2 sweeps | 5.3 / 7.1 | NaNs in 6 worlds |
| relaxation ≥ 6 | — | NaNs and launched objects |

Takeaways:

- **Relaxation 3 is the stability edge.** Averaging divides every contact
  row's step by at least 4 (its own contact's edge rows), so the usable
  relaxation is higher than without it. At 4 some worlds go NaN, and at 6
  most do. 2.5 keeps a margin, which matters because the planner's sampled
  controls are rougher than these motions.
- **2 sweeps beat 1 at every relaxation**, by about 1 mm.
- **Averaging costs little accuracy once tuned.** Weighting each row by how
  strongly it couples to the others (`(A_ee + α̃)/(A_ee + α̃ + Σ|A_ek|)`)
  reached 5.3 / 7.4 at 2 sweeps, against 5.7 / 7.8 here.
- **Compliance should be 0 or tiny** (measured with the coupling weights at 8
  and 16 sweeps, not re-run for averaging).
  - At 1e-3 the cube slips (about 5 mm off M5).
  - 1e-4 is a shade better on the cube and duck and worse on the ball.
  - So the preset is rigid for all tasks.
- **Cost:** one MPPI plan (256 worlds, 96 steps) takes 39 / 51 / 36 ms on the
  cube / duck / ball, against 41 / 54 / 29 ms for M3.
- **The old M4 let the duck sag** in the grasp (0.064 m after 2 s). Now it
  holds it at 0.088 m, against 0.088 m on M5.
- The `XPBD` simulator refuses sparse models (nv > 32), and supports only the
  Euler and implicitfast integrators.

## Running

Use the `contact_kamino` or `contact_modeling` conda environment; the top-level
`README.md` (Installation and system setup) has what each runs and how to build
it. A CUDA GPU is needed for the rollouts. For offscreen video on a headless
machine, set `MUJOCO_GL=egl`; the driver sets this itself when there's no
display. **M5 (Kamino) needs `contact_kamino`.**

**What M5 costs**, measured on an idle RTX 4090 under CUDA-graph replay (which
the planner and the eval sim use), with the earlier settings (`rho0=0.1`, a 0.9
warm start, 800 iterations, a 256-contact buffer). The tuned defaults need
several times fewer PADMM iterations; see [Why M5 objects stuck](#why-m5-objects-stuck).

| Case | Per physics step | Before the contact cap |
| --- | --- | --- |
| Cube rollout scene (4 ms step), 1 world | 2.3 ms | 196 ms |
| Cube rollout scene, 64 worlds | 4.7 ms | 370 ms |
| Cube rollout scene, 256 worlds | 9.3 ms | 857 ms |
| Cube eval scene (0.5 ms step), 1 world | 0.6 ms | 19 ms |
| MPPI plan, 256 samples, 80 steps, σ = 0.2 | about 50 ms (4 s per plan) | — |
| MPPI plan, σ = 0.01 | about 6 ms (0.5 s per plan) | — |

- **What made it fast:** Newton sizes the contact buffer for the worst case
  (4,844 contacts per world on the cube rollout scene, 19,992 on the eval
  scene), and Kamino's solve scales with that capacity rather than with the
  5–20 contacts actually active. `max_contacts_per_world` (default 128) caps
  it, which also lets dense dynamics, a blocked Cholesky factorization per
  step, run for any number of worlds.
  - The most contacts measured on any Leap scene, under exploratory controls,
    is 19 per world. With `convex_meshes=False` the low-fidelity duck reaches
    186, which needs a higher cap.
  - A world that reaches the cap drops contacts, and dropped contacts let
    fingers sink into the object (see below). Kamino prints "per-world contact
    capacity exceeded", and `GetState`/`Diagnostics` warn as well; raise the
    cap if you see it. At one world the cap costs nothing measurable (64, 128
    and 256 all step in the same time).
  - Results agree with the uncapped sparse solver to 7e-4 in joint angles,
    against 3e-4 between two identical uncapped runs and 1e-2 from tightening
    the PADMM tolerance to 1e-6.
- **Meshes collide as convex hulls (`convex_meshes`, default on), as in
  MuJoCo.** Newton imports mesh geoms as triangle meshes. Its triangle-mesh
  path put about 10× MuJoCo's contact points on the duck (32–39 per world
  against 3, on the same states), let the duck drift out of a held grasp,
  and cost 10× the step time.
  - Each colliding mesh is replaced by its hull, at most `maxhullvert`
    vertices, as MuJoCo builds it. Visual-only meshes are untouched.
  - Using the hull matters for speed as well: Kamino bounds a convex mesh by
    querying every vertex, every step, and the duck's hull files carry up to
    8,464 vertices.
  - On the same states the duck now has 1.9–4.0 contacts per world, against
    MuJoCo's 1.4–3.3, and the cube 2.0–4.9 against 2.1–5.6. Kamino still puts
    a few points on a face contact where MuJoCo puts one. These are point
    contacts, not a contact patch: three constraint rows each, with no
    torsional or rolling friction.
- **Planning is bound by PADMM iterations**, because a graph-replayed step
  waits for its slowest world. With the tuned defaults, exploratory controls
  (the driver's σ = 0.2) need about 27 iterations per step and a plan takes
  1.4 s; with the old `rho0=1`, 0.9 warm start and 1,000-iteration cap they
  needed about 220 and 9.7 s (measured on a shared GPU, so both are slow).
- **Eager stepping is 3–15× slower than graph replay**, because Kamino's
  inner loops synchronize with the host each iteration when not captured.

### Why M5 objects stuck

With Kamino as the eval simulator, a held cube would sometimes freeze in the
hand for seconds while the fingers kept moving, with a fingertip partly inside
it. Five settings combined to cause it; all are `KaminoConfig` fields now, and
the defaults are tuned (measured on the cube eval scene at its 0.5 ms step):

1. **PADMM did not converge on a grasp.** A held object leaves its internal
   squeeze forces underdetermined. PADMM's dual residual, which the penalty
   `padmm_rho0` scales, then stalls above the tolerance while the primal and
   complementarity residuals are already met. At `rho0=1` with a 0.9 warm
   start, 98% of squeezed-grasp steps ran to the iteration cap (1,000
   iterations did not help: still 72%). The truncated forces stay near the
   warm start. Over 0.5 s of finger rolling that put the cube up to 4 mm and
   6° from a converged solve at 100 iterations.
   - Fix: `padmm_rho0=0.1` and `padmm_warmstart_scale=1.0`. Grasps now
     converge in about 30 iterations, and the result matches a tight
     (5e-5) solve better than 1,000 iterations at `rho0=1` did.
   - `rho0` cannot go much lower: at 0.03 the primal side under-converges
     on impacts and fingertips sink up to 2.6 mm.
2. **An under-converged step lets fingers sink in.** Whenever the solve is cut
   short (too few iterations, `dynamics_solver="dvi"` at any useful speed,
   dropped contacts), fingertips penetrate 1–5 mm and stay there for hundreds
   of milliseconds.
3. **Penetration came out slowly.** Newton's contact stabilization (0.01)
   removes 1% of a penetration per step: a fingertip 8 mm inside, still
   pressing, took about 150 ms to come out.
   - Fix: `contact_stabilization=0.05`, about 30 ms, with no extra energy (the
     cube is pushed exactly as far). At 0.1 the error against a converged
     solve starts to grow.
4. **The contact buffer could overflow.** At 64 per world, anything that adds
   contacts (here a 1 mm `contact_gap`) dropped real ones. Fingers then passed
   into the cube and a third of the solves failed.
   - Fix: back to 128 (6× the most ever measured). Leave `contact_gap` at 0.
5. **The fingers came off their joints.** Kamino works in maximal
   coordinates: each link is a free body and each hinge a constraint in the
   same solve, with Newton's 0.01 joint stabilization. Steps cut off at the
   iteration cap let the links drift, about 1 mm after 0.25 s of squeezing
   and still growing. `qpos` (joint angles) then no longer says where the
   links are, and the video, the cost and the planner all read `qpos`. In a
   closed-loop episode with the old settings, MuJoCo forward kinematics of
   Kamino's `qpos` put the thumb 2 mm inside the cube, where Kamino's own
   bodies were 0.26 mm in. The penetration in the videos is largely this.
   - Fix: convergence (1.) keeps the drift under 0.01 mm, and
     `joint_stabilization=0.1` holds it to about 0.1 mm even when no step
     converges, at no cost in iterations (and with lower error against the
     converged reference).

**Result** on the same two closed-loop episodes (`--plan-on-eval`, M2
planner, seed 1; episode 0 / episode 1), old settings against new (run before
`joint_stabilization` went to 0.1, which changes neither iterations nor
speed):

| | `rho0=1`, 100 iterations | Tuned defaults |
| --- | --- | --- |
| Steps not converged | 38% / 69% | 1.9% / 11% |
| PADMM iterations per step | 75 / 89 | 14 / 22 |
| Eval time per 0.5 ms step | 6.9 / 9.7 ms | 2.1 / 2.6 ms |
| Worst penetration, Kamino's contacts | 0.19 / 0.26 mm | 0.09 / 0.09 mm |
| Worst penetration, MuJoCo FK of `qpos` | 0.50 / 1.97 mm | 0.06 / 0.11 mm |

On one world under graph replay the eval step is about 3.5× faster than the
old defaults (1.7 against 6.0 ms, on a shared GPU). `padmm_max_iterations=50`
is no faster on one world, but a little less accurate.

How each number was measured is in `refactor_progress.md` (Kamino sticking,
2026-10-08). `tests/test_kamino.py::test_a_squeezed_grasp_converges_and_a_sunk_fingertip_comes_out`
fails on either half of the old settings.

```bash
# one or more closed-loop episodes (every option: --help)
python ContactModelStudy/Drivers/run_episodes.py --task cube_reorient --n-episodes 5
python ContactModelStudy/Drivers/run_episodes.py --rollout-model M3 --eval-sim pinocchio --no-video
python ContactModelStudy/Drivers/run_episodes.py --hand-acc low --ctrl-time-step 0.032 --time-horizon 0.256

# the same episodes, run concurrently (same options, plus the pool's size)
python ContactModelStudy/Drivers/run_episodes_interwoven.py --n-episodes 10
python ContactModelStudy/Drivers/run_episodes_pooled.py --n-episodes 64 --gpus 0,1 --workers 12

# batches, grids and Bayesian optimization from a CSV, locally or on the HPC: see experiments/README.md
python experiments/run_episode_batches.py experiments/example_batches.csv
python experiments/run_bayes_opt.py experiments/example_bayes_opt.csv --cell 0

# where the time goes, per eval simulator
python test_scripts/profile_run_episodes.py -- --rollout-model M2

# tests (skips what the machine lacks: GPU, Pinocchio, Drake, old package)
python -m pytest tests                 # all, about 3.5 minutes on an RTX 4090
python -m pytest tests -m "not slow"   # without the end-to-end driver runs
```

Useful driver options:

| Option | Effect |
| --- | --- |
| `--task` | Which object: `cube_reorient`, `duck_reorient`, `ball_reorient` |
| `--rollout-model` | M1–M5 |
| `--eval-sim` | `mujoco`, `pinocchio`, `drake`, or a GPU contact model `M1`–`M5` |
| `--hand-acc` / `--obj-acc` | Rollout scene fidelity; for the duck, `obj_acc` can also be a `foam*` variant |
| `--cost-weight NAME=VALUE` | Override one cost weight; repeatable |
| `--stop-on-success` / `--no-stop-on-success` | Multi-goal episodes |
| `--results` / `--no-results` | Where to save; saved by default to `results/` |
| `--save-steps` | Also save the per-step `.npy` data |
| `--uncertainty` | Record the planner's uncertainty |
| `--plan-on-eval` | Plan on the eval scene at the eval timestep (`plan_on_eval_scene`) |
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
- **The main process** owns an `EpisodePool` and submits `EpisodeJob`s to it.
  `runPool`, behind both drivers, submits `--n-episodes` jobs and collects them
  into one results file, as `run_episodes.py` does.
- **Per-job settings.** The pool's processes live until it is closed, and a
  job can name its own rollout model (each planner process holds a planner
  per model the pool was built for), planner params and cost weights. The
  planner process applies them in place before each plan
  (`UpdateConfig`, `setCostWeights`), so nothing is rebuilt.
  `experiments/run_bayes_opt.py` keeps one pool for a whole search this way.

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
     `objectParams()`. If a contact model needs tuning for it, set
     `CONTACT_PARAMS`.
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
  2. Add a `ContactModel` to `CONTACT_MODELS` in `ContactModelPresets.py`
     (subclass it and override `_resolve` if a field depends on others, as
     M1's `StiffMujocoModel` does for its time constant).

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
