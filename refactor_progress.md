# Refactor Progress

Tracks the port from the legacy `contact_study/` package to the new
`ContactModelStudy/` layout described in [codebase_refactor.md](codebase_refactor.md).

The new tree lives **in parallel** with the old one. Nothing under
`contact_study/`, `experiments/`, `analysis/` or `tests/` has been modified, so
every existing script still runs while the port is in progress.

## Status legend

| Symbol | Meaning |
| --- | --- |
| DONE | Ported, imports cleanly, matches the spec in `codebase_refactor.md` |
| WIP | Partially ported |
| TODO | Directory/module exists but is empty |
| DEFER | Explicitly "ignore for now" in the refactor plan |

## Module status

### Simulators

| Module | Status | Ported from | Notes |
| --- | --- | --- | --- |
| `Simulator.py` | DONE | `contact_study/sim/base.py` | `Simulator` ABC + `SimulatorConfig` dataclass |
| `VectorizedSimulator.py` | DONE | `contact_study/contact_models/api.py` | `VectorizedSimulator` ABC + `VectorizedSimulatorConfig` |
| `Mujoco.py` | DONE | `contact_study/sim/mujoco_sim.py` | `Mujoco`; rendering left for `Renderers/` |
| `VectorizedMujoco.py` | DONE | `contact_study/contact_models/api.py` (MJWarp path) | `VectorizedMujoco` + `VectorizedMujocoConfig` |
| `ComFree.py` | TODO | `contact_study/contact_models/api.py` (M3 path) | |
| `XPBD.py` | TODO | `contact_study/contact_models/xpbd_backend.py` | 863 lines; largest single port |
| `Pinocchio.py` | TODO | `contact_study/contact_models/pinocchio_sim.py` | 1540 lines; largest file in the repo |
| `Drake.py` | TODO | `contact_study/contact_models/drake_sim.py` | |

### Tasks

| Module | Status | Ported from | Notes |
| --- | --- | --- | --- |
| `TaskBase.py` | DONE | `contact_study/tasks/base.py` | `TaskBase` + `TaskRole` |
| `LeapReorient.py` | DONE | `contact_study/tasks/grasp_reorient.py` | Shared leap machinery + the cost |
| `CubeReorient.py` | DONE | `contact_study/tasks/grasp_reorient.py` | The cube's numbers only |
| `BallReorient.py` | TODO | `contact_study/tasks/grasp_reorient.py` | Split out of the shared reorient task |
| `DuckReorient.py` | TODO | `contact_study/tasks/grasp_reorient.py` | Split out of the shared reorient task |
| `XML_Files/` | TODO | `scenes/leap/*.xml` | Scene XML moves in with the tasks |

### SamplingBasedPlanners

| Module | Status | Ported from | Notes |
| --- | --- | --- | --- |
| `SamplingBasedPlannerBase.py` | DONE | `contact_study/planners/base.py` | Planner no longer owns the physics |
| `MPPI.py` | DONE | `contact_study/planners/mppi.py` | `MPPI` + `MPPI_Config` |
| `CEM.py` | DEFER | `contact_study/planners/cem.py` | "Ignore for now" |
| `PS.py` | DEFER | `contact_study/planners/predictive_sampler.py` | "Ignore for now" |

### Renderers

| Module | Status | Ported from | Notes |
| --- | --- | --- | --- |
| `RendererBase.py` | DONE | `contact_study/sim/base.py` (`FrameClock`) | 4 classes: base + video, each with a config |
| `MujocoVideoRenderer.py` | DONE | `contact_study/sim/mujoco_sim.py` | Now a `VideoRendererBase` |
| `MujocoInteractiveRenderer.py` | TODO | new | `mujoco.viewer`; no user input back into the sim |

### Drivers / Utils / other

| Module | Status | Ported from | Notes |
| --- | --- | --- | --- |
| `Drivers/run_episodes.py` | DONE | `contact_study/drivers/run_eval_episode.py` | Closed-loop MPC episodes + video + JSON |
| `Drivers/run_episodes_parallel.py` | DEFER | `contact_study/drivers/run_async_eval_episode.py` | "Ignore for now" |
| `Utils/EpisodeIO.py` | TODO | `contact_study/evaluation/json_io.py` | |
| `Experiments/` `Results/` `Logs/` `Tests/` `Videos/` | DEFER | — | "Ignore for now" |

## What was done — Simulators/Simulator.py

Created `ContactModelStudy/Simulators/Simulator.py` with the two classes the
plan calls for.

**`SimulatorConfig`** — deliberately minimal: `timestep`, `substeps` and
`gravity`, plus a `control_timestep` property (`timestep * substeps`).
`__post_init__` rejects a non-positive timestep, `substeps < 1` and a gravity
vector that is not length 3, and coerces gravity to floats. Nothing
backend-specific lives here, so the same config object is valid for every
simulator M1-M4.

**`Simulator`** — the abstract base class. `__init__(xml, sim_config)` resolves
the model source and stores the config; the six spec'd operations
(`SetState`, `GetState`, `SetControl`, `GetControl`, `Step`, plus the `nq`/`nv`/`nu`
dimensions) are abstract.

### Decisions worth knowing about

1. **`SetState` takes `(q, q_dot=None)`, not one array.** The plan writes
   `SetState(Q)` against a `GetState` that returns the pair `(q, q_dot)`. A
   single packed array would force every caller to know the `nq`/`nv` split to
   build it, and the two differ whenever there is a free joint (quaternion in
   `qpos`, angular velocity in `qvel`). Taking them separately makes `SetState`
   the exact inverse of `GetState`; `q_dot=None` zeroes the velocities, which is
   the common case at episode reset.

2. **XML string vs. path is detected from content, not extension.** The plan
   says `__init__` takes "a string representing a XML file or the path to an XML
   file." `_resolve_model_source` sniffs for a leading `<?xml`/`<mujoco`/`<robot`
   tag; anything else is treated as a path, and a missing one raises
   `FileNotFoundError` immediately. Both forms populate `model_xml`, and the
   path form also populates `model_path`, which backends that can only parse
   from disk (Drake) and relative mesh references both need.

3. **Dimension properties (`nq`, `nv`, `nu`) are part of the interface.** Not in
   the plan's function list, but planners have to size their control sequences
   and rollout buffers, and every backend already knows these numbers. The
   alternative is each planner reaching into a backend-specific model handle,
   which is what the interface exists to prevent.

4. **`GetState`/`GetControl` must return copies.** Documented on the abstract
   methods. Rollout buffers hold these arrays across steps; returning a view
   into live backend memory would mutate stored history on the next `Step`.

5. **MuJoCo index layout is the interface contract.** Carried over from
   `contact_study/sim/base.py`, whose docstring already established it: state
   comes out in the MuJoCo `qpos`/`qvel` order and controls go in MuJoCo-ordered
   and absolute, whatever the backend uses internally. Pinocchio and Drake remap
   inside themselves.

6. **Rendering is *not* on `Simulator`.** The legacy `EvalSimulator` carried
   `render()`, `save_video()` and a `FrameClock`; `MujocoSimulator` owns a
   `mujoco.Renderer` and a frame list. The plan gives rendering its own
   directory, so none of that is on the new base class — it goes to
   `Renderers/` when those modules are written. `FrameClock` is the piece to
   carry over: it schedules frames on the *sim* clock, so video playback speed
   does not depend on control frequency.

7. **`Close()` and context-manager support are concrete no-ops.** So a driver
   can `with`-block or unconditionally close whatever simulator it was handed;
   GPU backends override it to free device memory.

## What was done — Simulators/VectorizedSimulator.py

Created `ContactModelStudy/Simulators/VectorizedSimulator.py` with the two
classes the plan calls for.

**`VectorizedSimulatorConfig`** — inherits `SimulatorConfig` (so `timestep`,
`substeps`, `gravity` and `control_timestep` carry over) and adds two fields a
parallel backend needs that a single-world one does not: `horizon` (the `H` of
the control sequences, fixed at construction so device buffers are allocated
once) and `device`. `__post_init__` chains to the parent's validation and adds
`horizon >= 1`.

**`VectorizedSimulator`** — extends `Simulator`. Adds `Step_GPU` (abstract) and
`SetControlSequence` (concrete), overrides `Step`, and re-declares the four
state/control abstracts so their batched shapes are documented on the method
the subclass actually implements.

Concrete helpers subclasses get for free:

| Member | Purpose |
| --- | --- |
| `N` / `n_worlds` | World count, validated `>= 1` at construction |
| `horizon` | `H`, read back from the config |
| `_to_worlds(x, dim, name)` | Coerce `(dim,)` or `(N, dim)` to `(N, dim)`, else raise |
| `SetControlSequence(U_n)` | Validate, store, reset the cursor |
| `ClearControlSequence()` | Drop the sequence, revert to the held control |
| `_validate_control_sequence(U_n)` | Shape checking alone, for a backend that stores on-device |
| `_advance_sequence()` | This step's *index*; advances the cursor (no host array) |
| `_has_control_sequence` | Whether a sequence is active; overridable |
| `_next_control()` | Host-side convenience: this step's `(N, nu)` slice |
| `Step(steps)` | Delegates to `Step_GPU` |

The last four came out of writing `VectorizedMujoco` — see that section for why
a single host-array accessor was not enough.

### Decisions worth knowing about

1. **Constructor is `(xml, sim_config, N)`, not `(time_step, sim_config, N)`.**
   The plan's signature passes a bare `time_step` alongside a config that
   already carries one, and its prose says the function "should take in a
   string representing a XML file or the path to an XML file" — so the first
   argument is the model. This keeps the model source in the same position as
   `Simulator.__init__`.

2. **`N` is a constructor argument, not a config field.** It decides how much
   device memory gets allocated, so it belongs at construction rather than in a
   config object that gets copied and passed around. This also matches the plan.

3. **Batched shapes: a leading world axis on everything.** `q` is `(N, nq)`,
   `q_dot` is `(N, nv)`, `u` is `(N, nu)`. Setters also accept the un-batched
   shape and broadcast, which is how an episode starts — every world seeded from
   the same measured state, then driven apart by different controls. `_to_worlds`
   raises on anything else rather than letting NumPy broadcast into the wrong
   axis, a bug that otherwise surfaces much later as a rollout that quietly did
   nothing.

4. **`U_n` is `(N, H, nu)`.** Matches what the existing MPPI already builds —
   `dim=(n_samples, horizon, nu)` in `contact_study/planners/mppi.py`. `(H, nu)`
   is accepted and broadcast, for a nominal or warm-start rollout.

5. **Stepping past `H` holds the last control.** A rollout that overruns its
   horizon should coast, not wrap around and silently restart its plan, and not
   crash mid-rollout. `SetControlSequence` resets the cursor, so re-planning
   replays from the start.

6. **`Step_GPU` does no host transfer; `GetState` is the explicit one.** The
   split the plan implies but does not state. `Step` is kept as a concrete
   delegate to `Step_GPU` so anything written against the `Simulator` interface
   runs unchanged on a vectorized backend.

7. **`SetControl` does not clear an active sequence.** The sequence wins on the
   next step; `ClearControlSequence()` is the explicit way back to the held
   control. Making a `SetControl` silently cancel a sequence would be a
   surprising action-at-a-distance in the middle of a rollout.

## What was done — Simulators/Mujoco.py

Created `ContactModelStudy/Simulators/Mujoco.py` with a single class, `Mujoco`:
standard CPU MuJoCo behind the `Simulator` interface. Since the interface's
index layout *is* MuJoCo's, every method is an identity map — no remapping, in
contrast with the Pinocchio and Drake wrappers still to come.

Ported from `contact_study/sim/mujoco_sim.py`, minus everything that was not
simulation. The old `MujocoSimulator` was 169 lines; this is the ~60 lines of it
that actually step physics.

### What was deliberately left out

| Dropped | Where it goes |
| --- | --- |
| `mujoco.Renderer`, `_build_camera`, `_capture`, `_frames`, `save_video`, `render()`, `FrameClock` | `Renderers/MujocoVideoRenderer.py` |
| `height`/`width`/`camera_name`/`use_mp4` constructor args | Renderer config |
| `_resolve_goal_mocap_id`, `set_goal_quat` | `Tasks/` — a goal marker is task state, not simulator state |
| `hard_contact` preset (`_apply_hard_contact_preset`) | A contact-model config; see the open question below |

A renderer gets what it needs from the public `mjm` and `mjd` handles, so the
simulator never has to know a renderer exists. Verified: a constructed `Mujoco`
carries only `config`, `mjm`, `mjd`, `model_path`, `model_xml`.

### Decisions worth knowing about

1. **Class is `Mujoco`, matching the filename**, following the plan's
   `Simulator.py`/`TaskBase.py`/`MPPI.py` convention rather than the old
   `MujocoSimulator` name.

2. **Compiled from the path when there is one.** `from_xml_path` resolves
   `<mesh>`, `<texture>` and `<include>` relative to the XML's own directory;
   `from_xml_string` cannot. Inline XML is therefore only good for
   self-contained scenes — the leap scenes must be passed as paths. Both forms
   are tested.

3. **The config overwrites `timestep` and `gravity`; nothing else is touched.**
   A scene file and a sweep config can't disagree about the two things
   `SimulatorConfig` actually carries, while integrator, solver, cone and
   contact `solref`/`solimp` stay exactly as the XML declares them.

4. **`SetState` runs `mj_forward`.** Contacts, site positions and sensors are
   consistent with the new state before anything reads it — including a
   renderer drawing a frame that was never stepped.

5. **`Step` does not clear the control.** MuJoCo's own zero-order hold, matching
   the base-class contract.

6. **`_check` validates lengths.** Assigning a wrong-length array straight into
   an `MjData` field raises a message naming only the buffer shape, which is
   hard to trace back to a call site inside a rollout loop.

7. **`time` is exposed; `SetState` does not rewind it.** Teleporting the state
   mid-episode is a perturbation, not a reset, so a driver reusing one simulator
   across episodes tracks episode time itself.

8. **`GetControl` can report a value the solver won't apply.** MuJoCo clamps to
   `ctrlrange` at step time, not at assignment; documented on `SetControl`
   rather than silently clamping, so the caller sees exactly what it wrote.

9. **`Mujoco` is not re-exported from `Simulators/__init__.py`.** Importing the
   subpackage should not drag in MuJoCo — and later, Warp, Drake or Pinocchio.
   Import it as `from ContactModelStudy.Simulators.Mujoco import Mujoco`.

### Verified

Loaded `scenes/leap/env_leap_eval_cube.xml` (nq=23, nv=22, nu=16) under MuJoCo
3.6.0 in the `contact_modeling` conda env: config timestep/gravity applied over
the XML's, state set/get roundtrip, `GetState` returns copies rather than live
views, `q_dot=None` zeroing, control set/get, `Step(10)` advancing `time` by
exactly `10 * dt`, control held across a multi-step, and both shape errors and
the missing-file error naming the offending argument. Also ran an inline-XML
pendulum for 200 steps to confirm the string path integrates real dynamics.

## What was done — Renderers/RendererBase.py and MujocoVideoRenderer.py

**Revised 2026-09-23** to match the updated plan, which split the renderer
interface in two and gave the config real camera control. What follows describes
the current code; the changes from the first version are called out at the end.

A renderer draws a *state*, not a simulator. It owns its own `MjModel`, compiled
from the task's MJCF, and is handed positions:

    renderer.RenderState(q)

So a renderer never touches a simulator, and one renderer can visualize any
backend — including those with no graphics of their own (Pinocchio, XPBD) and a
single world sampled out of a vectorized batch.

### The four classes

| Class | Role |
| --- | --- |
| `RendererBase` | Anything that draws a state: `RenderState(q)`, `Close()` |
| `RendererBaseConfig` | `width`, `height`, `fps`, `cam_name`, `cam_pos`, `cam_quat`, `cam_fovy` |
| `VideoRendererBase` | Adds `getStepsPerFrame()`, `Save(path)`, `Reset()` |
| `VideoRendererBaseConfig` | Adds `output_path` |

`MujocoVideoRenderer` now subclasses `VideoRendererBase`. An interactive viewer
will subclass `RendererBase` directly, which is what the split is for.

### Decisions worth knowing about

1. **Every camera field defaults to `None`, meaning "use what the MJCF says".**
   That lets a config override one aspect — nudge the field of view — without
   restating a pose the scene already gets right.

2. **Camera overrides resolve three ways.** With a `cam_name`, the model's
   camera entry is patched in place, which is exact. With no name but some
   override, a free `MjvCamera` is synthesized from the requested pose — a
   camera cannot be added to a compiled model, and `MjvCamera` is parameterized
   by lookat/azimuth/elevation rather than by pose, so the conversion happens in
   `_freeCamera`. With neither, it is `-1`, MuJoCo's default framing.

3. **The task fills in the config before the renderer is built**
   *(revised 2026-09-23; this used to be the task handing out a whole config
   via `getRendererConfig()`)*. The caller builds a config, passes it to
   `task.alignRendererConfigWithTask(cfg)` — which sets only the fields the task
   cares about, in place — and then constructs the renderer from it:

       cfg = VideoRendererBaseConfig(width=640, height=480)
       task.alignRendererConfigWithTask(cfg)
       renderer = MujocoVideoRenderer(task, cfg)

   A config passed to the renderer is used exactly as given; it is *not*
   re-aligned, because the caller may have overridden the task's choice since
   (the driver's `--camera` does exactly that). With `config=None` the renderer
   builds a default config and has the task align it, so the task's camera is
   still used when the caller has no settings of its own. A bare MJCF path has
   nothing to align with and now simply gets the defaults — the old "config
   required" error existed only because there was nothing to ask for one.

   The plan's Renderers section still says `RendererBase` calls the task's
   `getRendererConfig`; that line predates the change and is yours to update.

4. **A base config is up-converted, not rejected.** A task publishes one camera
   setup; a video renderer needs that plus an output path. `_coerceConfig`
   copies the shared fields into the renderer's own `CONFIG_CLASS` so no task
   has to know which renderer will consume its config.

5. **`getStepsPerFrame()` moved from the config to the renderer** and now reads
   the *task's* timestep, per the plan, rather than taking one as an argument.
   Its unit is one step of that timestep; a caller advancing several steps at a
   time divides by that stride, which is what the driver does.

6. **`Save` returns the written path** although the plan writes it as `-> None`.
   It is a superset — callers that ignore it are unaffected — and the driver
   logs it. Flagged rather than hidden.

7. **`height`, not `hight`.** The plan's attribute list has the typo; the code
   uses the correct spelling, consistent with `mujoco.Renderer`'s own argument.

### Knock-on changes to Tasks

The plan's renderer-config hook and the task-timestep requirement mean the task
layer had to grow two things:

  * **`TaskBase.timestep`** (constructor argument, default 0.002) — the physics
    timestep the task is posed at. The task carries it because things other than
    a simulator need it, and because a task's costs and initial state are only
    meaningful at the rate they were tuned for. A simulator may still run at a
    different one; this is the task's declared rate, not a constraint.
  * **`TaskBase.alignRendererConfigWithTask(config)`** (was
    `getRendererConfig()`) — a no-op by default; `LeapReorient` sets
    `cam_name = "demo-cam"`, which every leap scene defines, and touches
    nothing else.

`Tasks` now imports `Renderers` for the config type, done lazily inside the
method. It is a one-way dependency — `Renderers` still knows nothing about
`Tasks`, and still duck-types `getModelPath`.

### Call sites updated

`Drivers/run_episodes.py` and `test_scripts/render_finger_curl.py` both moved to
`VideoRendererBaseConfig` and `renderer.getStepsPerFrame()`. The driver's
`--camera` now defaults to `None` meaning "use the task's camera", with an
explicit name or `none` overriding it, and both tasks are constructed with the
CLI `--timestep` so the renderer's frame scheduling agrees with the simulator.

### Verified

Under MuJoCo 3.6.0 with `MUJOCO_GL=egl`, on `env_leap_eval_cube.xml`:

  * the class hierarchy and both config field lists are as specified, and a
    task's plain `RendererBaseConfig` up-converts to a
    `VideoRendererBaseConfig`;
  * `getStepsPerFrame()` tracks the task's timestep — 17 at dt=0.002/30 fps, 33
    at dt=0.001/30 fps, 8 at dt=0.002/60 fps, and floors at 1 when fps outruns
    the timestep;
  * all six camera paths render distinct frames (named, free, `cam_pos`,
    `cam_fovy`, `cam_quat`, free+pose), and — the strong test —
    **overriding with the scene's own authored pos/quat/fovy reproduces the
    un-overridden image bit for bit**, while a 2 cm camera nudge changes it;
  * `Save` to `.mp4` and `.gif`, `Reset` (frame count to 0, then `Save`
    returning `None`), `output_path` written on `Close`, and idempotent
    `Close`;
  * a bad `cam_name`, a task with no
    `timestep`, and every invalid config value raise with messages naming the
    problem;
  * end to end: the driver writes 16 frames at 30 fps for 0.48 s of sim and 21
    frames at 60 fps for 0.32 s — both correct for their cadence — and the curl
    script still runs from its new `test_scripts/` home.

### Verified — config alignment (2026-09-23)

  * `alignRendererConfigWithTask` returns `None` and changes exactly one field
    (`cam_name: None -> 'demo-cam'`), leaving width, height, fps, `cam_fovy` and
    `output_path` as the caller set them; the base task's version changes
    nothing, and the method is not abstract;
  * `config=None` gives an aligned `VideoRendererBaseConfig`, and its frame is
    **bit-identical** to one rendered from an explicit `cam_name="demo-cam"`;
  * an override applied after aligning survives construction; a bare path with
    no config gets plain defaults without error;
  * the driver's default run uses the task's camera and `--camera none` the
    free camera — the two videos differ; the curl script still runs.

### Changed from the first version

`RendererBaseConfig.camera` became `cam_name`; `cam_pos`/`cam_quat`/`cam_fovy`
are new; `steps_per_frame(timestep)` on the config became `getStepsPerFrame()`
on the renderer; `Save`/`Reset` are now declared on `VideoRendererBase`;
`MujocoVideoRendererConfig` is gone, replaced by the shared
`VideoRendererBaseConfig`; and `FrameClock` remains un-ported for the same
reason as before — it lived inside the simulator and fired from within `step()`,
which is exactly the coupling this refactor removes.

### Not done

`MujocoInteractiveRenderer.py` — the third renderer in the plan, wrapping
`mujoco.viewer`. It subclasses `RendererBase` rather than `VideoRendererBase`
(nothing to save), and needs a real display, so it could not be verified in this
environment.

## What was done — Simulators/VectorizedMujoco.py

Created `ContactModelStudy/Simulators/VectorizedMujoco.py`: `N` MuJoCo worlds
stepped together on the GPU through MJWarp, behind the `VectorizedSimulator`
interface. Ported from the MJWarp branch of
`contact_study/contact_models/api.py` and the rollout plumbing in
`contact_study/planners/base.py`.

**`VectorizedMujocoConfig`** — extends `VectorizedSimulatorConfig` with `cone`,
`solver`, `iterations`, `tolerance`, `nconmax`, `njmax`.

**`VectorizedMujoco`** — exposes `mjm` (host model), `m` (device model) and `d`
(device data, with `d.qpos`/`d.qvel`/`d.ctrl` as `(N, nq)`/`(N, nv)`/`(N, nu)`
float32 warp arrays).

### The base class needed revising

Writing the concrete backend exposed a real flaw in `VectorizedSimulator`:
`_next_control()` returned a **host** numpy slice, so a GPU backend would have
had to push controls across PCIe on every single step — defeating the entire
point of the vectorized path. The old planner avoided this with a device-side
`_assign_ctrl_kernel` reading a resident `(N, H, nu)` array, and the new base
class had no way to express that.

Fixed by splitting the base's sequence handling into three pieces:

| Member | Role |
| --- | --- |
| `_validate_control_sequence(U_n)` | Shape checking alone, reusable by a backend that stores the sequence on-device |
| `_advance_sequence()` | Returns the clamped step *index* and advances the cursor — no host array involved |
| `_next_control()` | The host-side convenience, now implemented on top of `_advance_sequence` |
| `_has_control_sequence` | Property, so a backend can report a sequence living somewhere other than the host |

`VectorizedMujoco` overrides `SetControlSequence` to upload once into a
preallocated device buffer and drops the host copy; `Step_GPU` writes each
step's slice with `_assign_ctrl_kernel`. Nothing crosses the bus per step. The
host-side path was re-tested afterwards and behaves exactly as before.

### Decisions worth knowing about

1. **Solver settings default to "leave the XML alone."** `solver`, `iterations`
   and `tolerance` are `None` by default, matching the policy already set in
   `Mujoco.py`, so the CPU and GPU wrappers agree about a scene unless told
   otherwise. Only `timestep`, `gravity` and the cone are written
   unconditionally. Note this differs from the old `api.put_model`, which always
   patched all four from `MujocoSolverParams`.

2. **The cone is forced to pyramidal, and `cone="elliptic"` is rejected.**
   MJWarp implements no elliptic cone on the GPU. It is named explicitly in the
   config rather than silently assumed because it is a real divergence from
   reference MuJoCo — a scene authored with an elliptic cone does not run
   unchanged here, and results must be reported as pyramidal. This is the same
   caveat the old `ContactModelConfig.M2` docstring spells out.

3. **`GetState` returns float32, not float64.** MJWarp is single precision
   throughout. Widening would imply a precision the simulator does not have and
   would hide genuine divergence from the CPU `Mujoco` simulator.

4. **The control buffer is allocated once at construction.** `N` and `H` are
   both fixed then, so a planner re-planning every control step does not
   allocate device memory on every call.

5. **No CUDA-graph capture in the simulator.** Graph capture spans reset +
   H steps + cost accumulation, which is the *planner's* structure, not the
   simulator's. `Step_GPU` is capture-safe — the sequence index resolves in
   Python at capture time, so each captured step carries its own constant index,
   which is what an unrolled rollout wants — so `MPPI.py` can capture around it.

6. **MJWarp is imported lazily**, through comfree_warp's vendored copy (there is
   no standalone `mujoco_warp` pinned in `pyproject.toml`), and
   `VectorizedMujoco` is not re-exported from `Simulators/__init__.py`, so
   importing the subpackage never initializes Warp or a CUDA context.

### Verified

On an RTX 4090 with Warp 1.13.0 / MuJoCo 3.6.0, scene
`scenes/leap/env_leap_rollout_cube_high_high.xml` (nq=23, nv=22, nu=16):

  * `SetState` broadcast seeds all `N` worlds identically and matches the task's
    `init_qpos`; a per-world control sequence drives them apart by an amount
    that increases monotonically with the per-world control offset;
  * **correction, found later while testing the planner:** this section
    originally said worlds "stay bit-identical under identical controls". That
    was verified over 10 steps and does not hold over a longer rollout — see
    "MJWarp is not reproducible" under the planner section. The divergence is
    MJWarp's, reproduced with no wrapper involved, but the original claim was
    broader than what was tested;
  * control sequences replay in order, hold the last slice past `H`, reset their
    cursor on re-set, broadcast from `(H, nu)`, and fall back to the held
    control after `ClearControlSequence`;
  * every bad shape and bad config value raises with a message naming the
    expectation;
  * **GPU vs CPU cross-check** — same scene, state and control on
    `VectorizedMujoco` and `Mujoco`: max |qpos| difference is 4.3e-08 after one
    step, growing to ~1e-3 by 50 steps. That is float32-vs-float64 divergence
    in a contact-rich scene, and is the strongest evidence the wrapper is
    wired up correctly;
  * **throughput scales with `N`**: 11.9k world-steps/s at N=64, 72.5k at
    N=512, 131.5k at N=2048.

### Note

MJWarp prints `opt.ccd_iterations, currently set to 35, needs to be increased`
on this scene. It comes from MJWarp's convex-collision solver, not from this
wrapper, and `ccd_iterations` is not currently exposed by
`VectorizedMujocoConfig` — worth adding if the duck/mesh scenes turn out to need
it.

## What was done — Tasks/TaskBase.py, LeapReorient.py, CubeReorient.py

**Revised 2026-09-23** to match the updated plan, which moved the cost onto the
simulator and added outcome checks. What follows describes the current code; the
changes from the first version are called out at the end.

The three-level hierarchy: `TaskBase` -> `LeapReorient` -> `CubeReorient`. A
task owns its scene path, its initial conditions, its cost, and what counts as
winning or losing. It owns no simulator and no renderer — it is handed a
simulator and reads what it needs from it.

### Interface

| Member | Kind | Notes |
| --- | --- | --- |
| `getModelPath()` | abstract | Scene for the task's `TaskRole` |
| `getInitialState()` | abstract | `(q0, q_dot0, u0)` — not in the plan, see below |
| `calcCosts(sim, terminal, out)` | abstract | **Device only**; takes a `VectorizedSimulator` |
| `isFailure(sim, out)` | abstract | `bool` for a `Simulator`, capturable bool array for a vectorized one |
| `isSuccess(sim, out)` | abstract | Same conventions |
| `alignRendererConfigWithTask(config)` | concrete | Fills in a renderer config in place; no-op by default, `LeapReorient` sets `demo-cam` |
| `setSimToInitialState(sim)` | concrete | Written once in the base on top of `getInitialState`; capturable when vectorized |
| `sampleNewGoal(current_goal=None)` | abstract | Draws a goal as a rotation *of the current goal*; adopts nothing |
| `setGoal(goal)` | abstract | Adopts a goal; cost and success test read it from then on |
| `setRendererToGoal(renderer)` | abstract | Turns the renderer's goal marker, so videos show the goal |
| `timestep` | attribute | The task's declared physics rate |

`LeapReorient` also exposes `goalErrors(q, q_dot)` — the pos/quat/vel errors
`isSuccess` thresholds — so a driver can report them. It is the success metric,
not the cost.

### Decisions worth knowing about

1. **The cost exists only on the device.** Per the plan, `calcCosts` takes a
   simulator and "should not reproduce the functionality on the CPU", so the
   host cost, `_cost_one`, `_forward_tips` and `calcCosts_GPU` are gone.
   `calcCosts(sim)` reads `sim.DeviceState()` itself and launches one kernel.
   Passing a CPU `Simulator` raises a `TypeError` that points at
   `isSuccess`/`isFailure`, which is how a CPU episode is judged now.

2. **`calcCosts` returns the device array, not numpy.** The plan writes
   `-> Numpy Array`, but the planner calls it inside a captured CUDA graph, and
   producing numpy would force a device-to-host sync that a graph cannot
   record. `.numpy()` on the result gives the array the plan describes.

3. **Outcome checks dispatch on the simulator type.** A `Simulator` gets a plain
   `bool` from its host state; a `VectorizedSimulator` gets an `(N,)` `wp.bool`
   array from a kernel. The kernels touch nothing on the host, and with `out=`
   (or after the first call, which allocates a cached buffer) they record into
   a CUDA graph — verified by capturing both, changing the state, and replaying.

4. **Thresholds reproduce the old task exactly**: success is position < 0.02 m,
   orientation error `1 - dot^2` < 0.04, and object 6-D speed < 0.1 — the last
   so a cube tumbling *through* the goal pose does not count. Failure is the
   object below z = 0.0, the floor. That is deliberately **not** `fallen_z`
   (0.08): `fallen_z` is the cost's drop *penalty*, set just under the palm,
   and ending the episode there would stop runs the planner could still save.

5. **Non-finite states fail and never succeed — a deliberate departure.** The
   old `has_failed` tested `xpos[2] < 0.0`, which is simply *false* for a NaN,
   so a blown-up simulation was reported as still running. Now any NaN or inf in
   positions or velocities is a failure. And because success only reads the
   object, a NaN in a *hand* joint used to leave a world both "succeeded" and
   "failed"; success now also requires a finite state, so the two are mutually
   exclusive. Verified on every finite state that both checks match the old
   logic exactly; the non-finite rows are the only differences.

6. **Failure reads `qpos`, not the body's `xpos`.** For a free-joint body they
   are the same point, and `qpos` is the post-step state where `xpos` is left
   over from the step's forward pass.

7. **`getInitialState()` stays on the interface** though the plan's list does
   not include it: every driver needs it, and the control belongs in the initial
   condition because a position-actuated hand left at `ctrl = 0` collapses
   before the planner's first command lands.

8. **`TaskBase.py` holds one class plus `TaskRole`.** The plan says "two class"
   but describes only `TaskBase`; nothing is common to every task and not
   already per-task, so a config would have been empty.

### Initial state and goals (added 2026-09-23; goal API revised twice the same day)

**`setSimToInitialState(sim)`** is written once, in `TaskBase`, on top of
`getInitialState` — a subclass says where it starts, not how a simulator is put
there. A single simulator gets `SetState` + `SetControl`. A vectorized one gets
`BroadcastState` from device copies of `(q0, v0, u0)` uploaded on the first
call, so every later call is pure kernel launches and records into a CUDA graph.
`BroadcastState` gained an optional `u` for this: on a position-actuated hand
the initial *control* is part of the initial state.

**Goals** are three functions. Choosing and applying are separate, so one
sampled goal can be handed to both the planner's rollout task and the
evaluator's eval task:

    g = eval_task.sampleNewGoal()
    eval_task.setGoal(g);  rollout_task.setGoal(g)
    eval_task.setRendererToGoal(renderer)

  * `sampleNewGoal(current_goal=None)` returns `current_goal * r` for a rotation
    `r` in the object's frame, chosen at `goal_difficulty`. With no argument it
    rotates from the task's reference — the last goal adopted, or the starting
    goal right after `setSimToInitialState`. It is pure apart from consuming
    randomness: the task's goal, device buffers and reference are untouched.
  * `setGoal(goal)` adopts it: normalizes, rebuilds the goal vector, writes it
    into the existing device buffer **in place** (so a planner's captured graph
    sees it on the next replay), and makes it the next sampling reference.
  * `setRendererToGoal(renderer)` turns the renderer's `goal` mocap marker. The
    renderer owns its own `MjData` and `RenderState` rewrites only positions,
    so the orientation set here persists across frames.

**`setSimToGoal` was removed**, per the plan. It turned the simulator's copy of
the marker, which has no effect: in all 32 leap scenes the marker is a mocap body
that collides with nothing and has zero mass, 200 steps with it at "R" and at "O"
ended in bit-identical states, and nothing in the package read it back.

#### Every goal is a rotation of the current goal

The difficulty levels keep their old numbers, but all ten now rotate *from the
current goal*. Levels 0, 1, 2 and 5 already did, and are unchanged — still
identical to the old sampler. Levels 3, 4, 6, 7, 8 and 9 used to build their goal
from the starting orientation (`canonical * face_rotation`) and now apply a
rotation to the current goal instead:

| Level | Old (from the start) | Now (from the current goal) |
| --- | --- | --- |
| 3 | one of 4 faces adjacent to the start's face, + twist | tip onto an adjacent face, + twist |
| 4 | any of 5 faces other than the start's, + twist | turn to any of the 5 other faces, + twist |
| 6 | adjacent to the start's face, no twist | tip onto an adjacent face, no twist |
| 7 | always show "O" | roll -90 degrees about object X |
| 8 | show "O" or "B" | roll +/-90 degrees about object X |
| 9 | always show "B" | roll +90 degrees about object X |

From the start, levels 7/8/9 still show "O"/either/"B" as their first goal;
after that they keep rolling — level 7 cycles R -> O -> S -> B -> R.

"Tip onto face `v`" is the object-frame rotation with `r(v) = up`: 90 degrees
about `v x up` for an adjacent face, 180 degrees about an in-face axis for the
opposite one, then an optional random 90-degree twist about `v`. The shown face
is read off the quaternion (`_upAxis`), so the face tables, the face index and
the face-from-quaternion bookkeeping of the previous version are all gone; the
only sampler state left is the reference orientation.

Consequences:

1. **No level can re-issue its starting goal**, because every `r` is a nonzero
   rotation. That resolves the open question about multi-goal mode counting
   re-issued goals: level 8 multi-goal now runs B -> S -> O with successes at
   steps 72 and 239 — two real reorientations — where before it produced
   B -> B -> B with successes a few steps apart.
2. **Levels 0 and 1 spin about the shown face's *signed* normal.** The old code
   used the unsigned axis, which reverses "clockwise" on a face whose normal is
   negative. The two agree whenever the shown face's normal is positive, which
   for levels 0/1 is always (they never change face), so their sequences still
   match the old sampler.
3. **An episode's first goal is rotated from the start.** `setSimToInitialState`
   moves the sampling reference back to the starting goal through a host-only
   `_onInitialState` hook. It deliberately does not touch the *adopted* goal —
   rewriting the device goal there would break capture — so the cost keeps the
   previous goal until the driver's next `setGoal`, which it issues immediately.

Verified:

  * for each level, 300 consecutive goals where every step's rotation
    `r = g^-1 * new` is exactly the level's (level 0: -90 degrees about the
    signed up axis; 5: 180 degrees onto the opposite face; 6: exactly 90 degrees
    onto an adjacent face; 7/8/9: +/-90 about object X; and so on), with **zero
    repeats** at every level; levels 3, 4 and 6 reach every face by chaining;
  * levels 0, 1, 2 and 5 still match the old `sample_new_goal` exactly;
  * `sampleNewGoal(current_goal=B)` rotates from B, and six samples leave the
    task's goal, reference and device vector untouched;
  * after a reset, episode 2's first goal is rotated from the start (R -> O at
    level 7) while the adopted goal is unchanged; `setSimToInitialState` still
    replays correctly from a captured graph;
  * a captured cost graph matches eager evaluation across four relative goals;
    the renderer's marker follows the goal;
  * the cost port is still bit-exact against the old kernel, and the planner
    keeps 11.8 ms/plan.

### Quirks in the original cost, preserved deliberately

1. **`w_velo` scales position error, not velocity.** The old kernel computes an
   object-velocity term and never adds it, while `weights[4]` — the slot named
   `w_velo` — multiplies `c_pos`, the L2 position error. (The line carries a
   `#<- BIG CHANGE!!!!!!` comment.) Renaming the key would silently change what
   every tuned weight set in the study means. It is 0.0 for the cube.
2. **Controls do not enter the cost** — `grasp_reorient_cost_wp` takes `ctrl`
   and never reads it.
3. The joint-velocity term indexes `qvel` with `robot_qpos_adr`. Identical here
   (both blocks start at 0) but conflates a qpos and a qvel address.

### Knock-on changes

  * **Planner** — `_rolloutBody` now calls `task.calcCosts(self.sim, terminal=...,
    out=...)`; it no longer fetches device state itself.
  * **Driver** — episodes are judged by `isSuccess`/`isFailure` on the eval
    simulator, not by a cost (there is no host cost to call). A failure always
    ends the episode; a success ends it unless `--no-stop-on-success`, matching
    the old driver's `fin_ep_on_success=True` default. It reports `end_reason`,
    `steps_to_success`, start/end `goalErrors`, and the planner's
    `last_min_cost` trace as a progress signal — explicitly the rollout model's
    cost, not an eval score.

### Verified

  * `calcCosts(sim)` **still matches `grasp_reorient_cost_wp` bit-for-bit** —
    max |Δ| = 0.000e+00 over 64 randomized states, running and terminal;
  * `calcCosts` on a CPU `Mujoco` raises; `calcCosts_GPU` no longer exists;
    `TaskBase`'s abstract set is exactly `calcCosts`, `getInitialState`,
    `getModelPath`, `isFailure`, `isSuccess`;
  * on a 32-world batch spanning every outcome (at goal; just inside and just
    outside each of the three thresholds; below `fallen_z`; below the floor;
    NaN and inf states): the vectorized and single-simulator paths agree on
    every world, both match the old `is_success`/`has_failed` on every finite
    state, and success and failure are never both true;
  * both outcome checks record into a CUDA graph and replay correctly against a
    changed state;
  * the planner keeps its 11.8 ms/plan with the cost inside the captured graph,
    and still closes the loop (object 0.0189 -> 0.0053 m from target).

### Changed from the first version

The host `calcCosts(q, q_dot, u, terminal, site_xpos)` and `calcCosts_GPU(...)`
merged into the device-only `calcCosts(sim, terminal, out)`. `isSuccess` and
`isFailure` are new. The earlier "host cost matches the old kernel to 1.4e-07"
verification no longer applies, since there is no host cost. The unused `_mjd`
went with `_forward_tips`.

### Deferred: Tasks/XML_Files

The plan puts the scene XML under `Tasks/XML_Files`. Not done: the leap scenes
reference meshes and `<include>` files by relative path, so moving the XML
without its asset tree stops it compiling. `LeapReorient.SCENES_DIR` is a single
constant, so the move is one line plus copying the assets.

## What was done — SamplingBasedPlanners/

`SamplingBasedPlannerBase.py` and `MPPI.py`, ported from
`contact_study/planners/base.py` and `mppi.py`.

**The structural change: the planner no longer owns the physics.** The old
`SamplingPlanner` called `api.put_model` / `api.make_data` itself and held
`self.m` / `self.d`. The new base is handed a `VectorizedSimulator` and a task
and drives both through their public interfaces, so swapping the contact model
under a planner is a matter of constructing a different simulator.

**`SamplingBasedPlannerConfig`** — `noise_sigma`, `n_iterations`, `warm_start`,
`control_mode`, `delta_range`, `seed`, `debug`. Notably *absent*: `N`, `H`,
`nu`, `substeps` and the device, which are read off the simulator. Duplicating
them invites a config and a simulator that disagree about the shape of the same
buffer.

**`SamplingBasedPlannerBase`** — owns the loop, the noise, the rollout, cost
accumulation, the control parameterization and the warm start. Subclasses
implement two hooks: `_buildSamples` (draw `V` from `theta`) and
`_updateParams` (fold costs back into `theta`).

**`MPPI_Config` / `MPPI`** — `temperature`, `adaptive_temp`, `adp_temp_params`,
`mean_cost_over_horizon`, `normalize_cost_by_samples`, plus the softmax weight
kernels. Same NaN handling as the original: a blown-up rollout gets exactly zero
weight so the valid samples still drive the update, and an all-NaN block leaves
the min-cost sentinel untouched, which is the degeneracy test.

### Three more simulator gaps this exposed

Same pattern as before — the concrete consumer reveals what the layer below
needs:

1. **The sequence cursor had to follow control steps, not physics steps.**
   `_advance_sequence` indexed `U` once per physics step, so with `substeps=4` a
   horizon-8 sequence was consumed in 2 control steps. It now advances every
   `substeps` physics steps, which is what makes a control sequence a
   zero-order hold at the control rate. Identical behaviour at `substeps=1`.

2. **`SetControlSequence` now accepts a warp array.** A planner's samples are
   already on the device; round-tripping `(N, H, nu)` through the host every
   plan would cost more than the rollout. A host array still takes the
   validated path.

3. **`DeviceState()` was added** — a small dataclass of the backend's live
   device arrays (`qpos`, `qvel`, `ctrl`, `site_xpos`). The task's GPU cost has
   to read the simulator's state without a transfer, and `GetState()` crosses
   PCIe. Concrete on `VectorizedMujoco`, raising with a clear message on the
   base, so the planner stays backend-agnostic instead of reaching into
   `sim.d`.

### Decisions worth knowing about

1. **`Plan` returns a ready-to-apply command.** Whatever the control mode, the
   driver gets the actuator command itself — `U_0`, `u + U_0` or
   `q_robot + U_0` — so it never has to know which parameterization the planner
   used; it just calls `SetControl`.

2. **Three control modes** *(revised 2026-09-23; this entry used to describe a
   single "relative" mode that offset by the plan-time pose — the one
   deliberate difference from the old planner, now gone)*. At rollout control
   step `t`:

   | Mode | Command | How it reaches the simulator |
   | --- | --- | --- |
   | `absolute` | `ctrl_t = U_t` | `SetControlSequence(V)` |
   | `ctrl_relative` | `ctrl_t = ctrl_{t-1} + U_t`, from the control currently applied | precomputed on the device (`_cumsum_commands_kernel`), then `SetControlSequence(A)` |
   | `pos_relative` *(default)* | `ctrl_t = q_t + U_t`, `q_t` each world's joints *at that step* | formed inside the rollout each control step (`_pos_relative_kernel`) and applied with `SetControl` |

   `pos_relative` is the old planner's default (`_assign_ctrl_relative_kernel`)
   and `ctrl_relative` its legacy `ctrl += delta` (`_assign_ctrl_kernel`); both
   are **bit-identical** to those kernels. `pos_relative` cannot be precomputed
   — its command depends on where each world has moved to — so for it the
   planner sets the control every step instead of handing over a sequence. That
   needed `VectorizedMujoco.SetControl` to accept a device array (a
   device-to-device copy, capturable), mirroring `SetControlSequence`.

   `ctrl_relative` needs the control currently applied, so `Plan` gained an
   optional `u` (not in the plan's signature). Without it, the planner uses the
   action it returned last; on the very first plan there is no such action, so
   it raises rather than guess. The driver passes `u=sim.GetControl()` every
   step, so an episode reset is never mistaken for a continuation.

   The returned tape (`last_action_seq`) is exact for `absolute` and
   `ctrl_relative`. For `pos_relative`, row 0 is exact and later rows assume the
   robot holds its measured pose — the rollouts themselves re-read it.

3. **The terminal cost replaces the last step's running cost**, rather than
   being added to it — matching the old `_launch_accumulate_costs(terminal=...)`,
   which routes that step's cost to the terminal buffer instead of the running
   one.

4. **Dropped from the old planner**, all documented rather than silently
   missing: time-constrained rollouts, convergence-terminated iteration
   (`convergence_tol` / `max_iterations`), spline-smoothed noise,
   `resample_interval` / `resample_per_iteration` (noise is redrawn once per
   plan), the async-driver tape machinery beyond `last_action_seq`, and
   time-based schedule resolution (`time_horizon` / `step_time` -> steps).

5. **No CUDA-graph capture** — the known performance gap. See below.

### Verified

On an RTX 4090, cube task, `N=256`, `H=8`, `substeps=4`:

  * **closed loop works**: 40 plans driving the CPU `Mujoco` eval simulator took
    the eval cost from 20.857 to 1.521, moved the cube from 0.0189 m to
    0.0059 m from its target, and kept it held (z=0.0908 above the 0.08 drop
    threshold);
  * `Plan` returns `(nu,)` float32; `last_action_seq` is `(H, nu)`;
    `last_plan_ok` tracks a real update;
  * (then-)relative mode puts the action within 0.0065 rad of the measured joint
    positions; absolute mode adds no offset; `delta_range=(-0.02, 0.02)` holds
    every returned delta inside the clip (max 0.00325) even at
    `noise_sigma=0.5`;
  * warm start zeroes the tail row of the mean;
  * `N`, `H`, `nu` and `substeps` are all read off the simulator, and
    `robot_qpos_adr` off the task's index vector;
  * bad `q`/`q_dot` shapes and every invalid config value raise with a message
    naming the problem.

### Verified — control modes (2026-09-23)

  * Driving the planner's **real** rollout body with fixed samples and
    recording, in every world at every control step, the control actually held:
    `absolute` matches `U_t` exactly, `ctrl_relative` matches
    `u + sum_{k<=t} U_k` to 1.2e-07 (float32 vs float64), `pos_relative` matches
    `q_t + U_t` exactly — with `q_t` re-read per step: the joints moved 0.089
    rad over the horizon, which a plan-time offset would have been off by;
  * `pos_relative` is bit-identical to the old `_assign_ctrl_relative_kernel`,
    and `ctrl_relative` to the old `_assign_ctrl_kernel` applied step by step;
  * returned action and tape follow each mode's formula;
  * all three capture into a CUDA graph, 15-17x faster than eager, with graph
    and eager rollout costs agreeing to ~2e-6 relative; the per-step
    `SetControl` in `pos_relative` costs nothing measurable (11.7 ms/plan);
  * `ctrl_relative` raises on a first plan with no `u`, and otherwise starts
    from the action it returned last.

Closed loop, 5 episodes each at `--substeps 32 --horizon 5 --temperature 5`:
`pos_relative` 2/5, `ctrl_relative` 0/5, `absolute` 0/5. Those settings were
found for position-relative control, so this does not rank the modes: in
particular `ctrl_relative` sums its deltas over the horizon, so the same
per-step noise spreads far wider, and would need its own `--noise-sigma`.

### MJWarp is not reproducible, and it is not the wrapper

Testing "same seed -> same action" failed, so this was chased down:

  * the **noise block is exactly reproducible** for a fixed seed (bitwise
    identical across planner instances), so the planner's own sampling is
    deterministic;
  * the **simulator is not**. Stepping raw MJWarp — `mjw.step` directly, no
    class of ours involved — from bit-identical state and controls twice gives
    `max|Δqpos| = 7.7e-04` after 32 steps;
  * worlds inside a *single* batch, given identical states and controls, also
    diverge by the same amount, and the magnitude varies non-monotonically with
    the world count (3.8e-04 at N=64, 3.7e-09 at N=128, 7.7e-04 at N=256 and
    512, each in a fresh process). It is unaffected by `nconmax`/`njmax`, so it
    is not contact-buffer saturation;
  * the practical consequence: a fixed seed reproduces the noise but not the
    plan. Two identically-seeded planners agree on the action to ~1.5e-04
    absolute, about 6e-05 relative.

This is a property of the study's existing GPU stack, not something the refactor
introduced — the old planner has it too. It matters because rollout costs carry
physics noise of that order, and because any test that assumes bitwise
reproducibility of a GPU rollout will be flaky.

### Performance: CUDA-graph capture (done)

Originally left as the known gap; now implemented. A plan at `N=256`, `H=8`,
`substeps=4` went from **194 ms to 11.8 ms — a 16.4x speedup**, turning a 5 Hz
planner into a 50+ Hz one.

Three pieces made it possible:

1. **`VectorizedSimulator.BroadcastState(q, q_dot)`** — the capturable twin of
   `SetState`. `SetState` takes host arrays, so it necessarily contains a
   host-to-device copy and a graph cannot record one; that was what blocked
   capture. `BroadcastState` reads a start state the caller has already staged
   on the device and does the job in two kernel launches. It deliberately skips
   `forward`, since the first `Step_GPU` recomputes everything derived anyway.

2. **`_rolloutBody`** — the rollout split out as a pure-device routine: reset,
   unroll, accumulate, with nothing touching the host. That is the body the
   graph records. `Plan` stages the start state into `_q0_wp`/`_v0_wp` outside
   the graph, and the graph broadcasts from them, so replays pick up each new
   plan's state and samples.

3. **Fewer launches per step** — `Step_GPU` now re-assigns `d.ctrl` only when
   the control index actually moves on, so a control step with `substeps=4`
   costs one launch instead of four.

Capture happens once, lazily, after a warm-up pass — the first call compiles
kernels and lets MJWarp make its internal allocations, neither of which belongs
in a recording. A capture failure is not fatal: it sets `_graph_failed`, warns,
and falls back to eager launches, so `--no-graph` and a capture-hostile backend
behave the same way.

Verified: graphed and eager rollouts agree to 6.6e-04 (the MJWarp
nondeterminism floor measured above, not a capture bug); three successive graph
replays produce three different cost vectors, confirming replays read the new
samples rather than frozen ones; and the closed loop still converges with
graphs on (cost 20.857 -> 1.213).

## What was done — Drivers/run_episodes.py

The first driver: closed-loop MPC episodes, wiring the whole package together.
A task supplies the scene, the initial state and the cost; `VectorizedMujoco`
rolls samples out on the GPU; `MPPI` plans; a CPU `Mujoco` stands in for
reality; `MujocoVideoRenderer` records it.

    python -m ContactModelStudy.Drivers.run_episodes
    python -m ContactModelStudy.Drivers.run_episodes --n-episodes 5 --steps 200
    python -m ContactModelStudy.Drivers.run_episodes --n-samples 512 --temperature 5 \
        --results results/run.json

### Decisions worth knowing about

1. **The planner and the evaluator load different scenes from the same task.**
   The planner gets `TaskRole.ROLLOUT` (with `--hand-acc` / `--obj-acc`
   fidelity), the evaluator gets `TaskRole.EVAL`. That gap is what the study
   measures, so it is the default rather than something to switch on, and the
   episode outcome is judged by the *eval* task — performance on the accurate
   scene, not on the planner's own model of it.

   *Revised 2026-09-23:* the outcome is now `isSuccess`/`isFailure` rather than
   an eval cost, since the task no longer has a host-side cost. See the Tasks
   section. The verification numbers below that quote an "eval cost" predate
   that change; they were real at the time, but the driver no longer prints
   them.

2. **The first plan is reported separately from the rest.** It pays kernel
   compilation and graph capture — ~1.3 s against a ~12 ms steady state — so
   averaging it in would produce a figure describing neither. The summary
   carries `plan_ms_first`, `plan_ms_mean` and `plan_ms_max`, the last two over
   the remaining steps.

3. **`--delta` defaults to unclipped**, matching every old driver: under
   relative control the delta *is* the position-servo error, so clipping it
   caps how hard the hand can grip.

4. **Video is per-episode**, with the index appended when running more than
   one, captured on the sim clock via `steps_per_frame` so playback is real
   time whatever the control rate.

### A defect this turned up: absolute control dropped the cube

Running `--control-mode absolute` sent the eval cost to 247 and dropped the
object. The cause was in the planner, not the driver: `Reset()` zeroed the mean,
and under absolute control the mean *is* the command — so the first plan
commanded every joint to zero, opening the hand. Under relative control zero is
correct (the sequence is a delta, so a zero mean holds the measured pose), which
is why the default path never showed it.

Fixed by giving `Reset` an optional `mean`, which the driver seeds with the
task's initial control in absolute mode. Absolute control now works and in fact
scores slightly better than relative on this task (final cost 1.093 vs 1.505
over 60 steps), which is worth knowing given that relative is the study's
default.

### Verified

  * single episode, 120 control steps at 125 Hz: eval cost 20.857 -> 1.357,
    object 0.0189 -> 0.0062 m from target, held;
  * `--n-episodes 2` writes `multi_ep0.mp4` / `multi_ep1.mp4`, both valid
    (21 frames, 640x480, 30 fps on readback), and prints an aggregate line;
  * `--results` writes a JSON summary with the full config and per-episode
    cost traces;
  * `--no-graph` falls back to eager and shows the expected slowdown
    (177 ms/plan vs 14.7 ms);
  * `--hand-acc low` loads `env_leap_rollout_cube_low_high.xml` and still
    converges;
  * `--no-video`, `--delta`, `--warm-start`, `--control-mode`, `--n-samples`,
    `--horizon`, `--temperature` and `--help` all behave.

## End-to-end check — test_scripts/render_finger_curl.py

A script that exercises the ported pieces together on the leap cube scene:
`Mujoco` steps the physics, `MujocoVideoRenderer` draws it, and the hand starts
where the study's own task starts it.

    python test_scripts/render_finger_curl.py
    python test_scripts/render_finger_curl.py --curl 0.8 --seconds 1.5 --out /tmp/curl.mp4

What it does: loads `scenes/leap/env_leap_eval_cube.xml`, sets the initial state
and control from the old task, curls every finger linearly inward over
`--seconds`, uncurls linearly back, and writes `videos/leap_cube_finger_curl.mp4`.

### Decisions worth knowing about

1. **The initial state is read live from the old task**, via
   `contact_study.tasks.grasp_reorient._OBJ_PARAMS["cube"]` — the same table
   `GraspReorientTask.get_inital_state` reads — rather than copied into the
   script, so it follows the task if those numbers are retuned. The task class
   itself is not instantiated: `initialize_task` builds Warp arrays on a GPU,
   which this test has no use for. Reaching into a private name is the tradeoff
   for not duplicating the numbers; it should become
   `LeapReorient.get_inital_state()` once `Tasks/` is ported.

2. **Curled joints are found by name, not hardcoded index.** The leap hand names
   every actuated joint `<digit>_<suffix>`; the script curls the flexion
   suffixes (`mcp`, `pip`, `dip`, `ipl`) and holds the rest — `rot` spreads the
   fingers sideways, and the thumb's `cmc`/`axl` position the whole thumb rather
   than bending it. 11 of 16 actuators curl. A scene that renames or reorders
   actuators cannot silently curl the wrong ones.

3. **Curl is a fraction of each joint's remaining range**, not a fixed angle, so
   every command stays inside its `ctrlrange` by construction whatever pose the
   task starts from.

4. **It is not named `test_*.py`.** Pytest would collect it, and it needs a GL
   context and writes a video — not something a unit-test run should do. It sits
   alongside the other script-style entries in `test_scripts/` (`compare_eval_sims.py`,
   `view_model_drake.py`).

5. **`MUJOCO_GL` defaults to `egl` when there is no DISPLAY**, set before mujoco
   is imported since that variable is read once. It only fills in a default and
   never overrides an explicit setting.

### Note on the task's initial data

`init_qpos` and `init_ctrl` disagree: `init_qpos[13]` (`th_axl`) is written as
`1.52604395e+00 + 1.0` in the task's table, so the thumb starts 0.9939 rad away
from what `init_ctrl` commands and the position servo pulls it there in the
first few steps. That is the task's data, not a porting artifact, but it means a
"did the hand return to its start pose?" check has to measure against the
commanded pose — which the script does, reporting 0.0458 rad at the end. Worth
deciding whether that `+ 1.0` is deliberate before `LeapReorient.py` is written.

### Verified

Ran under MuJoCo 3.6.0 (`contact_modeling` env, `MUJOCO_GL=egl`): 2000 steps at
dt=0.002 (4.0 s of sim), 119 frames at 30 fps. Frame-difference over the clip
rises monotonically to a peak at frame 63 of 118 and falls monotonically after —
the triangular curl profile, with the few-frame lag past the midpoint being the
position servos tracking the command. Inspecting the frames confirms the
fingertips close on the cube and open back out.

## Open questions

- **Where do the backend-specific parameters live?** Now a concrete gap:
  `Mujoco` ignores integrator/solver/cone and takes whatever the XML declares,
  and the M1 hard-contact preset that the old eval sim applied by default is
  simply gone. Nothing in the new plan's file list holds the parameters in
  `contact_study/contact_models/config.py` (hard-contact preset, ComFree
  stiffness, XPBD iterations). Suggestion: each backend module owns its own
  config dataclass extending `SimulatorConfig` — `Mujoco.py` holds
  `MujocoConfig`, `XPBD.py` holds `XPBDConfig` — rather than one union config
  that every backend ignores most of. Needed before `Mujoco` can serve as the
  M1 ground-truth eval sim the old code used it for.
- **The driver's default planning horizon is too short to reorient the cube.**
  `--substeps 4 --horizon 8` plans 64 ms ahead; the old driver planned 352 ms
  (`time_horizon`) with 64 ms control steps. With real goals, 0 of 3 episodes
  succeeded at the defaults, and 0 of 3 with the old temperature/noise but the
  short horizon; with `--substeps 32 --horizon 5` (320 ms) and temperature 5,
  **2 of 5 succeeded** within 5.1 s (orientation error 0.5 -> 0.0087). The
  defaults were not changed, because you have since edited the driver's other
  defaults by hand (`--steps 4000`, `--noise-sigma 0.1`, `--temperature 10`)
  and the schedule is yours to set alongside them.
- **Package installation.** `pyproject.toml` has
  `include = ["contact_study*"]`, so `ContactModelStudy` is not installed by
  `pip install -e .`. It imports fine from the repository root. Add it to the
  include list once enough of the port is usable.
