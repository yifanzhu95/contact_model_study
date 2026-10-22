# LEAP reorientation: PPO training and shared controller evaluation

I added a PPO workflow for the existing LEAP cube, duck and ball tasks, using the
same M1–M5 contact-model implementations as MPC. I use skrl for PPO and keep
training separate from controller evaluation. RL and MPPI share one evaluation
loop for case reset, command execution, stopping rules, metrics and optional video.

I have checked the interfaces and bounded execution paths. I have not established
a reliable learning baseline, sustained training throughput or hardware transfer.
The implementation supports a framework review before those experiments.

## What I implemented

| Area | Available behavior |
|---|---|
| Tasks and scenes | Cube, duck and ball; all 24 existing evaluation/rollout scene choices, including duck FOAM variants |
| Contact models | Native M1–M5 backends; CPU MuJoCo diagnostics; adapters for the original CPU evaluators |
| Environment | Independent indexed resets, owned state readout, per-world goals and command history, finite task clocks |
| PPO | skrl 2.1.0, all 16 actuators, multiple environments on one device, actor export and checkpoint restart |
| Execution | Explicit update/sample budgets, cooperative time/signal stops, periodic checkpoints and runtime preflight |
| Evaluation | Fixed actors or the existing MPPI, shared cases and scoring, cross-model matrices and explicit fault accounting |
| Video | Optional streaming MP4 per case, actual simulation timing, goal metadata and actual M5 rigid-body poses |
| Continuous goals | Optional evaluation sequences without resetting physics; one total deadline and per-goal results |
| Independent jobs | Select and plan or execute one isolated training job; the default only displays the plan |

The native simulators, contact presets and MPPI algorithm are unchanged. The
legacy MPC entry remains available with its original execution protocol.

## What I have not implemented or validated

- A validated PPO learning baseline or a fair learning-performance ranking of M1–M5.
- Actual domain randomization. The configuration and lifecycle are reserved;
  enabled or nonempty randomization requests fail before physics allocation.
- Joint multi-GPU PPO, a parallel pool for the shared evaluator, or qualified scaling.
- Automatic controller switching within an episode. A run selects RL or MPPI at its start.
- Site-specific HPC execution qualification, real sensor observations or hardware integration.
- Arbitrary robot/object import or demonstrated cross-object policy generalization.

Single-GPU batching is implemented. Selecting one independent job is not concurrent
GPU scheduling, and the original MPC pool does not provide multi-GPU PPO training.

## Runtime and installation

Use a Git checkout containing `.git` and `scenes/`, and install in editable mode.
I checked Python 3.12. GPU backends require a compatible NVIDIA driver and CUDA
PyTorch; the current adapters use one visible GPU as `cuda:0`.

| Extra | Purpose |
|---|---|
| `rl-inference` | Fixed-actor inference with Torch and Gymnasium |
| `rl` | Inference dependencies plus skrl PPO |
| `kamino` | Newton/Kamino dependencies for M5 |
| `rl-video` | Optional mediapy video writer; ffmpeg must also be on PATH |
| `dev` | Test and development tools |

```bash
python -m pip install -e '.[rl,dev]'
# Include M5 when needed:
python -m pip install -e '.[rl,kamino,dev]'
# Add video separately when needed:
python -m pip install -e '.[rl-video]'
```

The runtime checks enforce Torch 2.14.1, Gymnasium 1.4.0, skrl 2.1.0 and MuJoCo
3.6.0. GPU paths use Warp 1.17.0 and the project-pinned ComFree dependency; M5
also uses Newton 1.6.0. Native Pinocchio/Drake evaluation needs its optional engine
installation. There is no fallback to another model or device.

## Task and model selection

The `environment` object in each configuration contains these choices:

| Field | Meaning |
|---|---|
| `task_name` | `cube_reorient`, `duck_reorient` or `ball_reorient` |
| `scene_role` | `eval` or `rollout` |
| `hand_acc`, `obj_acc` | Existing rollout geometry selections |
| `physics_backend`, `contact_preset`, `device` | A compatible backend/model/device combination from the table below |
| `num_envs` | Parallel environments for single-device training; case evaluation resolves this to one |
| `physics_dt`, `control_dt` | Physical and control timesteps; control time must be an integer multiple of physical time |
| `episode_steps` | Finite task duration in control steps |
| `target_rate` | Maximum actuator-target change per second |
| `goal_difficulty`, `seed` | Original task goal rule and training/reset randomness |
| `cost_weights` | Optional finite, nonnegative overrides of the original named task costs |

| Model | Backend | Device |
|---|---|---|
| M1 / M2 | `mjwarp` | `cuda:0` |
| M3 | `comfree` | `cuda:0` |
| M4 | `xpbd` | `cuda:0` |
| M5 | `kamino` | `cuda:0` |
| M1 / M2 diagnostics | `mujoco_cpu_diagnostic` | `cpu` |
| Original CPU engines | `mujoco_eval`, `pinocchio_eval`, `drake_eval`, with preset `native` | `cpu` |

The 24 scenes are three evaluation scenes, three cube rollout scenes, three ball
rollout scenes and fifteen duck rollout scenes. Duck object fidelity includes
`high`, `foam4`, `foam16a`, `foam16b` and `foam64`, each with three hand fidelities.
Initial states, default costs and goal rules come from the original task classes.

```bash
python -m ContactModelStudy.Drivers.make_rl_cases --list-scenes
python -m ContactModelStudy.Drivers.make_rl_cases \
  --task cube_reorient --count 32 --seed 0 --goal-difficulty 8 \
  --output results/development_cases/cases.json
```

A CaseBank stores explicit initial states and targets, plus a content hash. Use
separate banks for training, development and final evaluation. The example above
creates development cases; it does not establish an independent test split.
Task-bound v2 banks reject a different task. Historical v1 banks remain cube-only.

Example configurations are under `configs/rl/`: `cube_mjwarp_smoke.json`,
`cube_kamino_smoke.json`, `duck_mjwarp_smoke.json`, `ball_mjwarp_smoke.json` and
`duck_foam4_mjwarp_smoke.json`. Their short durations check interfaces, not learning.
For example, eight control steps at 16 ms give only 0.128 seconds per episode.
I select a separate task duration, rollout batch and sample budget for learning;
increasing `--updates` alone does not turn a smoke configuration into a learning task.

## Observation, actions and task ending

I use the same 73-value observation encoding for training and inference: hand
position/velocity, object position/orientation, goal error, object twist, previous
actuator command and remaining time. Quaternions use wxyz order and observations
use world-frame conventions. These are simulated states, not sensor estimates.

PPO controls all 16 actuators. A latent Gaussian action is decoded as:

```text
new_target = clamp(previous_target + target_rate * control_dt * tanh(action), lower, upper)
```

This limits command changes, not physical joint velocity. PPO stores the latent
action; exported inference uses the deterministic actor mean and the same decoder.

The single-goal task ends on success, physical drop or its finite deadline. It
never resets automatically inside `step`. Selected-world reset preserves the
other worlds, and state/transition tensors are owned copies. M5 reads actual
integrated body state instead of reconstructing it from stale joint coordinates.

Success retains the original thresholds: position error below 2 cm,
`1 - dot(q, goal)^2 < 0.04` (about 23.07 degrees), and the original object-velocity
norm below 0.1. A physical drop takes priority. Nonfinite states or backend faults
stop execution and lock the environment; they are not ordinary training failures.

The finite reward uses the original running costs, normalized/clamped by
`cost_reference`, with gamma=1. Drop absorbs the remaining failure cost. The
original `w_velo` term weights position error; I retain that behavior for compatibility.
New M1/M2 runs use `native_mjwarp_v1` without extra contact ordering. Normal GPU
nondeterminism is accepted. Restart preserves an explicitly recorded historical
numerics profile rather than silently substituting the new default.

## Preflight, training and restart

These commands are usage examples, not an automatically executed training plan.
Select a meaningful task duration, independent evaluation cases and a training
budget before interpreting learning results.

```bash
# Check dependencies, scenes, device and budget compatibility; no physics steps.
python -m ContactModelStudy.Drivers.check_rl_runtime \
  --config configs/rl/cube_mjwarp_smoke.json
# Also reset/read at most two environments; still no physics steps or PPO updates.
python -m ContactModelStudy.Drivers.check_rl_runtime \
  --config configs/rl/cube_mjwarp_smoke.json --check-backend

# A bounded interface-training example; replace these settings for a learning study.
python -m ContactModelStudy.Drivers.train_rl \
  --config configs/rl/cube_mjwarp_smoke.json \
  --updates 1 --max-transitions 16 --max-seconds 300 --stop-margin-seconds 30 \
  --checkpoint-interval 1 --output results/interface_run

# Restore and export without sampling or updating PPO.
python -m ContactModelStudy.Drivers.train_rl \
  --resume results/interface_run/training.pt --updates 0 --output results/restore_only
```

`--updates` means additional updates for this invocation. The sampled batch is
`num_envs × steps_per_update`; the transition cap must fit complete PPO updates.
The example uses two worlds and eight rollout steps. It is not a learning budget.
`--case-bank` supplies an explicit training bank; otherwise the new-run entry makes
its documented development bank. Resume can infer its config and saved case bank;
changed environment/PPO/bank contracts are rejected.

A complete checkpoint contains `training.pt`, `training.resume.pt`, `actor.pt`,
`cases.json` and a final `checkpoint.json` hash/completion marker. Periodic complete
bundles also appear under `checkpoints/`. Restart restores network, optimizer and
RNG state, then begins new physical episodes and discards a partial rollout. It
does not restore a partly completed physical episode or guarantee cross-hardware
CUDA RNG compatibility. Outputs must be new directories; existing results are retained.

New actors and checkpoints include a fingerprint of the compiled source scene,
covering included XML, meshes and textures as well as the top-level XML hash.
The fingerprint is computed before backend-specific contact settings, once at
environment construction. It clears the compiler asset cache to avoid stale
mesh reads and does not change live physics/solver arrays. Moving the checkout
preserves the identity; changing compiled scene content is rejected by restart
and strict policy evaluation. Explicit scene-transfer evaluation can bypass the
scene identity check while retaining action, observation and clock checks.
Legacy checkpoint restart warns when only the older top-level XML check is available;
the identity of its historical included assets cannot be reconstructed.

`transitions.jsonl` records accepted samples before optimizer execution, so an
optimizer fault retains the last sampled step in failure counts and diagnostics.
These sample counts are distinct from completed PPO updates and complete saved
checkpoints. `ppo_updates.jsonl` saves the library's update diagnostics; consumed
diagnostics are released from memory at each update instead of accumulating for
the lifetime of a run. Detailed transition logs still grow with the sample count.
Nonfinite update diagnostics stop collection even without a logging callback.
I record validated completed-update progress before the subsequent environment
reset/readout. If that operation fails, the failure summary retains completed
update counts separately from the last complete saved checkpoint. Only the
reported complete checkpoint is available for restart.

Time limits and supported stop signals are checked between complete PPO updates.
A cooperative stop saves the last completed update and stops before starting another.
It cannot interrupt an active GPU call or guarantee saving before a scheduler's
hard kill. Cleanup errors do not replace the primary fault, and partial checkpoint
writes are not treated as complete bundles.

For independent jobs, `run_rl_job` selects one job from a JSON plan:

```bash
python -m ContactModelStudy.Drivers.run_rl_job \
  --plan configs/rl/independent_jobs_example.json --job-index 0 \
  --output-root results/independent_jobs
```

Without `--execute`, this only displays the plan and does not initialize physics
or train. With `--execute`, it checks the runtime and invokes the same bounded
trainer for that one job. It does not allocate GPUs, submit scheduler jobs, run a
concurrent pool or retry faults automatically. Within an allocated HPC job,
`experiments/hpc/run_rl_job.sh` forwards the selected runtime, GPU visibility and
signals. Site-specific resources, accounts and deployment validation remain separate.

## Shared RL/MPPI evaluation

Evaluation fixes the policy; it never updates PPO. The driver resets each explicit
case, asks the selected controller for a command, applies it to the environment,
checks termination and saves the result. Cases run sequentially.

```bash
python -m ContactModelStudy.Drivers.run_controller_episodes \
  --controller rl --policy results/interface_run/actor.pt \
  --case-bank results/development_cases/cases.json --save-steps --output results/shared_rl

python -m ContactModelStudy.Drivers.run_controller_episodes \
  --controller mppi --planner-config configs/rl/mppi_interface_smoke.json \
  --config configs/rl/cube_mjwarp_smoke.json \
  --case-bank results/development_cases/cases.json --save-steps --output results/shared_mppi
```

`run_episodes --protocol finite_case_bank` routes to the same executor. Omitting
that protocol retains legacy MPC behavior, including its own settling, 64 ms
default control period and start-of-step checks. Do not combine legacy MPC scores
with finite-case scores as though they were the same task.

The adapter invokes the existing `MPPI.Plan` with current measured state and the
actually applied previous command. The shared protocol uses `ctrl_relative` and
the same command-rate/actuator bounds; projections are recorded. Internal MPPI
prediction saturation and terminal costs can still differ from the RL objective.
Sharing execution does not establish a fair or optimal performance comparison.
The supplied four-sample/two-step planner configuration only checks the interface.

## Single-policy and model-matrix evaluation

```bash
python -m ContactModelStudy.Drivers.evaluate_rl \
  --policy results/interface_run/actor.pt \
  --case-bank results/development_cases/cases.json \
  --contact-preset M5 --save-steps --output results/eval_M5

python -m ContactModelStudy.Drivers.evaluate_rl_matrix \
  --policy results/interface_run/actor.pt \
  --case-bank results/development_cases/cases.json --models M1 M2 M3 M4 M5 \
  --output results/model_matrix
```

A model matrix can repeat `--policy` for several fixed actors. CPU engines can be
selected using `--eval-sim mujoco` in single evaluation, or `mujoco`, `pinocchio`
and `drake` in matrix `--models`. Evaluation also accepts an independent `--config`.
Shortcut M1/M2 choices retain CPU for CPU-diagnostic policies; provide an explicit
GPU environment configuration to move such a policy to MJWarp.

Observation/action encoding, clocks, target rate, task duration and scene/cost
contracts are checked against the exported actor. An eight-step example cannot
be used to evaluate a 256-step actor. Intentional scene/object transfer requires
`--config` plus `--allow-scene-transfer`; observation, action and clock checks
remain enforced. Allowing transfer does not establish generalization.

## Optional video and continuous targets

All shared, single-policy and matrix evaluation entries accept `--video`,
`--fps`, `--width`, `--height` and `--camera`. Video is off by default. For headless
NVIDIA rendering, set `MUJOCO_GL=egl` before starting Python. MP4 dimensions must
be positive even integers, and video settings require the explicit `--video` flag.

```bash
MUJOCO_GL=egl python -m ContactModelStudy.Drivers.run_controller_episodes \
  --controller rl --policy results/interface_run/actor.pt \
  --case-bank results/development_cases/cases.json \
  --video --fps 30 --width 640 --height 480 --output results/recorded_rl
```

Video streams one clip per case, including initial/final states. It holds actual
control samples on the simulation clock without interpolating motion or adding
physics steps. A final frame can add less than two frame periods to encoded duration;
use logged simulation time for timing analysis. Rendering/encoding adds overhead
outside the synchronized controller timer. M5 displays actual hand/object body
poses on renderer scratch data; the simulator is not modified.

Add the following options to an evaluation command for continuous targets:

```text
--goals-per-episode 3 --goal-seed 0
```

Default `1` preserves single-goal evaluation. The first goal comes from the case;
subsequent orientations use the original task sampler relative to the previous
goal. The independent goal seed and complete sequence are saved before execution.
Identical task, cases, difficulty, count and goal seed give identical sequences
across controllers/models, independently of policy training seeds or success times.

After success with time left, only the target changes. Physical/solver state,
actuator commands, RNG and elapsed steps continue. MPPI refreshes its goal and
internal plan; the fixed actor receives the new goal observation. The whole
sequence shares the original deadline; remaining-time observations do not restart.
Each goal requires at least one control step before checking success.

All requested goals must be reached for sequence success. Reaching an intermediate
goal on the last step ends the sequence at its deadline, preserving the completed
count. A later drop ends the sequence as a drop even after earlier successes.
The protocol is `continuous_goal_sequence_v1`, distinct from the actor's unchanged
single-goal training contract. Final error refers to the last attempted goal;
the episode minimum spans goals. Do not pool single-goal and sequence success rates.
These options apply to evaluation only; they do not change the PPO training task.

## Outputs and failure interpretation

| Output | Content |
|---|---|
| `manifest.json` | Runtime/source identity, resolved environment, controller and evaluation protocol |
| `cases.json` | Exact allocated initial states and first goals |
| `summary.json` | Execution status, success/drop/deadline/fault/not-run counts and explicit denominators |
| `episodes.jsonl` | Per-case errors, return, steps, controller time and command projections |
| `steps.jsonl` | Optional post-step canonical state, command and costs with `--save-steps` |
| `goal_sequences.json`, `goals.jsonl` | Continuous-mode target table and timestamped activations/outcomes |
| `videos/` | Optional MP4, target history, per-frame sample/video clocks and goal indices |
| `comparison.csv` | Incrementally saved policy/model comparison |

Continuous-mode episode records include each goal's outcome and attempted,
completed and unattempted counts. Faults retain active sequence progress. Video
faults are infrastructure errors; incomplete clips retain `.partial.mp4` labels.
All allocated cases remain in the success denominator, including failed/not-run
cases. Means are labeled as completed cases only. A fault stops the matrix without
an automatic retry. CUDA controller timing waits for device completion before
and after action selection; it is synchronized wall time, not pure kernel time.

## Components and validation

| Component | Responsibility |
|---|---|
| `Contracts.py`, `CaseBank.py` | Configuration and owned state/case contracts |
| `LeapVectorEnv.py`, `Encoding.py` | Reset, task/observation/action/reward and guarded evaluation target advance |
| `Backends.py` and native backend adapters | Physics selection, reset and state readout |
| `SkrlPPO.py`, `Policy.py` | Existing PPO integration, network and actor export |
| `RunIO.py`, `Execution.py`, `Preflight.py`, `Provenance.py` | Restart integrity, budgets and runtime identity |
| `Controllers/` and `Drivers/ControllerEpisodes.py` | Controller objects and one shared evaluation loop |
| `Drivers/ContinuousGoals.py`, `Drivers/EvaluationVideo.py` | Optional target sequences and streaming video |
| `Randomization.py` | Fixed-domain lifecycle and disabled randomization reservation |

I obtained 184 passing selected checks in both the working tree and a clean
imported checkout; two optional Pinocchio/Drake runtime checks were skipped.
Additional micro-runs covered three CPU object scenes, M1–M5 GPU backends, native
MPPI and matrix routing. Ten target switches preserved readout, commands and
clocks, and two continuous clips decoded and were visually checked. These used
untrained hold actors, easy initial goals and microsecond timesteps to check
boundaries. They do not demonstrate learned rotation or realistic task performance.
Further qualification and unresolved scope are described in [VALIDATION.md](VALIDATION.md).

The pre-training audit and follow-up passed all 266 available core tests, including
the GPU and real PPO-update tests; the same two optional engine tests were skipped.
The reviews fixed diagnostic retention, fault accounting and scene dependency
identity, and checked asynchronous episode endings and nonterminal rollout bootstrap.
A separate M2 driver check completed two updates, restarted one saved
update, restored without sampling and evaluated two independent cases. That check
used only 12 training transitions and checks execution, not learning.

Selected checks can be run without real PPO optimization or GPU physics tests:

```bash
python -m pytest tests/test_rl_environment.py tests/test_rl_ppo.py \
  tests/test_rl_run_io.py tests/test_controller_episodes.py \
  tests/test_continuous_goals.py tests/test_evaluation_video.py \
  tests/test_rl_randomization.py tests/test_rl_tasks.py \
  -m 'not gpu and not ppo_update' -q
```

Actual PPO updates are marked `ppo_update`; GPU physics checks are marked `gpu`.
Run those separately with an explicit budget. No trained policy or research
performance claim is included in this framework release.
