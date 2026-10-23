# RL: PPO policies for the LEAP reorientation tasks

A trained policy is one more planner. `run_episodes --planner rl` runs it in
MPPI's place, scored by the same eval simulator, recorder and video. Training
runs on the study's own simulators and task costs: `num_envs` worlds of the
task on one contact model (M1–M5), each world chasing its own goal.

The code here only covers what the shared layers don't already provide:

| File | Contents |
|---|---|
| `Config.py` | `EnvConfig` (the training environment, and the contract a trained actor carries) and `PPOConfig` |
| `Encoding.py` | The 73-value observation and the rate-limited action decoder, shared by training and inference |
| `TrainEnv.py` | skrl-compatible environment over `GetContactModelSim`, rewarded by the task's `calcCosts` |
| `Agent.py` | skrl PPO set up for `TrainEnv`, with a stop check after each update |
| `Policy.py` | The actor network and the `actor.pt` format |
| `PolicyPlanner.py` | `RLPolicyPlanner`, the `PlannerBase` that runs an `actor.pt` |
| `Execution.py`, `Provenance.py` | Clean stops on a time budget or signal; source and runtime identity for the manifest |

From the shared layers, training uses `VectorizedSimulator.ResetWorlds` (reset
finished worlds without touching the others), `CheckCapacity` (fail rather than
train on dropped contacts), and the `goals=` table of `LeapReorient.calcCosts` /
`isSuccess` (score each world against its own goal).

## Install

Use the `contact_kamino` env (Python 3.12, Warp 1.17, Newton for M5):

```bash
python -m pip install -e '.[rl]'            # training: torch, gymnasium, skrl
python -m pip install -e '.[rl-inference]'  # running a trained policy only
```

## Train

```bash
python -m ContactModelStudy.Drivers.train_rl_policy \
    --config configs/rl/cube_m2.json --timesteps 20000 --output results/rl/cube_m2
```

The config has an `environment` section (`EnvConfig`) and a `ppo` section
(`PPOConfig`). `configs/rl/cube_m*_smoke.json` are tiny configs that check the
pipeline on each contact model; they do not learn anything. `--timesteps`
counts control steps of the whole batch: a run collects
`timesteps × num_envs` transitions and makes `timesteps / rollout_steps` PPO
updates.

The output directory must be new. It holds:

| File | Content |
|---|---|
| `manifest.json` | Config, git commit, package versions, GPU |
| `agent.pt` | skrl checkpoint: policy, value network, optimizer |
| `actor.pt` | The policy for `run_episodes`, with its environment contract |
| `summary.json` | `completed`, `stopped` (with the reason) or `failed`; timesteps, PPO updates, seconds |
| `skrl/` | TensorBoard logs and checkpoints every `--checkpoint-interval` PPO updates |

`--max-seconds` and SIGTERM, SIGINT or SIGUSR1 stop the run after the current
PPO update and save it. `--resume <agent.pt>` continues from a checkpoint with
new episodes, keeping the networks and the optimizer. The resumed run's config
is used unless `--config` is given, and a different config is rejected. On a
cluster, `experiments/hpc/run_rl_job.sh` runs the driver inside an allocated job.

### The task as trained

- **Episodes.** Every episode starts from the task's initial grasp, held for
  `settle_s` (as `run_episodes --settle`), with a goal drawn by the task's own
  sampler. It ends on the first success, a drop (object below the floor), or
  after `episode_steps`.
- **Success.** The task's `isSuccess`, so the thresholds are the ones MPPI is
  judged by.
- **Reward.** The task's running cost divided by `cost_reference` and
  clamped to [0, 1], spread over the episode, with gamma = 1. A drop takes the
  full penalty for every step it cut off. The deadline is terminal rather than
  truncated, because the time left is observed.
- **Actions.** All 16 actuators.
  `target = clamp(previous + target_rate · control_dt · tanh(a), lower, upper)`.

## Evaluate

```bash
python -m ContactModelStudy.Drivers.run_episodes --planner rl \
    --policy results/rl/cube_m2/actor.pt --n-episodes 20 --eval-sim mujoco
```

The control step, episode length (`--steps`) and settle time (`--settle`)
come from the policy. Giving different values is an error, as are
`--no-stop-on-success` and a different `--task`. The options in the planner
group other than `--planner` and `--policy` are MPPI's and have no effect.
To compare contact models, repeat with `--eval-sim` set to `pinocchio`,
`drake` or `M1`–`M5`. The pooled and interwoven drivers run MPPI only.

## Not done

- A validated learning baseline. Nothing here shows that a policy learns the
  task, or how well.
- Domain randomization, multi-GPU training, and real-sensor observations or
  hardware.
