#!/usr/bin/env python3
"""Train a PPO policy for a LEAP reorientation task on one of the study's contact models.

skrl's ``SequentialTrainer`` runs PPO on a ``TrainEnv``: ``num_envs`` worlds of
the task on the chosen contact model (M1-M5), scored by the task's own cost.
The trained actor is run with ``run_episodes --planner rl --policy actor.pt``.

The config is a JSON file with an ``environment`` object (``RL.Config.EnvConfig``
fields) and a ``ppo`` object (``RL.Config.PPOConfig`` fields); see
``configs/rl/``. ``--timesteps`` counts control steps of the whole batch, so a
run collects ``timesteps * num_envs`` transitions and makes
``timesteps / rollout_steps`` PPO updates.

Examples::

    python -m ContactModelStudy.Drivers.train_rl_policy \\
        --config configs/rl/cube_m2.json --timesteps 20000 --output results/rl/cube_m2
    # Stop cleanly before a 4 h SLURM limit, saving every 50 updates:
    python -m ContactModelStudy.Drivers.train_rl_policy --config configs/rl/cube_m2.json \\
        --timesteps 1000000 --max-seconds 14400 --checkpoint-interval 50 --output results/rl/long
    # Continue from a checkpoint (its run's config is used unless --config is given):
    python -m ContactModelStudy.Drivers.train_rl_policy \\
        --resume results/rl/long/agent.pt --timesteps 200000 --output results/rl/long_2

The output directory must be new. It receives ``manifest.json`` (config and
source/runtime identity), ``agent.pt`` (skrl checkpoint: networks and
optimizer), ``actor.pt`` (the policy for ``run_episodes``), ``summary.json``,
and ``skrl/`` (TensorBoard logs and periodic checkpoints). A resumed run starts
new episodes; it keeps the networks and the optimizer, not the rollout in
progress or the random streams.

A stop requested by SIGTERM, SIGINT, SIGUSR1 or the ``--max-seconds`` budget
takes effect after the current PPO update, which is then saved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON with 'environment' and 'ppo'; optional with --resume")
    p.add_argument("--output", type=Path, required=True, help="new directory for this run")
    p.add_argument("--timesteps", type=int, required=True, help="control steps of the whole batch to train for")
    p.add_argument("--resume", type=Path, help="agent checkpoint (agent.pt, or one under skrl/checkpoints)")
    p.add_argument("--checkpoint-interval", type=int, default=50,
                   help="save a checkpoint every this many PPO updates; 0 saves only the final one")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="wall-time budget, setup included; checked after each PPO update")
    p.add_argument("--stop-margin-seconds", type=float, default=30.0,
                   help="stop this long before --max-seconds, to leave time for saving")
    return p


def _runManifest(checkpoint: Path) -> dict | None:
    """The ``manifest.json`` of the run that wrote ``checkpoint``, if it can be found."""
    for parent in list(checkpoint.resolve().parents)[:4]:
        path = parent / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text())
    return None


def resolveConfigs(args, parser):
    """``(EnvConfig, PPOConfig)`` from ``--config``, or from the resumed run's manifest."""
    from ContactModelStudy.RL.Config import EnvConfig, PPOConfig

    previous = _runManifest(args.resume) if args.resume else None
    if args.config is not None:
        raw = json.loads(args.config.read_text())
    elif previous is not None:
        raw = previous
    else:
        parser.error("give --config, or --resume a checkpoint whose run directory has a manifest.json")
    unknown = set(raw) - {"environment", "ppo"} - ({"runtime", "timesteps", "resume_from"} if raw is previous else set())
    if unknown:
        parser.error(f"unknown config section(s) {sorted(unknown)}")
    try:
        env_cfg, ppo_cfg = EnvConfig(**raw.get("environment", {})), PPOConfig(**raw.get("ppo", {}))
    except (TypeError, ValueError) as exc:
        parser.error(f"invalid config: {exc}")
    if previous is not None and (previous["environment"] != env_cfg.to_dict()
                                 or previous["ppo"] != json.loads(json.dumps(ppo_cfg.to_dict()))):
        parser.error("--config differs from the run being resumed; a resumed run keeps its environment and PPO settings")
    return env_cfg, ppo_cfg


def main(argv=None) -> int:
    started = time.monotonic()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timesteps < 1 or args.checkpoint_interval < 0:
        parser.error("--timesteps must be positive and --checkpoint-interval nonnegative")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"--resume {args.resume} does not exist")

    import torch
    from ContactModelStudy.RL.Agent import TrainingStopped, build_agent
    from ContactModelStudy.RL.Execution import BoundaryStop, require_single_process
    from ContactModelStudy.RL.Policy import export_actor
    from ContactModelStudy.RL.Provenance import source_runtime
    from ContactModelStudy.RL.TrainEnv import TrainEnv

    env_cfg, ppo_cfg = resolveConfigs(args, parser)
    try:
        require_single_process()
        stop = BoundaryStop(args.max_seconds, args.stop_margin_seconds, started=started)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(env_cfg.seed)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    manifest = {"environment": env_cfg.to_dict(), "ppo": ppo_cfg.to_dict(), "timesteps": args.timesteps,
                "resume_from": str(args.resume.resolve()) if args.resume else None,
                "runtime": source_runtime(_REPO_ROOT, env_cfg.device)}
    save("manifest.json", manifest)

    from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg

    env = None
    try:
        with stop:
            env = TrainEnv(env_cfg)
            agent, actor = build_agent(env, ppo_cfg, directory=args.output,
                                       checkpoint_interval=args.checkpoint_interval, should_stop=stop.reason)
            trainer = SequentialTrainer(env=env, agents=agent, cfg=SequentialTrainerCfg(
                timesteps=args.timesteps, headless=True, close_environment_at_exit=False))
            if args.resume:
                agent.load(str(args.resume))
            stop_reason, done = None, args.timesteps
            try:
                trainer.train()
            except TrainingStopped as exc:
                stop_reason, done = exc.reason, exc.timestep
            agent.save(str(args.output / "agent.pt"))
            export_actor(actor, env.describe(), args.output / "actor.pt")
    except BaseException as exc:
        save("summary.json", {"status": "failed", "exception": type(exc).__name__, "message": str(exc),
                              "run_seconds": time.monotonic() - started})
        raise
    finally:
        if env is not None:
            env.close()

    summary = {"status": "stopped" if stop_reason else "completed", "stop_reason": stop_reason,
               "timesteps": done, "transitions": done * env_cfg.num_envs,
               "ppo_updates": done // ppo_cfg.rollout_steps, "run_seconds": time.monotonic() - started}
    save("summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
