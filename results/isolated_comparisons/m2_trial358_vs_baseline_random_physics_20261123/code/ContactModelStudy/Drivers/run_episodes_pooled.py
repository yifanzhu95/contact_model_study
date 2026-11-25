#!/usr/bin/env python3
"""Run many episodes at once across the GPUs and CPU cores of a machine.

Like ``run_episodes.py``, with the same options, plus the size of the pool:

* ``--gpus``: which GPUs plan (default: every visible GPU);
* ``--planners-per-gpu``: planner processes per GPU (default 1). More than one
  lets a GPU work on one plan while another process is busy in Python;
* ``--workers``: eval-sim processes, one episode each (default: the cores
  available, minus one per planner, and no more than ``--n-episodes``).

A queue-based scheduler hands work out as it becomes possible: a free worker
takes the next episode, and a free planner takes the next plan request from any
episode. Each episode's planner state travels with its request, so no episode is
tied to a GPU. See ``EpisodePool`` for the design.

Sizing: planning costs about the same whatever the eval sim, while an eval step
costs from a few ms (MuJoCo) to tens of ms (Pinocchio) per control step (see
``test_scripts/profile_run_episodes.py``). Enough workers to keep the planners
fed is roughly ``planners x (1 + eval time / plan time)``; the end-of-run line
reports how busy the planners and workers were, so the balance can be tuned.

Each episode's goals and planner noise come from ``(--seed, episode index)``,
so results do not depend on the pool's size or on scheduling.

Examples::

    python ContactModelStudy/Drivers/run_episodes_pooled.py --n-episodes 64 --no-video
    python ContactModelStudy/Drivers/run_episodes_pooled.py --n-episodes 64 --gpus 0,1 --workers 12 --eval-sim pinocchio
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ContactModelStudy.Drivers import run_episodes as drv  # noqa: E402
from ContactModelStudy.Drivers.EpisodePool import availableCpus, runPool, visibleGpus  # noqa: E402


def build_parser():
    p = drv.build_parser()
    p.description = __doc__
    g = p.add_argument_group("pool")
    g.add_argument("--gpus", default="all",
                   help="comma-separated GPU ids, or 'all' (default) for every visible GPU")
    g.add_argument("--planners-per-gpu", type=int, default=1,
                   help="planner processes per GPU")
    g.add_argument("--workers", type=int, default=None,
                   help="eval-sim worker processes (concurrent episodes); default: available "
                        "cores minus one per planner, at most --n-episodes")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = drv.parseArgs(argv, parser=parser, results_prefix="run_episodes_pooled")
    gpus = visibleGpus() if args.gpus == "all" else [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        parser.error("no GPU found (set --gpus or CUDA_VISIBLE_DEVICES)")
    if args.planners_per_gpu < 1:
        parser.error("--planners-per-gpu must be >= 1")
    n_planners = len(gpus) * args.planners_per_gpu
    workers = args.workers if args.workers is not None else max(1, availableCpus() - n_planners)
    if workers < 1:
        parser.error("--workers must be >= 1")
    return runPool(args, n_workers=min(workers, args.n_episodes), gpus=gpus,
                   planners_per_gpu=args.planners_per_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
