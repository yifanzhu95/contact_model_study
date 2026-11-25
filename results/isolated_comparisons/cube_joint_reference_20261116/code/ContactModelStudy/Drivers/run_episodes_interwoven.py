#!/usr/bin/env python3
"""Run episodes two at a time, so the CPU and the GPU are both kept busy.

Like ``run_episodes.py``, with the same options, but with two episodes in
flight: while one episode's eval simulator steps on the CPU, the other
episode's plan runs on the GPU, and then they swap. When an episode ends, the
next one starts in its place, until ``--n-episodes`` have run.

It is the pooled engine (``EpisodePool``) at its smallest: one GPU planner
process and two CPU eval-sim workers. ``run_episodes_pooled.py`` sizes the
same engine to a whole node. How much it gains depends on the balance of the
two halves: with a cheap eval sim (MuJoCo) the GPU is the bottleneck and the
gain is small; with an expensive one (Pinocchio, Drake) it approaches 2x.

Each episode's goals and planner noise come from ``(--seed, episode index)``,
so results do not depend on which episode ran alongside which. They are not
episode-for-episode identical to ``run_episodes.py``, whose episodes share one
goal stream.

Examples::

    python ContactModelStudy/Drivers/run_episodes_interwoven.py --n-episodes 10 --eval-sim pinocchio
    python ContactModelStudy/Drivers/run_episodes_interwoven.py --n-episodes 20 --no-video --gpu 1
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
from ContactModelStudy.Drivers.EpisodePool import runPool, visibleGpus  # noqa: E402


def build_parser():
    p = drv.build_parser()
    p.description = __doc__
    g = p.add_argument_group("interwoven")
    g.add_argument("--gpu", default=None,
                   help="GPU id the planner runs on; default the first visible one")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = drv.parseArgs(argv, parser=parser, results_prefix="run_episodes_interwoven")
    gpus = [args.gpu] if args.gpu is not None else visibleGpus()[:1]
    if not gpus:
        parser.error("no GPU found (set --gpu or CUDA_VISIBLE_DEVICES)")
    return runPool(args, n_workers=2, gpus=gpus, planners_per_gpu=1)


if __name__ == "__main__":
    raise SystemExit(main())
