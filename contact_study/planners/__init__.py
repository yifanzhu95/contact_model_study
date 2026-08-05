"""Sampling-based predictive controllers, selectable by name.

All three planners share one rollout engine (`base.SamplingPlanner`): N control
sequences sampled from the planner's distribution, rolled out in parallel on the
GPU as N worlds, scored by the task's Warp cost. They differ only in how the
sampled costs update the distribution:

    mppi                softmax(-J/lambda)-weighted mean of the samples
    cem                 refit a Gaussian to the k lowest-cost (elite) samples
    predictive_sampler  take the single lowest-cost sample (greedy best-of-N)

Use the helpers below to go from a CLI string to a configured planner:

    cfg      = make_planner_config("cem", n_samples=256, time_horizon=0.25, ...)
    planner  = make_planner("cem", task=rollout_task, cfg=contact_cfg,
                            planner_cfg=cfg, rng=rng)
"""

from __future__ import annotations

import dataclasses

import numpy as np

from contact_study.contact_models.config import ContactModelConfig
from contact_study.planners.base import PlannerConfig, SamplingPlanner
from contact_study.planners.cem import CEMConfig, CEMController
from contact_study.planners.mppi import MPPIConfig, MPPIController
from contact_study.planners.predictive_sampler import (
    PredictiveSampler,
    PredictiveSamplerConfig,
)

__all__ = [
    "PlannerConfig", "SamplingPlanner",
    "MPPIConfig", "MPPIController",
    "CEMConfig", "CEMController",
    "PredictiveSamplerConfig", "PredictiveSampler",
    "PLANNERS", "PLANNER_ALIASES", "PLANNER_NAMES",
    "resolve_planner_name", "planner_config_cls", "planner_cls",
    "make_planner_config", "make_planner", "planner_name_for_config",
]

# name -> (config class, controller class)
PLANNERS: dict[str, tuple[type[PlannerConfig], type[SamplingPlanner]]] = {
    "mppi":               (MPPIConfig,              MPPIController),
    "cem":                (CEMConfig,               CEMController),
    "predictive_sampler": (PredictiveSamplerConfig, PredictiveSampler),
}

PLANNER_ALIASES = {"ps": "predictive_sampler", "sampler": "predictive_sampler"}

# Every string accepted on a command line, canonical names first.
PLANNER_NAMES = list(PLANNERS) + list(PLANNER_ALIASES)


def resolve_planner_name(name: str) -> str:
    """Canonicalize a planner name (resolving aliases); raise on unknown ones."""
    key = PLANNER_ALIASES.get(name, name)
    if key not in PLANNERS:
        raise KeyError(f"Unknown planner {name!r}. Available: {PLANNER_NAMES}")
    return key


def planner_config_cls(name: str) -> type[PlannerConfig]:
    return PLANNERS[resolve_planner_name(name)][0]


def planner_cls(name: str) -> type[SamplingPlanner]:
    return PLANNERS[resolve_planner_name(name)][1]


def planner_name_for_config(cfg: PlannerConfig) -> str:
    """Reverse lookup: which planner does this config belong to?

    Lets callers that pass only a config (the pre-`--planner` calling convention)
    still get the matching controller. Subclasses are matched most-specific
    first, so an exact type wins over a base-class match.
    """
    for name, (cfg_cls, _) in PLANNERS.items():
        if type(cfg) is cfg_cls:
            return name
    for name, (cfg_cls, _) in PLANNERS.items():
        if isinstance(cfg, cfg_cls):
            return name
    raise TypeError(f"{type(cfg).__name__} is not a known planner config")


def make_planner_config(name: str, **kwargs) -> PlannerConfig:
    """Build the config for `name`, ignoring kwargs it does not declare.

    Drivers parse one flat set of CLI knobs covering every planner
    (`--temperature`, `--elite_frac`, ...); this drops the ones that do not
    apply. Keys are filtered by name only — a None that IS a declared field is
    passed through, since None is meaningful for several of them (time_horizon,
    step_time, resample_interval, seed, ...). To fall back on a config's own
    default, leave the key out of kwargs entirely.
    """
    cfg_cls = planner_config_cls(name)
    fields  = {f.name for f in dataclasses.fields(cfg_cls)}
    return cfg_cls(**{k: v for k, v in kwargs.items() if k in fields})


def make_planner(
    name:        str,
    task,
    cfg:         ContactModelConfig,
    planner_cfg: PlannerConfig,
    rng:         np.random.Generator | None = None,
) -> SamplingPlanner:
    """Construct the planner `name` around a ROLLOUT task and a contact model."""
    key = resolve_planner_name(name)
    cfg_cls, ctrl_cls = PLANNERS[key]
    if not isinstance(planner_cfg, cfg_cls):
        raise TypeError(
            f"planner {key!r} needs a {cfg_cls.__name__}, got {type(planner_cfg).__name__}"
        )
    # Each controller names its config parameter after itself (mppi_cfg,
    # cem_cfg, ps_cfg), so pass it positionally.
    return ctrl_cls(task, cfg, planner_cfg, rng)
