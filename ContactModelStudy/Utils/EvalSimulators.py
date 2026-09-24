"""Building the evaluation simulator by name.

The eval simulator is the high-fidelity "real" world an episode is scored in.
Every choice runs the same eval MJCF with default solver settings, at the
timestep given. The planner's rollout simulators are a separate matter.

Pinocchio and Drake are imported only when chosen, so a machine with only
MuJoCo installed can still use the rest.
"""

from __future__ import annotations

from ContactModelStudy.Simulators.Simulator import Simulator


def _mujoco(xml: str, timestep: float) -> Simulator:
    from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
    return Mujoco(xml, MujocoConfig(timestep=timestep))


def _pinocchio(xml: str, timestep: float) -> Simulator:
    from ContactModelStudy.Simulators.Pinocchio import Pinocchio, PinocchioConfig
    return Pinocchio(xml, PinocchioConfig(timestep=timestep))


def _drake(xml: str, timestep: float) -> Simulator:
    from ContactModelStudy.Simulators.Drake import Drake, DrakeConfig
    return Drake(xml, DrakeConfig(timestep=timestep))


#: Eval simulator builders, by name.
EVAL_SIMS = {"mujoco": _mujoco, "pinocchio": _pinocchio, "drake": _drake}


def makeEvalSim(name: str, xml: str, timestep: float) -> Simulator:
    """Build the named eval simulator on ``xml``, with default solver settings.

    Args:
        name: One of ``EVAL_SIMS``: "mujoco", "pinocchio" or "drake".
        xml: Path to the eval MJCF.
        timestep: Physics timestep in seconds.

    Raises:
        ValueError: If ``name`` is not a known simulator.
    """
    if name not in EVAL_SIMS:
        raise ValueError(f"unknown eval simulator {name!r}; choose from {sorted(EVAL_SIMS)}")
    return EVAL_SIMS[name](xml, timestep)
