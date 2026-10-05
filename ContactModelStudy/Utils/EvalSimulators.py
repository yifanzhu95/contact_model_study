"""Building the evaluation simulator by name.

The eval simulator is the high-fidelity "real" world an episode is scored in.
Every choice runs the eval MJCF at the timestep given. Two kinds:

* **CPU simulators** (``EVAL_SIMS``): ``mujoco``, ``pinocchio``, ``drake``, each
  with its default solver settings.
* **GPU contact models** (``EVAL_PRESETS``): ``M1``-``M4``, exactly the rollout
  contact model of that name (``ContactModelPresets``), built with one world and
  wrapped in ``SingleWorld`` so it reads like a CPU simulator. Planning with one
  model and judging in another gives the rollout x eval matrix. The preset is
  resolved at the *eval* timestep, so M1's time constant, ``2 * dt``, is twice
  the eval step here.

Pinocchio, Drake and the GPU backends are imported only when chosen, so a
machine with only MuJoCo installed can still use the rest.
"""

from __future__ import annotations

from ContactModelStudy.Simulators.Simulator import Simulator


def _mujoco():
    from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
    return Mujoco, MujocoConfig


def _pinocchio():
    from ContactModelStudy.Simulators.Pinocchio import Pinocchio, PinocchioConfig
    return Pinocchio, PinocchioConfig


def _drake():
    from ContactModelStudy.Simulators.Drake import Drake, DrakeConfig
    return Drake, DrakeConfig


#: CPU eval simulators by name: each gives ``(simulator class, config class)``.
EVAL_SIMS = {"mujoco": _mujoco, "pinocchio": _pinocchio, "drake": _drake}

#: GPU contact models usable as the eval simulator, by preset name.
EVAL_PRESETS = ("M1", "M2", "M3", "M4")

#: Config fields set on a GPU eval preset on top of the preset itself. Empty:
#: the per-world contact and constraint buffers already cover the eval scenes.
EVAL_PRESET_OVERRIDES: dict = {}


def evalSimNames() -> list[str]:
    """Every name ``makeEvalSim`` accepts."""
    return sorted(EVAL_SIMS) + list(EVAL_PRESETS)


def isGpuEvalSim(name: str) -> bool:
    """Whether the named eval simulator runs on the GPU (and so needs one)."""
    return name in EVAL_PRESETS


def _check(name: str) -> None:
    if name not in EVAL_SIMS and name not in EVAL_PRESETS:
        raise ValueError(f"unknown eval simulator {name!r}; choose from {evalSimNames()}")


def evalSimConfig(name: str, timestep: float):
    """``(class name, config)`` of the named eval simulator, without building it.

    What ``makeEvalSim`` would build, for describing a run in a process that
    does not hold the simulator itself.
    """
    _check(name)
    if isGpuEvalSim(name):
        from ContactModelStudy.Utils.ContactModelPresets import CONTACT_MODELS, _presetConfig, _simulatorClasses
        sim_cls, _ = _simulatorClasses(CONTACT_MODELS[name]["simulator"])
        return sim_cls.__name__, _presetConfig(name, timestep=timestep, **EVAL_PRESET_OVERRIDES)
    sim_cls, cfg_cls = EVAL_SIMS[name]()
    return sim_cls.__name__, cfg_cls(timestep=timestep)


def makeEvalSim(name: str, xml: str, timestep: float) -> Simulator:
    """Build the named eval simulator on ``xml``.

    Args:
        name: A CPU simulator (``"mujoco"``, ``"pinocchio"``, ``"drake"``) or a
            GPU contact model (``"M1"``-``"M4"``).
        xml: Path to the eval MJCF.
        timestep: Physics timestep in seconds.

    Returns:
        A ``Simulator``. A GPU model comes wrapped in ``SingleWorld``.

    Raises:
        ValueError: If ``name`` is not a known simulator.
    """
    _check(name)
    if isGpuEvalSim(name):
        from ContactModelStudy.Simulators.SingleWorld import SingleWorld
        from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
        return SingleWorld(GetContactModelSim(name, xml, N=1, timestep=timestep, **EVAL_PRESET_OVERRIDES))
    sim_cls, cfg_cls = EVAL_SIMS[name]()
    return sim_cls(xml, cfg_cls(timestep=timestep))
