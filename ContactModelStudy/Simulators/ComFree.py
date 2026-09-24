"""ComFree Warp — complementarity-free contact (Jin 2024) — as a ``VectorizedSimulator``.

ComFree replaces MuJoCo's constraint solver with a closed-form,
complementarity-free contact force: no iterative QP, so one step costs the same
however many contacts are active. It runs on MJWarp's model and data layout —
``comfree_warp`` is MJWarp with its constraint solve swapped out — so this
simulator is ``VectorizedMujoco`` with the model upload and the step replaced.
Controls, state, control sequences and CUDA-graph capture are all inherited.

Only pyramidal friction cones, as for MJWarp.
"""

from __future__ import annotations

from dataclasses import dataclass

from ContactModelStudy.Simulators.VectorizedMujoco import (
    VectorizedMujoco,
    VectorizedMujocoConfig,
)


@dataclass
class ComFreeConfig(VectorizedMujocoConfig):
    """Physics parameters for the ComFree backend.

    Everything in ``VectorizedMujocoConfig``, plus ComFree's two contact
    parameters. Not every inherited setting reaches ComFree's contact force.
    All of them are still written to the model:

    * ``solimp_d``, ``solimp_midpoint``, ``solimp_power``: used. They set the
      per-row impedance, which scales each contact row's effective mass.
    * ``solimp_width``: no effect. comfree_warp hard-codes the width to 0.01
      to avoid a jump in force when an initial penetration is large.
    * ``solref_timeconst``, ``solref_dampratio``: no effect. They only feed
      MuJoCo's reference acceleration, which ComFree does not use; its
      ``stiffness`` and ``damping`` play that role instead.
    * ``solver``, ``iterations``, ``tolerance``: no effect, because ComFree
      has no iterative solve.

    Attributes:
        stiffness: Contact stiffness, dimensionless: ComFree divides it by the
            timestep, so the same value means the same thing at any ``dt``.
            Larger removes penetration faster and makes contact harder.
        damping: Contact damping, dimensionless and divided by the timestep in
            the same way. Larger dissipates more contact velocity per step.
    """

    stiffness: float = 0.2
    damping: float = 0.001

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stiffness < 0:
            raise ValueError(f"stiffness must be >= 0, got {self.stiffness}")
        if self.damping < 0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")


class ComFree(VectorizedMujoco):
    """``N`` parallel worlds on the GPU under ComFree contact.

    Same attributes as ``VectorizedMujoco``; ``m`` additionally carries
    ``comfree_stiffness`` and ``comfree_damping`` device arrays, and ``d``
    ComFree's extra scratch fields.
    """

    def __init__(self, xml, sim_config: ComFreeConfig | None = None, N: int = 1):
        """Compile a model, upload it under ComFree, and allocate ``N`` worlds.

        Args:
            xml: Path to an MJCF file, or the MJCF document itself.
            sim_config: Physics parameters. Defaults to ``ComFreeConfig()``.
            N: Number of parallel worlds.

        Raises:
            ImportError: If comfree_warp is not installed.
        """
        self._cfw = _comfree_warp()
        super().__init__(xml, sim_config if sim_config is not None else ComFreeConfig(), N)

    # -- backend hooks -------------------------------------------------------
    def _putModel(self):
        return self._cfw.put_model(
            self.mjm,
            comfree_stiffness=self.config.stiffness,
            comfree_damping=self.config.damping,
        )

    def _makeData(self, **kwargs):
        # comfree_warp's make_data adds the scratch fields its step needs.
        return self._cfw.make_data(self.mjm, nworld=self.N, **kwargs)

    def _stepPhysics(self) -> None:
        self._cfw.step(self.m, self.d)

    def _forwardPhysics(self) -> None:
        self._cfw.forward(self.m, self.d)


def _comfree_warp():
    """Import comfree_warp, with a message that names the package if it is missing."""
    try:
        import comfree_warp as cfw
    except ImportError as exc:
        raise ImportError(
            "ComFree needs the comfree_warp package (see pyproject.toml)."
        ) from exc
    return cfw
