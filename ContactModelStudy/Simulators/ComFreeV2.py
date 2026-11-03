"""ComFree-Sim (comfree_warp 0.3, asu-iris/comfree-warp-simbenchmark) as a ``VectorizedSimulator``.

The second ComFree, M6. Like M3 it starts from a closed-form, complementarity-
free contact force (comfree-0), but then refines the dynamics in at most L
passes, each a Newton step of MuJoCo's soft-constraint problem at fixed row
states (ComFree-C). A world stops early once its active rows settle, and its
acceleration is then MuJoCo's exact solution. L = 0 is the closed form alone.

comfree_warp 0.3 vendors MuJoCo Warp 3.14 and needs MuJoCo 3.14, while M3's
comfree_warp 0.1 needs MuJoCo 3.6, and both install as ``comfree_warp``. So the
two cannot share an env: M6 runs in ``contact_comfree2`` (see
``ContactModelStudy/README.md``), where every MJWarp model (M1, M2, M4) runs
on the 3.14 copy too, and M3 does not run at all.

Only pyramidal friction cones, as for MJWarp.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata

from ContactModelStudy.Simulators.VectorizedMujoco import (
    VectorizedMujoco,
    VectorizedMujocoConfig,
)


@dataclass
class ComFreeV2Config(VectorizedMujocoConfig):
    """Physics parameters for the ComFree-Sim backend.

    Everything in ``VectorizedMujocoConfig``, plus ComFree-Sim's three. Unlike
    M3, the scene's solref and solimp are used: comfree-0's reference
    acceleration takes each contact's stiffness, damping and impedance from
    them, and the passes solve MuJoCo's own problem with them. ``solver`` and
    ``iterations`` have no effect (the passes always use MuJoCo's Newton step,
    and ``refinement`` caps them); ``tolerance`` is used only to decide whether
    a world's warm start is already converged and can skip its passes.

    Attributes:
        alpha: comfree-0's relaxation α, in (0, 1]. Scales the closed-form force
            that decides which rows start active.
        gamma: comfree-0's reference gain γ, at least 1. The reference
            acceleration is ``-γ² K imp (pos - margin) - γ B vel``; at 10 a
            resting contact starts as sticking.
        refinement: L, the most passes after comfree-0, from 0 to 4.
    """

    alpha: float = 0.002  # comfree_warp 0.3's defaults; ContactModelPresets sets M6's own
    gamma: float = 10.0
    refinement: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.gamma < 1.0:
            raise ValueError(f"gamma must be >= 1, got {self.gamma}")
        if isinstance(self.refinement, bool) or not 0 <= int(self.refinement) <= 4:
            raise ValueError(f"refinement must be an integer in 0..4, got {self.refinement}")


class ComFreeV2(VectorizedMujoco):
    """``N`` parallel worlds on the GPU under ComFree-Sim contact.

    Same attributes as ``VectorizedMujoco``; ``m`` additionally carries
    comfree_warp's ``comfree_*`` settings, and ``d`` its pass scratch.
    """

    def __init__(self, xml, sim_config: ComFreeV2Config | None = None, N: int = 1):
        """Compile a model, upload it under ComFree-Sim, and allocate ``N`` worlds.

        Raises:
            ImportError: If comfree_warp 0.3 or later is not installed.
        """
        self._cfw = _comfree_warp()
        super().__init__(xml, sim_config if sim_config is not None else ComFreeV2Config(), N)

    # -- backend hooks -------------------------------------------------------
    def _putModel(self):
        return self._cfw.put_model(
            self.mjm,
            comfree_alpha=self.config.alpha,
            comfree_gamma=self.config.gamma,
            comfree_refinement=int(self.config.refinement),
        )

    def _makeData(self, **kwargs):
        return self._cfw.make_data(self.mjm, nworld=self.N, **kwargs)

    def _stepPhysics(self) -> None:
        self._cfw.step(self.m, self.d)

    def _forwardPhysics(self) -> None:
        self._cfw.forward(self.m, self.d)

    # -- state ---------------------------------------------------------------
    # The passes start from the warm start, the last step's acceleration, which
    # after a reseed belongs to some other state (for a planner, the end of
    # another world's rollout). Both seeders clear it, and the warm-start
    # check's backoff, so the first step from a new state starts from zero
    # acceleration in every world alike. Kernel and memset work only, so it
    # stays inside a captured rollout graph.
    def _clearWarmstart(self) -> None:
        self.d.qacc_warmstart.zero_()
        self._cfw.comfree_core._src.contact_solver.reset(self.d)

    def SetState(self, q, q_dot=None) -> None:
        self._clearWarmstart()
        super().SetState(q, q_dot)

    def BroadcastState(self, q, q_dot=None, u=None) -> None:
        super().BroadcastState(q, q_dot, u)
        self._clearWarmstart()


#: The first comfree_warp with the passes (``comfree_refinement``).
_MIN_VERSION = (0, 3)


def _comfree_warp():
    """Import comfree_warp, checking it is ComFree-Sim and not M3's 0.1."""
    try:
        import comfree_warp as cfw
        import comfree_warp.comfree_core._src.contact_solver  # noqa: F401 - used by _clearWarmstart
        version = metadata.version("comfree_warp")
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise ImportError(
            "ComFreeV2 (M6) needs comfree_warp >= 0.3 from asu-iris/comfree-warp-simbenchmark; "
            "use the contact_comfree2 env (ContactModelStudy/README.md)."
        ) from exc
    if tuple(int(p) for p in version.split(".")[:2]) < _MIN_VERSION:
        raise ImportError(
            f"ComFreeV2 (M6) needs comfree_warp >= 0.3, found {version} (M3's ComFree); "
            f"use the contact_comfree2 env (ContactModelStudy/README.md)."
        )
    return cfw
