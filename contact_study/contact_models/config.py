"""ContactModelConfig: central dispatch object for contact model variants.

Each Mk on the contact-model axis of the study is fully described by one of
these configs. The 'backend' field drives dispatch in api.py.

Scope note
----------
This config describes ONLY the contact model (M1..M4). Orthogonal study
axes — geometry fidelity and physics parameter noise — are handled OUTSIDE
this object:

  * Geometry variants live in XML files; pick one at task-load time via
    contact_study.tasks.base.BaseTask(geometry=GeometryVariant.CONVEX_HULL).
  * Physics noise is applied by contact_study.utils.physics_noise.apply_physics_noise
    to an MjModel before put_model is called.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional


class Backend(str, enum.Enum):
    """Top-level contact model backend selector."""
    MUJOCO_HARD  = "mujoco_hard"   # M1: stiff MuJoCo model
    MUJOCO_SOFT      = "mujoco_soft"       # M2: MJWarp default soft contact
    COMFREE          = "comfree"           # M3: Jin 2024 complementarity-free
    XPBD             = "xpbd"              # M4: XPBD-style penalty model


class GeometryVariant(str, enum.Enum):
    """Geometry fidelity level (selects which XML scene file is loaded).

    NOT part of ContactModelConfig. Pair any geometry variant with any Mk
    by passing it to BaseTask(geometry=...) before calling api.put_model.
    """
    ACCURATE         = "accurate"
    CONVEX_HULL      = "convex_hull"
    PRIMITIVE_UNION  = "primitive_union"
    LINEARIZED       = "linearized"


@dataclasses.dataclass
class MujocoSolverParams:
    """Parameters for MuJoCo's (MJWarp-hosted) built-in solver.

    MJWarp only supports pyramidal friction cones — elliptic is not
    implemented on the GPU backend. All MJWarp-backed configs must use
    cone='pyramidal'; api._patch_mujoco_options raises on anything else.

    The soft-vs-hard distinction between M1 and M2 is expressed via
    `hard_contact` rather than cone type. When hard_contact=True,
    api._apply_hard_contact_preset patches every geom's solref/solimp
    toward the hard-constraint limit (d → 1, timeconst → 2·dt), pushing
    MJWarp's regularized convex QP as close to the true Anitescu QP as
    the solver can express while remaining stable.
    """
    cone:       str   = "pyramidal"
    solver:     str   = "Newton"       # "PGS" | "CG" | "Newton"
    iterations: int   = 100
    tolerance:  float = 1e-8

    # --- Hard-contact preset (M1) ---------------------------------
    # When True, all geom/pair solref and solimp are overwritten with
    # the values below at put_model time. When False, the XML's own
    # per-geom solref/solimp are left untouched (M2 default).
    hard_contact:               bool  = False

    # solimp target: d → 1 means the contact row's regularizer R = 1/D → 0.
    # dmin == dmax collapses the impedance ramp to a flat near-one curve.
    hard_solimp_d:              float = 0.9999
    hard_solimp_width:          float = 1.0e-4
    hard_solimp_midpoint:       float = 0.5
    hard_solimp_power:          float = 2.0

    # solref target: timeconst = timeconst_mult · dt, clamped to ≥ 2·dt
    # by _apply_hard_contact_preset. Smaller timeconst → stiffer
    # position-level correction; < 2·dt is unstable for the implicit
    # integrator.
    hard_solref_timeconst_mult: float = 2.0
    hard_solref_dampratio:      float = 1.0


@dataclasses.dataclass
class ComfreeParams:
    """Parameters for Jin's complementarity-free model (M3)."""
    stiffness: float = 0.2
    damping:   float = 0.001


@dataclasses.dataclass
class XPBDParams:
    """Parameters for the XPBD-style contact model (M4).

    See contact_study.contact_models.xpbd_backend for what each field does.
    """
    substeps:            int   = 1
    vmax_depenetration:  float = 1.0
    iterations:          int   = 1


@dataclasses.dataclass
class ContactModelConfig:
    """Full specification of one contact model variant Mk (k ∈ 1..4).

    Usage::

        cfg = ContactModelConfig.M3()
        cfg.comfree.stiffness = 0.3
        m = api.put_model(mjm, cfg)
        d = api.make_data(mjm, m, nworld=1024)
        api.step(m, d)
    """
    backend: Backend = Backend.MUJOCO_SOFT

    mujoco:  MujocoSolverParams = dataclasses.field(default_factory=MujocoSolverParams)
    comfree: ComfreeParams      = dataclasses.field(default_factory=ComfreeParams)
    xpbd:    XPBDParams         = dataclasses.field(default_factory=XPBDParams)

    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.backend.value

    # ------------------------------------------------------------------
    # Factory methods — one per contact model in the study
    # ------------------------------------------------------------------

    @classmethod
    def M1(cls) -> "ContactModelConfig":
        """M1: Stiff-limit pyramidal contact.

        MJWarp's solver is a regularized convex QP, not a hard LCP, 
        and there is no way to
        disable the regularization entirely. What M1 does instead is
        push that QP toward its hard-constraint limit:

          * contact solimp → (0.9999, 0.9999, 1e-4, 0.5, 2)
            so the diagonal regularizer R = 1/efc_D is driven ~1000×
            smaller than MuJoCo's default d ≈ 0.9;
          * contact solref → (2·dt, 1.0)
            the tightest stable timeconst for the implicit integrator,
            producing critically-damped position-level correction
            within ~one step;
          * Newton solver, iterations=200, tolerance=1e-10
            to ensure the stiffened QP actually converges.

        The result is measurably different from M2 (soft defaults) and
        is the closest-to-Anitescu formulation that stays inside
        MJWarp's parallel-per-world solver. The paper should describe
        M1 as "stiff-limit pyramidal MJWarp," not "Anitescu verbatim."

        If the M1-vs-M2 gap turns out too small to be interesting, the
        next step is to write a dedicated projected-Newton Anitescu
        backend (see xpbd_backend.py for how a custom solver slots in).
        """
        return cls(
            backend=Backend.MUJOCO_HARD,
            mujoco=MujocoSolverParams(
                cone="pyramidal",
                solver="Newton",
                iterations=200,
                tolerance=1e-10,
                hard_contact=True,
            ),
            label="M1_stiff_pyramidal",
        )

    @classmethod
    def M2(cls) -> "ContactModelConfig":
        """M2: MJWarp default soft contact (pyramidal cone, Newton).

        CAVEAT — paper vs. implementation: the paper's M2 is "MuJoCo
        default soft contact," which in reference MuJoCo uses an
        *elliptic* friction cone. MJWarp does not implement elliptic
        cones on the GPU; only pyramidal. So the M2 available here is
        pyramidal + default solref/solimp, which is MuJoCo's soft
        formulation with a pyramidalized friction cone. This is an
        unavoidable approximation of the paper's M2 under the MJWarp
        backend and should be reported as such in the methodology.

        Concretely this means M1 and M2 differ only in the stiffness of
        the soft-contact formulation (hard vs. default regularization),
        not in cone type. That is not ideal for the taxonomy, but it is
        honest and it reflects the state of the GPU tooling.
        """
        return cls(
            backend=Backend.MUJOCO_SOFT,
            mujoco=MujocoSolverParams(
                cone="pyramidal",   # NOT elliptic — MJWarp limitation
                solver="Newton",
                hard_contact=False,
            ),
            label="M2_mjwarp_soft",
        )

    @classmethod
    def M3(cls) -> "ContactModelConfig":
        """M3: Complementarity-free model (Jin 2024)."""
        return cls(backend=Backend.COMFREE, label="M3_comfree")

    @classmethod
    def M4(cls) -> "ContactModelConfig":
        """M4: Decoupled XPBD-style penalty model."""
        return cls(backend=Backend.XPBD, xpbd=XPBDParams(), label="M4_xpbd")

    @classmethod
    def all_models(cls) -> list["ContactModelConfig"]:
        """Return the canonical list of all contact model configs."""
        return [cls.M1(), cls.M2(), cls.M3(), cls.M4()]