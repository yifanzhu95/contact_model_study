"""ContactModelConfig: central dispatch object for all contact model variants.

Each Mk in the study is fully described by one of these configs.
The 'backend' field drives dispatch in api.py; all other fields are
backend-specific parameters with sensible defaults.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional


class Backend(str, enum.Enum):
    """Top-level contact model backend selector."""
    MUJOCO_ANITESCU  = "mujoco_anitescu"   # M1: Anitescu / Newton-CG solver
    MUJOCO_SOFT      = "mujoco_soft"       # M2: MuJoCo default soft-contact (PGS)
    COMFREE          = "comfree"           # M3: Jin 2024 complementarity-free
    XPBD             = "xpbd"             # M4: decoupled penalty / XPBD-style


class GeometryVariant(str, enum.Enum):
    """Geometry fidelity level (controls which XML scene file is loaded)."""
    ACCURATE         = "accurate"          # original mesh / accurate geometry
    CONVEX_HULL      = "convex_hull"       # single convex hull per body
    PRIMITIVE_UNION  = "primitive_union"   # union of spheres / capsules / boxes
    LINEARIZED       = "linearized"        # polyhedral linearization of curved geoms


@dataclasses.dataclass
class MujocoSolverParams:
    """Parameters for MuJoCo's built-in solvers (M1 and M2).

    For M1 (Anitescu): set cone=pyramidal, solver=Newton or CG.
    For M2 (soft):     set cone=elliptic,  solver=PGS.
    These map directly to MuJoCo XML <option> attributes and are written
    into the model at load time by api.put_model().
    """
    cone:       str   = "pyramidal"   # "pyramidal" (M1) | "elliptic" (M2)
    solver:     str   = "Newton"      # "PGS" | "CG" | "Newton"
    iterations: int   = 100
    tolerance:  float = 1e-8
    # Soft-contact impedance (used by M2; ignored if cone=pyramidal)
    solimp_dmin:  float = 0.9
    solimp_dmax:  float = 0.95
    solimp_width: float = 0.001


@dataclasses.dataclass
class ComfreeParams:
    """Parameters for Jin's complementarity-free model (M3).

    These are injected onto the Warp model object by comfree_warp.api.put_model().
    """
    stiffness: float = 0.2
    damping:   float = 0.001


@dataclasses.dataclass
class XPBDParams:
    """Parameters for the decoupled XPBD-style contact model (M4).
 
    Normal contacts are resolved via XPBD compliance projection
    (position-level, per-contact, unilateral clamp).  Friction uses
    regularised Coulomb:
 
        f_t = −μ · f_n · v_t / √(‖v_t‖² + ε)
 
    Non-contact constraints (equality, limits, joint friction) are
    handled with the same compliance parameter, bilateral for equalities
    and unilateral for limits.
 
    Parameters
    ----------
    compliance : float
        XPBD positional compliance α.  Smaller → stiffer.  The effective
        compliance per timestep is α̃ = α / dt².  Default 1e-4 gives
        firm contact at dt = 0.002.
    friction_reg_eps : float
        Regularisation ε in the friction denominator.  Controls the
        cross-over from Coulomb to viscous behaviour near zero slip.
        Larger values → more damping at low speed.
    """
    compliance:       float = 1e-4 #1e-4
    friction_reg_eps: float = 1e-6
    iterations:       int   = 1

@dataclasses.dataclass
class PhysicsNoiseParams:
    """Perturbations applied to physical parameters (M7–M10).

    All noise is multiplicative: true_value * (1 + N(0, sigma)).
    Set sigma=0.0 to disable.
    """
    mass_sigma:     float = 0.0
    inertia_sigma:  float = 0.0
    friction_sigma: float = 0.0
    com_sigma:      float = 0.0   # additive noise on CoM position (meters)


@dataclasses.dataclass
class ContactModelConfig:
    """Full specification of a contact model variant Mk.

    Usage::

        cfg = ContactModelConfig(
            backend=Backend.COMFREE,
            geometry=GeometryVariant.CONVEX_HULL,
            comfree=ComfreeParams(stiffness=0.3),
            physics_noise=PhysicsNoiseParams(friction_sigma=0.1),
        )
        m, d = api.put_model(mjm, cfg), api.make_data(mjm, m)
        api.step(m, d)
    """
    backend:       Backend          = Backend.MUJOCO_SOFT
    geometry:      GeometryVariant  = GeometryVariant.ACCURATE

    # Per-backend parameter blocks (only the relevant one is used)
    mujoco:        MujocoSolverParams  = dataclasses.field(default_factory=MujocoSolverParams)
    comfree:       ComfreeParams       = dataclasses.field(default_factory=ComfreeParams)
    xpbd:          XPBDParams          = dataclasses.field(default_factory=XPBDParams)

    # Physics parameter noise (shared across all backends)
    physics_noise: PhysicsNoiseParams  = dataclasses.field(default_factory=PhysicsNoiseParams)

    # Human-readable label used in plots / result tables
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = f"{self.backend.value}_{self.geometry.value}"

    # ------------------------------------------------------------------
    # Factory methods for each Mk in the study
    # ------------------------------------------------------------------

    @classmethod
    def M1(cls) -> "ContactModelConfig":
        """M1: Anitescu model (pyramidal cone, Newton solver)."""
        return cls(
            backend=Backend.MUJOCO_ANITESCU,
            mujoco=MujocoSolverParams(cone="pyramidal", solver="Newton"),
            label="M1_anitescu",
        )

    @classmethod
    def M2(cls) -> "ContactModelConfig":
        """M2: MuJoCo default soft-contact (elliptic cone, Newton)."""
        return cls(
            backend=Backend.MUJOCO_SOFT,
            mujoco=MujocoSolverParams(cone="elliptic", solver="Newton"),
            label="M2_mujoco_soft",
        )

    @classmethod
    def M3(cls) -> "ContactModelConfig":
        """M3: Complementarity-free model (Jin 2024)."""
        return cls(backend=Backend.COMFREE, label="M3_comfree")

    @classmethod
    def M4(cls) -> "ContactModelConfig":
        """M4: Decoupled XPBD-style penalty model."""
        return cls(
            backend=Backend.XPBD,
            xpbd=XPBDParams(),
            label="M4_xpbd",
        )

    @classmethod
    def M5(cls, geom: GeometryVariant = GeometryVariant.CONVEX_HULL) -> "ContactModelConfig":
        """M5: Degraded geometry + accurate contact model."""
        return cls(backend=Backend.MUJOCO_SOFT, geometry=geom, label=f"M5_{geom.value}")

    @classmethod
    def M6(cls, geom: GeometryVariant = GeometryVariant.CONVEX_HULL) -> "ContactModelConfig":
        """M6: Degraded geometry + M3/M4."""
        return cls(backend=Backend.XPBD, geometry=geom, label=f"M6_{geom.value}")

    @classmethod
    def M7(cls, friction_sigma: float = 0.2) -> "ContactModelConfig":
        """M7: Inaccurate physical parameters + accurate contact model."""
        return cls(
            backend=Backend.MUJOCO_SOFT,
            physics_noise=PhysicsNoiseParams(friction_sigma=friction_sigma, mass_sigma=0.1),
            label="M7_phys_noise",
        )

    @classmethod
    def M8(cls, friction_sigma: float = 0.2) -> "ContactModelConfig":
        """M8: Inaccurate physical parameters + M3/M4."""
        return cls(
            backend=Backend.XPBD,
            physics_noise=PhysicsNoiseParams(friction_sigma=friction_sigma, mass_sigma=0.1),
            label="M8_xpbd_phys_noise",
        )

    @classmethod
    def M9(cls, geom: GeometryVariant = GeometryVariant.CONVEX_HULL,
           friction_sigma: float = 0.2) -> "ContactModelConfig":
        """M9: Degraded geometry + inaccurate physics + accurate contact model."""
        return cls(
            backend=Backend.MUJOCO_SOFT,
            geometry=geom,
            physics_noise=PhysicsNoiseParams(friction_sigma=friction_sigma, mass_sigma=0.1),
            label=f"M9_{geom.value}_phys_noise",
        )

    @classmethod
    def M10(cls, geom: GeometryVariant = GeometryVariant.CONVEX_HULL,
            friction_sigma: float = 0.2) -> "ContactModelConfig":
        """M10: Degraded geometry + inaccurate physics + M3/M4."""
        return cls(
            backend=Backend.XPBD,
            geometry=geom,
            physics_noise=PhysicsNoiseParams(friction_sigma=friction_sigma, mass_sigma=0.1),
            label=f"M10_{geom.value}_phys_noise",
        )

    @classmethod
    def all_models(cls) -> list["ContactModelConfig"]:
        """Return the canonical list of all Mk configs for a full study run."""
        return [
            cls.M1(), cls.M2(), cls.M3(), cls.M4(), cls.M4(damping_friction=True),
            cls.M5(GeometryVariant.CONVEX_HULL), cls.M5(GeometryVariant.PRIMITIVE_UNION),
            cls.M6(GeometryVariant.CONVEX_HULL),
            cls.M7(), cls.M8(), cls.M9(), cls.M10(),
        ]
