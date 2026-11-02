"""XPBD contact on MJWarp, as a ``VectorizedSimulator``.

This backend keeps MJWarp's collision detection and constraint rows, but
replaces its constraint solver with XPBD (Macklin et al. 2016) relaxation
sweeps. The sweeps cover every active row: equality, limit, friction loss and
contact. Like ComFree, it runs on MJWarp's model and data layout, so this
simulator is ``VectorizedMujoco`` with the step and forward replaced. Controls,
state, control sequences and CUDA-graph capture are all inherited.

Each substep solves for the impulses ``λ`` that make every row's constraint
hold at the end of the step, to within its compliance ``α``. With ``M̂`` the
matrix the integrator advances velocity with (see below), the per-row update,
one parallel (Jacobi) launch over all rows, is::

    v_e   = J_e · qvel_pred                       # row velocity, from M̂
    A_e   = J_e M̂⁻¹ J_eᵀ                          # exact effective inverse mass
    α̃     = α / dt²                                # compliance, per row type
    C_e   = efc_pos[e]                            # row residual
    Δλ    = -ω_e · (v_e + C_e/dt + α̃ λ_old) / (A_e + α̃)
    λ_new = clamp(λ_old + Δλ, [lo, hi])           # bounds depend on row type
    qfrc += J_eᵀ · (Δλ / dt)

``C_e + dt · v_e`` is the row's constraint at the predicted end-of-step
position, so this is XPBD's ``Δλ = (-C - α̃λ) / (∇C M⁻¹ ∇Cᵀ + α̃)`` with ``λ``
held as an impulse. MuJoCo's own regularizer (``efc_D``, from solimp) and
reference acceleration (solref) are not used: the compliance is XPBD's.

``M̂`` is the integrator's velocity-update matrix, so that the velocity the
solve predicts is the velocity the step produces. Under Euler it is
``M + dt·diag(damping)``, because MuJoCo integrates joint damping implicitly;
under implicitfast it is ``M - dt·∂f/∂v``. Each substep factors it once,
computes ``A_e`` from that factor, and advances the state with it.

Row types:

  EQUALITY ............... bilateral, no clamp,              joint_compliance
  LIMIT_JOINT/TENDON ..... unilateral (λ ≥ 0),               joint_compliance
  FRICTION_DOF/TENDON .... box clamp ±frictionloss·dt, no C,  rigid
  CONTACT_FRICTIONLESS ... unilateral (λ ≥ 0),               contact_compliance
  CONTACT_PYRAMIDAL ...... unilateral (λ ≥ 0),               contact_compliance
  CONTACT_ELLIPTIC ....... skipped (MJWarp is pyramidal-only anyway)

Why contact rows are averaged: many rows act on the same body. A pyramidal
cone has 4 edge rows per contact, and a grasp can have several contacts on the
object. In a parallel Jacobi sweep, every row corrects the same error at once,
so the combined correction overshoots by roughly the number of rows that share
it. As in Newton's XPBD (``rigid_contact_con_weighting``), each contact row's
step is divided by the number of contact rows on its bodies::

    ω_e = relaxation / max(n_a, n_b)     contact rows
    ω_e = relaxation                     every other row

where ``n_a`` and ``n_b`` count the active contact rows touching the contact's
two bodies. A body welded to the world is static and not counted. Rows are
counted rather than contacts because a contact's 4 edges all push along its
normal. Newton divides each body's correction by its own count; with one
impulse per row, ``max`` is the version that keeps rows' impulses consistent
forces. The weights change how fast the sweeps converge, not what they
converge to.

Pyramidal Coulomb friction: for condim=3, MJWarp emits 4 edge rows
``J_n ± μ J_t``. Keeping each one unilateral (λ ≥ 0) enforces ``|λ_t| ≤ μ λ_n``
through the row arithmetic, with no explicit Coulomb clamp. Compliance is per
row, so a condim-3 contact's 4 edges act as 4 springs in parallel along the
normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ContactModelStudy.Simulators.VectorizedMujoco import (
    VectorizedMujoco,
    VectorizedMujocoConfig,
    _mujoco_warp,
)

# Imported at module level because the kernels below compile in constants
# that come from MJWarp's types. The whole module needs MJWarp, and
# VectorizedMujoco already imports it.
_mujoco_warp()
from comfree_warp.mujoco_warp._src import collision_driver  # noqa: E402
from comfree_warp.mujoco_warp._src import constraint as _mjw_constraint  # noqa: E402
from comfree_warp.mujoco_warp._src import derivative  # noqa: E402
from comfree_warp.mujoco_warp._src import sensor  # noqa: E402
from comfree_warp.mujoco_warp._src import smooth  # noqa: E402
from comfree_warp.mujoco_warp._src.forward import (  # noqa: E402
    _advance,
    fwd_acceleration,
    fwd_actuation,
    fwd_velocity,
)
from comfree_warp.mujoco_warp._src.types import (  # noqa: E402
    ConstraintType,
    DisableBit,
    EnableBit,
    IntegratorType,
)


# -- constraint type constants ----------------------------------------------

def _ct(name: str, default: int) -> int:
    """This build's integer for a ConstraintType, or a value no row will match.

    A name missing from this MJWarp build maps to a sentinel, so rows of that
    type fall through to the bilateral default and do not crash. As a result, a
    typo in a name here quietly downgrades how that row type is treated.
    """
    try:
        return int(getattr(ConstraintType, name))
    except AttributeError:
        return default


_CT_EQUALITY             = wp.constant(_ct("EQUALITY",             -101))
_CT_FRICTION_DOF         = wp.constant(_ct("FRICTION_DOF",         -102))
_CT_FRICTION_TENDON      = wp.constant(_ct("FRICTION_TENDON",      -103))
_CT_LIMIT_JOINT          = wp.constant(_ct("LIMIT_JOINT",          -104))
_CT_LIMIT_TENDON         = wp.constant(_ct("LIMIT_TENDON",         -105))
_CT_CONTACT_FRICTIONLESS = wp.constant(_ct("CONTACT_FRICTIONLESS", -106))
_CT_CONTACT_PYRAMIDAL    = wp.constant(_ct("CONTACT_PYRAMIDAL",    -107))
_CT_CONTACT_ELLIPTIC     = wp.constant(_ct("CONTACT_ELLIPTIC",     -108))


# -- kernels -----------------------------------------------------------------

@wp.kernel
def _copy_1d(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _scale_1d(
    src: wp.array(dtype=float),
    scale: float,
    dst: wp.array(dtype=float),
):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def _predict_qvel(
    opt_timestep: wp.array(dtype=float),
    qvel: wp.array2d(dtype=float),
    qacc: wp.array2d(dtype=float),
    qvel_pred: wp.array2d(dtype=float),  # [out]
):
    """qvel_pred = qvel + qacc · dt."""
    w, i = wp.tid()
    dt = opt_timestep[w % opt_timestep.shape[0]]
    qvel_pred[w, i] = qvel[w, i] + qacc[w, i] * dt


@wp.kernel
def _sum2(
    a: wp.array2d(dtype=float),
    b: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    w, i = wp.tid()
    out[w, i] = a[w, i] + b[w, i]


@wp.kernel
def _add_damping_dense(
    opt_timestep: wp.array(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    qM_int: wp.array3d(dtype=float),  # in/out
):
    """qM_int += dt · diag(damping), Euler's implicit damping."""
    w, i = wp.tid()
    dt = opt_timestep[w % opt_timestep.shape[0]]
    qM_int[w, i, i] += dt * dof_damping[w % dof_damping.shape[0], i]


@wp.func
def _row_compliance(ctype: int, contact_compliance: float, joint_compliance: float) -> float:
    """XPBD compliance α of a row of this type (friction loss is rigid)."""
    if ctype == _CT_CONTACT_FRICTIONLESS or ctype == _CT_CONTACT_PYRAMIDAL:
        return contact_compliance
    if ctype == _CT_FRICTION_DOF or ctype == _CT_FRICTION_TENDON:
        return 0.0
    return joint_compliance


@wp.func
def _dynamic_body(geom: int, geom_bodyid: wp.array(dtype=int), body_weldid: wp.array(dtype=int)) -> int:
    """The body of ``geom``, or -1 if it has none or is welded to the world."""
    if geom < 0:
        return -1
    body = geom_bodyid[geom]
    if body_weldid[body] == 0:
        return -1
    return body


@wp.kernel
def _count_contact_rows_per_body(
    geom_bodyid: wp.array(dtype=int),
    body_weldid: wp.array(dtype=int),
    efc_type: wp.array2d(dtype=int),
    efc_id: wp.array2d(dtype=int),
    nefc: wp.array(dtype=int),
    contact_geom: wp.array(dtype=wp.vec2i),
    body_rows: wp.array2d(dtype=float),  # [out], zeroed by the caller
):
    """Count the active contact rows on each dynamic body."""
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return
    ctype = efc_type[worldid, efcid]
    if not (ctype == _CT_CONTACT_FRICTIONLESS or ctype == _CT_CONTACT_PYRAMIDAL):
        return
    geoms = contact_geom[efc_id[worldid, efcid]]
    for j in range(2):
        body = _dynamic_body(geoms[j], geom_bodyid, body_weldid)
        if body >= 0:
            wp.atomic_add(body_rows, worldid, body, 1.0)


@wp.kernel
def _efc_average_weight(
    relaxation: float,
    geom_bodyid: wp.array(dtype=int),
    body_weldid: wp.array(dtype=int),
    efc_type: wp.array2d(dtype=int),
    efc_id: wp.array2d(dtype=int),
    nefc: wp.array(dtype=int),
    contact_geom: wp.array(dtype=wp.vec2i),
    body_rows: wp.array2d(dtype=float),
    efc_omega: wp.array2d(dtype=float),  # [out]
):
    """Per-row Jacobi relaxation, averaged over the contact rows on the row's bodies.

    ω_e = relaxation / max(n_a, n_b) for contact rows, ``relaxation`` for the rest.
    """
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return
    ctype = efc_type[worldid, efcid]
    n = float(1.0)
    if ctype == _CT_CONTACT_FRICTIONLESS or ctype == _CT_CONTACT_PYRAMIDAL:
        geoms = contact_geom[efc_id[worldid, efcid]]
        for j in range(2):
            body = _dynamic_body(geoms[j], geom_bodyid, body_weldid)
            if body >= 0:
                n = wp.max(n, body_rows[worldid, body])
    efc_omega[worldid, efcid] = relaxation / n


@wp.kernel
def _efc_inv_mass_dense(
    nv: int,
    efc_J: wp.array3d(dtype=float),
    nefc: wp.array(dtype=int),
    L: wp.array3d(dtype=float),
    z: wp.array3d(dtype=float),       # scratch, one nv-vector per row
    efc_A: wp.array2d(dtype=float),   # [out]
):
    """A_e = J_e M̂⁻¹ J_eᵀ = |L⁻¹ J_eᵀ|², with M̂ = L Lᵀ a dense Cholesky factor.

    One forward substitution per row. L is block-diagonal by kinematic tree,
    and the entries between trees are zero, so the full lower triangle can be
    walked as is.
    """
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return
    a = float(0.0)
    for i in range(nv):
        s = efc_J[worldid, efcid, i]
        for k in range(i):
            s -= L[worldid, i, k] * z[worldid, efcid, k]
        zi = s / L[worldid, i, i]
        z[worldid, efcid, i] = zi
        a += zi * zi
    efc_A[worldid, efcid] = a


@wp.kernel
def _xpbd_unified_sweep(
    opt_timestep: wp.array(dtype=float),
    nv: int,
    contact_compliance: float,
    joint_compliance: float,
    efc_J: wp.array3d(dtype=float),
    efc_pos: wp.array2d(dtype=float),
    efc_A: wp.array2d(dtype=float),
    efc_omega: wp.array2d(dtype=float),
    efc_type: wp.array2d(dtype=int),
    efc_frictionloss: wp.array2d(dtype=float),
    nefc: wp.array(dtype=int),
    qvel_pred: wp.array2d(dtype=float),
    # in/out
    lambda_efc: wp.array2d(dtype=float),
    efc_force: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
):
    """One XPBD relaxation sweep over every active constraint row.

    The update and the per-type treatment are in the module docstring.
    """
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return

    ctype = efc_type[worldid, efcid]
    if ctype == _CT_CONTACT_ELLIPTIC:
        return

    dt = opt_timestep[worldid % opt_timestep.shape[0]]

    is_contact = ctype == _CT_CONTACT_FRICTIONLESS or ctype == _CT_CONTACT_PYRAMIDAL
    is_limit = ctype == _CT_LIMIT_JOINT or ctype == _CT_LIMIT_TENDON
    is_friction_loss = ctype == _CT_FRICTION_DOF or ctype == _CT_FRICTION_TENDON

    C_e = efc_pos[worldid, efcid]
    # Dry friction resists velocity; it is not a position constraint.
    if is_friction_loss:
        C_e = 0.0
    is_unilateral = is_contact or is_limit
    relax = efc_omega[worldid, efcid]

    alpha = _row_compliance(ctype, contact_compliance, joint_compliance) / (dt * dt)
    denom = efc_A[worldid, efcid] + alpha
    if denom <= 1.0e-12:
        # A row no free DOF can move, held rigidly.
        return

    v_e = float(0.0)
    for i in range(nv):
        v_e += efc_J[worldid, efcid, i] * qvel_pred[worldid, i]

    old_l = lambda_efc[worldid, efcid]
    d_lambda = -relax * (v_e + C_e / dt + alpha * old_l) / denom
    new_l = old_l + d_lambda

    if is_friction_loss:
        # frictionloss is a force bound and λ is an impulse, so the impulse
        # bound is frictionloss·dt.
        max_l = efc_frictionloss[worldid, efcid] * dt
        if max_l <= 0.0:
            new_l = 0.0
        elif new_l > max_l:
            new_l = max_l
        elif new_l < -max_l:
            new_l = -max_l
    elif is_unilateral and new_l < 0.0:
        new_l = 0.0

    d_lambda = new_l - old_l
    lambda_efc[worldid, efcid] = new_l

    if d_lambda != 0.0:
        d_f = d_lambda / dt
        for i in range(nv):
            wp.atomic_add(qfrc_constraint, worldid, i, efc_J[worldid, efcid, i] * d_f)

    # Diagnostic: the running total, in force units.
    efc_force[worldid, efcid] = new_l / dt


# -- config ------------------------------------------------------------------

@dataclass
class XPBDConfig(VectorizedMujocoConfig):
    """Physics parameters for the XPBD backend.

    Everything in ``VectorizedMujocoConfig``, plus XPBD's own settings. XPBD
    bypasses MuJoCo's solver, so the inherited ``solver``, ``iterations`` and
    ``tolerance`` have no effect, and neither do the solref and solimp
    settings: contact softness is ``contact_compliance``. They are all still
    written to the model.

    The two XPBD counts carry an ``xpbd_`` prefix. That keeps them separate
    from the inherited ``substeps``, which counts simulator steps per control
    step, and from ``iterations``, MuJoCo's solver iteration limit.

    Attributes:
        xpbd_substeps: Physics substeps within one ``timestep``. Each substep
            runs the whole pipeline (collision, constraint rows, solve,
            integrate) at ``timestep / xpbd_substeps``. More substeps give
            stiffer and more stable contact, and cost roughly linearly more.
        xpbd_iterations: Relaxation sweeps per substep. Each sweep after the
            first recomputes the predicted velocity from the constraint forces
            accumulated so far.
        relaxation: Multiplier on every row's Jacobi weight ``ω_e`` (see the
            module docstring). It sets how much of the remaining error each
            sweep removes; the compliant solution the sweeps converge to does
            not depend on it.
        contact_compliance: XPBD compliance of each contact row, in m/N (the
            inverse of a stiffness). 0 is rigid contact. A condim-3 contact has
            4 rows, so its normal stiffness is ``4 / contact_compliance``.
        joint_compliance: XPBD compliance of equality and joint/tendon limit
            rows, in the row's units per N (m/N or rad/(N·m)). 0 is rigid.
    """

    xpbd_substeps: int = 1
    xpbd_iterations: int = 2
    relaxation: float = 1.0
    contact_compliance: float = 0.0
    joint_compliance: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.xpbd_substeps < 1:
            raise ValueError(f"xpbd_substeps must be >= 1, got {self.xpbd_substeps}")
        if self.xpbd_iterations < 1:
            raise ValueError(f"xpbd_iterations must be >= 1, got {self.xpbd_iterations}")
        if self.relaxation <= 0:
            raise ValueError(f"relaxation must be > 0, got {self.relaxation}")
        if self.contact_compliance < 0:
            raise ValueError(
                f"contact_compliance must be >= 0, got {self.contact_compliance}"
            )
        if self.joint_compliance < 0:
            raise ValueError(f"joint_compliance must be >= 0, got {self.joint_compliance}")


# -- simulator ---------------------------------------------------------------

class XPBD(VectorizedMujoco):
    """``N`` parallel worlds on the GPU under XPBD contact.

    Has the same attributes as ``VectorizedMujoco``. ``m`` and ``d`` are plain
    MJWarp model and data. The XPBD scratch buffers live on the simulator
    itself.
    """

    def __init__(self, xml, sim_config: XPBDConfig | None = None, N: int = 1):
        """Compile a model, upload it, and allocate ``N`` worlds plus XPBD scratch.

        Args:
            xml: Path to an MJCF file, or the MJCF document itself.
            sim_config: Physics parameters. Defaults to ``XPBDConfig()``.
            N: Number of parallel worlds.

        Raises:
            NotImplementedError: If MJWarp stores the model sparse (``nv > 32``
                or ``jacobian="sparse"``), or its integrator is not Euler or
                implicitfast.
        """
        super().__init__(xml, sim_config if sim_config is not None else XPBDConfig(), N)
        m, d = self.m, self.d
        if m.is_sparse:
            raise NotImplementedError(
                f"the XPBD backend reads efc.J and the mass factor as dense, but MJWarp "
                f"stores this model sparse (nv={self.mjm.nv}, jacobian={self.mjm.opt.jacobian})"
            )
        if m.opt.integrator not in (IntegratorType.EULER, IntegratorType.IMPLICITFAST):
            raise NotImplementedError(
                f"integrator {m.opt.integrator} is not supported by the XPBD backend"
            )
        device = d.qpos.device
        nv = self.mjm.nv
        # One slot per constraint row, however many rows are active.
        self._njmax_pad = int(d.efc.J.shape[1])
        # Velocity predicted from the forces so far (substep-local).
        self._qvel_pred = wp.zeros((self.N, nv), dtype=float, device=device)
        # qfrc_smooth + qfrc_constraint.
        self._qfrc_total = wp.zeros((self.N, nv), dtype=float, device=device)
        # M̂⁻¹ times a force: the acceleration the integrator applies.
        self._qacc_int = wp.zeros((self.N, nv), dtype=float, device=device)
        # Per-row Lagrange multipliers (impulses), reset every substep.
        self._lambda_efc = wp.zeros((self.N, self._njmax_pad), dtype=float, device=device)
        # Per-row J M̂⁻¹ Jᵀ, and the forward-substitution scratch it needs.
        self._efc_A = wp.zeros((self.N, self._njmax_pad), dtype=float, device=device)
        self._efc_z = wp.zeros((self.N, self._njmax_pad, nv), dtype=float, device=device)
        # Per-row Jacobi relaxation, averaged over the contact rows on each body.
        self._efc_omega = wp.zeros((self.N, self._njmax_pad), dtype=float, device=device)
        self._body_rows = wp.zeros((self.N, self.mjm.nbody), dtype=float, device=device)
        # m.opt.timestep is patched down to the substep dt while stepping. The
        # original is kept here and restored afterwards.
        self._timestep_orig = wp.zeros(
            int(m.opt.timestep.shape[0]), dtype=float, device=device
        )

        # The integrator's velocity-update matrix M̂. It is M itself, and its
        # factor d.qLD, unless the integrator adds velocity derivatives.
        if m.opt.integrator == IntegratorType.EULER:
            self._int_kind = ("euler_damp" if self.mjm.dof_damping.any()
                              and not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER))
                              else "mass")
        else:
            self._int_kind = ("implicit" if (m.opt.disableflags & (DisableBit.ACTUATION
                              | DisableBit.SPRING | DisableBit.DAMPER))
                              != (DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)
                              else "mass")
        if self._int_kind == "mass":
            self._qM_int = self._qLD_int = self._qLDiagInv_int = None
        else:
            # Zeroed once: the dense factor only writes the per-tree blocks.
            self._qM_int = wp.zeros(d.qM.shape, dtype=float, device=device)
            self._qLD_int = wp.zeros(d.qLD.shape, dtype=float, device=device)
            self._qLDiagInv_int = wp.zeros(d.qLDiagInv.shape, dtype=float, device=device)

    # -- backend hooks -------------------------------------------------------
    def _stepPhysics(self) -> None:
        """Advance every world one ``timestep`` in ``xpbd_substeps`` substeps.

        Nothing here reads from the device, so the call can be captured. The
        timestep is patched and restored by kernels, not on the host.
        """
        m, d = self.m, self.d
        n_sub = self.config.xpbd_substeps
        n_ts = int(m.opt.timestep.shape[0])
        if n_sub > 1:
            wp.launch(_copy_1d, dim=n_ts, inputs=[m.opt.timestep, self._timestep_orig])
            wp.launch(_scale_1d, dim=n_ts,
                      inputs=[self._timestep_orig, 1.0 / float(n_sub)],
                      outputs=[m.opt.timestep])
        try:
            for _ in range(n_sub):
                self._substep()
        finally:
            if n_sub > 1:
                wp.launch(_copy_1d, dim=n_ts, inputs=[self._timestep_orig, m.opt.timestep])

        # Sensors once per full step, not once per substep.
        d.sensordata.zero_()
        sensor.sensor_pos(m, d)
        if m.opt.enableflags & EnableBit.ENERGY:
            if m.sensor_e_potential == 0:
                sensor.energy_pos(m, d)
        else:
            d.energy.zero_()
        sensor.sensor_vel(m, d)
        if m.opt.enableflags & EnableBit.ENERGY:
            if m.sensor_e_kinetic == 0:
                sensor.energy_vel(m, d)
        sensor.sensor_acc(m, d)

    def _forwardPhysics(self) -> None:
        """A single forward pass, with no substepping and no integration."""
        m, d = self.m, self.d
        self._positionPhase()

        d.sensordata.zero_()
        sensor.sensor_pos(m, d)
        if m.opt.enableflags & EnableBit.ENERGY:
            if m.sensor_e_potential == 0:
                sensor.energy_pos(m, d)
        else:
            d.energy.zero_()

        fwd_velocity(m, d)
        sensor.sensor_vel(m, d)
        if m.opt.enableflags & EnableBit.ENERGY:
            if m.sensor_e_kinetic == 0:
                sensor.energy_vel(m, d)

        self._actuationAndSolve()

    # -- pipeline ------------------------------------------------------------
    def _positionPhase(self) -> None:
        """Kinematics through constraint rows, in ComFree's forward order.

        Collision has to come after ``factor_m`` and before ``make_constraint``,
        which reads the contacts. It runs every substep. The old backend tried
        detecting collisions once per step and reusing them across substeps
        (Macklin §4.2), but that call returned zero contacts on this MJWarp
        build.
        """
        m, d = self.m, self.d
        smooth.kinematics(m, d)
        smooth.com_pos(m, d)
        smooth.camlight(m, d)
        smooth.flex(m, d)
        smooth.tendon(m, d)
        smooth.crb(m, d)
        smooth.tendon_armature(m, d)
        smooth.factor_m(m, d)
        if not (m.opt.disableflags & DisableBit.CONSTRAINT):
            collision_driver.collision(m, d)
        _mjw_constraint.make_constraint(m, d)
        smooth.transmission(m, d)

    def _actuationAndSolve(self) -> None:
        """Compute actuation and smooth acceleration, then run the XPBD solve."""
        m, d = self.m, self.d
        if not (m.opt.disableflags & DisableBit.ACTUATION):
            if m.callback.control:
                m.callback.control(m, d)
        fwd_actuation(m, d)
        fwd_acceleration(m, d, factorize=True)
        self._solve()

    def _substep(self) -> None:
        """One substep: position phase, velocity, solve, integrate.

        ``m.opt.timestep`` must already hold the substep dt.
        """
        m, d = self.m, self.d
        self._positionPhase()
        fwd_velocity(m, d)
        self._actuationAndSolve()
        # The solve already holds M̂⁻¹ · (total force), the acceleration Euler
        # or implicitfast would compute from it, so advance with that directly.
        _advance(m, d, self._qacc_int)

    def _factorIntegrationMatrix(self) -> tuple:
        """Factor M̂ for this substep and return ``(L, D⁻¹)`` for ``solve_LD``."""
        m, d = self.m, self.d
        if self._int_kind == "mass":
            return d.qLD, d.qLDiagInv
        if self._int_kind == "euler_damp":
            wp.copy(self._qM_int, d.qM)
            wp.launch(_add_damping_dense, dim=(self.N, m.nv),
                      inputs=[m.opt.timestep, m.dof_damping], outputs=[self._qM_int])
        else:
            derivative.deriv_smooth_vel(m, d, self._qM_int)
        # factor_solve_i also solves; the solve here is the one _solve needs next.
        smooth.factor_solve_i(m, d, self._qM_int, self._qLD_int, self._qLDiagInv_int,
                              self._qacc_int, d.qfrc_smooth)
        return self._qLD_int, self._qLDiagInv_int

    def _solve(self) -> None:
        """The XPBD constraint solve, replacing MJWarp's solver.

        1. Factor M̂ and predict qvel from the smooth forces alone:
           qvel + dt · M̂⁻¹ qfrc_smooth.
        2. Zero qfrc_constraint and λ, which the sweep accumulates into.
        3. Compute each row's J M̂⁻¹ Jᵀ, and its Jacobi weight from the
           number of contact rows on its bodies.
        4. Sweep ``xpbd_iterations`` times. Between sweeps, re-predict qvel
           from the constraint forces accumulated so far.
        5. Solve qacc = M⁻¹ (qfrc_smooth + qfrc_constraint), MuJoCo's qacc, and
           M̂⁻¹ of the same force, which the integrator applies.
        """
        m, d = self.m, self.d
        cfg = self.config
        nv, nw = m.nv, self.N
        njmax_pad = self._njmax_pad

        L, Dinv = self._factorIntegrationMatrix()
        if self._int_kind == "mass":
            wp.copy(self._qacc_int, d.qacc_smooth)
        wp.launch(_predict_qvel, dim=(nw, nv),
                  inputs=[m.opt.timestep, d.qvel, self._qacc_int],
                  outputs=[self._qvel_pred])

        d.qfrc_constraint.zero_()
        self._lambda_efc.zero_()

        if njmax_pad > 0:
            wp.launch(_efc_inv_mass_dense, dim=(nw, njmax_pad),
                      inputs=[nv, d.efc.J, d.nefc, L, self._efc_z],
                      outputs=[self._efc_A])
            self._body_rows.zero_()
            wp.launch(_count_contact_rows_per_body, dim=(nw, njmax_pad),
                      inputs=[m.geom_bodyid, m.body_weldid, d.efc.type, d.efc.id, d.nefc,
                              d.contact.geom],
                      outputs=[self._body_rows])
            wp.launch(_efc_average_weight, dim=(nw, njmax_pad),
                      inputs=[float(cfg.relaxation), m.geom_bodyid, m.body_weldid, d.efc.type,
                              d.efc.id, d.nefc, d.contact.geom, self._body_rows],
                      outputs=[self._efc_omega])

        n_iter = cfg.xpbd_iterations
        for it in range(n_iter):
            if njmax_pad > 0:
                wp.launch(
                    _xpbd_unified_sweep,
                    dim=(nw, njmax_pad),
                    inputs=[
                        m.opt.timestep, nv,
                        float(cfg.contact_compliance), float(cfg.joint_compliance),
                        d.efc.J, d.efc.pos, self._efc_A, self._efc_omega, d.efc.type,
                        d.efc.frictionloss, d.nefc,
                        self._qvel_pred,
                    ],
                    outputs=[self._lambda_efc, d.efc.force, d.qfrc_constraint],
                )
            # Skip after the last sweep: the solves below do this anyway.
            if it < n_iter - 1:
                wp.launch(_sum2, dim=(nw, nv),
                          inputs=[d.qfrc_smooth, d.qfrc_constraint],
                          outputs=[self._qfrc_total])
                smooth.solve_LD(m, d, L, Dinv, self._qacc_int, self._qfrc_total)
                wp.launch(_predict_qvel, dim=(nw, nv),
                          inputs=[m.opt.timestep, d.qvel, self._qacc_int],
                          outputs=[self._qvel_pred])

        wp.launch(_sum2, dim=(nw, nv),
                  inputs=[d.qfrc_smooth, d.qfrc_constraint],
                  outputs=[self._qfrc_total])
        smooth.solve_m(m, d, d.qacc, self._qfrc_total)
        if self._int_kind == "mass":
            wp.copy(self._qacc_int, d.qacc)
        else:
            smooth.solve_LD(m, d, L, Dinv, self._qacc_int, self._qfrc_total)

    # -- lifecycle -----------------------------------------------------------
    def Close(self) -> None:
        """Drop the device allocations, including the XPBD scratch buffers."""
        super().Close()
        self._qvel_pred = None
        self._qfrc_total = None
        self._qacc_int = None
        self._lambda_efc = None
        self._efc_A = None
        self._efc_z = None
        self._efc_omega = None
        self._body_rows = None
        self._timestep_orig = None
        self._qM_int = None
        self._qLD_int = None
        self._qLDiagInv_int = None
