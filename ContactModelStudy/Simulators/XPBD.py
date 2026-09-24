"""XPBD-style contact on MJWarp, as a ``VectorizedSimulator``.

This backend keeps MJWarp's collision detection and constraint rows, but
replaces its constraint solver with one XPBD relaxation sweep. The sweep covers
every active row: equality, limit, friction loss and contact. Like ComFree, it
runs on MJWarp's model and data layout, so this simulator is
``VectorizedMujoco`` with the step and forward replaced. Controls, state,
control sequences and CUDA-graph capture are all inherited.

Ported from ``contact_study/contact_models/xpbd_backend.py``. The kernels and
the order of the pipeline are unchanged.

Per-row update, one parallel launch over all rows::

    v_e   = J_e · qvel_pred                       # row velocity
    denom = 1 / efc_D[e]  = J_e M⁻¹ J_eᵀ + R      # effective mass + regularizer
    C_e   = efc_pos[e]                            # row residual
    Δλ    = -relax · (v_e + C_e/dt) / denom
    λ_new = clamp(λ_old + Δλ, [lo, hi])           # bounds depend on row type
    qfrc += J_eᵀ · (Δλ / dt)

Row types:

  EQUALITY ............... bilateral, no clamp,              relax = 1
  LIMIT_JOINT/TENDON ..... unilateral (λ ≥ 0),               relax = 1
  FRICTION_DOF/TENDON .... box clamp ±frictionloss·dt, no C,  relax = 1
  CONTACT_FRICTIONLESS ... unilateral, vmax cap, adaptive SOR
  CONTACT_PYRAMIDAL ...... unilateral, vmax cap, adaptive SOR
  CONTACT_ELLIPTIC ....... skipped (MJWarp is pyramidal-only anyway)

Why contact rows get their own relaxation: many contact rows act on the same
body. A pyramidal cone has 4 edge rows per contact, and a grasp can have
several contacts on the cube. In a parallel Jacobi sweep, every row corrects
the same error at once, so the combined correction is too large by roughly the
number of rows. Each contact row is therefore divided by the number of contact
rows on its most-loaded DOF. Non-contact rows rarely share a DOF this way, so
they use relax = 1.

Pyramidal Coulomb friction: for condim=3, MJWarp emits 4 edge rows
``J_n ± μ J_t``. Keeping each one unilateral (λ ≥ 0) enforces ``|λ_t| ≤ μ λ_n``
through the row arithmetic, with no explicit Coulomb clamp.
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
from comfree_warp.mujoco_warp._src import sensor  # noqa: E402
from comfree_warp.mujoco_warp._src import smooth  # noqa: E402
from comfree_warp.mujoco_warp._src.forward import (  # noqa: E402
    euler,
    fwd_acceleration,
    fwd_actuation,
    fwd_velocity,
    implicit,
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
def _count_contact_rows_per_dof(
    nv: int,
    efc_J: wp.array3d(dtype=float),
    efc_type: wp.array2d(dtype=int),
    nefc: wp.array(dtype=int),
    dof_con_count: wp.array2d(dtype=float),  # [out]
):
    """For every DOF, count the contact rows whose Jacobian touches it."""
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return
    ctype = efc_type[worldid, efcid]
    if not (ctype == _CT_CONTACT_FRICTIONLESS or ctype == _CT_CONTACT_PYRAMIDAL):
        return
    for i in range(nv):
        if wp.abs(efc_J[worldid, efcid, i]) > 1.0e-9:
            wp.atomic_add(dof_con_count, worldid, i, 1.0)


@wp.kernel
def _xpbd_unified_sweep(
    opt_timestep: wp.array(dtype=float),
    nv: int,
    vmax_depen: float,
    relax_contact: float,
    efc_J: wp.array3d(dtype=float),
    efc_pos: wp.array2d(dtype=float),
    efc_D: wp.array2d(dtype=float),
    efc_type: wp.array2d(dtype=int),
    efc_frictionloss: wp.array2d(dtype=float),
    nefc: wp.array(dtype=int),
    qvel_pred: wp.array2d(dtype=float),
    dof_con_count: wp.array2d(dtype=float),
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

    D_e = efc_D[worldid, efcid]
    if D_e <= 1.0e-10:
        return
    denom = 1.0 / D_e

    v_e = float(0.0)
    for i in range(nv):
        v_e += efc_J[worldid, efcid, i] * qvel_pred[worldid, i]

    dt = opt_timestep[worldid % opt_timestep.shape[0]]

    is_contact = ctype == _CT_CONTACT_FRICTIONLESS or ctype == _CT_CONTACT_PYRAMIDAL
    is_limit = ctype == _CT_LIMIT_JOINT or ctype == _CT_LIMIT_TENDON
    is_friction_loss = ctype == _CT_FRICTION_DOF or ctype == _CT_FRICTION_TENDON

    C_e = efc_pos[worldid, efcid]
    relax = float(1.0)
    is_unilateral = False

    if is_contact:
        # vmax cap (Macklin §4.3): remove at most vmax·dt of penetration
        # per substep.
        if C_e < -vmax_depen * dt:
            C_e = -vmax_depen * dt
        # Also clamp the row velocity. Without this, a large relative
        # velocity (e.g. from an unstable rollout) gives an extreme Δλ, which
        # can drive qfrc_constraint and then qvel/qpos to NaN.
        if v_e > vmax_depen:
            v_e = vmax_depen
        elif v_e < -vmax_depen:
            v_e = -vmax_depen
        # Adaptive SOR: divide by the number of contact rows on this row's
        # most-loaded DOF, to undo the Jacobi over-correction.
        # relax_contact is left as a safety multiplier on top.
        n_red = float(1.0)
        for i in range(nv):
            if wp.abs(efc_J[worldid, efcid, i]) > 1.0e-9:
                c = dof_con_count[worldid, i]
                if c > n_red:
                    n_red = c
        relax = relax_contact / n_red
        is_unilateral = True
    elif is_limit:
        is_unilateral = True
    elif is_friction_loss:
        # Dry friction resists velocity; it is not a position constraint.
        C_e = 0.0

    d_lambda = -relax * (v_e + C_e / dt) / denom

    old_l = lambda_efc[worldid, efcid]
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

    Everything in ``VectorizedMujocoConfig``, plus XPBD's own four settings.
    XPBD bypasses MuJoCo's solver, so the inherited ``solver``, ``iterations``
    and ``tolerance`` have no effect. They are still written to the model. The
    solimp settings do matter, because they set ``efc_D``, the regularized
    effective mass each row divides by. The solref settings do not: XPBD
    corrects the raw ``efc_pos`` and never uses MuJoCo's reference
    acceleration.

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
            accumulated so far. Values above 1 tend to oscillate with
            unilateral clamps instead of converging; prefer ``xpbd_substeps``
            for refinement.
        relaxation: Safety multiplier on the contact-row relaxation. It is
            applied after the adaptive division by rows per DOF. Too low gives
            soft, mushy contact; too high throws bodies.
        vmax_depenetration: Speed cap in m/s. It limits how fast existing
            penetration is removed, and the contact-row velocity used in the
            update. Without it, a large initial overlap at small ``dt`` pushes
            bodies apart explosively.
    """

    xpbd_substeps: int = 1
    xpbd_iterations: int = 2
    relaxation: float = 0.1
    vmax_depenetration: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.xpbd_substeps < 1:
            raise ValueError(f"xpbd_substeps must be >= 1, got {self.xpbd_substeps}")
        if self.xpbd_iterations < 1:
            raise ValueError(f"xpbd_iterations must be >= 1, got {self.xpbd_iterations}")
        if self.relaxation <= 0:
            raise ValueError(f"relaxation must be > 0, got {self.relaxation}")
        if self.vmax_depenetration <= 0:
            raise ValueError(
                f"vmax_depenetration must be > 0, got {self.vmax_depenetration}"
            )


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
        """
        super().__init__(xml, sim_config if sim_config is not None else XPBDConfig(), N)
        device = self.d.qpos.device
        nv = self.mjm.nv
        # One slot per constraint row, however many rows are active.
        self._njmax_pad = int(self.d.efc.J.shape[1])
        # Velocity predicted from the smooth dynamics only (substep-local).
        self._qvel_pred = wp.zeros((self.N, nv), dtype=float, device=device)
        # qfrc_smooth + qfrc_constraint, passed to solve_m and the integrators.
        self._qfrc_total = wp.zeros((self.N, nv), dtype=float, device=device)
        # Per-row Lagrange multipliers, reset every substep.
        self._lambda_efc = wp.zeros((self.N, self._njmax_pad), dtype=float, device=device)
        # Contact rows touching each DOF, for the adaptive SOR.
        self._dof_con_count = wp.zeros((self.N, nv), dtype=float, device=device)
        # m.opt.timestep is patched down to the substep dt while stepping. The
        # original is kept here and restored afterwards.
        self._timestep_orig = wp.zeros(
            int(self.m.opt.timestep.shape[0]), dtype=float, device=device
        )

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

        if d.njmax == 0 or m.nv == 0:
            wp.copy(d.qacc, d.qacc_smooth)
            wp.copy(self._qfrc_total, d.qfrc_smooth)
        else:
            self._solve()

    def _substep(self) -> None:
        """One substep: position phase, velocity, solve, integrate.

        ``m.opt.timestep`` must already hold the substep dt.
        """
        m, d = self.m, self.d
        self._positionPhase()
        fwd_velocity(m, d)
        self._actuationAndSolve()

        # Both integrators read the total force from efc.Ma, as ComFree does.
        wp.copy(d.efc.Ma, self._qfrc_total)
        if m.opt.integrator == IntegratorType.EULER:
            euler(m, d)
        elif m.opt.integrator == IntegratorType.IMPLICITFAST:
            implicit(m, d)
        else:
            raise NotImplementedError(
                f"integrator {m.opt.integrator} is not supported by the XPBD backend"
            )

    def _solve(self) -> None:
        """The XPBD constraint solve, replacing MJWarp's solver.

        1. Predict qvel from the smooth dynamics alone: qvel + dt · qacc_smooth.
        2. Zero qfrc_constraint and λ, which the sweep accumulates into.
        3. Count the contact rows on each DOF, for the adaptive SOR.
        4. Sweep ``xpbd_iterations`` times. Between sweeps, re-predict qvel
           from the constraint forces accumulated so far.
        5. Solve qacc = M⁻¹ (qfrc_smooth + qfrc_constraint).
        """
        m, d = self.m, self.d
        cfg = self.config
        nv, nw = m.nv, self.N
        njmax_pad = self._njmax_pad

        wp.launch(_predict_qvel, dim=(nw, nv),
                  inputs=[m.opt.timestep, d.qvel, d.qacc_smooth],
                  outputs=[self._qvel_pred])

        d.qfrc_constraint.zero_()
        self._lambda_efc.zero_()

        self._dof_con_count.zero_()
        if njmax_pad > 0:
            wp.launch(_count_contact_rows_per_dof, dim=(nw, njmax_pad),
                      inputs=[nv, d.efc.J, d.efc.type, d.nefc],
                      outputs=[self._dof_con_count])

        n_iter = cfg.xpbd_iterations
        for it in range(n_iter):
            if njmax_pad > 0:
                wp.launch(
                    _xpbd_unified_sweep,
                    dim=(nw, njmax_pad),
                    inputs=[
                        m.opt.timestep, nv,
                        float(cfg.vmax_depenetration), float(cfg.relaxation),
                        d.efc.J, d.efc.pos, d.efc.D, d.efc.type,
                        d.efc.frictionloss, d.nefc,
                        self._qvel_pred, self._dof_con_count,
                    ],
                    outputs=[self._lambda_efc, d.efc.force, d.qfrc_constraint],
                )
            # Skip after the last sweep: the solve_m below does this anyway.
            if it < n_iter - 1:
                wp.launch(_sum2, dim=(nw, nv),
                          inputs=[d.qfrc_smooth, d.qfrc_constraint],
                          outputs=[self._qfrc_total])
                smooth.solve_m(m, d, d.qacc, self._qfrc_total)
                wp.launch(_predict_qvel, dim=(nw, nv),
                          inputs=[m.opt.timestep, d.qvel, d.qacc],
                          outputs=[self._qvel_pred])

        wp.launch(_sum2, dim=(nw, nv),
                  inputs=[d.qfrc_smooth, d.qfrc_constraint],
                  outputs=[self._qfrc_total])
        smooth.solve_m(m, d, d.qacc, self._qfrc_total)

    # -- lifecycle -----------------------------------------------------------
    def Close(self) -> None:
        """Drop the device allocations, including the XPBD scratch buffers."""
        super().Close()
        self._qvel_pred = None
        self._qfrc_total = None
        self._lambda_efc = None
        self._dof_con_count = None
        self._timestep_orig = None
