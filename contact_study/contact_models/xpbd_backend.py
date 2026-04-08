"""XPBD-style decoupled contact model (M4).

Contact resolution follows Macklin et al., "Small Steps in Physics
Simulation" (SCA 2019), §4.3 and §4.4.

  Normal contact — hard unilateral constraint (NO compliance α).
    Per contact, accumulate λ_n with unilateral clamp λ_n ≥ 0:

        Δλ_n = −(v_n + C_n/dt) / w_n,    C_n = min(pos, 0)
        λ_n  ← max(λ_n + Δλ_n, 0)
        f_n  = (λ_n_new − λ_n_old) / dt        (velocity-level impulse)

  Friction — Coulomb box clamp (paper Eq. 11/12):

        Δλ_t  = −v_t / w_t
        λ_t  ← λ_t + Δλ_t
        (λ_t1, λ_t2) projected onto disk of radius μ · λ_n    (cone clamp)

    Tangent directions are recovered from the pyramidal-cone edge rows
    that MJWarp populates in efc_J.  Tangent effective mass is
    approximated as w_t ≈ 1/(2 μ² D_edge), which comes from
    J_± = J_n ± μ J_t ⇒ w_tt = (w_{++} − 2 w_{+−} + w_{−−})/(4μ²)
    under the (typical) assumption that the cross term w_{+−} is small.

  Non-contact constraints (equality, limits, joint/tendon friction)
    are delegated to MJWarp's native constraint solver.  This backend
    is NOT intended for soft-body / deformable simulation — it only
    overrides contact treatment.

Pipeline
--------
    kinematics → collision → make_constraint → factor_m →
    transmission → fwd_velocity → fwd_actuation → fwd_acceleration →
    **mjw_solver.solve**                (native solve, all rows)
    **remove_native_contact_qfrc**      (undo native contact forces)
    **xpbd_contact_solve**              (hard-contact XPBD)
    **solve_m**                         (final qacc)
    → sensor_acc → euler
"""

from __future__ import annotations

import mujoco
import warp as wp

from .config import XPBDParams

# ── MJWarp imports ───────────────────────────────────────────────
import comfree_warp.mujoco_warp as _mjw
from comfree_warp.mujoco_warp._src import collision_driver
from comfree_warp.mujoco_warp._src import constraint as _mjw_constraint
from comfree_warp.mujoco_warp._src import sensor
from comfree_warp.mujoco_warp._src import smooth
from comfree_warp.mujoco_warp._src import solver as _mjw_solver  # native constraint solver
from comfree_warp.mujoco_warp._src.forward import (
    euler,
    fwd_acceleration,
    fwd_actuation,
    fwd_velocity,
    implicit,
)
from comfree_warp.mujoco_warp._src.types import (
    ConstraintType,
    ContactType,
    DisableBit,
    EnableBit,
    IntegratorType,
    vec5,
)

wp.set_module_options({"enable_backward": False})


# ═══════════════════════════════════════════════════════════════════
# Warp kernels
# ═══════════════════════════════════════════════════════════════════

@wp.kernel
def _zero_array2d(a: wp.array2d(dtype=float)):
    worldid, i = wp.tid()
    a[worldid, i] = 0.0


@wp.kernel
def _predict_qvel(
    opt_timestep: wp.array(dtype=float),
    qvel:         wp.array2d(dtype=float),
    qacc:         wp.array2d(dtype=float),
    # out
    qvel_pred:    wp.array2d(dtype=float),
):
    """qvel_pred = qvel + qacc * dt.

    qacc here is post-native-solve (MJWarp's solution with non-contact
    rows applied and contact rows zeroed), so qvel_pred is the velocity
    the body would have if only equality/limit/friction-loss constraints
    were active — exactly the prediction the XPBD contact projection
    should correct.
    """
    worldid, dofid = wp.tid()
    dt = opt_timestep[worldid % opt_timestep.shape[0]]
    qvel_pred[worldid, dofid] = qvel[worldid, dofid] + qacc[worldid, dofid] * dt


# ── remove native contact contributions from qfrc ────────────────

@wp.kernel
def _remove_native_contact_qfrc(
    efc_J:      wp.array3d(dtype=float),
    efc_type:   wp.array2d(dtype=int),
    nefc:       wp.array(dtype=int),
    nv:         int,
    # in/out
    efc_force:       wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
):
    """Subtract J^T · f from qfrc_constraint for every contact efc row,
    then zero those efc_force entries.

    After this, qfrc_constraint contains only non-contact constraint
    forces (equality, limit, joint/tendon friction) as produced by the
    MJWarp native solver.
    """
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return

    ctype = efc_type[worldid, efcid]
    is_contact = (
        ctype == int(ConstraintType.CONTACT_PYRAMIDAL)
        or ctype == int(ConstraintType.CONTACT_FRICTIONLESS)
        or ctype == int(ConstraintType.CONTACT_ELLIPTIC)
    )
    if not is_contact:
        return

    f = efc_force[worldid, efcid]
    if f != 0.0:
        for i in range(nv):
            wp.atomic_sub(
                qfrc_constraint, worldid, i, efc_J[worldid, efcid, i] * f
            )
    efc_force[worldid, efcid] = 0.0


# ── XPBD hard contact + box-clamp Coulomb friction ───────────────

@wp.kernel
def _xpbd_contact(
    # model
    opt_timestep:     wp.array(dtype=float),
    nv:               int,
    # contact data
    contact_friction: wp.array(dtype=vec5),
    contact_worldid:  wp.array(dtype=int),
    contact_dim:      wp.array(dtype=int),
    contact_type:     wp.array(dtype=int),
    contact_efc_adr:  wp.array2d(dtype=int),   # (naconmax, 5)
    nacon:            wp.array(dtype=int),     # (1,)
    # constraint data
    efc_J:            wp.array3d(dtype=float),
    efc_pos:          wp.array2d(dtype=float),
    efc_D:            wp.array2d(dtype=float),
    nefc:             wp.array(dtype=int),
    qvel_pred:        wp.array2d(dtype=float),
    # accumulated lambdas (in-out, per contact, zeroed each step)
    lambda_n:         wp.array(dtype=float),   # (naconmax,)
    lambda_t1:        wp.array(dtype=float),
    lambda_t2:        wp.array(dtype=float),
    # out
    efc_force:        wp.array2d(dtype=float),
    qfrc_constraint:  wp.array2d(dtype=float),
):
    """Small Steps §4.3 + §4.4 — hard XPBD contact with cone-clamp friction.

    Operates in velocity-level impulse form (Δλ is an impulse; force =
    Δλ/dt).  Only the DELTA is scattered each iteration so accumulated
    forces don't compound across iterations.
    """
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    if not (contact_type[cid] & int(ContactType.CONSTRAINT)):
        return

    worldid = contact_worldid[cid]
    dt = opt_timestep[worldid % opt_timestep.shape[0]]

    normal_efc = contact_efc_adr[cid, 0]
    if normal_efc < 0 or normal_efc >= nefc[worldid]:
        return

    mu = contact_friction[cid][0]

    # ── normal effective mass w_n = 1/D ──────────────────────────
    D_n = efc_D[worldid, normal_efc]
    if D_n <= 1.0e-10:
        return
    w_n = 1.0 / D_n

    # ── normal constraint velocity ───────────────────────────────
    v_n = float(0.0)
    for i in range(nv):
        v_n += efc_J[worldid, normal_efc, i] * qvel_pred[worldid, i]

    # ── hard XPBD normal (no compliance) ─────────────────────────
    # paper Eq. (7) with α̃ = 0, velocity-level form
    C_n = wp.min(efc_pos[worldid, normal_efc], 0.0)
    d_lambda_n = -(v_n + C_n / dt) / w_n

    old_ln = lambda_n[cid]
    new_ln = wp.max(old_ln + d_lambda_n, 0.0)        # unilateral clamp
    d_lambda_n = new_ln - old_ln
    lambda_n[cid] = new_ln

    # scatter normal impulse
    d_fn = d_lambda_n / dt
    efc_force[worldid, normal_efc] = new_ln / dt     # total for diagnostics
    for i in range(nv):
        wp.atomic_add(
            qfrc_constraint, worldid, i, efc_J[worldid, normal_efc, i] * d_fn
        )

    # ── friction (paper §4.4) ────────────────────────────────────
    if mu < 1.0e-8 or contact_dim[cid] < 3:
        return

    e1p = contact_efc_adr[cid, 1]
    e1m = contact_efc_adr[cid, 2]
    if e1p < 0 or e1m < 0:
        return

    inv_2mu = 0.5 / mu

    # tangent basis from pyramidal edges: J_t = (J_+ − J_−) / (2μ)
    # effective mass:  w_t ≈ 1/(2 μ² D_edge)  (see module docstring)
    D_e1 = efc_D[worldid, e1p]
    if D_e1 <= 1.0e-10:
        return
    w_t1 = 1.0 / (2.0 * mu * mu * D_e1)

    # tangent-1 velocity
    vt1 = float(0.0)
    for i in range(nv):
        jt1_i = (efc_J[worldid, e1p, i] - efc_J[worldid, e1m, i]) * inv_2mu
        vt1 += jt1_i * qvel_pred[worldid, i]

    # tangent-2 (if available)
    e2p = contact_efc_adr[cid, 3]
    e2m = contact_efc_adr[cid, 4]
    has_t2 = e2p >= 0 and e2m >= 0

    vt2 = float(0.0)
    w_t2 = float(0.0)
    if has_t2:
        D_e2 = efc_D[worldid, e2p]
        if D_e2 > 1.0e-10:
            w_t2 = 1.0 / (2.0 * mu * mu * D_e2)
            for i in range(nv):
                jt2_i = (efc_J[worldid, e2p, i] - efc_J[worldid, e2m, i]) * inv_2mu
                vt2 += jt2_i * qvel_pred[worldid, i]
        else:
            has_t2 = False

    # Δλ_t = −v_t / w_t  (velocity-level XPBD, zero-slip target)
    d_lambda_t1 = -vt1 / w_t1
    d_lambda_t2 = float(0.0)
    if has_t2:
        d_lambda_t2 = -vt2 / w_t2

    # accumulate then cone-clamp: ‖λ_t‖ ≤ μ · λ_n
    new_lt1 = lambda_t1[cid] + d_lambda_t1
    new_lt2 = lambda_t2[cid] + d_lambda_t2

    lt_mag = wp.sqrt(new_lt1 * new_lt1 + new_lt2 * new_lt2)
    lt_max = mu * new_ln
    if lt_mag > lt_max and lt_mag > 1.0e-12:
        scale = lt_max / lt_mag
        new_lt1 *= scale
        new_lt2 *= scale

    d_lambda_t1 = new_lt1 - lambda_t1[cid]
    d_lambda_t2 = new_lt2 - lambda_t2[cid]
    lambda_t1[cid] = new_lt1
    lambda_t2[cid] = new_lt2

    # scatter tangent impulses as force
    d_ft1 = d_lambda_t1 / dt
    d_ft2 = d_lambda_t2 / dt
    for i in range(nv):
        jt1_i = (efc_J[worldid, e1p, i] - efc_J[worldid, e1m, i]) * inv_2mu
        frc_i = jt1_i * d_ft1
        if has_t2:
            jt2_i = (efc_J[worldid, e2p, i] - efc_J[worldid, e2m, i]) * inv_2mu
            frc_i += jt2_i * d_ft2
        wp.atomic_add(qfrc_constraint, worldid, i, frc_i)


# ── final qfrc assembly ───────────────────────────────────────────

@wp.kernel
def _sum_qfrc(
    qfrc_smooth:     wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
    qfrc_total:      wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    qfrc_total[worldid, dofid] = (
        qfrc_smooth[worldid, dofid] + qfrc_constraint[worldid, dofid]
    )


# ═══════════════════════════════════════════════════════════════════
# Model / Data wrappers
# ═══════════════════════════════════════════════════════════════════

class XPBDModel:
    """Wraps a MJWarp Model with XPBD parameters."""

    def __init__(self, mjw_model, xpbd_params: XPBDParams):
        self._m = mjw_model
        self.xpbd_params = xpbd_params
        self.contact_cfg = None  # set by api.put_model

    def __getattr__(self, name):
        return getattr(self._m, name)


class XPBDData:
    """Wraps a MJWarp Data with scratch buffers for the XPBD contact solve."""

    def __init__(self, mjw_data, nworld: int, nv: int, naconmax: int):
        self._d = mjw_data
        device = mjw_data.qpos.device
        self.qvel_pred       = wp.zeros((nworld, nv), dtype=float, device=device)
        self.qfrc_total      = wp.zeros((nworld, nv), dtype=float, device=device)
        # per-contact accumulated lambdas (zeroed each step)
        self.lambda_n        = wp.zeros(naconmax, dtype=float, device=device)
        self.lambda_t1       = wp.zeros(naconmax, dtype=float, device=device)
        self.lambda_t2       = wp.zeros(naconmax, dtype=float, device=device)

    def __getattr__(self, name):
        return getattr(self._d, name)


# ═══════════════════════════════════════════════════════════════════
# Public backend surface
# ═══════════════════════════════════════════════════════════════════

def put_model(mjm: mujoco.MjModel, xpbd_params: XPBDParams) -> XPBDModel:
    mjw_m = _mjw.put_model(mjm)
    return XPBDModel(mjw_m, xpbd_params)


def make_data(
    mjm: mujoco.MjModel,
    m: XPBDModel,
    nworld: int = 1,
    nconmax: int | None = None,
    njmax: int | None = None,
) -> XPBDData:
    kw = dict(nworld=nworld)
    if nconmax is not None:
        kw["nconmax"] = nconmax
    if njmax is not None:
        kw["njmax"] = njmax
    mjw_d = _mjw.make_data(mjm, **kw)
    naconmax = getattr(mjw_d, "naconmax", (nconmax or mjm.nconmax) * nworld)
    return XPBDData(mjw_d, nworld, mjm.nv, naconmax)


def put_data(
    mjm: mujoco.MjModel,
    mjd: mujoco.MjData,
    m: XPBDModel,
    nworld: int = 1,
    nconmax: int | None = None,
    njmax: int | None = None,
) -> XPBDData:
    kw = dict(nworld=nworld)
    if nconmax is not None:
        kw["nconmax"] = nconmax
    if njmax is not None:
        kw["njmax"] = njmax
    mjw_d = _mjw.put_data(mjm, mjd, **kw)
    naconmax = getattr(mjw_d, "naconmax", (nconmax or mjm.nconmax) * nworld)
    return XPBDData(mjw_d, nworld, mjm.nv, naconmax)


def get_data_into(
    mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData, mjd: mujoco.MjData
):
    return _mjw.get_data_into(mjm, m._m, d._d, mjd)


def reset_data(mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData):
    return _mjw.reset_data(mjm, m._m, d._d)


# ═══════════════════════════════════════════════════════════════════
# Constraint solve  (native non-contact + XPBD contact)
# ═══════════════════════════════════════════════════════════════════

def _xpbd_solve(m: XPBDModel, d: XPBDData):
    """Hybrid non-contact-native / contact-XPBD constraint solve.

    Steps:
      1. Native MJWarp solve.  This populates inner_d.qacc, inner_d.efc.force
         and inner_d.qfrc_constraint for ALL rows (eq + limit + friction_loss
         + contact + contact-friction edges).
      2. Remove the native solver's contact contributions from
         qfrc_constraint by subtracting J^T·f for every contact row, then
         zero those efc.force entries.  After this, qfrc_constraint reflects
         only non-contact constraint forces.
      3. Rebuild qacc for the "no contacts yet" state and derive qvel_pred
         from it — this is the velocity the body would have under gravity,
         actuation, and NON-contact constraints alone.
      4. Run the XPBD contact kernel with that qvel_pred, adding hard-contact
         + cone-clamp friction impulses to qfrc_constraint.
      5. Caller (forward) will do the final solve_m to get post-contact qacc.
    """
    inner_m = m._m
    inner_d = d._d
    params  = m.xpbd_params
    nv = inner_m.nv
    nw = inner_d.nworld

    # ── 1. native solve (handles every constraint type) ─────────
    _mjw_solver.solve(inner_m, inner_d)

    # The MJWarp native solver internally factorizes the *constrained*
    # system matrix A = M + Jᵀ R J into qLD / qLDiagInv as part of its
    # Newton / CG updates, clobbering the clean M factorization that
    # fwd_acceleration produced.  Any subsequent `smooth.solve_m` would
    # therefore NOT compute M⁻¹ — it would invert whatever the native
    # solver left behind (observed symptom: qacc exploding to ~1e6 when
    # qfrc_total is ~10, i.e. an "inverse mass" of ~1e5).  Restore a
    # clean factorization of M before we do any of our own solve_m calls.
    smooth.factor_m(inner_m, inner_d)

    # ── 2. strip contact contributions from qfrc_constraint ─────
    njmax_pad = inner_d.efc.J.shape[1]
    wp.launch(
        _remove_native_contact_qfrc,
        dim=(nw, njmax_pad),
        inputs=[
            inner_d.efc.J,
            inner_d.efc.type,
            inner_d.nefc,
            nv,
        ],
        outputs=[inner_d.efc.force, inner_d.qfrc_constraint],
    )

    # ── 3. qacc ← M⁻¹ (qfrc_smooth + qfrc_constraint_noncon) ────
    wp.launch(
        _sum_qfrc,
        dim=(nw, nv),
        inputs=[inner_d.qfrc_smooth, inner_d.qfrc_constraint],
        outputs=[d.qfrc_total],
    )
    # solve_m wants arrays that live on inner_d; reuse the scratch slot.
    wp.copy(inner_d.qfrc_constraint, d.qfrc_total)
    smooth.solve_m(inner_m, inner_d, inner_d.qacc, inner_d.qfrc_constraint)

    # qvel_pred = qvel + qacc · dt  (no-contact prediction)
    wp.launch(
        _predict_qvel,
        dim=(nw, nv),
        inputs=[inner_m.opt.timestep, inner_d.qvel, inner_d.qacc],
        outputs=[d.qvel_pred],
    )

    # Reset qfrc_constraint to contain only NON-contact forces again.
    # (We clobbered it above with qfrc_total to feed solve_m.)
    # Easiest: recompute qfrc_constraint = qfrc_total − qfrc_smooth.
    # But we don't need it — the XPBD kernel only ADDS to qfrc_constraint
    # and we'll reassemble qfrc_total from scratch afterwards.  So zero it.
    wp.launch(_zero_array2d, dim=(nw, nv), inputs=[inner_d.qfrc_constraint])

    # We still need the non-contact forces to add to qfrc_smooth later.
    # Derive them as (qfrc_total − qfrc_smooth) and stash in d.qfrc_total.
    # d.qfrc_total currently holds (qfrc_smooth + qfrc_constraint_noncon),
    # which is exactly what we want added to the contact forces below.

    # ── 4. XPBD hard-contact + box-clamp friction ───────────────
    # Zero per-contact accumulators at start of step.
    d.lambda_n.zero_()
    d.lambda_t1.zero_()
    d.lambda_t2.zero_()

    # (Optionally loop iterations; paper recommends 1 per substep.)
    for _iter in range(max(1, params.iterations)):
        if inner_d.naconmax > 0:
            wp.launch(
                _xpbd_contact,
                dim=inner_d.naconmax,
                inputs=[
                    inner_m.opt.timestep,
                    nv,
                    inner_d.contact.friction,
                    inner_d.contact.worldid,
                    inner_d.contact.dim,
                    inner_d.contact.type,
                    inner_d.contact.efc_address,
                    inner_d.nacon,
                    inner_d.efc.J,
                    inner_d.efc.pos,
                    inner_d.efc.D,
                    inner_d.nefc,
                    d.qvel_pred,
                    d.lambda_n,
                    d.lambda_t1,
                    d.lambda_t2,
                ],
                outputs=[inner_d.efc.force, inner_d.qfrc_constraint],
            )

    # ── 5. final qfrc_total = (smooth + non-contact) + contact ──
    # d.qfrc_total already = qfrc_smooth + qfrc_constraint_noncon,
    # and inner_d.qfrc_constraint now holds ONLY the XPBD contact forces.
    wp.launch(
        _sum_qfrc,
        dim=(nw, nv),
        inputs=[d.qfrc_total, inner_d.qfrc_constraint],   # noncon_total + contact
        outputs=[d.qfrc_total],
    )


# ═══════════════════════════════════════════════════════════════════
# Forward / Step
# ═══════════════════════════════════════════════════════════════════

def forward(m: XPBDModel, d: XPBDData):
    """Full forward pass: native non-contact solve + XPBD contact."""
    inner_m = m._m
    inner_d = d._d

    # ── forward position ─────────────────────────────────────────
    smooth.kinematics(inner_m, inner_d)
    smooth.com_pos(inner_m, inner_d)
    smooth.camlight(inner_m, inner_d)
    smooth.flex(inner_m, inner_d)
    smooth.tendon(inner_m, inner_d)
    smooth.crb(inner_m, inner_d)
    smooth.tendon_armature(inner_m, inner_d)
    smooth.factor_m(inner_m, inner_d)

    if not (inner_m.opt.disableflags & DisableBit.CONSTRAINT):
        collision_driver.collision(inner_m, inner_d)

    _mjw_constraint.make_constraint(inner_m, inner_d)
    smooth.transmission(inner_m, inner_d)

    inner_d.sensordata.zero_()
    sensor.sensor_pos(inner_m, inner_d)

    if inner_m.opt.enableflags & EnableBit.ENERGY:
        if inner_m.sensor_e_potential == 0:
            sensor.energy_pos(inner_m, inner_d)
    else:
        inner_d.energy.zero_()

    # ── forward velocity ─────────────────────────────────────────
    fwd_velocity(inner_m, inner_d)
    sensor.sensor_vel(inner_m, inner_d)

    if inner_m.opt.enableflags & EnableBit.ENERGY:
        if inner_m.sensor_e_kinetic == 0:
            sensor.energy_vel(inner_m, inner_d)

    # ── actuation + smooth acceleration ──────────────────────────
    if not (inner_m.opt.disableflags & DisableBit.ACTUATION):
        if inner_m.callback.control:
            inner_m.callback.control(inner_m, inner_d)
    fwd_actuation(inner_m, inner_d)
    # factorize=True matches comfree_warp's forward_comfree.  factor_m is
    # idempotent on already-factored qLD, so the second factorization here
    # is harmless (a small waste).  My earlier "double factor is a bug"
    # diagnosis was wrong.
    fwd_acceleration(inner_m, inner_d, factorize=True)

    # ── constraint solve (native non-contact + XPBD contact) ────
    if inner_d.njmax == 0 or inner_m.nv == 0:
        wp.copy(inner_d.qacc, inner_d.qacc_smooth)
    else:
        _xpbd_solve(m, d)
        # d.qfrc_total = qfrc_smooth + qfrc_noncontact + qfrc_contact.
        # solve_m reads from a real Data field:
        wp.copy(inner_d.qfrc_constraint, d.qfrc_total)
        smooth.solve_m(inner_m, inner_d, inner_d.qacc, inner_d.qfrc_constraint)

    sensor.sensor_acc(inner_m, inner_d)


def step(m: XPBDModel, d: XPBDData):
    """Advance one timestep.  Dispatches on m.opt.integrator to match
    comfree_warp's step_comfree behaviour.
    """
    inner_m = m._m
    inner_d = d._d

    forward(m, d)

    # Both EULER and IMPLICITFAST integrators read efc.Ma; comfree stashes
    # qfrc_total there unconditionally before integrating.
    wp.copy(inner_d.efc.Ma, d.qfrc_total)

    if inner_m.opt.integrator == IntegratorType.EULER:
        euler(inner_m, inner_d)
    elif inner_m.opt.integrator == IntegratorType.IMPLICITFAST:
        implicit(inner_m, inner_d)
    else:
        raise NotImplementedError(
            f"integrator {inner_m.opt.integrator} not supported by xpbd backend"
        )