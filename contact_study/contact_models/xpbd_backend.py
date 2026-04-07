"""XPBD-style decoupled contact model (M4).

Contact resolution:
  Normal — XPBD position-level constraint projection with compliance α.
    Uses efc_J from MJWarp's ``make_constraint`` and the constraint mass
    efc.D for correct articulated-body effective mass.  Accumulated impulse
    clamping λ ≥ 0 enforces unilateral contacts.

  Friction — Regularised Coulomb (velocity-proportional + Coulomb cap):
    p_t = −μ · λ_n · v_t / √(‖v_t‖² + ε)
    Tangent Jacobians are recovered from the pyramidal-cone edge rows
    that MJWarp already populates in efc_J.

  Non-contact constraints (equality, limits, joint/tendon friction) —
    Same XPBD compliance projection, but BILATERAL (no unilateral clamp).

Architecture:
  Pure MJWarp base — no comfree_core dependency.
  Pipeline: kinematics → collision → make_constraint → factor_m →
    transmission → fwd_velocity → fwd_actuation → fwd_acceleration →
    **xpbd_solve** → smooth.solve_m → sensor_acc

The put_model / make_data / step surface matches the api.py dispatch.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp

from .config import XPBDParams

# ── MJWarp imports ───────────────────────────────────────────────
import comfree_warp.mujoco_warp as _mjw
from comfree_warp.mujoco_warp._src import collision_driver
from comfree_warp.mujoco_warp._src import constraint as _mjw_constraint
from comfree_warp.mujoco_warp._src import sensor
from comfree_warp.mujoco_warp._src import smooth
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
    vec5,
)

wp.set_module_options({"enable_backward": False})

_FRICTION_EPS = float(1e-6)


# ═══════════════════════════════════════════════════════════════════
# Warp kernels
# ═══════════════════════════════════════════════════════════════════

@wp.kernel
def _predict_qvel_and_zero(
    opt_timestep:     wp.array(dtype=float),
    qvel:             wp.array2d(dtype=float),
    qacc_smooth:      wp.array2d(dtype=float),
    # out
    qvel_pred:        wp.array2d(dtype=float),
    qfrc_constraint:  wp.array2d(dtype=float),
):
    """qvel_pred = qvel + qacc_smooth * dt;  zero qfrc_constraint."""
    worldid, dofid = wp.tid()
    dt = opt_timestep[worldid % opt_timestep.shape[0]]
    qvel_pred[worldid, dofid] = (
        qvel[worldid, dofid] + qacc_smooth[worldid, dofid] * dt
    )
    qfrc_constraint[worldid, dofid] = 0.0


# ── XPBD for all non-contact constraint rows (bilateral) ─────────

@wp.kernel
def _xpbd_noncontact(
    opt_timestep:     wp.array(dtype=float),
    efc_J:            wp.array3d(dtype=float),   # (nworld, njmax_pad, nv)
    efc_pos:          wp.array2d(dtype=float),   # signed distance
    efc_D:            wp.array2d(dtype=float),   # constraint mass
    efc_type:         wp.array2d(dtype=int),
    qvel_pred:        wp.array2d(dtype=float),
    nv:               int,
    nefc:             wp.array(dtype=int),
    compliance:       float,
    # out
    efc_force:        wp.array2d(dtype=float),
    qfrc_constraint:  wp.array2d(dtype=float),
):
    """XPBD compliance projection for equality / limit / friction-loss rows.

    Bilateral: force can be positive or negative.
    Limit rows are unilateral (clamped ≥ 0).
    """
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return

    ctype = efc_type[worldid, efcid]

    # Skip contact rows — handled by _xpbd_contact kernel
    is_contact = (
        ctype == int(ConstraintType.CONTACT_PYRAMIDAL)
        or ctype == int(ConstraintType.CONTACT_FRICTIONLESS)
        or ctype == int(ConstraintType.CONTACT_ELLIPTIC)
    )
    if is_contact:
        return

    dt = opt_timestep[worldid % opt_timestep.shape[0]]

    # Constraint-space velocity
    efc_vel = float(0.0)
    for i in range(nv):
        efc_vel += efc_J[worldid, efcid, i] * qvel_pred[worldid, i]

    # XPBD: Δλ = -(v_c + pos/dt) / (1/D + α̃),   α̃ = α/dt²
    alpha_tilde = compliance / (dt * dt)
    D = efc_D[worldid, efcid]
    w_inv = float(0.0)
    if D > 1.0e-10:
        w_inv = 1.0 / D

    c = efc_vel + efc_pos[worldid, efcid] / dt
    d_lambda = -c / (w_inv + alpha_tilde)

    # Convert impulse → force
    frc = d_lambda / dt

    # Unilateral clamp for limit constraints
    is_limit = (
        ctype == int(ConstraintType.LIMIT_JOINT)
        or ctype == int(ConstraintType.LIMIT_TENDON)
    )
    if is_limit:
        frc = wp.max(frc, 0.0)

    efc_force[worldid, efcid] = frc
    for i in range(nv):
        wp.atomic_add(qfrc_constraint, worldid, i, efc_J[worldid, efcid, i] * frc)


# ── XPBD contact: accumulated-lambda normal + regularised Coulomb ─

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
    contact_efc_adr:  wp.array2d(dtype=int),    # (naconmax, nmaxpyramid)
    nacon:            wp.array(dtype=int),       # (1,)
    # constraint data
    efc_J:            wp.array3d(dtype=float),
    efc_pos:          wp.array2d(dtype=float),
    efc_D:            wp.array2d(dtype=float),   # constraint-space effective mass
    nefc:             wp.array(dtype=int),
    qvel_pred:        wp.array2d(dtype=float),
    # XPBD params
    compliance:       float,
    # accumulated lambda (in-out, per contact)
    lambda_acc:       wp.array(dtype=float),     # (naconmax,)
    # out
    efc_force:        wp.array2d(dtype=float),
    qfrc_constraint:  wp.array2d(dtype=float),
):
    """Per-contact XPBD with accumulated lambda + regularised Coulomb.

    Standard XPBD iteration:
      Δλ = −(c + α̃·λ_acc) / (w + α̃)
      λ_acc ← max(λ_acc + Δλ, 0)      (unilateral clamp)
      Δλ = λ_acc_new − λ_acc_old       (actual applied delta)

    Effective mass w = 1/D from make_constraint (constraint-space,
    includes articulated-body inertia through the Jacobian).

    Only the DELTA impulse is scattered each iteration, so accumulated
    forces don't compound — this is what prevents the divergence.
    """
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    if not (contact_type[cid] & int(ContactType.CONSTRAINT)):
        return

    worldid = contact_worldid[cid]
    dt = opt_timestep[worldid % opt_timestep.shape[0]]

    # ── locate normal efc row ────────────────────────────────────
    normal_efc = contact_efc_adr[cid, 0]
    if normal_efc < 0 or normal_efc >= nefc[worldid]:
        return

    mu = contact_friction[cid][0]

    # ── effective inverse mass from efc.D ────────────────────────
    D = efc_D[worldid, normal_efc]
    w = float(0.0)
    if D > 1.0e-10:
        w = 1.0 / D    # w = J M^{-1} J^T (approximately, with impedance)

    # ── normal constraint velocity ───────────────────────────────
    v_n = float(0.0)
    for i in range(nv):
        v_n += efc_J[worldid, normal_efc, i] * qvel_pred[worldid, i]

    # ── XPBD accumulated-lambda update ───────────────────────────
    alpha_tilde = compliance / (dt * dt)
    dist = efc_pos[worldid, normal_efc]
    c = v_n + wp.min(dist, 0.0) / dt

    old_lambda = lambda_acc[cid]
    d_lambda = -(c + alpha_tilde * old_lambda) / (w + alpha_tilde)

    # Unilateral clamp on ACCUMULATED lambda
    new_lambda = wp.max(old_lambda + d_lambda, 0.0)
    d_lambda = new_lambda - old_lambda
    lambda_acc[cid] = new_lambda

    # ── scatter DELTA normal force ───────────────────────────────
    d_force = d_lambda / dt
    efc_force[worldid, normal_efc] = new_lambda / dt   # store total for diagnostics
    for i in range(nv):
        wp.atomic_add(
            qfrc_constraint, worldid, i, efc_J[worldid, normal_efc, i] * d_force
        )

    # ── regularised Coulomb friction ─────────────────────────────
    # Uses the ACCUMULATED normal impulse for the Coulomb cap
    if mu < 1.0e-8 or new_lambda < 1.0e-12 or contact_dim[cid] < 3:
        return

    edge1p = contact_efc_adr[cid, 1]
    edge1m = contact_efc_adr[cid, 2]
    if edge1p < 0 or edge1m < 0:
        return

    inv_2mu = 0.5 / mu

    # tangent-1 velocity
    vt1 = float(0.0)
    for i in range(nv):
        jt1_i = (efc_J[worldid, edge1p, i] - efc_J[worldid, edge1m, i]) * inv_2mu
        vt1 += jt1_i * qvel_pred[worldid, i]

    # tangent-2 velocity
    vt2 = float(0.0)
    edge2p = contact_efc_adr[cid, 3]
    edge2m = contact_efc_adr[cid, 4]
    has_t2 = edge2p >= 0 and edge2m >= 0
    if has_t2:
        for i in range(nv):
            jt2_i = (efc_J[worldid, edge2p, i] - efc_J[worldid, edge2m, i]) * inv_2mu
            vt2 += jt2_i * qvel_pred[worldid, i]

    # f_t = −μ · (λ_acc/dt) · v_t / √(‖v_t‖² + ε)
    vt_sq = vt1 * vt1 + vt2 * vt2
    denom = wp.sqrt(vt_sq + _FRICTION_EPS)
    f_n_total = new_lambda / dt
    fric_scale = mu * f_n_total / denom

    fric_t1 = -fric_scale * vt1
    fric_t2 = -fric_scale * vt2

    for i in range(nv):
        jt1_i = (efc_J[worldid, edge1p, i] - efc_J[worldid, edge1m, i]) * inv_2mu
        frc_i = jt1_i * fric_t1
        if has_t2:
            jt2_i = (efc_J[worldid, edge2p, i] - efc_J[worldid, edge2m, i]) * inv_2mu
            frc_i += jt2_i * fric_t2
        wp.atomic_add(qfrc_constraint, worldid, i, frc_i)


# ── combine smooth + constraint forces ────────────────────────────

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


# ── update qvel_pred from qacc (between XPBD iterations) ─────────

@wp.kernel
def _repredict_from_qacc(
    opt_timestep: wp.array(dtype=float),
    qvel:         wp.array2d(dtype=float),
    qacc:         wp.array2d(dtype=float),
    # out
    qvel_pred:    wp.array2d(dtype=float),
):
    """qvel_pred = qvel + qacc * dt."""
    worldid, dofid = wp.tid()
    dt = opt_timestep[worldid % opt_timestep.shape[0]]
    qvel_pred[worldid, dofid] = qvel[worldid, dofid] + qacc[worldid, dofid] * dt


@wp.kernel
def _zero_array2d(a: wp.array2d(dtype=float)):
    worldid, i = wp.tid()
    a[worldid, i] = 0.0


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
    """Wraps a MJWarp Data with scratch buffers for the XPBD solve."""

    def __init__(self, mjw_data, nworld: int, nv: int, naconmax: int):
        self._d = mjw_data
        device = mjw_data.qpos.device
        # Scratch arrays that MJWarp Data doesn't provide
        self.qvel_pred       = wp.zeros((nworld, nv), dtype=float, device=device)
        self.qfrc_constraint = wp.zeros((nworld, nv), dtype=float, device=device)
        self.qfrc_total      = wp.zeros((nworld, nv), dtype=float, device=device)
        # Per-contact accumulated normal impulse (zeroed each step)
        self.lambda_acc      = wp.zeros(naconmax, dtype=float, device=device)

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
    naconmax = getattr(mjw_d, 'naconmax', nconmax or mjm.nconmax * nworld)
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
    naconmax = getattr(mjw_d, 'naconmax', nconmax or mjm.nconmax * nworld)
    return XPBDData(mjw_d, nworld, mjm.nv, naconmax)


def get_data_into(
    mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData, mjd: mujoco.MjData
):
    return _mjw.get_data_into(mjm, m._m, d._d, mjd)


def reset_data(mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData):
    return _mjw.reset_data(mjm, m._m, d._d)


# ═══════════════════════════════════════════════════════════════════
# XPBD constraint solve
# ═══════════════════════════════════════════════════════════════════

def _xpbd_solve(m: XPBDModel, d: XPBDData):
    """Iterative XPBD constraint solve.

    Each iteration:
      1. Zero qfrc_constraint
      2. Compute all constraint forces from current qvel_pred
      3. Update qvel_pred += diag(M^{-1}) * qfrc_constraint * dt

    After all iterations, form qfrc_total and let smooth.solve_m
    compute the final qacc with the full (non-diagonal) M^{-1}.
    """
    inner_m = m._m
    inner_d = d._d
    params  = m.xpbd_params
    njmax_pad = inner_d.efc.J.shape[1]
    nv = inner_m.nv
    nw = inner_d.nworld

    # Initial prediction: qvel_pred = qvel + qacc_smooth * dt
    wp.launch(
        _predict_qvel_and_zero,
        dim=(nw, nv),
        inputs=[inner_m.opt.timestep, inner_d.qvel, inner_d.qacc_smooth],
        outputs=[d.qvel_pred, d.qfrc_constraint],
    )

    # Zero accumulated contact impulses for this step
    d.lambda_acc.zero_()

    for _iter in range(params.iterations):

        # Zero constraint forces for this iteration
        wp.launch(_zero_array2d, dim=(nw, nv), inputs=[d.qfrc_constraint])

        # Non-contact constraints (equality, limits, joint friction)
        wp.launch(
            _xpbd_noncontact,
            dim=(nw, njmax_pad),
            inputs=[
                inner_m.opt.timestep,
                inner_d.efc.J,
                inner_d.efc.pos,
                inner_d.efc.D,
                inner_d.efc.type,
                d.qvel_pred,
                nv,
                inner_d.nefc,
                params.compliance,
            ],
            outputs=[inner_d.efc.force, d.qfrc_constraint],
        )

        # Contact constraints (accumulated-lambda XPBD + friction)
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
                    params.compliance,
                    d.lambda_acc,
                ],
                outputs=[inner_d.efc.force, d.qfrc_constraint],
            )

        # Update qvel_pred for next iteration via full M^{-1} solve
        if _iter < params.iterations - 1:
            # qfrc_total = qfrc_smooth + qfrc_constraint
            wp.launch(
                _sum_qfrc,
                dim=(nw, nv),
                inputs=[inner_d.qfrc_smooth, d.qfrc_constraint],
                outputs=[d.qfrc_total],
            )
            # qacc = M^{-1} * qfrc_total  (via Data-field copy trick)
            wp.copy(inner_d.qfrc_constraint, d.qfrc_total)
            smooth.solve_m(inner_m, inner_d, inner_d.qacc, inner_d.qfrc_constraint)
            # qvel_pred = qvel + qacc * dt
            wp.launch(
                _repredict_from_qacc,
                dim=(nw, nv),
                inputs=[inner_m.opt.timestep, inner_d.qvel, inner_d.qacc],
                outputs=[d.qvel_pred],
            )

    # Final: qfrc_total = qfrc_smooth + qfrc_constraint
    wp.launch(
        _sum_qfrc,
        dim=(nw, nv),
        inputs=[inner_d.qfrc_smooth, d.qfrc_constraint],
        outputs=[d.qfrc_total],
    )


# ═══════════════════════════════════════════════════════════════════
# Forward / Step
# ═══════════════════════════════════════════════════════════════════

def forward(m: XPBDModel, d: XPBDData):
    """Full forward pass with XPBD constraint resolution."""
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

    if inner_m.opt.run_collision_detection:
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
    fwd_acceleration(inner_m, inner_d, factorize=True)

    # ── XPBD constraint solve ────────────────────────────────────
    if inner_d.njmax == 0 or inner_m.nv == 0:
        wp.copy(inner_d.qacc, inner_d.qacc_smooth)
    else:
        _xpbd_solve(m, d)
        # solve_m may only work with arrays that live on the Data struct.
        # Copy qfrc_total into inner_d.qfrc_constraint (a real Data field),
        # then solve:  qacc = M^{-1} * qfrc_total.
        wp.copy(inner_d.qfrc_constraint, d.qfrc_total)
        smooth.solve_m(inner_m, inner_d, inner_d.qacc, inner_d.qfrc_constraint)

    sensor.sensor_acc(inner_m, inner_d)


def step(m: XPBDModel, d: XPBDData):
    """Advance one timestep with XPBD contact."""
    inner_m = m._m
    inner_d = d._d

    forward(m, d)

    # euler reads inner_d.qacc (set by solve_m in forward).
    # efc.Ma is only needed by the implicit integrator.
    wp.copy(inner_d.efc.Ma, d.qfrc_total)
    euler(inner_m, inner_d)