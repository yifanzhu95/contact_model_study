"""XPBD-style decoupled penalty contact model (M4).

Architecture:
  - All kinematics, collision detection, and mass-matrix utilities are
    borrowed from the vendored mujoco_warp (no re-implementation).
  - Contact force resolution replaces MJWarp's global LCP/comfree solve
    with a per-contact XPBD position correction loop.
  - Friction can be either Coulomb (default) or viscous damping (M4 variant).

The put_model / make_data / step surface mirrors comfree_warp.api so that
contact_study/contact_models/api.py can dispatch uniformly.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import warp as wp

from .config import XPBDParams

# Borrow everything from the vendored mujoco_warp
import comfree_warp.mujoco_warp as _mjw
from comfree_warp.mujoco_warp._src import smooth, collision_driver, sensor
from comfree_warp.mujoco_warp._src.types import EnableBit, DisableBit


# ---------------------------------------------------------------------------
# XPBD contact resolution kernel
# ---------------------------------------------------------------------------

@wp.kernel
def _xpbd_resolve_contacts(
    # Model
    nv:             int,
    comfree_compliance:  wp.array(dtype=float),    # (1,)
    comfree_damping_k:   wp.array(dtype=float),    # (1,)
    damping_friction:    bool,
    # Contact data (from MJWarp collision pipeline)
    nacon:          wp.array(dtype=int),           # (nworld,)
    contact_dist:   wp.array(dtype=float),         # (naconmax,)
    contact_pos:    wp.array(dtype=wp.vec3),        # (naconmax,)
    contact_frame:  wp.array(dtype=wp.mat33),       # (naconmax,)  row0=normal
    contact_friction: wp.array(dtype=wp.vec5f),     # (naconmax,)
    contact_worldid: wp.array(dtype=int),           # (naconmax,)
    # State
    qvel:           wp.array2d(dtype=float),        # (nworld, nv)
    # Output: per-contact generalized force increment
    qfrc_contact:   wp.array2d(dtype=float),        # (nworld, nv)
    # Jacobian scratch (filled by smooth.jac, read here)
    jacp:           wp.array3d(dtype=float),        # (nworld, naconmax, 3*nv) - jacp for contact point
):
    """Resolve one contact per thread using decoupled XPBD."""
    contact_id = wp.tid()
    worldid = contact_worldid[contact_id]

    if contact_id >= nacon[worldid]:
        return

    dist = contact_dist[contact_id]
    if dist >= 0.0:
        return  # no penetration

    frame = contact_frame[contact_id]
    normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])

    compliance = comfree_compliance[0]
    damping    = comfree_damping_k[0]

    # Normal impulse via XPBD:  lambda_n = -dist / (w + compliance)
    # Effective mass w is approximated as 1/m_eff (diagonal, cheap).
    # For a full study implementation, replace with jac^T M^{-1} jac.
    w = 1.0   # placeholder: effective inverse mass
    lambda_n = -dist / (w + compliance)
    lambda_n = wp.max(lambda_n, 0.0)

    # Normal force in world frame
    fn = lambda_n * normal

    # Friction
    friction_coeff = contact_friction[contact_id][0]   # mu
    if damping_friction:
        # Viscous damping: f_t = -damping * v_t (no Coulomb cone)
        # v_t extracted from qvel via contact frame (simplified: world-frame approx)
        # Full implementation: compute v_t = frame[1:] @ jac @ qvel
        ft = wp.vec3(0.0, 0.0, 0.0)   # placeholder: implement via Jacobian
    else:
        # Coulomb: project tangential impulse onto friction cone
        # Full: compute v_t, apply positional friction correction
        ft = wp.vec3(0.0, 0.0, 0.0)   # placeholder: implement via Jacobian

    # Accumulate generalized force: qfrc += J^T * (fn + ft)
    # In the simplified scalar version, we scatter via contact normal.
    # Full implementation requires jacp to be pre-populated by smooth.jac().
    # (This kernel is the integration point -- fill in Jacobian scatter here.)
    _ = ft  # suppress unused warning until Jacobian path is wired


@wp.kernel
def _apply_contact_forces(
    qfrc_contact: wp.array2d(dtype=float),   # (nworld, nv)
    qacc_smooth:  wp.array2d(dtype=float),   # (nworld, nv)  smooth accel (no contact)
    qacc_out:     wp.array2d(dtype=float),   # (nworld, nv)
):
    """qacc = qacc_smooth + M^{-1} qfrc_contact  (M^{-1} pre-applied in qfrc_contact)."""
    worldid, dofid = wp.tid()
    qacc_out[worldid, dofid] = qacc_smooth[worldid, dofid] + qfrc_contact[worldid, dofid]


# ---------------------------------------------------------------------------
# Model / Data wrappers
# ---------------------------------------------------------------------------

class XPBDModel:
    """Thin wrapper around a MJWarp Model with XPBD parameters attached."""

    def __init__(self, mjw_model, xpbd_params: XPBDParams):
        self._m = mjw_model
        device = mjw_model.opt.timestep.device
        self.xpbd_compliance   = wp.array([xpbd_params.compliance], dtype=wp.float32, device=device)
        self.xpbd_damping      = wp.array([xpbd_params.damping],    dtype=wp.float32, device=device)
        self.xpbd_iterations   = xpbd_params.iterations
        self.damping_friction  = xpbd_params.damping_friction
        # contact_cfg attached by api.put_model
        self.contact_cfg       = None

    def __getattr__(self, name):
        # Transparent proxy to the underlying MJWarp model
        return getattr(self._m, name)


class XPBDData:
    """Thin wrapper around MJWarp Data with XPBD scratch arrays."""

    def __init__(self, mjw_data, nworld: int, nv: int, naconmax: int):
        self._d = mjw_data
        device = mjw_data.qpos.device
        self.qfrc_contact = wp.zeros((nworld, nv), dtype=wp.float32, device=device)

    def __getattr__(self, name):
        return getattr(self._d, name)


# ---------------------------------------------------------------------------
# Public backend surface
# ---------------------------------------------------------------------------

def put_model(mjm: mujoco.MjModel, xpbd_params: XPBDParams) -> XPBDModel:
    mjw_m = _mjw.put_model(mjm)
    return XPBDModel(mjw_m, xpbd_params)


def make_data(mjm: mujoco.MjModel, m: XPBDModel, nworld: int = 1, nconmax=None, njmax=None) -> XPBDData:
    kw = dict(nworld=nworld)
    if nconmax is not None: kw['nconmax'] = nconmax
    if njmax is not None: kw['njmax'] = njmax
    mjw_d = _mjw.make_data(mjm, **kw)
    return XPBDData(mjw_d, nworld, mjm.nv, mjm.nconmax)


def put_data(mjm: mujoco.MjModel, mjd: mujoco.MjData, m: XPBDModel, nworld: int = 1, nconmax=None, njmax=None) -> XPBDData:
    kw = dict(nworld=nworld)
    if nconmax is not None: kw['nconmax'] = nconmax
    if njmax is not None: kw['njmax'] = njmax
    mjw_d = _mjw.put_data(mjm, mjd, **kw)
    return XPBDData(mjw_d, nworld, mjm.nv, mjm.nconmax)


def get_data_into(mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData, mjd: mujoco.MjData):
    return _mjw.get_data_into(mjm, m._m, d._d, mjd)


def reset_data(mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData):
    return _mjw.reset_data(mjm, m._m, d._d)


def forward(m: XPBDModel, d: XPBDData):
    """Run full forward pass with XPBD contact resolution."""
    inner_m = m._m
    inner_d = d._d

    # --- standard MJWarp pipeline up to smooth acceleration ---
    smooth.kinematics(inner_m, inner_d)
    smooth.com_pos(inner_m, inner_d)
    smooth.camlight(inner_m, inner_d)
    smooth.tendon(inner_m, inner_d)
    smooth.factor_m(inner_m, inner_d)

    if inner_m.opt.run_collision_detection:
        collision_driver.collision(inner_m, inner_d)

    # Constraint Jacobians (needed for XPBD Jacobian scatter)
    # make_constraint still fills efc_J for equality constraints;
    # contact forces are handled separately below.
    from comfree_warp.mujoco_warp._src import constraint as _con
    _con.make_constraint(inner_m, inner_d)

    smooth.transmission(inner_m, inner_d)
    smooth.crb(inner_m, inner_d)

    from comfree_warp.mujoco_warp._src.forward import (
        fwd_velocity, fwd_actuation, fwd_acceleration,
    )
    fwd_velocity(inner_m, inner_d)
    fwd_actuation(inner_m, inner_d)
    fwd_acceleration(inner_m, inner_d, factorize=True)
    # inner_d.qacc_smooth now holds constraint-free acceleration

    # --- XPBD contact resolution (replaces LCP / comfree solve) ---
    d.qfrc_contact.zero_()

    for _ in range(m.xpbd_iterations):
        nworld = inner_d.nworld
        naconmax = inner_d.naconmax
        if naconmax > 0:
            wp.launch(
                _xpbd_resolve_contacts,
                dim=naconmax,
                inputs=[
                    inner_m.nv,
                    m.xpbd_compliance,
                    m.xpbd_damping,
                    m.damping_friction,
                    inner_d.nacon,
                    inner_d.contact.dist,
                    inner_d.contact.pos,
                    inner_d.contact.frame,
                    inner_d.contact.friction,
                    inner_d.contact.worldid,
                    inner_d.qvel,
                    d.qfrc_contact,
                    # jacp: populate via smooth.jac before this launch
                    wp.zeros((nworld, naconmax, 3 * inner_m.nv), dtype=float),
                ],
                outputs=[],
            )

    # Combine smooth accel + contact forces -> qacc
    wp.launch(
        _apply_contact_forces,
        dim=(inner_d.nworld, inner_m.nv),
        inputs=[d.qfrc_contact, inner_d.qacc_smooth],
        outputs=[inner_d.qacc],
    )

    sensor.sensor_acc(inner_m, inner_d)


def step(m: XPBDModel, d: XPBDData):
    """Advance one timestep with XPBD contact."""
    from comfree_warp.mujoco_warp._src.forward import euler
    forward(m, d)
    euler(m._m, d._d)
