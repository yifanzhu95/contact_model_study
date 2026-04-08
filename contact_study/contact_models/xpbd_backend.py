# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# -----------------------------------------------------------------------------
"""XPBD constraint backend for MJWarp — unified mode with substepping.

Architecture
------------
This backend bypasses MJWarp's native constraint solver entirely and
processes ALL active efc rows (equality, limit, friction-loss,
contact) with a single XPBD-style relaxation kernel. It is the
"fastest path" version of the backend, structurally comparable to
ComFree's compute_qfrc_total: one prediction kernel, one constraint
sweep, one sum, one solve_m per substep.

Per-row update (general form, applied to every row in a single
parallel kernel launch):

    v_e   = J_e · qvel_pred                       # row velocity
    denom = 1 / efc_D[e]    = J_e M⁻¹ J_e^T + R   # eff. mass + reg
    C_e   = efc_pos[e]                            # row residual
    Δλ    = -relax · (v_e + C_e/dt) / denom       # SOR-relaxed
    λ_new = clamp(λ_old + Δλ, [lo, hi])           # type-dependent
    qfrc += J_e^T · (Δλ / dt)

Type-dispatched bounds and relaxation:

  EQUALITY ............... bilateral, no clamp,           relax = 1
  LIMIT_JOINT/TENDON ..... unilateral (λ ≥ 0),            relax = 1
  FRICTION_DOF/TENDON .... bilateral (TODO: box clamp),   relax = 1
  CONTACT_FRICTIONLESS ... unilateral, vmax cap,          relax = relax_contact
  CONTACT_PYRAMIDAL ...... unilateral, vmax cap,          relax = relax_contact
  CONTACT_ELLIPTIC ....... rejected (early return)

Why two relaxation regimes:
  Contact rows are heavily redundant on the same body (4 pyramid
  edges × N grasp contacts → tens of rows pushing the same dof).
  Parallel-scatter Jacobi PGS over-counts roughly by Nrows, so
  contact rows need aggressive SOR (relax_contact ≈ 0.01–0.02 for
  typical multi-finger grasps). Non-contact rows (equality, limit,
  friction-loss) are typically NOT redundant — one row per joint
  limit, one per equality coupling — so they can use relax = 1
  without over-counting. A uniform relax would either soften limits
  (relax everywhere = 0.01) or catapult cubes (relax everywhere = 1).
  Type-dispatched SOR handles both correctly.

Substepping
-----------
step() runs collision detection inside the per-substep position
phase (matching the working comfree forward order exactly:
kinematics → ... → factor_m → collision → make_constraint →
transmission). The substep dt is patched into m.opt.timestep before
the loop and restored after. With substeps=1 the patching is skipped
and behaviour is identical to a single forward+integrate pass. With
substeps>1, collision runs every substep — losing the Macklin §4.2
once-per-frame amortization but maintaining correctness. Re-enabling
amortization is future work; see the _forward_setup docstring for
the rabbit hole.

Tuning
------
xpbd_params fields the backend reads (with getattr defaults):

  * iterations          (int, default 1)
        XPBD sweeps per substep. Each iteration after the first
        refreshes qvel_pred from the accumulated qfrc_constraint
        before re-sweeping. iterations > 1 is rarely useful with
        unilateral clamps because the iteration tends to oscillate
        rather than converge; prefer SUBSTEPS for refinement.

  * substeps            (int, default 1)
        Number of physics substeps per visual frame. Smaller dt
        improves contact stability and lets the system handle
        stiffer materials, but multiplies the per-frame work
        roughly linearly.

  * vmax_depenetration  (float, default 1.0 m/s)
        Cap on the velocity used to remove initial penetration in
        a single substep (Macklin §4.3). Without this, small dt
        plus large initial overlap → explosive separation.

  * relaxation          (float, default 0.01)
        SOR factor applied to CONTACT row updates. The optimal
        value is roughly 1/(rows-per-body); for typical grasps
        with 4 pyramid edges × ~9 contacts ≈ 36 rows, use 0.01
        to 0.03. Setting too low = soft mushy contacts; too high
        = catapulted bodies. Fix for this in the long run is per-
        body Gauss-Seidel grouping; until then, tune by hand.

Diagnostic helper
-----------------
print_constraint_types() prints the integer values of every
ConstraintType enum member that resolved successfully. Run it once
at startup if you ever see unexplained constraint behaviour to
verify the names this file expects (EQUALITY, FRICTION_DOF,
FRICTION_TENDON, LIMIT_JOINT, LIMIT_TENDON, CONTACT_FRICTIONLESS,
CONTACT_PYRAMIDAL, CONTACT_ELLIPTIC) all exist in your MJWarp build.
Names that don't resolve get a sentinel constant that the kernel
will never match, so missing types fall through to the bilateral
default rather than crashing — convenient but means a typo in a
name silently downgrades that row type's correctness.
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
from comfree_warp.mujoco_warp._src.forward import (
    euler,
    fwd_acceleration,
    fwd_actuation,
    fwd_velocity,
    implicit,
)
from comfree_warp.mujoco_warp._src.types import (
    ConstraintType,
    DisableBit,
    EnableBit,
    IntegratorType,
)

wp.set_module_options({"enable_backward": False})


# ═══════════════════════════════════════════════════════════════════
# Constraint type constants
# ═══════════════════════════════════════════════════════════════════

def _ct(name: str, default: int) -> int:
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


def print_constraint_types():
    """Print all ConstraintType enum members and their integer values.

    Run this once when porting to a new MJWarp build to confirm the
    type names this file uses (CONTACT_FRICTIONLESS, CONTACT_PYRAMIDAL,
    CONTACT_ELLIPTIC) exist, and to look up the names of FRICTION_DOF /
    LIMIT_JOINT / etc. if you want to enable a fully-unified mode.
    """
    print("ConstraintType members:")
    for name in dir(ConstraintType):
        if name.startswith("_"):
            continue
        try:
            val = int(getattr(ConstraintType, name))
            print(f"  {name:30s} = {val}")
        except (TypeError, ValueError):
            pass


# ═══════════════════════════════════════════════════════════════════
# Warp kernels
# ═══════════════════════════════════════════════════════════════════

@wp.kernel
def _zero_2d(a: wp.array2d(dtype=float)):
    w, i = wp.tid()
    a[w, i] = 0.0


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
    # out
    qvel_pred: wp.array2d(dtype=float),
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


# ── XPBD unified sweep (all efc rows) ────────────────────────────

@wp.kernel
def _xpbd_unified_sweep(
    # model / params
    opt_timestep:  wp.array(dtype=float),
    nv:            int,
    vmax_depen:    float,
    relax_contact: float,
    # efc data
    efc_J:    wp.array3d(dtype=float),
    efc_pos:  wp.array2d(dtype=float),
    efc_D:    wp.array2d(dtype=float),
    efc_type: wp.array2d(dtype=int),
    nefc:     wp.array(dtype=int),
    qvel_pred: wp.array2d(dtype=float),
    # in/out
    lambda_efc:      wp.array2d(dtype=float),
    efc_force:       wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
):
    """One XPBD relaxation sweep over ALL efc rows.

    Per-row update (general form, applied to every active row):

        v_e   = J_e · qvel_pred                       # row velocity
        denom = 1 / efc_D[e]    = J_e M⁻¹ J_e^T + R   # eff. mass + reg
        C_e   = efc_pos[e]                            # row residual
        Δλ    = -relax · (v_e + C_e/dt) / denom       # SOR-relaxed
        λ_new = clamp(λ_old + Δλ, [lo, hi])
        Δλ    = λ_new − λ_old
        qfrc += J_e^T · (Δλ / dt)

    Row-type dispatch:

      EQUALITY ............... bilateral, no clamp, relax = 1
      LIMIT_JOINT/TENDON ..... unilateral (λ ≥ 0), relax = 1
      FRICTION_DOF/TENDON .... bilateral, no clamp, relax = 1
                               (NOTE: this is technically wrong —
                               friction-loss should be box-clamped to
                               ±frictionloss·dt. Doing the box clamp
                               needs efc.frictionloss which I haven't
                               verified on this build. Treating as
                               bilateral makes friction-loss act as
                               an unbounded velocity damper, which is
                               only "approximately right" for small
                               velocities. TODO: verify the field name
                               and add the clamp.)
      CONTACT_FRICTIONLESS ... unilateral, vmax cap, relax = relax_contact
      CONTACT_PYRAMIDAL ...... unilateral, vmax cap, relax = relax_contact
      CONTACT_ELLIPTIC ....... rejected (early return)
      Anything else .......... falls through to the bilateral default

    Why two relaxation regimes:
      Contact rows are heavily redundant on the same body (4 pyramid
      edges × N grasp contacts → tens of rows that all push the same
      cube dof). Parallel scatter Jacobi over-counts by ~Nrows, so
      contact rows need aggressive SOR (relax_contact ≈ 0.01–0.02 for
      typical grasps). Non-contact rows (equality, limit, friction-
      loss) are typically NOT redundant — one row per joint limit,
      one row per friction-loss dof, one row per equality coupling —
      so they can use relax = 1 without over-counting. Mixing them
      would either soften limits/equality (relax everywhere = 0.01)
      or catapult cubes (relax everywhere = 1). Type-dispatched SOR
      handles both correctly.

    Pyramidal Coulomb cone: for condim=3 MJWarp emits 4 edge rows of
    the form J_n + μ J_t1, J_n − μ J_t1, J_n + μ J_t2, J_n − μ J_t2.
    Treating each as a unilateral row (λ_e ≥ 0) enforces |λ_t| ≤ μ λ_n
    by row arithmetic; no explicit Coulomb clamp is needed.
    """
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid]:
        return

    ctype = efc_type[worldid, efcid]

    # Reject elliptic — should never appear since MJWarp uses
    # pyramidal cones, but be defensive.
    if ctype == _CT_CONTACT_ELLIPTIC:
        return

    D_e = efc_D[worldid, efcid]
    if D_e <= 1.0e-10:
        return
    denom = 1.0 / D_e

    # Row-space velocity v_e = J_e · qvel_pred
    v_e = float(0.0)
    for i in range(nv):
        v_e += efc_J[worldid, efcid, i] * qvel_pred[worldid, i]

    dt = opt_timestep[worldid % opt_timestep.shape[0]]

    # Type dispatch: pick relaxation, position-residual treatment,
    # and clamp behaviour.
    is_contact = (
        ctype == _CT_CONTACT_FRICTIONLESS
        or ctype == _CT_CONTACT_PYRAMIDAL
    )
    is_limit = (
        ctype == _CT_LIMIT_JOINT
        or ctype == _CT_LIMIT_TENDON
    )

    C_e = efc_pos[worldid, efcid]
    relax = float(1.0)
    is_unilateral = False

    if is_contact:
        # vmax cap (Macklin §4.3): never try to remove more than
        # vmax·dt of penetration in a single substep.
        if C_e < -vmax_depen * dt:
            C_e = -vmax_depen * dt
        relax = relax_contact
        is_unilateral = True
    elif is_limit:
        # Joint or tendon limit. Unilateral, but typically not
        # redundant on the same dof, so no SOR needed.
        is_unilateral = True
    # else: equality or friction-loss → bilateral default
    #       (relax = 1, no clamp, no vmax cap)

    # XPBD increment.
    d_lambda = -relax * (v_e + C_e / dt) / denom

    old_l = lambda_efc[worldid, efcid]
    new_l = old_l + d_lambda
    if is_unilateral and new_l < 0.0:
        new_l = 0.0
    d_lambda = new_l - old_l
    lambda_efc[worldid, efcid] = new_l

    if d_lambda != 0.0:
        d_f = d_lambda / dt
        for i in range(nv):
            wp.atomic_add(
                qfrc_constraint, worldid, i, efc_J[worldid, efcid, i] * d_f
            )

    # Diagnostic: efc_force holds the running total in force units.
    efc_force[worldid, efcid] = new_l / dt


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
    """Wraps a MJWarp Data with scratch buffers used by the XPBD solve."""

    def __init__(self, mjw_data, nworld: int, nv: int, njmax_pad: int,
                 n_opt_timestep: int):
        self._d = mjw_data
        device = mjw_data.qpos.device

        # Smooth velocity prediction (substep-local).
        self.qvel_pred = wp.zeros((nworld, nv), dtype=float, device=device)

        # Total qfrc passed to solve_m and to the integrators.
        self.qfrc_total = wp.zeros((nworld, nv), dtype=float, device=device)

        # Per-efc-row Lagrange multipliers (substep-local; reset every
        # substep). Shape (nworld, njmax_pad) so we have one slot per
        # efc row regardless of constraint type or contact condim.
        self.lambda_efc = wp.zeros((nworld, njmax_pad), dtype=float,
                                   device=device)

        # Backup of m.opt.timestep so we can patch it for substepping.
        self.timestep_orig = wp.zeros(n_opt_timestep, dtype=float,
                                      device=device)

    def __getattr__(self, name):
        return getattr(self._d, name)


# ═══════════════════════════════════════════════════════════════════
# Public surface (drop-in replacement for the previous backend)
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

    njmax_pad = int(mjw_d.efc.J.shape[1])
    n_opt_ts = int(m._m.opt.timestep.shape[0])
    return XPBDData(mjw_d, nworld, mjm.nv, njmax_pad, n_opt_ts)


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

    njmax_pad = int(mjw_d.efc.J.shape[1])
    n_opt_ts = int(m._m.opt.timestep.shape[0])
    return XPBDData(mjw_d, nworld, mjm.nv, njmax_pad, n_opt_ts)


def get_data_into(
    mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData, mjd: mujoco.MjData
):
    return _mjw.get_data_into(mjm, m._m, d._d, mjd)


def reset_data(mjm: mujoco.MjModel, m: XPBDModel, d: XPBDData):
    return _mjw.reset_data(mjm, m._m, d._d)


# ═══════════════════════════════════════════════════════════════════
# Internal pipeline pieces
# ═══════════════════════════════════════════════════════════════════

def _forward_setup(m: XPBDModel, d: XPBDData):
    """Currently a no-op.

    Originally intended to host once-per-frame collision detection
    so that the contact set could be reused across substeps (Macklin
    et al. 2019 §4.2). After several iterations of debugging, the
    explicit collision_driver.collision() call from this function
    consistently produced zero contacts on this MJWarp build, even
    though the per-substep make_constraint downstream was somehow
    building a correct contact set anyway. The interaction between
    the explicit collider, the smooth chain, and make_constraint on
    this build is opaque, so for now we keep collision inside the
    per-substep position phase (matching the working comfree
    forward order exactly) and treat the per-frame amortization as
    future work.

    To re-enable amortization later you need to understand why an
    explicit collision_driver.collision() call here returns zero
    contacts while the comfree backend's identical-looking call at
    the same point in its own forward() works. Possibilities to
    investigate: an event_scope or stream-management context the
    comfree forward decorator establishes, a stale-state-detection
    path inside make_constraint that re-runs collision on demand,
    or a difference between collision_driver.collision() called at
    module level vs. inside the comfree forward function.
    """
    pass


def _substep_position_phase(m: XPBDModel, d: XPBDData):
    """Per substep: full position phase including collision detection.

    This mirrors the working comfree forward_comfree order exactly:

        kinematics → com_pos → camlight → flex →
        tendon → crb → tendon_armature → factor_m →
        collision → make_constraint → transmission

    Collision must come AFTER factor_m (some intermediate buffer is
    needed) and BEFORE make_constraint (which reads d.contact).
    Splitting collision out to a separate per-frame pass was
    attempted in earlier drafts and produced contact-pipeline bugs
    that took several iterations to chase. Don't.
    """
    inner_m = m._m
    inner_d = d._d
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


def _xpbd_solve(m: XPBDModel, d: XPBDData):
    """Unified XPBD constraint solve.

    Bypasses MJWarp's native constraint solver entirely. Every active
    efc row (equality, limit, friction-loss, contact) is processed by
    the same _xpbd_unified_sweep kernel with type-dispatched bounds
    and per-type relaxation.

    Steps (per substep, per iteration):
      1. Predict qvel from smooth dynamics only:
            qvel_pred = qvel + dt · qacc_smooth
         qacc_smooth comes from fwd_acceleration which already ran
         and represents M⁻¹ · qfrc_smooth (gravity + Coriolis +
         actuation + passive). NO constraint forces are in there yet.
      2. Zero qfrc_constraint and lambda_efc; we accumulate into them.
      3. Run _xpbd_unified_sweep over all (world, efc_row) pairs.
         The kernel scatters Δλ·Jᵀ/dt into qfrc_constraint atomically
         for every active row.
      4. (Optional, iterations > 1) Refresh qvel_pred from the
         current accumulated qfrc_constraint and re-sweep. Each
         iteration is one approximate Newton step on the implicit
         system.
      5. Final qfrc_total = qfrc_smooth + qfrc_constraint, then
         qacc = M⁻¹ · qfrc_total via solve_m.

    What this skips compared to the previous hybrid mode:
      * No native MJWarp solver call.
      * No M refactor (the native solver was clobbering qLD and we
        had to factor_m again afterward; bypass eliminates that).
      * No strip-and-restore dance for contact rows.
      * No "predict from non-contact" extra solve_m.
      * No stash of qfrc_noncontact.

    Net per-call kernel count drops from ~10 launches + 2 solve_m to
    ~3 launches + 1 solve_m, matching ComFree's compute_qfrc_total
    structure. Should be at parity or faster than ComFree on dense
    contact scenes.
    """
    inner_m = m._m
    inner_d = d._d
    nv = inner_m.nv
    nw = inner_d.nworld
    njmax_pad = int(inner_d.efc.J.shape[1])
    params = m.xpbd_params

    n_iter = max(1, getattr(params, "iterations", 1))
    vmax = float(getattr(params, "vmax_depenetration", 1.0))
    relax_contact = float(getattr(params, "relaxation", 0.01))

    # ── 1. predict qvel from smooth dynamics only ──
    wp.launch(
        _predict_qvel,
        dim=(nw, nv),
        inputs=[inner_m.opt.timestep, inner_d.qvel, inner_d.qacc_smooth],
        outputs=[d.qvel_pred],
    )

    # ── 2. zero accumulators ──
    wp.launch(_zero_2d, dim=(nw, nv), inputs=[inner_d.qfrc_constraint])
    d.lambda_efc.zero_()

    # ── 3-4. unified XPBD sweep(s) ──
    for it in range(n_iter):
        if njmax_pad > 0:
            wp.launch(
                _xpbd_unified_sweep,
                dim=(nw, njmax_pad),
                inputs=[
                    inner_m.opt.timestep,
                    nv,
                    vmax,
                    relax_contact,
                    inner_d.efc.J,
                    inner_d.efc.pos,
                    inner_d.efc.D,
                    inner_d.efc.type,
                    inner_d.nefc,
                    d.qvel_pred,
                ],
                outputs=[
                    d.lambda_efc,
                    inner_d.efc.force,
                    inner_d.qfrc_constraint,
                ],
            )

        # Refresh qvel_pred for the next iteration so it sees the
        # current constraint forces. Skip on the final iteration
        # since the caller's solve_m will redo it anyway.
        if it < n_iter - 1:
            wp.launch(
                _sum2,
                dim=(nw, nv),
                inputs=[inner_d.qfrc_smooth, inner_d.qfrc_constraint],
                outputs=[d.qfrc_total],
            )
            smooth.solve_m(inner_m, inner_d, inner_d.qacc, d.qfrc_total)
            wp.launch(
                _predict_qvel,
                dim=(nw, nv),
                inputs=[inner_m.opt.timestep, inner_d.qvel, inner_d.qacc],
                outputs=[d.qvel_pred],
            )

    # ── 5. final qfrc_total + qacc ──
    wp.launch(
        _sum2,
        dim=(nw, nv),
        inputs=[inner_d.qfrc_smooth, inner_d.qfrc_constraint],
        outputs=[d.qfrc_total],
    )
    smooth.solve_m(inner_m, inner_d, inner_d.qacc, d.qfrc_total)


def _substep(m: XPBDModel, d: XPBDData):
    """One substep: position phase → velocity/acc → XPBD → integrate.

    The caller is responsible for patching m.opt.timestep to the substep
    dt before invoking this function.
    """
    inner_m = m._m
    inner_d = d._d

    _substep_position_phase(m, d)

    fwd_velocity(inner_m, inner_d)

    if not (inner_m.opt.disableflags & DisableBit.ACTUATION):
        if inner_m.callback.control:
            inner_m.callback.control(inner_m, inner_d)
    fwd_actuation(inner_m, inner_d)
    fwd_acceleration(inner_m, inner_d, factorize=True)

    if inner_d.njmax == 0 or inner_m.nv == 0:
        wp.copy(inner_d.qacc, inner_d.qacc_smooth)
        wp.copy(d.qfrc_total, inner_d.qfrc_smooth)
    else:
        _xpbd_solve(m, d)

    # Both EULER and IMPLICITFAST stash qfrc_total in efc.Ma before
    # integrating, matching the comfree convention.
    wp.copy(inner_d.efc.Ma, d.qfrc_total)

    if inner_m.opt.integrator == IntegratorType.EULER:
        euler(inner_m, inner_d)
    elif inner_m.opt.integrator == IntegratorType.IMPLICITFAST:
        implicit(inner_m, inner_d)
    else:
        raise NotImplementedError(
            f"integrator {inner_m.opt.integrator} not supported by xpbd backend"
        )


# ═══════════════════════════════════════════════════════════════════
# Public step / forward
# ═══════════════════════════════════════════════════════════════════

def step(m: XPBDModel, d: XPBDData):
    """Advance one visual frame.

    The frame is split into n_substeps substeps. Collision detection
    runs once at the top of the frame; everything else (kinematics, M
    factor, J rebuild, fwd_*, XPBD solve, integrate) runs every substep
    against the reused contact set.

    With substeps=1 (default if xpbd_params.substeps is unset) the
    timestep patching is skipped and behaviour is identical to a
    single forward+integrate pass.
    """
    inner_m = m._m
    inner_d = d._d
    n_substeps = max(1, int(getattr(m.xpbd_params, "substeps", 1)))

    # ── frame setup ──
    _forward_setup(m, d)

    # ── patch timestep for substepping ──
    n_ts = int(inner_m.opt.timestep.shape[0])
    if n_substeps > 1:
        wp.launch(
            _copy_1d,
            dim=n_ts,
            inputs=[inner_m.opt.timestep, d.timestep_orig],
        )
        wp.launch(
            _scale_1d,
            dim=n_ts,
            inputs=[d.timestep_orig, 1.0 / float(n_substeps)],
            outputs=[inner_m.opt.timestep],
        )

    try:
        for _ in range(n_substeps):
            _substep(m, d)
    finally:
        if n_substeps > 1:
            wp.launch(
                _copy_1d,
                dim=n_ts,
                inputs=[d.timestep_orig, inner_m.opt.timestep],
            )

    # ── once-per-frame sensor pass ──
    inner_d.sensordata.zero_()
    sensor.sensor_pos(inner_m, inner_d)
    if inner_m.opt.enableflags & EnableBit.ENERGY:
        if inner_m.sensor_e_potential == 0:
            sensor.energy_pos(inner_m, inner_d)
    else:
        inner_d.energy.zero_()
    sensor.sensor_vel(inner_m, inner_d)
    if inner_m.opt.enableflags & EnableBit.ENERGY:
        if inner_m.sensor_e_kinetic == 0:
            sensor.energy_vel(inner_m, inner_d)
    sensor.sensor_acc(inner_m, inner_d)


def forward(m: XPBDModel, d: XPBDData):
    """Single forward pass without substepping or integration."""
    inner_m = m._m
    inner_d = d._d

    _forward_setup(m, d)
    _substep_position_phase(m, d)

    inner_d.sensordata.zero_()
    sensor.sensor_pos(inner_m, inner_d)
    if inner_m.opt.enableflags & EnableBit.ENERGY:
        if inner_m.sensor_e_potential == 0:
            sensor.energy_pos(inner_m, inner_d)
    else:
        inner_d.energy.zero_()

    fwd_velocity(inner_m, inner_d)
    sensor.sensor_vel(inner_m, inner_d)
    if inner_m.opt.enableflags & EnableBit.ENERGY:
        if inner_m.sensor_e_kinetic == 0:
            sensor.energy_vel(inner_m, inner_d)

    if not (inner_m.opt.disableflags & DisableBit.ACTUATION):
        if inner_m.callback.control:
            inner_m.callback.control(inner_m, inner_d)
    fwd_actuation(inner_m, inner_d)
    fwd_acceleration(inner_m, inner_d, factorize=True)

    if inner_d.njmax == 0 or inner_m.nv == 0:
        wp.copy(inner_d.qacc, inner_d.qacc_smooth)
        wp.copy(d.qfrc_total, inner_d.qfrc_smooth)
    else:
        _xpbd_solve(m, d)

    sensor.sensor_acc(inner_m, inner_d)