"""Unified contact model API.

Single entry point for M1-M4. Mirrors the comfree_warp.api surface
(put_model, make_data, put_data, get_data_into, reset_data, step,
forward) and dispatches on ContactModelConfig.backend.

Scope
-----
put_model takes an MjModel AS-IS and installs it on the GPU under the
requested contact backend. It does NOT apply physics noise and does NOT
swap geometry. To run any Mk against:

  * a degraded geometry  → load the corresponding XML variant via
    contact_study.tasks.base.BaseTask(geometry=...)
  * noisy physics params → call contact_study.utils.physics_noise
    .apply_physics_noise(mjm, PhysicsNoiseParams(...)) first, then
    pass the perturbed MjModel in here.

MJWarp limitation
-----------------
MJWarp only supports pyramidal friction cones on the GPU. Any config
with cone != 'pyramidal' is rejected at put_model time. The paper's
"elliptic" M2 cannot be run through this backend; see config.M2's
docstring.

Side effect warning
-------------------
_patch_mujoco_options, _apply_hard_contact_preset and
_apply_solref_override mutate the incoming MjModel in place (cone,
solver, iterations, tolerance, and for M1 or any config with an
explicit solref_timeconst also geom_solref/geom_solimp/pair_*). If you reuse the
same MjModel across multiple contact configs, later put_model calls
will NOT restore the previous settings. For benchmarks that sweep
over configs, re-load or deep-copy the MjModel between calls.
"""

from __future__ import annotations

import warnings

import mujoco

from .config import Backend, ContactModelConfig, MujocoSolverParams

# ---------------------------------------------------------------------------
# Lazy backend imports – only pay the import cost for what's actually used
# ---------------------------------------------------------------------------

def _mujoco_warp():
    import comfree_warp.mujoco_warp as mjw
    return mjw

def _comfree_warp():
    import comfree_warp as cfw
    return cfw

def _xpbd_backend():
    from . import xpbd_backend
    return xpbd_backend


# ---------------------------------------------------------------------------
# MuJoCo option patching (M1 / M2 solver params)
# ---------------------------------------------------------------------------

# Elliptic intentionally omitted — MJWarp only implements pyramidal cones.
_CONE_MAP = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
_SOLVER_MAP = {
    "PGS":    mujoco.mjtSolver.mjSOL_PGS,
    "CG":     mujoco.mjtSolver.mjSOL_CG,
    "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
}


def _patch_mujoco_options(mjm: mujoco.MjModel, cfg: ContactModelConfig) -> None:
    """Write MujocoSolverParams into the MjModel in place.

    Patches mjm.opt (cone, solver, iterations, tolerance); then — if
    cfg.mujoco.hard_contact is True — every geom's solref/solimp via
    _apply_hard_contact_preset; then — if cfg.mujoco.solref_timeconst /
    solref_dampratio are set — the solref columns via
    _apply_solref_override. The override runs last so it wins over the
    preset for M1 as well as over the XML defaults for M2/M3/M4.

    Raises ValueError if cone != 'pyramidal' since MJWarp does not
    support any other cone type on GPU.
    """
    p = cfg.mujoco
    if p.cone not in _CONE_MAP:
        raise ValueError(
            f"MJWarp only supports pyramidal friction cones; got "
            f"cone={p.cone!r}. All MJWarp-backed contact models (M1, "
            f"M2, and the MJWarp path of M3/M4) must use "
            f"cone='pyramidal'."
        )
    mjm.opt.cone       = _CONE_MAP[p.cone]
    mjm.opt.solver     = _SOLVER_MAP[p.solver]
    mjm.opt.iterations = p.iterations
    mjm.opt.tolerance  = p.tolerance

    if p.hard_contact:
        _apply_hard_contact_preset(mjm, p)

    # Runs after the preset so an explicit timeconst overrides it.
    _apply_solref_override(mjm, p)


def _apply_solref_override(mjm: mujoco.MjModel, p: MujocoSolverParams) -> None:
    """Overwrite the solref columns with explicit values, if any are set.

    solref is (timeconst, dampratio) per contact row: timeconst sets how
    fast position-level penetration is driven out, so shrinking it
    stiffens contact and growing it softens it. This helper writes those
    two numbers directly, independent of `hard_contact`, and leaves
    solimp alone — so sweeping `solref_timeconst` varies contact
    stiffness *only*, without also collapsing the constraint
    regularizer the way _apply_hard_contact_preset does.

    Each field is applied independently: setting only solref_timeconst
    keeps whatever dampratio the XML (or the hard preset) supplied.

    Stability
    ---------
    timeconst < 2·dt makes the semi-implicit integrator ring or diverge.
    Unlike _apply_hard_contact_preset, which clamps, this warns and
    applies the requested value anyway: a sweep probing the stiff limit
    should get the cell it asked for, and the resulting failure is data.

    Patches the same rows as the hard preset — mjm.geom_solref (ngeom, 2)
    and, when npair > 0, mjm.pair_solref (npair, 2) for contacts arising
    from explicit <pair> declarations.

    Side effect: mutates mjm in place (put_model saves and restores it).
    """
    if p.solref_timeconst is None and p.solref_dampratio is None:
        return

    if p.solref_timeconst is not None:
        dt = float(mjm.opt.timestep)
        if p.solref_timeconst < 2.0 * dt:
            warnings.warn(
                f"solref_timeconst={p.solref_timeconst:g}s is below the "
                f"2*dt={2.0 * dt:g}s stability floor for the semi-implicit "
                f"integrator (mjm.opt.timestep={dt:g}s); contact may ring or "
                f"diverge. Applying it as requested (no clamp).",
                stacklevel=3,
            )
        mjm.geom_solref[:, 0] = p.solref_timeconst
        if mjm.npair > 0:
            mjm.pair_solref[:, 0] = p.solref_timeconst

    if p.solref_dampratio is not None:
        mjm.geom_solref[:, 1] = p.solref_dampratio
        if mjm.npair > 0:
            mjm.pair_solref[:, 1] = p.solref_dampratio


def _apply_hard_contact_preset(mjm: mujoco.MjModel, p: MujocoSolverParams) -> None:
    """Push all contact rows toward the hard-constraint limit (M1).

    MuJoCo's constraint solver is a regularized convex QP:

        minimize  ½ vᵀ A v − vᵀ b     s.t.   v in friction cone

    where the diagonal regularizer R = 1/efc_D and the reference
    acceleration aref are derived per-row from solref/solimp. The
    regularized row-space equation is

        (J M⁻¹ Jᵀ + R) λ  =  aref − J v_free

    Driving solimp.d → 1 collapses R → 0 on contact rows; shrinking
    solref.timeconst tightens aref so position-level penetration is
    corrected within ~one step. In the joint limit R → 0 and the
    position correction becomes one-step tight, the regularized QP
    approaches the hard pyramidal-cone QP that Anitescu's formulation
    targets. This is NOT a bit-exact Anitescu solve — MJWarp keeps the
    convex-QP form throughout — but it is the stiff limit of MuJoCo's
    own math and stays fully parallel on GPU.

    Stability note
    --------------
    solref timeconst < 2·dt causes the semi-implicit integrator to
    ring or blow up. We clamp to max(timeconst_mult · dt, 2 · dt).

    What gets patched
    -----------------
    * mjm.geom_solref       shape (ngeom, mjNREF=2)
    * mjm.geom_solimp       shape (ngeom, mjNIMP=5)
    * mjm.pair_solref       shape (npair, 2)    (if npair > 0)
    * mjm.pair_solimp       shape (npair, 5)    (if npair > 0)

    Per-geom values are used when a contact is generated by the broad
    phase. Per-pair values override per-geom for contacts involving
    an explicit <pair> declared in XML. Patching both covers every
    contact path.

    Side effect: this mutates mjm in place. Calling put_model later
    with a different config will NOT restore the original XML values.
    """
    dt = float(mjm.opt.timestep)
    timeconst = max(p.hard_solref_timeconst_mult * dt, 2.0 * dt)

    solref_row = [timeconst, p.hard_solref_dampratio]
    solimp_row = [
        p.hard_solimp_d,         # dmin
        p.hard_solimp_d,         # dmax == dmin → flat, nearly-one impedance
        p.hard_solimp_width,
        p.hard_solimp_midpoint,
        p.hard_solimp_power,
    ]

    # Per-geom: numpy broadcast assigns the same row to every geom.
    mjm.geom_solref[:] = solref_row
    mjm.geom_solimp[:] = solimp_row

    # Per-pair overrides for explicit <pair> contacts in the XML.
    if mjm.npair > 0:
        mjm.pair_solref[:] = solref_row
        mjm.pair_solimp[:] = solimp_row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def put_model(mjm: mujoco.MjModel, cfg: ContactModelConfig):
    """Create a device-side model for the given contact config.

    Args:
        mjm:  Host-side MuJoCo model. Any geometry/physics-parameter
              degradation should already be baked into this MjModel
              before it reaches here.
        cfg:  Contact model configuration (selects backend + params).

    Returns:
        Backend-specific model object with a .contact_cfg attribute
        carrying the ContactModelConfig for downstream dispatch.
    """
    # Save all mjm fields that _patch_mujoco_options / _apply_hard_contact_preset
    # may mutate so the caller sees no side effects (e.g. when sweeping over
    # multiple configs with the same MjModel in the episodes loop).
    _saved_opt = (mjm.opt.cone, mjm.opt.solver, mjm.opt.iterations, mjm.opt.tolerance)
    _saved_geom_solref = mjm.geom_solref.copy() if mjm.ngeom > 0 else None
    _saved_geom_solimp = mjm.geom_solimp.copy() if mjm.ngeom > 0 else None
    _saved_pair_solref = mjm.pair_solref.copy() if mjm.npair > 0 else None
    _saved_pair_solimp = mjm.pair_solimp.copy() if mjm.npair > 0 else None

    try:
        if cfg.backend in (Backend.MUJOCO_HARD, Backend.MUJOCO_SOFT):
            _patch_mujoco_options(mjm, cfg)
            m = _mujoco_warp().put_model(mjm)

        elif cfg.backend == Backend.COMFREE:
            _patch_mujoco_options(mjm, cfg)
            m = _comfree_warp().put_model(
                mjm,
                comfree_stiffness=cfg.comfree.stiffness,
                comfree_damping=cfg.comfree.damping,
            )

        elif cfg.backend == Backend.XPBD:
            _patch_mujoco_options(mjm, cfg)
            mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            m = _xpbd_backend().put_model(mjm, cfg.xpbd)

        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")

    finally:
        # Restore mjm to its pre-call state regardless of success or failure.
        mjm.opt.cone, mjm.opt.solver, mjm.opt.iterations, mjm.opt.tolerance = _saved_opt
        if _saved_geom_solref is not None:
            mjm.geom_solref[:] = _saved_geom_solref
            mjm.geom_solimp[:] = _saved_geom_solimp
        if _saved_pair_solref is not None:
            mjm.pair_solref[:] = _saved_pair_solref
            mjm.pair_solimp[:] = _saved_pair_solimp

    m.contact_cfg = cfg
    return m


def make_data(mjm: mujoco.MjModel, m, nworld: int = 1,
              nconmax: int | None = None, njmax: int | None = None):
    """Allocate device-side data matching the model's backend."""
    cfg = m.contact_cfg
    kwargs = dict(nworld=nworld)
    if nconmax is not None:
        kwargs["nconmax"] = nconmax
    if njmax is not None:
        kwargs["njmax"] = njmax
    if cfg.backend in (Backend.MUJOCO_HARD, Backend.MUJOCO_SOFT):
        return _mujoco_warp().make_data(mjm, **kwargs)
    if cfg.backend == Backend.COMFREE:
        return _comfree_warp().make_data(mjm, **kwargs)
    if cfg.backend == Backend.XPBD:
        return _xpbd_backend().make_data(mjm, m, **kwargs)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def put_data(mjm: mujoco.MjModel, mjd: mujoco.MjData, m,
             nworld: int = 1, nconmax: int | None = None, njmax: int | None = None):
    """Upload host-side MjData to device."""
    cfg = m.contact_cfg
    kwargs = dict(nworld=nworld)
    if nconmax is not None:
        kwargs["nconmax"] = nconmax
    if njmax is not None:
        kwargs["njmax"] = njmax
    if cfg.backend in (Backend.MUJOCO_HARD, Backend.MUJOCO_SOFT):
        return _mujoco_warp().put_data(mjm, mjd, **kwargs)
    if cfg.backend == Backend.COMFREE:
        return _comfree_warp().put_data(mjm, mjd, **kwargs)
    if cfg.backend == Backend.XPBD:
        return _xpbd_backend().put_data(mjm, mjd, m, **kwargs)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def get_data_into(mjm: mujoco.MjModel, m, d, mjd: mujoco.MjData):
    """Download device-side data back to host."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_HARD, Backend.MUJOCO_SOFT, Backend.COMFREE):
        # library signature: get_data_into(result, mjm, d, world_id=0)
        # comfree_warp's wrapper incorrectly post-processes the None return value,
        # so call the underlying mjwarp function directly for all three backends.
        return _mujoco_warp().get_data_into(mjd, mjm, d)
    if cfg.backend == Backend.XPBD:
        return _xpbd_backend().get_data_into(mjm, m, d, mjd)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def reset_data(mjm: mujoco.MjModel, m, d):
    """Reset device-side data to the model default state."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_HARD, Backend.MUJOCO_SOFT):
        return _mujoco_warp().reset_data(m, d)
    if cfg.backend == Backend.COMFREE:
        return _comfree_warp().reset_data(m, d)
    if cfg.backend == Backend.XPBD:
        return _xpbd_backend().reset_data(mjm, m, d)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def step(m, d):
    """Advance simulation by one timestep."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_HARD, Backend.MUJOCO_SOFT):
        return _mujoco_warp().step(m, d)
    if cfg.backend == Backend.COMFREE:
        return _comfree_warp().step(m, d)
    if cfg.backend == Backend.XPBD:
        return _xpbd_backend().step(m, d)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def forward(m, d):
    """Run forward kinematics + dynamics (no integration)."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_HARD, Backend.MUJOCO_SOFT):
        return _mujoco_warp().forward(m, d)
    if cfg.backend == Backend.COMFREE:
        return _comfree_warp().forward(m, d)
    if cfg.backend == Backend.XPBD:
        return _xpbd_backend().forward(m, d)
    raise ValueError(f"Unknown backend: {cfg.backend}")