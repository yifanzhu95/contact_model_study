"""Unified contact model API.

This module is the single entry point for all contact model variants.
It mirrors the comfree_warp.api surface (put_model, make_data, put_data,
get_data_into, reset_data, step, forward) but accepts a ContactModelConfig
to dispatch to the correct backend.

Usage::

    from contact_study.contact_models.api import put_model, make_data, step
    from contact_study.contact_models.config import ContactModelConfig

    cfg  = ContactModelConfig.M3()
    m    = put_model(mjm, cfg)
    d    = make_data(mjm, m)

    for _ in range(horizon):
        step(m, d)
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp

from .config import Backend, ContactModelConfig, PhysicsNoiseParams

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
# Physics noise injection
# ---------------------------------------------------------------------------

def _apply_physics_noise(mjm: mujoco.MjModel, noise: PhysicsNoiseParams, rng: np.random.Generator) -> mujoco.MjModel:
    """Return a copy of mjm with noise applied to physical parameters.

    All perturbations are multiplicative: x <- x * (1 + N(0, sigma)).
    """
    if all(getattr(noise, f) == 0.0 for f in ("mass_sigma", "inertia_sigma", "friction_sigma", "com_sigma")):
        return mjm  # fast path: no noise

    # mujoco.MjModel is not directly copyable; we go via XML roundtrip.
    # For batched rollouts a pre-perturbed array of models is preferred;
    # see utils/rollout.py for the batched variant.
    import copy
    mjm_noisy = copy.deepcopy(mjm)

    if noise.mass_sigma > 0.0:
        mjm_noisy.body_mass[:] *= (1.0 + rng.normal(0, noise.mass_sigma, mjm.nbody))

    if noise.inertia_sigma > 0.0:
        mjm_noisy.body_inertia[:] *= (1.0 + rng.normal(0, noise.inertia_sigma, (mjm.nbody, 3)))

    if noise.friction_sigma > 0.0:
        mjm_noisy.geom_friction[:] *= (1.0 + rng.normal(0, noise.friction_sigma, mjm_noisy.geom_friction.shape))
        mjm_noisy.geom_friction[:] = np.clip(mjm_noisy.geom_friction, 0.01, None)

    if noise.com_sigma > 0.0:
        mjm_noisy.body_ipos[:] += rng.normal(0, noise.com_sigma, mjm_noisy.body_ipos.shape)

    return mjm_noisy


# ---------------------------------------------------------------------------
# MuJoCo option patching (M1 / M2 solver params)
# ---------------------------------------------------------------------------

_CONE_MAP = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic":  mujoco.mjtCone.mjCONE_ELLIPTIC,
}
_SOLVER_MAP = {
    "PGS":    mujoco.mjtSolver.mjSOL_PGS,
    "CG":     mujoco.mjtSolver.mjSOL_CG,
    "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
}

def _patch_mujoco_options(mjm: mujoco.MjModel, cfg: ContactModelConfig) -> None:
    """Write MujocoSolverParams into the MjModel option in-place."""
    p = cfg.mujoco
    mjm.opt.cone       = _CONE_MAP[p.cone]
    mjm.opt.solver     = _SOLVER_MAP[p.solver]
    mjm.opt.iterations = p.iterations
    mjm.opt.tolerance  = p.tolerance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def put_model(
    mjm: mujoco.MjModel,
    cfg: ContactModelConfig,
    rng: np.random.Generator | None = None,
):
    """Create a device-side model for the given contact config.

    Args:
        mjm:  Host-side MuJoCo model.
        cfg:  Contact model configuration (selects backend + params).
        rng:  RNG for physics noise (M7-M10). If None a default RNG is created.

    Returns:
        Backend-specific model object. Always has a .contact_cfg attribute
        carrying the ContactModelConfig for downstream dispatch.

    Note:
        nworld is NOT a model-level concept upstream — it belongs to
        make_data / put_data, matching the comfree_warp / mujoco_warp API.
    """
    if rng is None:
        rng = np.random.default_rng()

    mjm = _apply_physics_noise(mjm, cfg.physics_noise, rng)

    if cfg.backend in (Backend.MUJOCO_ANITESCU, Backend.MUJOCO_SOFT):
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
        m = _xpbd_backend().put_model(mjm, cfg.xpbd)

    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    m.contact_cfg = cfg
    return m


def make_data(mjm: mujoco.MjModel, m, nworld: int = 1, nconmax: int | None = None, njmax: int | None = None):
    """Allocate device-side data matching the model's backend."""
    cfg = m.contact_cfg
    kwargs = dict(nworld=nworld)
    if nconmax is not None:
        kwargs["nconmax"] = nconmax
    if njmax is not None:
        kwargs["njmax"] = njmax
    if cfg.backend in (Backend.MUJOCO_ANITESCU, Backend.MUJOCO_SOFT):
        return _mujoco_warp().make_data(mjm, **kwargs)
    elif cfg.backend == Backend.COMFREE:
        return _comfree_warp().make_data(mjm, **kwargs)
    elif cfg.backend == Backend.XPBD:
        return _xpbd_backend().make_data(mjm, m, **kwargs)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def put_data(mjm: mujoco.MjModel, mjd: mujoco.MjData, m, nworld: int = 1, nconmax: int | None = None, njmax: int | None = None):
    """Upload host-side MjData to device."""
    cfg = m.contact_cfg
    kwargs = dict(nworld=nworld)
    if nconmax is not None:
        kwargs["nconmax"] = nconmax
    if njmax is not None:
        kwargs["njmax"] = njmax
    if cfg.backend in (Backend.MUJOCO_ANITESCU, Backend.MUJOCO_SOFT):
        return _mujoco_warp().put_data(mjm, mjd, **kwargs)
    elif cfg.backend == Backend.COMFREE:
        return _comfree_warp().put_data(mjm, mjd, **kwargs)
    elif cfg.backend == Backend.XPBD:
        return _xpbd_backend().put_data(mjm, mjd, m, **kwargs)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def get_data_into(mjm: mujoco.MjModel, m, d, mjd: mujoco.MjData):
    """Download device-side data back to host."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_ANITESCU, Backend.MUJOCO_SOFT):
        return _mujoco_warp().get_data_into(mjm, m, d, mjd)
    elif cfg.backend == Backend.COMFREE:
        return _comfree_warp().get_data_into(mjm, m, d, mjd)
    elif cfg.backend == Backend.XPBD:
        return _xpbd_backend().get_data_into(mjm, m, d, mjd)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def reset_data(mjm: mujoco.MjModel, m, d):
    """Reset device-side data to the model default state."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_ANITESCU, Backend.MUJOCO_SOFT):
        return _mujoco_warp().reset_data(mjm, m, d)
    elif cfg.backend == Backend.COMFREE:
        return _comfree_warp().reset_data(mjm, m, d)
    elif cfg.backend == Backend.XPBD:
        return _xpbd_backend().reset_data(mjm, m, d)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def step(m, d):
    """Advance simulation by one timestep."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_ANITESCU, Backend.MUJOCO_SOFT):
        return _mujoco_warp().step(m, d)
    elif cfg.backend == Backend.COMFREE:
        return _comfree_warp().step(m, d)
    elif cfg.backend == Backend.XPBD:
        return _xpbd_backend().step(m, d)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def forward(m, d):
    """Run forward kinematics + dynamics (no integration)."""
    cfg = m.contact_cfg
    if cfg.backend in (Backend.MUJOCO_ANITESCU, Backend.MUJOCO_SOFT):
        return _mujoco_warp().forward(m, d)
    elif cfg.backend == Backend.COMFREE:
        return _comfree_warp().forward(m, d)
    elif cfg.backend == Backend.XPBD:
        return _xpbd_backend().forward(m, d)
    raise ValueError(f"Unknown backend: {cfg.backend}")
