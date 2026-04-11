"""Physics parameter noise injection.

Orthogonal to contact models. Call `apply_physics_noise` on an MjModel
BEFORE handing it to contact_study.contact_models.api.put_model, to run
any Mk (M1..M4) against a perturbed copy of the true model.

example usage: 
    from contact_study.contact_models.config import ContactModelConfig
    from contact_study.contact_models import api
    from contact_study.utils.physics_noise import (
        PhysicsNoiseParams, apply_physics_noise,
    )

    noise = PhysicsNoiseParams(mass_sigma=0.1, friction_sigma=0.2)
    mjm_noisy = apply_physics_noise(mjm, noise, rng)

    cfg = ContactModelConfig.M3()         # any contact model
    m   = api.put_model(mjm_noisy, cfg)

Perturbations are multiplicative except CoM which is additive:

    mass, inertia, friction:   x  <-  x * (1 + N(0, sigma))
    body_ipos (CoM position):  x  <-  x + N(0, sigma)
"""

from __future__ import annotations

import copy
import dataclasses

import mujoco
import numpy as np


@dataclasses.dataclass
class PhysicsNoiseParams:
    mass_sigma:     float = 0.0
    inertia_sigma:  float = 0.0
    friction_sigma: float = 0.0
    com_sigma:      float = 0.0   # additive noise on CoM position (meters)


def apply_physics_noise(
    mjm: mujoco.MjModel,
    noise: PhysicsNoiseParams,
    rng: np.random.Generator | None = None,
) -> mujoco.MjModel:
    """Return a deep-copied MjModel with noise applied to physical params.

    Fast path: if all sigmas are zero, returns the input unchanged
    (no copy). Otherwise a deepcopy is made so the caller's model is
    never mutated.
    """
    if all(getattr(noise, f) == 0.0
           for f in ("mass_sigma", "inertia_sigma", "friction_sigma", "com_sigma")):
        return mjm

    rng = rng or np.random.default_rng()
    mjm_noisy = copy.deepcopy(mjm)

    if noise.mass_sigma > 0.0:
        mjm_noisy.body_mass[:] *= (1.0 + rng.normal(0, noise.mass_sigma, mjm.nbody))

    if noise.inertia_sigma > 0.0:
        mjm_noisy.body_inertia[:] *= (
            1.0 + rng.normal(0, noise.inertia_sigma, (mjm.nbody, 3))
        )

    if noise.friction_sigma > 0.0:
        mjm_noisy.geom_friction[:] *= (
            1.0 + rng.normal(0, noise.friction_sigma, mjm_noisy.geom_friction.shape)
        )
        mjm_noisy.geom_friction[:] = np.clip(mjm_noisy.geom_friction, 0.01, None)

    if noise.com_sigma > 0.0:
        mjm_noisy.body_ipos[:] += rng.normal(
            0, noise.com_sigma, mjm_noisy.body_ipos.shape
        )

    return mjm_noisy