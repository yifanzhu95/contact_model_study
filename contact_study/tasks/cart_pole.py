"""Cart-pole swing-up task — testing vehicle for the Drake eval pipeline.

The MuJoCo *rollout* model is built from scenes/cart_pole.urdf via MjSpec (a
force actuator is added on the cart slider, a pole-tip site is added, and the
compiled model is saved as scenes/cart_pole.xml). The *eval* environment is
Drake, loading scenes/cart_pole.sdf (already exposes one CartSlider actuator and
welds the cart to the world). The URDF and SDF must stay parameter-consistent.

Moved out of tests/test_drake.py and registered so it loads through the standard
task registry. The pole hangs down at hinge angle 0 and is upright at ±π.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask
from .config import (
    ContactComplexity,
    TaskConfig,
    EvalSimulatorKind,
)

SCENES   = Path(__file__).parents[2] / "scenes"
URDF     = SCENES / "cart_pole.urdf"   # source for the MuJoCo rollout model
MJCF_OUT = SCENES / "cart_pole.xml"    # generated MJCF ("save as MJCF")
SDF      = SCENES / "cart_pole.sdf"    # Drake "real" env model

TIMESTEP  = 0.02          # MuJoCo planning timestep (s); also the control dt
FORCE_MAX = 100.0         # cart actuator force limit (N)
UPRIGHT   = float(np.pi)  # pole angle (rad) that points straight up


# ---------------------------------------------------------------------------
# Warp cost — angle-from-upright + cart-distance-from-zero
# ---------------------------------------------------------------------------
@wp.func
def cart_pole_cost_wp(qpos:      wp.array(dtype=float),
                      qvel:      wp.array(dtype=float),
                      ctrl:      wp.array(dtype=float),
                      site_xpos: wp.array(dtype=wp.vec3),
                      site_xmat: wp.array(dtype=wp.mat33),
                      terminal:  bool,
                      goal:      wp.array(dtype=float),
                      indices:   wp.array(dtype=int),
                      weights:   wp.array(dtype=float)) -> float:
    slider_qpos_adr = indices[0]
    hinge_qpos_adr  = indices[1]
    slider_dof_adr  = indices[2]
    hinge_dof_adr   = indices[3]

    x     = qpos[slider_qpos_adr]
    theta = qpos[hinge_qpos_adr]

    # Wrap-safe angle from upright: 0 when the pole points up (theta = +/- pi),
    # +/- pi when it hangs straight down. atan2(sin, cos) maps to (-pi, pi].
    d       = theta - goal[1]
    ang_err = wp.atan2(wp.sin(d), wp.cos(d))
    pos_err = x - goal[0]

    if terminal:
        return weights[2] * ang_err * ang_err + weights[3] * pos_err * pos_err

    cart_vel = qvel[slider_dof_adr]
    pole_vel = qvel[hinge_dof_adr]
    return (
        weights[0] * ang_err * ang_err
        + weights[1] * pos_err * pos_err
        + weights[4] * cart_vel * cart_vel
        + weights[5] * pole_vel * pole_vel
    )


class CartPoleTask(BaseTask):
    """Cart-pole swing-up backed by a URDF-derived MuJoCo rollout model."""

    # Initial pole angle (rad). 0.0 = hanging straight down (swing-up). Set to
    # ~pi to start near upright (balance-only). Overridable before load().
    init_angle: float = 0.0

    def __init__(self, geometry=None, role=None):
        # Tolerate get_task passing geometry/role; BaseTask defaults handle None.
        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if role is not None:
            kwargs["role"] = role
        super().__init__(**kwargs)

        self.config = TaskConfig(
            name               = "cart_pole",
            complexity         = ContactComplexity.LOW,
            max_steps          = 1000,
            success_thresholds = {"angle": 0.20, "pos": 0.20, "vel": 1.0},
            # [angle_run, pos_run, angle_term, pos_term, cart_vel, pole_vel]
            cost_weights       = {
                "angle": 100.0, "pos": 75.0,
                "angle_term": 100.0, "pos_term": 100.0,
                "cart_vel": 2.0, "pole_vel": 2.0,
            },
            rollout_model_path = str(URDF),
            rollout_is_urdf    = True,
            mjcf_out_path      = str(MJCF_OUT),
            eval_sim           = EvalSimulatorKind.DRAKE,
            eval_model_path    = str(SDF),
            cam_pos            = (0.0, -2.5, 0.25),
            cam_fps            = 30.0,
            timestep           = TIMESTEP,
            force_limits       = (-FORCE_MAX, FORCE_MAX),
        )

    # --- load: URDF -> add actuator -> compile -> save MJCF -----------------
    def load(self, full_path: str | None = None):
        """Build the MuJoCo model from the URDF and persist it as MJCF."""
        spec = mujoco.MjSpec.from_file(str(URDF))
        spec.option.timestep   = self.config.timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # Force actuator on the cart slider (force = ctrl; a default general
        # actuator is gain-fixed/bias-none, i.e. a motor).
        act = spec.add_actuator()
        act.name        = "cart_force"
        act.target      = "CartSlider"
        act.trntype     = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype    = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm     = [1.0] + list(act.gainprm[1:])
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange   = list(self.config.force_limits)

        # A pole-tip site keeps the cost kernel's site arrays non-empty.
        tip = spec.body("Pole").add_site()
        tip.name = "pole_tip"
        tip.pos  = [0.0, 0.0, -0.5]

        self._mjm = spec.compile()
        self._mjd = mujoco.MjData(self._mjm)
        Path(self.config.mjcf_out_path).write_text(spec.to_xml())

        self.initialize_task()
        return self._mjm, self._mjd

    def initialize_task(self):
        mjm = self.mjm
        slider = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "CartSlider")
        hinge  = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "PolePin")

        # [slider_qposadr, hinge_qposadr, slider_dofadr, hinge_dofadr]
        self.index_vector = np.array([
            mjm.jnt_qposadr[slider],
            mjm.jnt_qposadr[hinge],
            mjm.jnt_dofadr[slider],
            mjm.jnt_dofadr[hinge],
        ], dtype=np.int32)

        # [cart_target_x, upright_angle]
        self.goal_vector = np.array([0.0, UPRIGHT], dtype=np.float32)

        w = self.config.cost_weights
        weights_list = [
            w["angle"], w["pos"], w["angle_term"], w["pos_term"],
            w["cart_vel"], w["pole_vel"],
        ]

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32,   device="cuda")
        self.goal_vector_wp  = wp.array(self.goal_vector,  dtype=wp.float32, device="cuda")
        self.weights_wp      = wp.array(weights_list,      dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        q0 = np.zeros(self.mjm.nq, dtype=np.float64)
        q0[self.index_vector[0]] = 0.0                                  # cart at origin
        q0[self.index_vector[1]] = self.init_angle + rng.uniform(-0.02, 0.02)
        v0 = np.zeros(self.mjm.nv, dtype=np.float64)
        u0 = np.zeros(self.mjm.nu, dtype=np.float64)
        return q0, v0, u0

    @property
    def cost_fn_wp(self) -> wp.func:
        return cart_pole_cost_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        x     = mjd.qpos[self.index_vector[0]]
        theta = mjd.qpos[self.index_vector[1]]
        d     = theta - UPRIGHT
        ang_err = np.arctan2(np.sin(d), np.cos(d))
        thr = self.config.success_thresholds
        return bool(
            abs(ang_err) < thr["angle"]
            and abs(x) < thr["pos"]
            and np.linalg.norm(mjd.qvel) < thr["vel"]
        )

    # --- eval simulator -----------------------------------------------------
    def make_eval_simulator(self, video_path: str | None = None, render: bool = True):
        if self.config.eval_sim == EvalSimulatorKind.DRAKE:
            from contact_study.contact_models.drake_sim import (
                DrakeSimulator, DrakeJointChannel,
            )
            iv = self.index_vector
            channels = [
                DrakeJointChannel("CartSlider", "prismatic", q_adr=int(iv[0]), v_adr=int(iv[2])),
                DrakeJointChannel("PolePin",    "revolute",  q_adr=int(iv[1]), v_adr=int(iv[3])),
            ]
            return DrakeSimulator(
                model_path     = self.config.eval_model_path,
                config         = self.config,
                nq             = self.mjm.nq,
                nv             = self.mjm.nv,
                joint_channels = channels,
                n_ctrl         = self.mjm.nu,
                weld_base      = False,   # SDF already welds the cart to world
                video_path     = video_path,
            )
        return super().make_eval_simulator(video_path=video_path, render=render)


# Register at import time (keeps the decorator-free constructor above readable).
from .base import register  # noqa: E402
register("cart_pole")(CartPoleTask)
