"""Actuator-test task — a 2-joint hinge chain, each joint driven by its own
position actuator (scenes/actuator_test.xml), used to compare how each
simulator's actuator model tracks a commanded joint position. Not a
manipulation task: is_success and has_failed always return False (there is
no goal to reach or fail), and the cost is simply the sum of each joint's
squared distance from the zero position.

Joint/actuator discovery is generic (loops over every non-free joint in the
loaded model), so this scales to however many joints the XML declares — it is
not hardcoded to 2.

Rollout and eval both load the same static MJCF (no {geometry} template, no
on-the-fly build). Eval defaults to MuJoCo; a Pinocchio eval simulator is also
available (PID/PD gains mirror grasp_reorient.py's hand actuation constants).
"""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, register, SCENES_DIR
from .config import TaskConfig, EvalSimulatorKind

ACTUATOR_TEST_SCENE_XML = "actuator_test.xml"

# Position-control stiffness, mirroring grasp_reorient.py's hand PID/PD gains
# (also matches this scene's own <position kp="3.0" kv="0.01"/> default).
_PID_KP = 3.0

# Pinocchio eval PD: same scheme as grasp_reorient.py — kd is either derived
# per-joint from the mass matrix (critically damped) or applied directly.
_PIN_ZETA          = 1.0
_PIN_USE_DIRECT_KD = True
_PIN_KD            = 0.1 + 0.01
_PIN_GRAVITY_COMP  = False
_PIN_ARMATURE      = 0.001


@wp.func
def actuator_test_cost_wp(qpos: wp.array(dtype=float),
                           qvel: wp.array(dtype=float),
                           ctrl: wp.array(dtype=float),
                           site_xpos: wp.array(dtype=wp.vec3),
                           site_xmat: wp.array(dtype=wp.mat33),
                           terminal: bool,
                           goal: wp.array(dtype=float),
                           indices: wp.array(dtype=int),
                           weights: wp.array(dtype=float)) -> float:
    n_joints = indices[0]
    cost = float(0.0)
    for i in range(n_joints):
        dq = qpos[indices[1 + i]] - goal[i]
        cost = cost + dq * dq
    return weights[0] * cost


@register("actuator_test")
class ActuatorTestTask(BaseTask):
    """Drive a chain of hinge joints and compare actuator tracking across sims.

    Contact complexity: LOW (a simple hinge chain, no contacts of interest).
    """

    def __init__(self, geometry=None, role=None):
        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if role is not None:
            kwargs["role"] = role
        super().__init__(**kwargs)

        self.config = TaskConfig(
            name               = "actuator_test",
            complexity         = ContactComplexity.LOW,
            max_steps          = 200,
            success_thresholds = {},
            cost_weights       = {"w_joint": 1.0},
            xml_path_template  = ACTUATOR_TEST_SCENE_XML,
            rollout_model_path = str(SCENES_DIR / ACTUATOR_TEST_SCENE_XML),
            rollout_is_urdf    = False,
            eval_sim           = EvalSimulatorKind.MUJOCO,
            eval_model_paths   = {
                EvalSimulatorKind.MUJOCO:    str(SCENES_DIR / ACTUATOR_TEST_SCENE_XML),
                EvalSimulatorKind.PINOCCHIO: str(SCENES_DIR / ACTUATOR_TEST_SCENE_XML),
            },
            timestep                  = 0.0001,
            eval_substeps_per_rollout = 10,
        )

    def _hinge_joint_ids(self) -> list[int]:
        mjm = self.mjm
        return [j for j in range(mjm.njnt) if mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]

    def initialize_task(self):
        mjm = self.mjm
        qpos_adrs = [int(mjm.jnt_qposadr[j]) for j in self._hinge_joint_ids()]
        n = len(qpos_adrs)

        self.index_vector = np.array([n, *qpos_adrs], dtype=np.int32)
        self.goal_vector = np.zeros(n, dtype=np.float32)

        w = self.config.cost_weights
        weights_list = [w["w_joint"]]

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")
        self.weights_wp = wp.array(weights_list, dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0 = np.zeros(mjm.nq, dtype=np.float64)
        v0 = np.zeros(mjm.nv, dtype=np.float64)
        ctrl0 = np.zeros(mjm.nu, dtype=np.float64)
        for a in range(mjm.nu):
            jid = int(mjm.actuator(a).trnid[0])
            upper = float(mjm.actuator_ctrlrange[a, 1])
            q0[mjm.jnt_qposadr[jid]] = upper
            ctrl0[a] = upper
        return q0, v0, ctrl0

    @property
    def cost_fn_wp(self) -> wp.func:
        return actuator_test_cost_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        return False

    def has_failed(self, mjd: mujoco.MjData) -> bool:
        return False

    # --- eval simulator -------------------------------------------------
    def make_eval_simulator(self, video_path: str | None = None, render: bool = True):
        if self.config.eval_sim == EvalSimulatorKind.PINOCCHIO:
            return self._make_pinocchio_simulator(video_path=video_path, render=render)
        return super().make_eval_simulator(video_path=video_path, render=render)

    def _make_pinocchio_simulator(self, video_path: str | None = None, render: bool = True):
        """Pinocchio eval simulator for the hinge chain (no free bodies, no
        contacts of interest — each adjacent body pair is excluded via the
        MJCF's own <contact><exclude> tags). PD gains mirror grasp_reorient.py's
        hand actuation."""
        from contact_study.contact_models.pinocchio_sim import (
            PinocchioSimulator, PinocchioJointChannel, PinocchioPdActuation,
        )

        mjm = self.mjm
        joint_channels = [
            PinocchioJointChannel(
                pin_name=mjm.joint(j).name,
                q_adr=int(mjm.jnt_qposadr[j]),
                v_adr=int(mjm.jnt_dofadr[j]),
            )
            for j in self._hinge_joint_ids()
        ]
        ctrl_joint_names = [
            mjm.joint(int(mjm.actuator(a).trnid[0])).name for a in range(mjm.nu)
        ]
        # Per-actuator torque limit from each joint's actuatorfrcrange, so the
        # Pinocchio PD saturates like MuJoCo. This scene declares no limit, so
        # every entry is (-inf, inf) and no clamp is applied — but wiring it here
        # keeps the pattern identical to grasp_reorient.
        force_limit = np.array([
            mjm.jnt_actfrcrange[int(mjm.actuator(a).trnid[0])]
            if mjm.jnt_actfrclimited[int(mjm.actuator(a).trnid[0])]
            else (-np.inf, np.inf)
            for a in range(mjm.nu)
        ], dtype=np.float64)
        pid = PinocchioPdActuation(
            ctrl_joint_names=ctrl_joint_names,
            kp=_PID_KP, zeta=_PIN_ZETA,
            gravity_comp=_PIN_GRAVITY_COMP,
            armature=_PIN_ARMATURE,
            use_direct_kd=_PIN_USE_DIRECT_KD,
            kd=_PIN_KD,
            force_limit=force_limit,
        )

        return PinocchioSimulator(
            model_path     = self.config.eval_model_paths[self.config.eval_sim],
            config         = self.config,
            nq             = self.mjm.nq,
            nv             = self.mjm.nv,
            pid            = pid,
            joint_channels = joint_channels,
            free_channels  = [],
            video_path     = video_path,
            render         = render,
        )
