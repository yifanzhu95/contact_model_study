"""Reorienting a rubber duck held in the LEAP hand.

Everything specific to the duck: where it starts, where it should end up, how
low it can drop before the cost calls it fallen, and its cost weights. All the
machinery is in ``LeapReorient``.

The numbers come from ``_OBJ_PARAMS["duck"]`` in
``contact_study/tasks/grasp_reorient.py`` and are reproduced verbatim,
including the hand-applied offsets in ``init_qpos`` (see ``objectParams``).

The duck has more rollout scenes than the cube: besides ``obj_acc`` "low",
"med" and "high", it has foam variants ("foam4", "foam16a", "foam16b",
"foam64").
"""

from __future__ import annotations

import numpy as np

from ContactModelStudy.Tasks.LeapReorient import LeapReorient
from ContactModelStudy.Utils.Quaternions import quatMul


class DuckReorient(LeapReorient):
    """Reorient the duck to a target pose with the LEAP hand.

    Scene: 16 actuated hand joints and a free-joint duck (nq=23, nv=22, nu=16).
    The duck is lighter than the cube (0.05 kg against 0.14) and about 1.5 cm
    taller, so it starts higher and its target sits higher too.
    """

    OBJECT = "duck"

    #: This object's tuning of the contact models, over their presets; see
    #: ``TaskBase.CONTACT_PARAMS``. Empty: every model runs its preset.
    CONTACT_PARAMS = {
        # "M3": {"stiffness": 0.75, "damping": 0.01},
    }

    #: The duck's centre of mass in its body frame: the ``<inertial pos>`` that
    #: every duck scene (eval, rollout and foam variants) declares.
    COM_OFFSET = np.array([-0.0088857, -0.0000003, -0.0152658])

    #: Where the duck's COM should end up, whatever the goal orientation: where
    #: it rests in the ``init_ctrl`` grasp, settled in the M5 eval sim.
    COM_TARGET = np.array([0.025, 0.041, 0.073])

    def objectParams(self) -> dict:
        """The duck's start, grasp and target, re-chosen for the duck.

        ``init_qpos`` is a resting state: the duck sitting in the grasp that
        ``init_ctrl`` commands, as settled in the M5 (Kamino) eval sim, with
        every hand joint inside its range, so the driver's settle period is
        close to a no-op. ``init_ctrl`` is the ball's grasp, which came out of
        the same search for the duck: it dropped nothing under MPPI-shaped
        random control walks and let the duck wander least.

        ``target_pos`` is the target for the body *origin*, which the cost and
        the success test measure. The origin sits 1.8 cm from the COM, so where
        it rests depends on how the duck is turned. A single origin target puts
        some goal orientations up to ~3.5 cm from where the duck could rest in
        them. So ``COM_TARGET`` is what is fixed, and ``setGoal`` moves the
        origin target with each goal orientation (see ``originTarget``).

        ``fallen_z`` is 0.05, down from 0.075. It too is tested on the origin,
        which is as low as ~0.066 when the duck is held upside down, so at 0.075
        the cost charged the drop penalty for some goals while the duck was
        still in the hand.

        The old table's entry is kept commented out below it. It started the
        thumb (``th_axl``, joint 13) at ``1.34131893 + 1.0``, past its joint
        range, which swung the thumb through the duck while it settled. Its
        target was ~3.5 cm from where the duck rests.

        The goal orientation is the identity, as for every object in the old
        task (``_TARGET_QUAT``). The duck does not start there: its start
        quaternion is roughly a quarter turn about z.

        The cost weights were tuned against the old target.
        """
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        return {
            # 16 hand joints, then the duck's in-palm pos(3) + quat(4).
            "init_qpos": np.array([
                0.56820, -0.46091,  1.09432,  1.15528,  0.38783, -0.06142,  0.70435,  0.97591,
                0.57785,  0.01632,  0.96194,  0.95323,  0.73765,  1.24852,  1.21775,  0.46786,

                0.02606,  0.04974,  0.08836,
                0.73385,  0.01031,  0.00912,  0.67917,
            ]),
            "init_ctrl": np.array([
                0.58968, -0.46068,  1.09597,  1.15426,  0.41104, -0.06144,  0.70906,  0.97582,
                0.60184,  0.01615,  0.96447,  0.95257,  0.73813,  1.25667,  1.21173,  0.46540,
            ]),
            "target_pos": self.originTarget(target_quat),
            "target_quat": target_quat,
            "fallen_z": 0.05,
            # The old table's start, grasp, target and drop height:
            # "init_qpos": np.array([
            #     5.80303253e-01, -4.60995160e-01,  1.09418735e+00,  9.71266541e-01,
            #     3.86906512e-01, -1.10955454e-05,  7.04409919e-01,  1.01898232e+00,
            #     5.78769646e-01,  2.09500294e-01,  9.62015366e-01,  1.01933705e+00,
            #     7.38906405e-01,  1.34131893e+00 + 1.0,  1.28450986e+00,  6.59284046e-01,
            #
            #     2.56294614e-02 - 0.02,  4.96063104e-02,  8.82105693e-02 + 0.05,
            #     6.97889476e-01,  1.36622853e-02, -3.23292221e-02,  7.15344982e-01,
            # ]),
            # "init_ctrl": np.array([
            #     0.60184,  -0.46068,   1.09597,   0.97044,   0.41104,   0.0,
            #     0.709056,  1.01884,   0.60184,   0.2094,    0.964465,  1.0186,
            #     0.738135,  1.251165,  1.2778,    0.6564,
            # ]),
            # "target_pos": np.array([0.03, 0.015, 0.09]),
            # "fallen_z": 0.075,
            # Retuned for the duck in the old study.
            "cost_weights": {
                "w_quat":         100.0,#23.0224,
                "w_pos_x":        0.01,#0.01,
                "w_pos_y":        100.0,#100.0,
                "w_pos_z":        0.6222,#0.361846,
                "w_velo":         0.0,
                "w_contact":      100.0,
                "w_joint":        0.1095,#6.20467,
                "w_joint_velo":   0.0,
                "w_fallen":       24.887,#93.6095,
                "w_quat_term":    39.5931,#145.812,
                "w_pos_term":     1000.0,#500.0,
                "w_fallen_term":  0.0,
            },
        }

    @classmethod
    def originTarget(cls, quat: np.ndarray) -> np.ndarray:
        """Body-origin position that puts the COM on ``COM_TARGET`` at orientation ``quat``.

        ``COM_TARGET - R(quat) @ COM_OFFSET``, with the rotation applied as
        ``quat * (0, v) * conj(quat)``.
        """
        q = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
        conj = q * np.array([1.0, -1.0, -1.0, -1.0])
        offset = quatMul(quatMul(q, np.array([0.0, *cls.COM_OFFSET])), conj)[1:]
        return cls.COM_TARGET - offset

    def setGoal(self, goal: np.ndarray) -> None:
        """Adopt ``goal`` and move the origin target with it; see ``LeapReorient.setGoal``.

        The target position is set before the base class rebuilds the goal
        vector and writes it to the device, so both change in the same call.
        """
        q = self._unitQuat(goal)
        self.params["target_pos"] = self.originTarget(q)
        super().setGoal(q)
