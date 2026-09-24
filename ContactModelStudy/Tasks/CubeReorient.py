"""Reorienting a lettered cube held in the LEAP hand.

The concrete task: everything specific to the cube — where it starts, where it
should end up, how far it can fall before the grasp counts as failed, and the
cost weights the study tuned for it. All the machinery is in ``LeapReorient``.

The numbers come from ``_OBJ_PARAMS["cube"]`` in
``contact_study/tasks/grasp_reorient.py`` and are reproduced here verbatim,
including the two hand-applied offsets in ``init_qpos`` (see ``objectParams``).
"""

from __future__ import annotations

import numpy as np

from ContactModelStudy.Tasks.LeapReorient import LeapReorient


class CubeReorient(LeapReorient):
    """Reorient the cube to a target pose with the LEAP hand.

    Scene: 16 actuated hand joints and a free-joint cube (nq=23, nv=22, nu=16).

    Example::

        task = CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT,
                                               hand_acc="high", obj_acc="high"))
        sim = Mujoco(task.getModelPath())
        q0, v0, u0 = task.getInitialState()
        sim.SetState(q0, v0)
        sim.SetControl(u0)
    """

    OBJECT = "cube"

    def objectParams(self) -> dict:
        """The cube's entry, carried over from the old task's table.

        Two offsets in ``init_qpos`` are written the way the old table writes
        them, as ``value + offset``, rather than folded into a single number:
        they were applied by hand on top of a pose read out of the viewer, and
        keeping them visible is the only record of that.

        ``th_axl`` (joint 13) is ``1.52604395 + 1.0``. Note this leaves the
        thumb 0.9939 rad away from what ``init_ctrl`` commands for the same
        joint, so the position servo pulls it there over the first few steps —
        the initial state is *not* an equilibrium of the initial control. That
        is inherited behaviour, flagged here because it looks like drift in any
        "did the hand hold its pose?" check.
        """
        return {
            # 16 hand joints, then the cube's settled in-palm pos(3) + quat(4).
            "init_qpos": np.array([
                7.41953443e-01, -5.14095650e-01,  6.97705793e-01,  5.73857360e-01,
                3.11686592e-01, -2.08901684e-05,  7.04119781e-01,  1.01887562e+00,
                7.14271347e-01,  2.63610945e-01,  6.97700993e-01,  6.10100133e-01,
                7.00288255e-01,  1.52604395e+00 + 1.0,  1.33871871e+00,  8.68983906e-01,

                1.70374863e-02,  3.65435775e-02,  8.36225067e-02 + 0.02,
                1.0, 0.0, 0.0, 0.0,
            ]),
            "init_ctrl": np.array([
                0.7672,   -0.51303,   0.701455,  0.573897,  0.33472,   0.0,
                0.709056,  1.01884,   0.74176,   0.26175,   0.701455,  0.610097,
                0.69912,   1.53211,   1.33179,   0.8657,
            ]),
            "target_pos": np.array([0.02, 0.035, 0.085]),
            # Identity: the study's goal orientation is the cube's canonical
            # face-up pose. The old code derived this from _TARGET_EULER =
            # (0, 0, 0), which is the identity quaternion.
            "target_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "fallen_z": 0.08,
            "cost_weights": {
                "w_quat":        23.3929,
                "w_pos_x":        5.99412,
                "w_pos_y":       50.0,
                "w_pos_z":        1.0,
                # Scales the L2 position error, not a velocity term — see
                # LeapReorient.COST_WEIGHT_KEYS. Zero for the cube either way.
                "w_velo":         0.0,
                "w_contact":      2.48369,
                "w_joint":       20.0,
                "w_joint_velo":   0.0,
                "w_fallen":     100.0,
                "w_quat_term":  100.0,
                "w_pos_term":   300.0,
                "w_fallen_term":  0.0,
            },
        }
