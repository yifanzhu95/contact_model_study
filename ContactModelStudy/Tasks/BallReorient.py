"""Reorienting a ball held in the LEAP hand.

Everything specific to the ball: where it starts, where it should end up, how
low it can drop before the cost calls it fallen, and its cost weights. All the
machinery is in ``LeapReorient``.

The numbers come from ``_OBJ_PARAMS["ball"]`` in
``contact_study/tasks/grasp_reorient.py`` and are reproduced verbatim,
including the hand-applied offsets in ``init_qpos`` (see ``objectParams``).

The ball is a sphere of radius 0.0485 m (9.7 cm across, against the cube's 7 cm)
with the cube's mass, 0.14 kg. A sphere looks the same to the contact solver in
every orientation, so the reorientation goal is only visible through the ball's
cube-mapped texture.
"""

from __future__ import annotations

import numpy as np

from ContactModelStudy.Tasks.LeapReorient import LeapReorient


class BallReorient(LeapReorient):
    """Reorient the ball to a target pose with the LEAP hand.

    Scene: 16 actuated hand joints and a free-joint ball (nq=23, nv=22, nu=16).
    """

    OBJECT = "ball"

    def objectParams(self) -> dict:
        """The ball's start, grasp and target, re-chosen for the ball.

        ``init_qpos`` is a resting state: the ball sitting in the grasp that
        ``init_ctrl`` commands, as settled in the M5 (Kamino) eval sim, with
        every hand joint inside its range. So the driver's settle period is
        close to a no-op and every eval simulator starts from the same contact
        state. The orientation is the identity, which for a sphere changes
        nothing about the contact.

        ``init_ctrl`` came out of a search over 256 grasp commands, each settled
        in M5 and then shaken with MPPI-shaped random control walks. This one
        dropped nothing, let the ball wander least, and keeps all four
        fingertips on it. ``target_pos`` is where it rests, so as for the cube
        an episode starts with no position error and the task is reorientation
        only.

        The old table's entry is kept commented out below it. It started the
        thumb (``th_axl``, joint 13) at ``1.35324943 + 1.0``, past its joint
        range, which swung the thumb through the ball while it settled. It used
        the duck's grasp command, and put the target 2.3 cm in y from where that
        grasp lets the ball rest, outside the 2 cm success threshold.

        The cost weights were tuned against the old target.
        """
        return {
            # 16 hand joints, then the ball's in-palm pos(3) + quat(4).
            "init_qpos": np.array([
                0.57368, -0.46952,  1.04771,  1.13727,  0.37649, -0.06141,  0.70413,  0.97580,
                0.47866,  0.09236,  0.84379,  0.88495,  0.71404,  1.31701,  1.20278,  0.47122,

                0.02498,  0.04687,  0.08647,
                1.0, 0.0, 0.0, 0.0,
            ]),
            "init_ctrl": np.array([
                0.58968, -0.46068,  1.09597,  1.15426,  0.41104, -0.06144,  0.70906,  0.97582,
                0.60184,  0.01615,  0.96447,  0.95257,  0.73813,  1.25667,  1.21173,  0.46540,
            ]),
            "target_pos": np.array([0.025, 0.047, 0.0865]),
            # The old table's start, grasp and target:
            # "init_qpos": np.array([
            #     5.74644705e-01, -4.67896711e-01,  1.06154201e+00,  9.49754192e-01,
            #     3.39887676e-01, -7.54712363e-04,  6.93497978e-01,  1.01159852e+00,
            #     5.37995866e-01,  2.37394401e-01,  8.84233738e-01,  9.67669103e-01,
            #     6.78255227e-01,  1.35324943e+00 + 1.0,  1.21785619e+00,  6.34822484e-01,
            #
            #     2.58181221e-02 - 0.02,  4.08290136e-02,  8.46989861e-02 + 0.05,
            #     1.0, 0.0, 0.0, 0.0,
            # ]),
            # "init_ctrl": np.array([
            #     0.60184,  -0.46068,   1.09597,   0.97044,   0.41104,   0.0,
            #     0.709056,  1.01884,   0.60184,   0.2094,    0.964465,  1.0186,
            #     0.738135,  1.251165,  1.2778,    0.6564,
            # ]),
            # "target_pos": np.array([0.025, 0.015, 0.085]),
            "target_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "fallen_z": 0.08,
            # "cost_weights": {
            #     "w_quat":         9.29366,
            #     "w_pos_x":      100.0,
            #     "w_pos_y":       38.8977,
            #     "w_pos_z":      100.0,
            #     "w_velo":         0.0,
            #     "w_contact":      0.0879367,
            #     "w_joint":        1.53065,
            #     "w_joint_velo":   0.0,
            #     "w_fallen":     165.111,
            #     "w_quat_term":  195.532,
            #     "w_pos_term":   290.442,
            #     "w_fallen_term":  0.0,
            # },
            "cost_weights": {
                "w_quat":         34.148,#1.797,#9.29366,
                "w_pos_x":        0.0107,#100.0,
                "w_pos_y":        100.0,#50,#38.8977,
                "w_pos_z":        100.0,#1.0,#100.0,
                "w_velo":         0.0,
                "w_contact":      100.0,#100.0,#0.0879367,
                "w_joint":        0.1268,#0.1,#1.53065,
                "w_joint_velo":   0.0,
                "w_fallen":     68.321,#100.0,
                "w_quat_term":  10.0,#100.0,
                "w_pos_term":   572.98,#300.0,
                "w_fallen_term":  0.0,
            },
        }
