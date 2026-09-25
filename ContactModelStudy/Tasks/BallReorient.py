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
        """The ball's entry, carried over from the old task's table.

        The offsets in ``init_qpos`` are kept as ``value + offset``, the way the
        old table wrote them. ``th_axl`` (joint 13) is ``1.35324943 + 1.0``,
        which, as for the cube, starts the thumb past its joint range. The ball
        starts 2 cm back in x and 5 cm higher than where it was read off, at
        the identity orientation.

        ``init_ctrl`` is the duck's command, not the cube's. The old table's
        comment calls every number here "the cube's", but only ``fallen_z``
        (0.08) is. The hand pose, target position and cost weights were all
        changed for the ball.
        """
        return {
            # 16 hand joints, then the ball's in-palm pos(3) + quat(4).
            "init_qpos": np.array([
                5.74644705e-01, -4.67896711e-01,  1.06154201e+00,  9.49754192e-01,
                3.39887676e-01, -7.54712363e-04,  6.93497978e-01,  1.01159852e+00,
                5.37995866e-01,  2.37394401e-01,  8.84233738e-01,  9.67669103e-01,
                6.78255227e-01,  1.35324943e+00 + 1.0,  1.21785619e+00,  6.34822484e-01,

                2.58181221e-02 - 0.02,  4.08290136e-02,  8.46989861e-02 + 0.05,
                1.0, 0.0, 0.0, 0.0,
            ]),
            "init_ctrl": np.array([
                0.60184,  -0.46068,   1.09597,   0.97044,   0.41104,   0.0,
                0.709056,  1.01884,   0.60184,   0.2094,    0.964465,  1.0186,
                0.738135,  1.251165,  1.2778,    0.6564,
            ]),
            "target_pos": np.array([0.025, 0.015, 0.085]),
            "target_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "fallen_z": 0.08,
            "cost_weights": {
                "w_quat":         9.29366,
                "w_pos_x":      100.0,
                "w_pos_y":       38.8977,
                "w_pos_z":      100.0,
                "w_velo":         0.0,
                "w_contact":      0.0879367,
                "w_joint":        1.53065,
                "w_joint_velo":   0.0,
                "w_fallen":     165.111,
                "w_quat_term":  195.532,
                "w_pos_term":   290.442,
                "w_fallen_term":  0.0,
            },
        }
