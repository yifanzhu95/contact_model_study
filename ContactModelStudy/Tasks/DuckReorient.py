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


class DuckReorient(LeapReorient):
    """Reorient the duck to a target pose with the LEAP hand.

    Scene: 16 actuated hand joints and a free-joint duck (nq=23, nv=22, nu=16).
    The duck is lighter than the cube (0.05 kg against 0.14) and about 1.5 cm
    taller, so it starts higher and its target sits higher too.
    """

    OBJECT = "duck"

    def objectParams(self) -> dict:
        """The duck's entry, carried over from the old task's table.

        The offsets in ``init_qpos`` are kept as ``value + offset``, the way the
        old table wrote them: they were applied by hand on top of a pose read
        out of the viewer. ``th_axl`` (joint 13) is ``1.34131893 + 1.0``, which,
        as for the cube, starts the thumb past its joint range. The duck starts
        2 cm back in x and 5 cm higher than where it was read off.

        The goal orientation is the identity, as for every object in the old
        task (``_TARGET_QUAT``). The duck does not start there: its start
        quaternion is roughly a quarter turn about z.
        """
        return {
            # 16 hand joints, then the duck's in-palm pos(3) + quat(4).
            "init_qpos": np.array([
                5.80303253e-01, -4.60995160e-01,  1.09418735e+00,  9.71266541e-01,
                3.86906512e-01, -1.10955454e-05,  7.04409919e-01,  1.01898232e+00,
                5.78769646e-01,  2.09500294e-01,  9.62015366e-01,  1.01933705e+00,
                7.38906405e-01,  1.34131893e+00 + 1.0,  1.28450986e+00,  6.59284046e-01,

                2.56294614e-02 - 0.02,  4.96063104e-02,  8.82105693e-02 + 0.05,
                6.97889476e-01,  1.36622853e-02, -3.23292221e-02,  7.15344982e-01,
            ]),
            "init_ctrl": np.array([
                0.60184,  -0.46068,   1.09597,   0.97044,   0.41104,   0.0,
                0.709056,  1.01884,   0.60184,   0.2094,    0.964465,  1.0186,
                0.738135,  1.251165,  1.2778,    0.6564,
            ]),
            "target_pos": np.array([0.03, 0.015, 0.09]),
            "target_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "fallen_z": 0.075,
            # Retuned for the duck in the old study.
            "cost_weights": {
                "w_quat":        23.0224,
                "w_pos_x":        0.01,
                "w_pos_y":      100.0,
                "w_pos_z":        0.361846,
                "w_velo":         0.0,
                "w_contact":    100.0,
                "w_joint":        6.20467,
                "w_joint_velo":   0.0,
                "w_fallen":      93.6095,
                "w_quat_term":  145.812,
                "w_pos_term":   500.0,
                "w_fallen_term":  0.0,
            },
        }
