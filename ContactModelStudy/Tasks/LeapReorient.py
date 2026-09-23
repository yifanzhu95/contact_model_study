"""Base task for reorienting a grasped object with the LEAP hand.

Everything shared by the cube, ball and duck variants lives here: the scene
layout (16 hand joints, one free-joint object, four fingertip sites), the cost
function, and the scene-path templates. A subclass supplies only what differs
per object — its name, where it starts, where it should go, and its cost
weights — through ``objectParams``.

Ported from ``contact_study/tasks/grasp_reorient.py``. The cost is a faithful
translation of ``grasp_reorient_cost_wp``, including two quirks that are
preserved deliberately rather than tidied up; see ``COST_WEIGHT_KEYS`` and
``calcCosts``.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np

from ContactModelStudy.Tasks.TaskBase import TaskBase, TaskRole

# Where the scene XML actually lives. The refactor plan puts these under
# Tasks/XML_Files, but the leap scenes reference meshes and <include> files by
# relative path (../leap_hand/..., textures/...), so moving the XML without its
# asset tree breaks compilation. Kept pointing at the existing tree for now;
# moving it is a one-line change here plus copying the assets alongside.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = _REPO_ROOT / "scenes"

# Scene filename templates, as in the old TaskConfig. The rollout scene encodes
# the geometry-fidelity axis of the study (hand accuracy, object accuracy); the
# eval scene is always the accurate one.
EVAL_XML_TEMPLATE = "leap/env_leap_eval_{obj}.xml"
ROLLOUT_XML_TEMPLATE = "leap/env_leap_rollout_{obj}_{hand_acc}_{obj_acc}.xml"

N_HAND_JOINTS = 16
FINGERTIP_SITES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
OBJECT_BODY = "obj"
OBJECT_JOINT = "obj_joint"

# Cost weights, in the order the cost indexes them. THE ORDER IS LOAD-BEARING.
#
# Note "w_velo": despite the name it multiplies the L2 position error, not a
# velocity term. The old kernel computes an object-velocity term and then never
# adds it to the cost, while weights[4] — the slot named w_velo — scales
# `c_pos`. That is carried over verbatim: renaming the key would silently
# change what every tuned weight set in the study means. See calcCosts.
COST_WEIGHT_KEYS = (
    "w_quat", "w_pos_x", "w_pos_y", "w_pos_z", "w_velo", "w_contact",
    "w_joint", "w_joint_velo", "w_fallen",
    "w_quat_term", "w_pos_term", "w_fallen_term",
)


class LeapReorient(TaskBase):
    """Base class for LEAP-hand reorientation tasks.

    The scene is 16 actuated hand joints followed by one free-joint object, so
    ``nq = 16 + 7 = 23`` and ``nv = 16 + 6 = 22``. Subclasses define the object.
    """

    #: Scene-variant object name, e.g. "cube". Set by the subclass.
    OBJECT: str = ""

    def __init__(
        self,
        role: TaskRole | str = TaskRole.EVAL,
        hand_acc: str = "high",
        obj_acc: str = "high",
        scenes_dir: str | Path | None = None,
    ):
        """Create the task for one scene role and geometry fidelity.

        Args:
            role: ``EVAL`` (accurate scene) or ``ROLLOUT`` (planner's scene).
            hand_acc: Hand mesh fidelity in the rollout scene — "low", "med" or
                "high". Ignored for the eval scene, which has only one fidelity.
            obj_acc: Object mesh fidelity in the rollout scene.
            scenes_dir: Root holding the scene XML. Defaults to the repo's
                ``scenes/``.

        Raises:
            NotImplementedError: If the subclass did not set ``OBJECT``.
        """
        super().__init__(role)
        if not self.OBJECT:
            raise NotImplementedError(
                f"{type(self).__name__} must set OBJECT to a scene-variant object "
                f"name (e.g. 'cube')."
            )
        self.hand_acc = hand_acc
        self.obj_acc = obj_acc
        self.scenes_dir = Path(scenes_dir) if scenes_dir is not None else SCENES_DIR

        self.params = self.objectParams()
        self._validate_params()

        # Resolved lazily: building them compiles the scene, which a caller that
        # only wants getModelPath() should not pay for.
        self._mjm = None
        self._mjd = None
        self._indices = None
        self._goal = None
        self._weights = None
        self._gpu = None

    # -- subclass hook -------------------------------------------------------
    @abc.abstractmethod
    def objectParams(self) -> dict:
        """Return this object's parameters.

        Required keys:
            init_qpos:    (nq,) initial state — 16 hand joints then obj pos(3)+quat(4).
            init_ctrl:    (16,) initial hand command. Doubles as the cost's joint
                          "home" target, so the joint term penalizes drift away
                          from the commanded grasp.
            target_pos:   (3,) goal position for the object.
            target_quat:  (4,) goal orientation, wxyz.
            fallen_z:     scalar. Below this object height the state counts as
                          dropped. Per-object because a squatter object would
                          otherwise read as fallen while still held.
            cost_weights: dict with exactly the keys of ``COST_WEIGHT_KEYS``.
        """
        ...

    def _validate_params(self) -> None:
        """Check the subclass's table now, not on the first cost evaluation."""
        required = {"init_qpos", "init_ctrl", "target_pos", "target_quat",
                    "fallen_z", "cost_weights"}
        missing = required - set(self.params)
        if missing:
            raise KeyError(f"{type(self).__name__}.objectParams() is missing {sorted(missing)}")

        shapes = {"init_qpos": 23, "init_ctrl": N_HAND_JOINTS, "target_pos": 3,
                  "target_quat": 4}
        for key, n in shapes.items():
            arr = np.asarray(self.params[key], dtype=float).ravel()
            if arr.shape != (n,):
                raise ValueError(
                    f"{type(self).__name__}.objectParams()[{key!r}] must have "
                    f"{n} elements, got {arr.shape[0]}"
                )
            self.params[key] = arr

        w = self.params["cost_weights"]
        missing_w = set(COST_WEIGHT_KEYS) - set(w)
        unknown_w = set(w) - set(COST_WEIGHT_KEYS)
        if missing_w or unknown_w:
            raise KeyError(
                f"{type(self).__name__} cost_weights: missing {sorted(missing_w)}, "
                f"unknown {sorted(unknown_w)}. The cost indexes them positionally, "
                f"so the set must be exactly {list(COST_WEIGHT_KEYS)}."
            )

    # -- scene ---------------------------------------------------------------
    def getModelPath(self) -> str:
        """Path to this task's MJCF, chosen by ``role``."""
        if self.role is TaskRole.EVAL:
            rel = EVAL_XML_TEMPLATE.format(obj=self.OBJECT)
        else:
            rel = ROLLOUT_XML_TEMPLATE.format(
                obj=self.OBJECT, hand_acc=self.hand_acc, obj_acc=self.obj_acc
            )
        return self._resolve_path(self.scenes_dir / rel)

    def getInitialState(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(q0, q_dot0, u0)``: the object's table entry, at rest."""
        q0 = self.params["init_qpos"].copy()
        u0 = self.params["init_ctrl"].copy()
        return q0, np.zeros(self.nv), u0

    # -- lazily compiled model ----------------------------------------------
    @property
    def mjm(self):
        """The compiled ``MjModel`` for this task's scene."""
        if self._mjm is None:
            import mujoco

            self._mjm = mujoco.MjModel.from_xml_path(self.getModelPath())
            self._mjd = mujoco.MjData(self._mjm)
        return self._mjm

    @property
    def nq(self) -> int:
        return self.mjm.nq

    @property
    def nv(self) -> int:
        return self.mjm.nv

    @property
    def nu(self) -> int:
        return self.mjm.nu

    @property
    def indices(self) -> np.ndarray:
        """``[obj_qpos_adr, obj_qvel_adr, robot_qpos_adr, n_manip, obj_id, *tip_site_ids]``.

        Same layout as the old ``index_vector``, so the ported cost can be
        compared against the original term by term.
        """
        if self._indices is None:
            import mujoco

            mjm = self.mjm
            obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT)
            if obj_jnt < 0:
                raise ValueError(f"scene {self.getModelPath()} has no {OBJECT_JOINT!r} joint")
            tips = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, s) for s in FINGERTIP_SITES]
            if any(t < 0 for t in tips):
                missing = [s for s, t in zip(FINGERTIP_SITES, tips) if t < 0]
                raise ValueError(f"scene {self.getModelPath()} is missing sites {missing}")
            self._indices = np.array([
                mjm.jnt_qposadr[obj_jnt],
                mjm.jnt_dofadr[obj_jnt],
                0,                       # robot joints lead qpos
                N_HAND_JOINTS,
                mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY),
                *tips,
            ], dtype=np.int32)
        return self._indices

    @property
    def goal(self) -> np.ndarray:
        """``[target_pos(3), target_quat(4), home(16), fallen_z(1)]``, float32."""
        if self._goal is None:
            self._goal = np.concatenate([
                self.params["target_pos"],
                self.params["target_quat"],
                self.params["init_ctrl"],
                [float(self.params["fallen_z"])],
            ]).astype(np.float32)
        return self._goal

    @property
    def weights(self) -> np.ndarray:
        """Cost weights in ``COST_WEIGHT_KEYS`` order, float32."""
        if self._weights is None:
            w = self.params["cost_weights"]
            self._weights = np.array([float(w[k]) for k in COST_WEIGHT_KEYS], dtype=np.float32)
        return self._weights

    # -- cost, host ----------------------------------------------------------
    def calcCosts(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        u: np.ndarray | None = None,
        terminal: bool = False,
        site_xpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cost of ``(q, q_dot)``; see ``TaskBase.calcCosts`` for shapes.

        Terms, matching ``grasp_reorient_cost_wp`` exactly:

        =============  ==========================================================
        ``c_quat``     ``1 - dot(q_target, q_obj)^2`` — orientation error
        ``c_pos_*``    per-axis ``|Δp|``, weighted independently in X/Y/Z
        ``c_pos``      ``||Δp||_2``, scaled by the slot named ``w_velo``
        ``c_joint``    squared drift of each hand joint from its home command
        ``c_joint_v``  squared hand joint velocity
        ``c_contact``  summed distance from the object to each of 4 fingertips
        ``fallen``     1 when the object is below ``fallen_z``
        =============  ==========================================================

        Two faithfully-preserved quirks:

        1. **``u`` is ignored.** The old kernel takes ``ctrl`` and never reads
           it — this cost does not penalize control effort. The argument stays
           in the signature because ``TaskBase`` defines it and another task
           will use it.
        2. **``w_velo`` scales position, not velocity.** The object-velocity
           term is computed in the old kernel and then dropped on the floor,
           while ``weights[4]`` multiplies ``c_pos``. Reproduced as-is: every
           tuned weight set in the study was fitted against this behaviour.

        Args:
            site_xpos: ``(..., nsite, 3)`` fingertip positions, if the caller
                already has them. Omitted, they are computed here with
                ``mj_forward`` per state — correct but slow, so pass them on any
                hot path.
        """
        import mujoco

        q = np.asarray(q, dtype=float)
        q_dot = np.asarray(q_dot, dtype=float)
        batch = q.shape[:-1]
        if q.shape[-1] != self.nq:
            raise ValueError(f"q must have trailing dim {self.nq}, got {q.shape[-1]}")
        if q_dot.shape[-1] != self.nv:
            raise ValueError(f"q_dot must have trailing dim {self.nv}, got {q_dot.shape[-1]}")
        if q_dot.shape[:-1] != batch:
            raise ValueError(f"q and q_dot batch shapes differ: {batch} vs {q_dot.shape[:-1]}")

        qf = q.reshape(-1, self.nq)
        vf = q_dot.reshape(-1, self.nv)
        if site_xpos is not None:
            tips = np.asarray(site_xpos, dtype=float).reshape(len(qf), -1, 3)
        else:
            tips = self._forward_tips(qf, mujoco)

        costs = np.array([
            self._cost_one(qf[i], vf[i], tips[i], terminal) for i in range(len(qf))
        ])
        return costs.reshape(batch)

    def _forward_tips(self, qf: np.ndarray, mujoco) -> np.ndarray:
        """Fingertip world positions for each state, via forward kinematics."""
        _ = self.mjm                                  # compile + allocate mjd
        tip_ids = self.indices[5:9]
        out = np.empty((len(qf), len(tip_ids), 3))
        for i, qi in enumerate(qf):
            self._mjd.qpos[:] = qi
            mujoco.mj_kinematics(self._mjm, self._mjd)
            out[i] = self._mjd.site_xpos[tip_ids]
        return out

    def _cost_one(self, q, v, tips, terminal: bool) -> float:
        """Scalar cost of one state. The direct translation of the warp func."""
        idx, goal, w = self.indices, self.goal, self.weights
        obj_q, obj_v, robot_q, n_manip = int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3])

        p_obj = q[obj_q:obj_q + 3]
        quat_obj = q[obj_q + 3:obj_q + 7]
        p_target, q_target = goal[0:3], goal[3:7]

        dot = float(np.dot(q_target, quat_obj))
        c_quat = 1.0 - dot * dot

        d = p_obj - p_target
        c_pos_x, c_pos_y, c_pos_z = abs(d[0]), abs(d[1]), abs(d[2])
        c_pos = float(np.linalg.norm(d))

        home = goal[7:7 + n_manip]
        c_joint = float(np.sum((q[robot_q:robot_q + n_manip] - home) ** 2))
        # Indexes qvel with the *qpos* address, exactly as the old kernel does.
        # Identical here (both robot blocks start at 0) and kept so the two
        # implementations cannot drift apart.
        c_joint_velo = float(np.sum(v[robot_q:robot_q + n_manip] ** 2))

        c_contact = float(np.sum(np.linalg.norm(p_obj - tips, axis=-1)))
        fallen = 1.0 if q[obj_q + 2] < goal[7 + n_manip] else 0.0

        if terminal:
            return float(w[9] * c_quat + w[10] * c_pos + w[11] * fallen)
        return float(
            w[0] * c_quat
            + w[1] * c_pos_x + w[2] * c_pos_y + w[3] * c_pos_z
            + w[4] * c_pos                      # slot named w_velo; see docstring
            + w[5] * c_contact
            + w[6] * c_joint
            + w[7] * c_joint_velo
            + w[8] * fallen
        )

    # -- cost, device --------------------------------------------------------
    def calcCosts_GPU(self, q, q_dot, u=None, terminal: bool = False, out=None,
                      site_xpos=None, device: str = "cuda"):
        """Cost of a batch of states on the device; see ``TaskBase``.

        Args:
            q: ``(N, nq)`` warp array. ``q_dot``: ``(N, nv)``.
            site_xpos: ``(N, nsite)`` warp array of ``wp.vec3`` fingertip
                positions — required, and normally ``sim.d.site_xpos`` straight
                from the vectorized simulator. Forward kinematics on the device
                needs an MJWarp model, which is the simulator's to own, not the
                task's.
            out: ``(N,)`` warp array to write into; allocated if omitted.
        """
        import warp as wp

        if site_xpos is None:
            raise ValueError(
                "calcCosts_GPU needs site_xpos (pass sim.d.site_xpos). The device "
                "cost cannot run forward kinematics itself — that needs the "
                "simulator's MJWarp model."
            )
        n = q.shape[0]
        if out is None:
            out = wp.zeros(n, dtype=wp.float32, device=device)
        g = self._gpu_arrays(device)
        wp.launch(
            _leap_reorient_cost_kernel,
            dim=n,
            inputs=[q, q_dot, site_xpos, g["goal"], g["indices"], g["weights"],
                    1 if terminal else 0],
            outputs=[out],
        )
        return out

    def _gpu_arrays(self, device: str) -> dict:
        """Upload the goal/index/weight vectors once and cache them."""
        import warp as wp

        if self._gpu is None or self._gpu["device"] != device:
            self._gpu = {
                "device": device,
                "goal": wp.array(self.goal, dtype=wp.float32, device=device),
                "indices": wp.array(self.indices, dtype=wp.int32, device=device),
                "weights": wp.array(self.weights, dtype=wp.float32, device=device),
            }
        return self._gpu

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(object={self.OBJECT!r}, role={self.role.value!r}, "
            f"geometry={self.hand_acc}_{self.obj_acc})"
        )


# The kernel is defined at import time only if warp is importable, so that a
# CPU-only caller can use the host cost without warp installed.
try:
    import warp as wp

    @wp.kernel
    def _leap_reorient_cost_kernel(
        qpos: wp.array2d(dtype=float),          # (N, nq)
        qvel: wp.array2d(dtype=float),          # (N, nv)
        site_xpos: wp.array2d(dtype=wp.vec3),   # (N, nsite)
        goal: wp.array(dtype=float),
        indices: wp.array(dtype=int),
        weights: wp.array(dtype=float),
        terminal: int,
        out: wp.array(dtype=float),             # (N,)
    ):
        """Device translation of LeapReorient._cost_one, one world per thread."""
        n = wp.tid()
        obj_q = indices[0]
        robot_q = indices[2]
        n_manip = indices[3]

        p_obj = wp.vec3(qpos[n, obj_q], qpos[n, obj_q + 1], qpos[n, obj_q + 2])
        quat_obj = wp.vec4(qpos[n, obj_q + 3], qpos[n, obj_q + 4],
                           qpos[n, obj_q + 5], qpos[n, obj_q + 6])
        p_target = wp.vec3(goal[0], goal[1], goal[2])
        q_target = wp.vec4(goal[3], goal[4], goal[5], goal[6])

        dot = wp.dot(q_target, quat_obj)
        c_quat = 1.0 - dot * dot

        d = p_obj - p_target
        c_pos_x = wp.abs(d[0])
        c_pos_y = wp.abs(d[1])
        c_pos_z = wp.abs(d[2])
        c_pos = wp.length(d)

        c_joint = float(0.0)
        c_joint_velo = float(0.0)
        for i in range(n_manip):
            dq = qpos[n, robot_q + i] - goal[7 + i]
            c_joint += dq * dq
            dv = qvel[n, robot_q + i]
            c_joint_velo += dv * dv

        c_contact = float(0.0)
        for i in range(5, 9):
            c_contact += wp.length(p_obj - site_xpos[n, indices[i]])

        fallen = float(0.0)
        if qpos[n, obj_q + 2] < goal[7 + n_manip]:
            fallen = 1.0

        if terminal == 1:
            out[n] = weights[9] * c_quat + weights[10] * c_pos + weights[11] * fallen
        else:
            out[n] = (
                weights[0] * c_quat
                + weights[1] * c_pos_x + weights[2] * c_pos_y + weights[3] * c_pos_z
                + weights[4] * c_pos          # slot named w_velo; see calcCosts
                + weights[5] * c_contact
                + weights[6] * c_joint
                + weights[7] * c_joint_velo
                + weights[8] * fallen
            )

except ImportError:  # pragma: no cover - warp is optional for the host path
    _leap_reorient_cost_kernel = None
