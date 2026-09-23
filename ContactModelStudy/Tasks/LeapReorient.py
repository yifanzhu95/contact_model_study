"""Base task for reorienting a grasped object with the LEAP hand.

Everything shared by the cube, ball and duck variants lives here: the scene
layout (16 hand joints, one free-joint object, four fingertip sites), the cost,
the success and failure tests, and the scene-path templates. A subclass supplies
only what differs per object — its name, where it starts, where it should go,
and its cost weights — through ``objectParams``.

Ported from ``contact_study/tasks/grasp_reorient.py``. The cost is a faithful
translation of ``grasp_reorient_cost_wp``, including two quirks preserved
deliberately rather than tidied up; see ``COST_WEIGHT_KEYS`` and ``calcCosts``.
Success and failure reproduce the old ``is_success`` / ``has_failed``. Goal
sampling keeps the old ten difficulty levels but draws every goal as a rotation
of the *current* goal; see ``GOAL_DIFFICULTIES``.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np

from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator
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

#: Camera every leap scene defines, framed on the hand and the object.
CAMERA_NAME = "demo-cam"

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


def _axis_angle_quat(axis, angle: float) -> np.ndarray:
    """wxyz quaternion for a rotation of ``angle`` (rad) about unit ``axis``."""
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    return np.array([c, s * axis[0], s * axis[1], s * axis[2]], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of wxyz quaternions — MuJoCo's ``mju_mulQuat``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# The six signed object axes, in a fixed order so sampling is reproducible.
# Each is the outward normal of one face; "which face is up" is "which of these
# the goal points at world +Z".
_SIGNED_AXES = tuple(
    np.array(v, dtype=float) for v in
    ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])
)
_X_AXIS = np.array([1.0, 0.0, 0.0])

# Letter on the face whose outward normal is each signed object axis — what
# goalFace() reads off a goal quaternion.
_AXIS_LETTERS = {(0, 1): "I", (0, -1): "I", (1, 1): "B", (1, -1): "O",
                 (2, 1): "R", (2, -1): "S"}

def _upAxis(quat: np.ndarray) -> tuple[int, int]:
    """(axis, sign) of the object axis that ``quat`` points at world +Z."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    # Third row of the rotation matrix: world +Z expressed in the object frame.
    up = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    axis = int(np.argmax(np.abs(up)))
    return axis, (1 if up[axis] > 0 else -1)


#: What each goal difficulty samples. Every level is a rotation applied to the
#: *current* goal in the object's own frame — never a pose picked relative to
#: the start. Same numbering as the old task; levels 3, 4 and 6-9 differ from it,
#: which built its goals from the starting orientation instead.
GOAL_DIFFICULTIES = {
    0: "90-degree clockwise spin about the shown face's normal",
    1: "+/-90-degree spin about the shown face's normal (face unchanged)",
    2: "+/-90-degree rotation about a random object axis",
    3: "tip onto one of the 4 adjacent faces, then a random 90-degree twist",
    4: "turn to any of the 5 other faces, then a random 90-degree twist",
    5: "flip: 180 degrees about an in-face axis (opposite face shown)",
    6: "tip onto one of the 4 adjacent faces, no twist",
    7: "roll -90 degrees about the object's X axis",
    8: "roll +/-90 degrees about the object's X axis, chosen uniformly",
    9: "roll +90 degrees about the object's X axis",
}


class LeapReorient(TaskBase):
    """Base class for LEAP-hand reorientation tasks.

    The scene is 16 actuated hand joints followed by one free-joint object, so
    ``nq = 16 + 7 = 23`` and ``nv = 16 + 6 = 22``. Subclasses define the object.
    """

    #: Scene-variant object name, e.g. "cube". Set by the subclass.
    OBJECT: str = ""

    #: Success requires every error below its threshold, as in the old
    #: ``is_success``: object position (m, L2), orientation (``1 - dot^2``),
    #: and object speed (norm of the 6-D linear+angular velocity) — the last so
    #: that a cube tumbling *through* the goal pose does not count.
    SUCCESS_THRESHOLDS = {"pos": 0.02, "quat": 0.04, "vel": 0.1}

    #: Object height (m) below which the episode has failed — the object has
    #: left the hand and hit the floor. Deliberately not ``fallen_z``: that is
    #: the cost's drop *penalty*, set just under the palm, and firing a
    #: terminal failure there would end episodes the planner could still save.
    FAILURE_Z = 0.0

    def __init__(
        self,
        role: TaskRole | str = TaskRole.EVAL,
        hand_acc: str = "high",
        obj_acc: str = "high",
        scenes_dir: str | Path | None = None,
        timestep: float = 0.002,
        goal_difficulty: int = 8,
        seed: int | None = None,
    ):
        """Create the task for one scene role and geometry fidelity.

        Args:
            role: ``EVAL`` (accurate scene) or ``ROLLOUT`` (planner's scene).
            hand_acc: Hand mesh fidelity in the rollout scene — "low", "med" or
                "high". Ignored for the eval scene, which has only one fidelity.
            obj_acc: Object mesh fidelity in the rollout scene.
            scenes_dir: Root holding the scene XML. Defaults to the repo's
                ``scenes/``.
            timestep: Physics timestep this task is posed at.
            goal_difficulty: Which goal sampler ``sampleNewGoal`` uses; see
                ``GOAL_DIFFICULTIES``. Defaults to 8, the old task's default.
            seed: Seed for goal sampling. ``None`` draws from fresh entropy.

        Raises:
            NotImplementedError: If the subclass did not set ``OBJECT``.
            ValueError: If ``goal_difficulty`` is not one of the ten levels.
        """
        super().__init__(role, timestep=timestep)
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

        if goal_difficulty not in GOAL_DIFFICULTIES:
            raise ValueError(
                f"goal_difficulty must be one of {sorted(GOAL_DIFFICULTIES)}, got {goal_difficulty}"
            )
        self.goal_difficulty = int(goal_difficulty)
        self._rng = np.random.default_rng(seed)
        # The starting goal: the object's orientation at the start of an
        # episode, and the goal a fresh task holds.
        self._start_quat = self.params["target_quat"].copy()
        # What sampleNewGoal rotates from when not told otherwise: the goal last
        # adopted, or the starting goal again after setSimToInitialState.
        self._sample_ref = self._start_quat.copy()

        # Resolved lazily: building them compiles the scene, which a caller that
        # only wants getModelPath() should not pay for.
        self._mjm = None
        self._indices = None
        self._goal = None
        self._weights = None
        self._gpu = None
        # Output buffers handed back when a caller passes no `out`, keyed by
        # (kind, N, device) so one task can serve simulators of several sizes.
        self._buffers: dict = {}

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
            fallen_z:     scalar. Below this object height the cost's drop
                          penalty applies. Per-object because a squatter object
                          would otherwise read as dropped while still held.
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
    def alignRendererConfigWithTask(self, config) -> None:
        """Point the renderer at the leap scenes' own camera.

        Every leap scene defines ``demo-cam``, positioned to frame the hand and
        the grasped object. Only the camera name is set: pose and field of view
        are left as the caller had them — ``None``, normally, meaning the
        scene's values as authored — and so is everything else.
        """
        config.cam_name = CAMERA_NAME

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
        """The compiled ``MjModel`` for this task's scene.

        Needed only to resolve names to indices (the fingertip sites, the
        object's joint); nothing is ever simulated on it.
        """
        if self._mjm is None:
            import mujoco

            self._mjm = mujoco.MjModel.from_xml_path(self.getModelPath())
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

    # -- goal ----------------------------------------------------------------
    def sampleNewGoal(self, current_goal: np.ndarray | None = None) -> np.ndarray:
        """Draw a new target orientation (wxyz), rotated from the current goal.

        Args:
            current_goal: The orientation to rotate from. ``None`` uses the
                task's reference: the goal last adopted with ``setGoal``, or the
                starting goal after ``setSimToInitialState``.

        Every level returns ``current_goal * r`` for a rotation ``r`` in the
        object's own frame (see ``GOAL_DIFFICULTIES``), so a new goal is always
        a move *from where the last goal left the object* — never a pose chosen
        relative to the start. That is also why no level can re-issue the goal
        it started from: every ``r`` is a nonzero rotation.

        Does not adopt the result; pass it to ``setGoal``. Consumes randomness
        and changes nothing else. The target *position* is per-object and
        fixed, as in the old task — only orientation is sampled.
        """
        g = self._sample_ref if current_goal is None else self._unitQuat(current_goal)
        up = self._upVector(g)
        d = self.goal_difficulty

        if d == 0:
            r = _axis_angle_quat(up, -np.pi / 2.0)
        elif d == 1:
            r = _axis_angle_quat(up, self._rng.choice([np.pi / 2.0, -np.pi / 2.0]))
        elif d == 2:
            axis = np.zeros(3)
            axis[self._rng.integers(0, 3)] = 1.0
            r = _axis_angle_quat(axis, self._rng.choice([np.pi / 2.0, -np.pi / 2.0]))
        elif d == 3:
            r = self._turnTo(self._pick(self._adjacent(up)), up, twist=True)
        elif d == 5:
            r = np.array([0.0, *self._inFaceAxis(up)])        # 180 degrees
        elif d == 6:
            r = self._turnTo(self._pick(self._adjacent(up)), up, twist=False)
        elif d == 7:
            r = _axis_angle_quat(_X_AXIS, -np.pi / 2.0)
        elif d == 8:
            r = _axis_angle_quat(_X_AXIS, self._rng.choice([-np.pi / 2.0, np.pi / 2.0]))
        elif d == 9:
            r = _axis_angle_quat(_X_AXIS, np.pi / 2.0)
        else:                                                # level 4
            others = self._adjacent(up) + [-up]
            r = self._turnTo(self._pick(others), up, twist=True)
        return _quat_mul(g, r)

    def setGoal(self, goal: np.ndarray) -> None:
        """Adopt ``goal``, a wxyz target orientation; see ``TaskBase.setGoal``.

        Normalized on the way in, and becomes the orientation the next
        ``sampleNewGoal`` rotates from.
        """
        q = self._unitQuat(goal)
        self.params["target_quat"] = q
        self._sample_ref = q.copy()
        self._goal = None                    # rebuilt from params on next access
        if self._gpu is not None:
            # In place, never reallocated: a planner's captured CUDA graph holds
            # this buffer's address, and must see the new goal on replay.
            self._gpu["goal"].assign(self.goal)

    def setRendererToGoal(self, renderer) -> None:
        """Turn ``renderer``'s goal marker to the current goal, so frames show it.

        A MuJoCo renderer keeps its own ``MjData``; ``RenderState`` rewrites
        only positions, so the marker orientation set here persists across every
        frame until the next call. Resolved against the *renderer's* model, not
        this task's, in case the two were built from different scenes. Reaches
        into the renderer's ``mjd`` because ``RendererBase`` has no mocap
        accessor; a renderer without one, or a scene with no marker, is left
        alone.
        """
        mjd = getattr(renderer, "mjd", None)
        mid = self._mocapIdIn(getattr(renderer, "mjm", None))
        if mjd is not None and mid is not None:
            mjd.mocap_quat[mid] = self.params["target_quat"]

    def goalFace(self, goal: np.ndarray | None = None) -> str:
        """Letter shown on top when the object sits at ``goal`` (default: current).

        Read off the quaternion — which object axis the goal points at world +Z
        — so it is right however the goal was set.
        """
        q = self.params["target_quat"] if goal is None else np.asarray(goal, dtype=float)
        return _AXIS_LETTERS[_upAxis(q)]

    def _onInitialState(self) -> None:
        """Sample from the starting goal again: a reset object is back at its start.

        Moves only the sampling reference — host bookkeeping, no device write —
        so ``setSimToInitialState`` stays capturable. The adopted goal, which
        the cost reads, is left as it was until the next ``setGoal``. Without
        this, an episode's first goal would be rotated from wherever the
        previous episode's last goal happened to leave off, not from where the
        object actually starts.
        """
        self._sample_ref = self._start_quat.copy()

    # -- goal sampling helpers ----------------------------------------------
    @staticmethod
    def _unitQuat(goal) -> np.ndarray:
        """Validate and normalize a wxyz quaternion."""
        q = np.asarray(goal, dtype=float).ravel()
        if q.shape != (4,):
            raise ValueError(f"goal must be a (4,) wxyz quaternion, got shape {q.shape}")
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"goal quaternion must be finite and non-zero, got {q}")
        return q / norm

    @staticmethod
    def _upVector(g: np.ndarray) -> np.ndarray:
        """The signed object axis ``g`` points at world +Z — the shown face's normal."""
        axis, sign = _upAxis(g)
        v = np.zeros(3)
        v[axis] = float(sign)
        return v

    @staticmethod
    def _adjacent(up: np.ndarray) -> list[np.ndarray]:
        """The four signed axes perpendicular to ``up``: the adjacent faces."""
        return [a for a in _SIGNED_AXES if abs(float(np.dot(a, up))) < 0.5]

    @staticmethod
    def _inFaceAxis(up: np.ndarray) -> np.ndarray:
        """First cardinal axis lying in the shown face — the flip axis, as before."""
        axis = np.zeros(3)
        axis[next(i for i in range(3) if abs(up[i]) < 0.5)] = 1.0
        return axis

    def _pick(self, candidates: list[np.ndarray]) -> np.ndarray:
        """Uniform choice among candidate face normals."""
        return candidates[int(self._rng.choice(len(candidates)))]

    def _turnTo(self, v: np.ndarray, up: np.ndarray, twist: bool) -> np.ndarray:
        """Object-frame rotation that brings face ``v`` up in place of face ``up``.

        We need ``r`` with ``r(v) = up``: then ``g * r`` sends ``v`` to where
        ``g`` sent ``up`` — world +Z. For an adjacent face that is 90 degrees
        about ``v x up``; for the opposite face, 180 degrees about an axis in the
        shown face. An optional twist spins the result about ``v``, now the up
        axis, by a random multiple of 90 degrees.
        """
        if float(np.dot(v, up)) < -0.5:
            r = np.array([0.0, *self._inFaceAxis(up)])
        else:
            r = _axis_angle_quat(np.cross(v, up), np.pi / 2.0)
        if twist:
            angle = int(self._rng.integers(0, 4)) * (np.pi / 2)
            r = _quat_mul(r, _axis_angle_quat(v, angle))
        return r

    def _mocapIdIn(self, mjm) -> int | None:
        """Mocap index of the goal marker in ``mjm``, or ``None`` if there is none.

        ``"goal"`` first, then ``"obj_target"`` — the name older scenes use —
        matching the old ``_update_goal``. A body is only used if it is actually
        mocap: some scenes carry a non-mocap ``"goal"`` body.
        """
        if mjm is None:
            return None
        import mujoco

        for name in ("goal", "obj_target"):
            bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0 and mjm.body_mocapid[bid] >= 0:
                return int(mjm.body_mocapid[bid])
        return None

    # -- cost ----------------------------------------------------------------
    def calcCosts(self, sim, terminal: bool = False, out=None):
        """Score every world of ``sim`` on the device; see ``TaskBase``.

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

        1. **Controls are not read.** The old kernel takes ``ctrl`` and never
           uses it — this cost does not penalize control effort.
        2. **``w_velo`` scales position, not velocity.** The object-velocity
           term is computed in the old kernel and then dropped, while
           ``weights[4]`` multiplies ``c_pos``. Reproduced as-is: every tuned
           weight set in the study was fitted against this behaviour.

        Kernel launches only, so it can be recorded into a CUDA graph as long
        as ``out`` is supplied (or the cached buffer already exists).

        Raises:
            TypeError: If ``sim`` is not vectorized. Evaluate a CPU episode with
                ``isSuccess`` / ``isFailure``; the cost is not reproduced on the
                host.
        """
        import warp as wp

        self._requireVectorized(sim, "calcCosts")
        state = sim.DeviceState()
        if state.site_xpos is None:
            raise ValueError(
                f"{type(sim).__name__} exposes no site_xpos, which the contact "
                f"term needs (the fingertip-to-object distances)."
            )
        n, device = sim.N, sim.config.device
        if out is None:
            out = self._buffer("cost", n, device, wp.float32)
        g = self._gpu_arrays(device)
        wp.launch(
            _leap_reorient_cost_kernel,
            dim=n,
            inputs=[state.qpos, state.qvel, state.site_xpos, g["goal"], g["indices"],
                    g["weights"], 1 if terminal else 0],
            outputs=[out],
        )
        return out

    # -- outcome -------------------------------------------------------------
    def goalErrors(self, q: np.ndarray, q_dot: np.ndarray) -> dict[str, float]:
        """Distance to the goal for one state, keyed like ``SUCCESS_THRESHOLDS``.

        The same three quantities ``isSuccess`` thresholds, so the two cannot
        drift apart — and a driver or a hyperparameter search can report or
        scalarize them directly. Not the cost: these are the pass/fail metrics.
        """
        obj_q, obj_v = int(self.indices[0]), int(self.indices[1])
        q = np.asarray(q, dtype=float).ravel()
        q_dot = np.asarray(q_dot, dtype=float).ravel()
        pos = q[obj_q:obj_q + 3]
        quat = q[obj_q + 3:obj_q + 7]
        vel = q_dot[obj_v:obj_v + 6]
        return {
            "pos": float(np.linalg.norm(pos - self.params["target_pos"])),
            "quat": float(1.0 - np.dot(quat, self.params["target_quat"]) ** 2),
            "vel": float(np.linalg.norm(vel)),
        }

    def isSuccess(self, sim, out=None):
        """Every goal error below its threshold; see ``TaskBase.isSuccess``.

        A state with a NaN or inf anywhere — hand or object, position or
        velocity — is never a success, even when the object's own pose happens
        to sit inside the thresholds. That keeps success and failure mutually
        exclusive: ``isFailure`` flags the same states, and a world reported as
        having both succeeded and failed would be a trap for any caller that
        checks only one of them.
        """
        if not isinstance(sim, VectorizedSimulator):
            q, q_dot = sim.GetState()
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(q_dot))):
                return False
            err = self.goalErrors(q, q_dot)
            thr = self.SUCCESS_THRESHOLDS
            return bool(all(err[k] < thr[k] for k in thr))

        import warp as wp

        state = sim.DeviceState()
        n, device = sim.N, sim.config.device
        if out is None:
            out = self._buffer("success", n, device, wp.bool)
        g = self._gpu_arrays(device)
        thr = self.SUCCESS_THRESHOLDS
        wp.launch(
            _leap_success_kernel,
            dim=n,
            inputs=[state.qpos, state.qvel, g["goal"], int(self.indices[0]),
                    int(self.indices[1]), float(thr["pos"]), float(thr["quat"]),
                    float(thr["vel"])],
            outputs=[out],
        )
        return out

    def isFailure(self, sim, out=None):
        """The object is on the floor, or the state has blown up.

        The first is the old ``has_failed``: object height below
        ``FAILURE_Z``. The second is new — a world whose positions or
        velocities contain a NaN or inf is also a failure. The old check read ``xpos[2] < 0.0``, which is
        simply *false* for a NaN, so a blown-up simulation was silently
        reported as still running; for a batch of rollouts that is exactly the
        case a planner needs flagged.

        Reads the object height from ``qpos`` rather than the body's ``xpos``:
        for a free-joint body they are the same point, and ``qpos`` is the
        post-step state where ``xpos`` is left over from the step's forward pass.
        """
        if not isinstance(sim, VectorizedSimulator):
            q, q_dot = sim.GetState()
            q = np.asarray(q, dtype=float)
            finite = np.all(np.isfinite(q)) and np.all(np.isfinite(q_dot))
            return bool(not finite or q[int(self.indices[0]) + 2] < self.FAILURE_Z)

        import warp as wp

        state = sim.DeviceState()
        n, device = sim.N, sim.config.device
        if out is None:
            out = self._buffer("failure", n, device, wp.bool)
        wp.launch(
            _leap_failure_kernel,
            dim=n,
            inputs=[state.qpos, state.qvel, int(self.indices[0]), float(self.FAILURE_Z)],
            outputs=[out],
        )
        return out

    # -- device plumbing -----------------------------------------------------
    @staticmethod
    def _requireVectorized(sim, what: str) -> None:
        if not isinstance(sim, VectorizedSimulator):
            raise TypeError(
                f"{what} needs a VectorizedSimulator, got {type(sim).__name__}. The "
                f"cost exists only on the device; judge a CPU episode with "
                f"isSuccess / isFailure instead."
            )

    def _buffer(self, kind: str, n: int, device: str, dtype):
        """A reused ``(n,)`` output array for callers that pass no ``out``.

        Allocated on first use and handed back on every later call with the same
        size and device — so it is overwritten each time, and a caller that
        needs to keep a result copies it. Being stable is what makes the calls
        graph-safe: after the first one there is no allocation left to record.
        """
        import warp as wp

        key = (kind, n, str(device))
        if key not in self._buffers:
            self._buffers[key] = wp.zeros(n, dtype=dtype, device=device)
        return self._buffers[key]

    def _gpu_arrays(self, device: str) -> dict:
        """Upload the goal/index/weight vectors once and cache them."""
        import warp as wp

        if self._gpu is None or self._gpu["device"] != str(device):
            self._gpu = {
                "device": str(device),
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


# The kernels are defined only if warp is importable, so a CPU-only caller can
# still use the task's scene, initial state and single-simulator outcome checks.
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
        """Device translation of grasp_reorient_cost_wp, one world per thread."""
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

        # The joint-velocity term indexes qvel with the *qpos* address, exactly
        # as the old kernel does. Identical here (both robot blocks start at 0).
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

    @wp.kernel
    def _leap_success_kernel(
        qpos: wp.array2d(dtype=float),   # (N, nq)
        qvel: wp.array2d(dtype=float),   # (N, nv)
        goal: wp.array(dtype=float),     # target pos(3) + quat(4) lead the vector
        obj_q: int,
        obj_v: int,
        pos_thr: float,
        quat_thr: float,
        vel_thr: float,
        out: wp.array(dtype=wp.bool),    # (N,)
    ):
        """Device twin of goalErrors + the threshold test, one world per thread."""
        n = wp.tid()
        d = wp.vec3(qpos[n, obj_q] - goal[0], qpos[n, obj_q + 1] - goal[1],
                    qpos[n, obj_q + 2] - goal[2])
        quat = wp.vec4(qpos[n, obj_q + 3], qpos[n, obj_q + 4],
                       qpos[n, obj_q + 5], qpos[n, obj_q + 6])
        dot = wp.dot(quat, wp.vec4(goal[3], goal[4], goal[5], goal[6]))
        speed2 = float(0.0)
        for i in range(6):
            speed2 += qvel[n, obj_v + i] * qvel[n, obj_v + i]
        ok = (wp.length(d) < pos_thr) and (1.0 - dot * dot < quat_thr) \
            and (wp.sqrt(speed2) < vel_thr)
        # A non-finite state anywhere rules success out, keeping it exclusive
        # of _leap_failure_kernel, which flags the same states.
        for i in range(qpos.shape[1]):
            if not wp.isfinite(qpos[n, i]):
                ok = False
        for i in range(qvel.shape[1]):
            if not wp.isfinite(qvel[n, i]):
                ok = False
        out[n] = ok

    @wp.kernel
    def _leap_failure_kernel(
        qpos: wp.array2d(dtype=float),   # (N, nq)
        qvel: wp.array2d(dtype=float),   # (N, nv)
        obj_q: int,
        fail_z: float,
        out: wp.array(dtype=wp.bool),    # (N,)
    ):
        """Object below the floor, or any position or velocity non-finite."""
        n = wp.tid()
        bad = qpos[n, obj_q + 2] < fail_z
        for i in range(qpos.shape[1]):
            if not wp.isfinite(qpos[n, i]):
                bad = True
        for i in range(qvel.shape[1]):
            if not wp.isfinite(qvel[n, i]):
                bad = True
        out[n] = bad

except ImportError:  # pragma: no cover - warp is optional for the host path
    _leap_reorient_cost_kernel = None
    _leap_success_kernel = None
    _leap_failure_kernel = None
