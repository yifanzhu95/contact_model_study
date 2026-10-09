"""Kamino: Newton's full-NCP contact solver (``SolverKamino``, PADMM) as a vectorized simulator.

Contact model M5. Kamino solves the nonlinear complementarity problem of
frictional contact with a proximal ADMM iteration, on maximal coordinates, for
``N`` replicated worlds at once. Where M1-M4 relax contact (soft constraints,
complementarity-free, XPBD), Kamino aims at the hard NCP, which makes it the
study's high-accuracy reference, and by far its most expensive model.

Ported from the collaborator's feasibility package in ``kamino_reference/``
(``target_adapter.py``, ``batched_target_rollout.py``, ``mjcf_probe.py``): the
scene import with site recording, the asset-path fallback, the replicated
build, the FK reset, and the frozen solver policy that sets the defaults below.

Everything the rest of the study reads is in MuJoCo's layout. Newton's joint
coordinates and site shapes are mapped to MuJoCo ``qpos``/``qvel`` addresses and
site ids by name, from a MuJoCo compile of the same file, so a task's cost
kernel and success test (written against MuJoCo indices) run unchanged on
``DeviceState()``. Two conventions differ and are converted on the device:

* free-joint quaternions: MuJoCo ``wxyz``, Newton ``xyzw``;
* free-joint angular velocity: MuJoCo body frame, Newton world frame (linear
  velocity agrees). The reference adapter passed ``qvel`` through unchanged.

Mesh geoms collide as their convex hulls, as in MuJoCo (``convex_meshes``).
Newton imports them as triangle meshes, whose contact path puts tens of points
on each mesh pair (about 10x MuJoCo's count on the duck's hulls) and let the
duck drift out of a held grasp; as convex meshes they take Newton's GJK/MPR
path.

Needs ``newton`` 1.6 (with Warp 1.17), which is imported only when a simulator
is built, so ``KaminoConfig`` is importable, and recordable, anywhere.

Speed comes from the contact buffer. Newton sizes it for the worst case of
every geometry pair (4,844 contacts per world on the cube rollout scene, 19,992
on the eval scene), and Kamino's solve costs scale with that capacity, not with
the ~5-20 contacts actually active. ``max_contacts_per_world`` caps it (128 by
default; the most measured on any Leap scene is 19 with convex meshes, and 186,
the low-fidelity duck, with ``convex_meshes=False``, which needs a higher cap), so
the dense Delassus matrix stays small and dense dynamics, a blocked Cholesky
factorization per step, works for any number of worlds.

Cost, measured on an idle RTX 4090 under CUDA-graph replay (``rho0=0.1``, a
256-contact cap, a 0.9 warm start): ~2.3 ms
per 4 ms step for one world on the cube rollout scene, ~4.7 ms for 64 worlds,
~9.3 ms for 256; ~0.6 ms per 0.5 ms step on the cube eval scene. The uncapped
sparse solve took 196, 370, 857 and 19 ms. Eager stepping is 3-15x slower than
graph replay, because Kamino's inner loops synchronize with the host each
iteration when not captured.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from ContactModelStudy.Simulators.VectorizedSimulator import (
    DeviceState,
    VectorizedSimulator,
    VectorizedSimulatorConfig,
)

_STORAGES = ("sparse", "dense")
_LINEAR_SOLVERS = ("auto", "CRF", "CR")
_PENALTY_UPDATES = ("fixed", "balanced")
_DYNAMICS_SOLVERS = ("padmm", "dvi")
_INTEGRATORS = ("euler", "moreau")


@dataclass
class KaminoConfig(VectorizedSimulatorConfig):
    """Kamino's solver settings; everything physical comes from the MJCF.

    The defaults are tuned against objects sticking in a grasp (see the README,
    "Why M5 objects stuck"): fixed penalty ``rho0=0.1``, full warm start,
    tolerance ``5e-4``, at most 100 iterations, a 128-contact buffer, and
    stronger contact (5x) and joint (10x) stabilization than Newton's. Grasps
    then converge in tens of iterations, where ``rho0=1`` with a 0.9 warm start
    ran nearly every grasp step to the iteration cap, and an eval step is about
    3.5x faster.

    The settings that matter, from the tuning:

    * ``padmm_rho0`` and ``padmm_warmstart_scale`` decide whether a grasp
      converges at all. A held object leaves its internal squeeze forces
      underdetermined, and PADMM's dual residual, which ``rho`` scales, stalls
      above the tolerance (``rho0=1``: 98% of grasp steps at the cap); shrinking
      the warm start makes it re-find those forces every step. Too small a
      ``rho0`` (0.03) instead under-converges the primal side, and fingers
      sink into the object on impact.
    * ``max_contacts_per_world``: a world past it drops contacts, and dropped
      contacts let fingers pass into the object, after which PADMM fails to
      converge and the fingertip stays embedded for hundreds of milliseconds.
    * ``contact_gap`` above 0 fills that buffer (a 1 mm gap overflowed 64)
      and slows convergence; leave it at 0.
    * ``contact_stabilization`` is how fast a penetration is pushed back out
      (Newton's 0.01 takes ~150 ms for 8 mm at a 0.5 ms step; 0.05 takes
      ~30 ms). Much higher adds energy and error.
    * ``joint_stabilization``: Kamino works in maximal coordinates, so the
      hand's hinges are constraints in the same solve. Steps cut off at the
      iteration cap let the links drift off their joints (fingertips 1 mm and
      growing after 0.25 s of an unconverged squeeze), and ``qpos``, which is
      what the video, the cost and the planner see, then no longer says where
      the links are. 0.1 (Newton's is 0.01) holds that drift to ~0.1 mm even
      when nothing converges, at no cost in iterations.
    * ``dynamics_solver="dvi"`` is a fixed-cost alternative, but at any useful
      speed it under-converges and penetrates (1-4 mm), so it is not the
      default.

    Attributes:
        dynamics_storage: ``"dense"`` (blocked Cholesky; the default) or
            ``"sparse"`` (iterative conjugate residual) Jacobian and dynamics.
            Dense needs ``max_contacts_per_world``: its matrix grows with the
            square of the contact capacity.
        max_contacts_per_world: Contact buffer per world, or ``None`` for
            Newton's worst-case estimate (thousands, and dense then overflows
            beyond a few worlds). Contacts past the cap are dropped; Kamino
            prints a warning, and ``GetState``/``Diagnostics`` warn when a
            world has reached it. The most measured on a Leap scene is 19.
        sparse_linear_solver: ``"CRF"`` (fused conjugate residual), ``"CR"``,
            or ``"auto"``: CRF, falling back to CR, with a warning, when the
            GPU cannot give the fused kernel the shared memory it asks for.
        dynamics_solver: ``"padmm"`` (proximal ADMM; the default) or ``"dvi"``
            (direct joint solve plus projected Gauss-Seidel on contacts).
        padmm_tolerance: Primal, dual and complementarity tolerance.
        padmm_max_iterations: Cap on PADMM iterations per step. A converging
            step needs 5-50; the cap bounds the rare one that does not, which
            a graph-replayed batch of worlds waits for.
        padmm_rho0: Initial (and, with ``"fixed"``, constant) penalty.
        padmm_penalty_update: ``"fixed"`` or ``"balanced"`` (sparse only).
        padmm_use_acceleration: Nesterov-accelerated PADMM.
        padmm_warmstart_scale: Scale on the previous step's constraint forces
            when warm-starting, in ``[0, 1]``.
        dvi_max_iterations: DVI outer iterations per step (``"dvi"`` only).
        dvi_sweeps: Gauss-Seidel sweeps per DVI iteration.
        dvi_tolerance: DVI tolerance on the projected update size.
        contact_stabilization: Baumgarte factor on contact penetration (the
            fraction removed per step), in ``[0, 1]``.
        joint_stabilization: Baumgarte factor on joint constraint drift.
        limit_stabilization: Baumgarte factor on joint-limit violation.
        contact_penetration_margin: Dead zone (m) on contact distance, so
            near-touching contacts are not corrected for float noise.
        integrator: ``"euler"`` (semi-implicit) or ``"moreau"`` (midpoint
            Moreau-Jean).
        contact_gap: Contact detection gap (m) for geoms without one. 0 is
            MuJoCo's default; Newton's own default is 0.1 m.
        collision_detection: Detect contacts at all. Off is for tests.
        convex_meshes: Collide mesh geoms as their convex hulls, as MuJoCo
            does (and so M1-M4). Off keeps Newton's triangle-mesh treatment,
            which generates several times more contact points per mesh pair.
    """

    dynamics_storage: str = "dense"
    max_contacts_per_world: int | None = 128
    sparse_linear_solver: str = "auto"
    dynamics_solver: str = "padmm"
    padmm_tolerance: float = 5e-4
    padmm_max_iterations: int = 1000#100
    padmm_rho0: float = 0.1
    padmm_penalty_update: str = "fixed"
    padmm_use_acceleration: bool = True
    padmm_warmstart_scale: float = 1.0
    dvi_max_iterations: int = 24
    dvi_sweeps: int = 2
    dvi_tolerance: float = 1e-5
    contact_stabilization: float = 0.05
    joint_stabilization: float = 0.1
    limit_stabilization: float = 0.01
    contact_penetration_margin: float = 1e-6
    integrator: str = "euler"
    contact_gap: float = 0.0
    collision_detection: bool = True
    convex_meshes: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dynamics_storage not in _STORAGES:
            raise ValueError(f"dynamics_storage must be one of {_STORAGES}, got {self.dynamics_storage!r}")
        if self.dynamics_solver not in _DYNAMICS_SOLVERS:
            raise ValueError(f"dynamics_solver must be one of {_DYNAMICS_SOLVERS}, got {self.dynamics_solver!r}")
        if self.integrator not in _INTEGRATORS:
            raise ValueError(f"integrator must be one of {_INTEGRATORS}, got {self.integrator!r}")
        for name in ("contact_stabilization", "joint_stabilization", "limit_stabilization", "padmm_warmstart_scale"):
            value = getattr(self, name)
            if not (np.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not (np.isfinite(self.contact_penetration_margin) and self.contact_penetration_margin >= 0):
            raise ValueError(f"contact_penetration_margin must be >= 0, got {self.contact_penetration_margin}")
        for name in ("dvi_max_iterations", "dvi_sweeps"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if not (np.isfinite(self.dvi_tolerance) and self.dvi_tolerance >= 0):
            raise ValueError(f"dvi_tolerance must be >= 0, got {self.dvi_tolerance}")
        if (self.dynamics_solver == "padmm" and self.padmm_penalty_update != "fixed"
                and self.dynamics_storage != "sparse"):
            raise ValueError("padmm_penalty_update='balanced' needs dynamics_storage='sparse'")
        if self.sparse_linear_solver not in _LINEAR_SOLVERS:
            raise ValueError(f"sparse_linear_solver must be one of {_LINEAR_SOLVERS}, "
                             f"got {self.sparse_linear_solver!r}")
        if self.padmm_penalty_update not in _PENALTY_UPDATES:
            raise ValueError(f"padmm_penalty_update must be one of {_PENALTY_UPDATES}, "
                             f"got {self.padmm_penalty_update!r}")
        if not (np.isfinite(self.padmm_tolerance) and self.padmm_tolerance > 0):
            raise ValueError(f"padmm_tolerance must be positive, got {self.padmm_tolerance}")
        if int(self.padmm_max_iterations) != self.padmm_max_iterations or self.padmm_max_iterations < 1:
            raise ValueError(f"padmm_max_iterations must be a positive integer, got {self.padmm_max_iterations}")
        if not (np.isfinite(self.padmm_rho0) and self.padmm_rho0 > 0):
            raise ValueError(f"padmm_rho0 must be positive, got {self.padmm_rho0}")
        cap = self.max_contacts_per_world
        if cap is not None and (int(cap) != cap or cap < 1):
            raise ValueError(f"max_contacts_per_world must be a positive integer or None, got {cap}")
        if not (np.isfinite(self.contact_gap) and self.contact_gap >= 0):
            raise ValueError(f"contact_gap must be >= 0, got {self.contact_gap}")


def _newton():
    try:
        import newton
    except ImportError as exc:
        raise ImportError(
            "The Kamino simulator (M5) needs Newton 1.6 with Warp 1.17 on Python >= 3.11: "
            "use the `contact_kamino` conda env (see ContactModelStudy/README.md)."
        ) from exc
    return newton


def _leaf(label: str) -> str:
    """The MJCF name at the end of a Newton label (``.../palm/if_bs/if_mcp`` -> ``if_mcp``)."""
    return str(label).rsplit("/", 1)[-1]


class _ScenePathResolver:
    """Resolve MJCF assets the way MuJoCo finds them for the Leap scenes.

    Newton 1.6 resolves a mesh file against a different base than MuJoCo, which
    applies the ``meshdir`` of an included file: ``../../objects/ducks/...``
    from the hand's asset directory lands outside the repo for Newton. When the
    direct path does not exist, the file's path with its ``..``/``.`` parts
    removed is tried under the scene's directory, dropping leading components
    until one exists (so ``objects/cube.obj`` and ``objects/ducks/.../x.obj``
    are both found under ``scenes/leap``).
    """

    def __init__(self, scene: Path):
        self.scene = Path(scene).resolve()

    def __call__(self, base_dir: str | None, file_path: str) -> str:
        base = Path(base_dir) if base_dir is not None else self.scene.parent
        candidate = (base / file_path).resolve()
        if candidate.exists():
            return str(candidate)
        parts = [p for p in Path(file_path).parts if p not in ("..", ".")]
        for k in range(len(parts)):
            alternate = self.scene.parent / Path(*parts[k:])
            if alternate.exists():
                return str(alternate)
        return str(candidate)


class Kamino(VectorizedSimulator):
    """``N`` worlds of an MJCF scene stepped by Newton's ``SolverKamino``.

    Attributes:
        mjm: A MuJoCo compile of the same file: the source of names, addresses
            and the actuator-to-joint map.
        model: The finalized Newton model (all ``N`` worlds).
        solver: The ``SolverKamino`` instance.
        linear_solver: The sparse linear solver actually in use (after "auto").
    """

    def __init__(self, xml: str | Path, sim_config: KaminoConfig | None = None, N: int = 1):
        super().__init__(xml, sim_config if sim_config is not None else KaminoConfig(), N)
        if self.model_path is None:
            raise ValueError("Kamino needs the MJCF as a file path (Newton resolves its assets from it)")
        newton = _newton()
        import mujoco

        cfg = self.config
        self._newton = newton
        self.mjm = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.device = wp.get_device(cfg.device)

        with wp.ScopedDevice(self.device):
            template, self._recorded_sites = self._templateBuilder(newton)
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            builder.rigid_gap = cfg.contact_gap
            newton.solvers.SolverKamino.register_custom_attributes(builder)
            builder.replicate(template, world_count=self.N)
            self.model = builder.finalize()
            self.model.set_gravity(tuple(float(g) for g in cfg.gravity))
            self._buildMaps(template)
            self.linear_solver = self._makeSolver(
                "CRF" if cfg.sparse_linear_solver == "auto" else cfg.sparse_linear_solver)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self._allocate()

        # The scene's own initial state, then a probe step: compiles the solver
        # and, for "auto", finds out whether the fused CR solve fits the GPU.
        self.SetState(self.mjm.qpos0, np.zeros(self.mjm.nv))
        self._probeSolver()
        self.SetState(self.mjm.qpos0, np.zeros(self.mjm.nv))

    # -- construction --------------------------------------------------------
    def _templateBuilder(self, newton):
        """One world of the scene, recording every site's shape index as it is imported.

        With ``convex_meshes``, each colliding mesh shape becomes a convex mesh
        here, before the template is replicated, holding the mesh's convex hull
        (at most ``maxhullvert`` vertices, as MuJoCo builds it). The hull, not
        the raw vertices, matters for speed too: Kamino bounds a convex mesh by
        querying every vertex, every step, and the duck's hull files carry up to
        8,464. Visual-only meshes are left as they are.
        """
        recorded: list[tuple[str, int]] = []

        class _Builder(newton.ModelBuilder):
            def add_site(self, body, **kwargs):
                shape = super().add_site(body, **kwargs)
                recorded.append((_leaf(kwargs.get("label") or f"site_{shape}"), shape))
                return shape

        b = _Builder(up_axis=newton.Axis.Z)
        b.rigid_gap = self.config.contact_gap
        newton.solvers.SolverKamino.register_custom_attributes(b)
        b.add_mjcf(str(self.model_path), path_resolver=_ScenePathResolver(Path(self.model_path)))
        if self.config.convex_meshes:
            collides, hulls = int(newton.ShapeFlags.COLLIDE_SHAPES), {}
            for i, t in enumerate(b.shape_type):
                if t == newton.GeoType.MESH and b.shape_flags[i] & collides:
                    mesh = b.shape_source[i]
                    if id(mesh) not in hulls:                   # meshes are shared, e.g. the fingertips
                        hulls[id(mesh)] = mesh.compute_convex_hull()
                    b.shape_source[i] = hulls[id(mesh)]
                    b.shape_type[i] = newton.GeoType.CONVEX_MESH
        return b, recorded

    def _buildMaps(self, template) -> None:
        """Index maps between Newton's per-world layout and MuJoCo's, by name."""
        import mujoco

        mjm, model = self.mjm, self.model
        jpw = model.joint_count // self.N                       # joints per world
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        labels = list(model.joint_label)[:jpw]
        self._nq_nw = int(model.joint_coord_count // self.N)
        self._nv_nw = int(model.joint_dof_count // self.N)
        if (self._nq_nw, self._nv_nw) != (mjm.nq, mjm.nv):
            raise RuntimeError(f"Newton imported {self._nq_nw} coordinates / {self._nv_nw} dofs per world, "
                               f"MuJoCo {mjm.nq} / {mjm.nv}")

        def span(starts, j, total):
            return int(starts[j]), int(starts[j + 1]) if j + 1 < len(starts) else int(total)

        newton_joint = {}
        for j, label in enumerate(labels):
            newton_joint[_leaf(label)] = (span(q_start, j, self._nq_nw), span(qd_start, j, self._nv_nw))

        q_src = np.full(mjm.nq, -1, dtype=np.int32)         # newton coord -> mujoco qpos index
        d_src = np.full(mjm.nv, -1, dtype=np.int32)         # newton dof   -> mujoco qvel index
        free_mj_q, free_mj_d, free_nw_q, free_nw_d = [], [], [], []
        for jid in range(mjm.njnt):
            name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name not in newton_joint:
                raise RuntimeError(f"MuJoCo joint {name!r} has no Newton joint")
            (q0, q1), (d0, d1) = newton_joint[name]
            mq, md = int(mjm.jnt_qposadr[jid]), int(mjm.jnt_dofadr[jid])
            if mjm.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                if (q1 - q0, d1 - d0) != (7, 6):
                    raise RuntimeError(f"free joint {name!r} is not 7/6 in Newton")
                # position xyz, then quaternion: Newton xyzw <- MuJoCo wxyz
                for k, src in enumerate((0, 1, 2, 4, 5, 6, 3)):
                    q_src[q0 + k] = mq + src
                for k in range(6):
                    d_src[d0 + k] = md + k            # angular frame converted in the kernels
                free_mj_q.append(mq), free_mj_d.append(md), free_nw_q.append(q0), free_nw_d.append(d0)
            else:
                nq_j = {int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(mjm.jnt_type[jid]), 1)
                if q1 - q0 != nq_j:
                    raise RuntimeError(f"joint {name!r}: Newton has {q1 - q0} coordinates, MuJoCo {nq_j}")
                q_src[q0:q1] = np.arange(mq, mq + nq_j)
                d_src[d0:d1] = np.arange(md, md + (d1 - d0))
        if (q_src < 0).any() or (d_src < 0).any():
            raise RuntimeError("some Newton coordinates have no MuJoCo counterpart")
        q_dst, d_dst = np.argsort(q_src).astype(np.int32), np.argsort(d_src).astype(np.int32)

        # Actuators, in MuJoCo control order, to Newton joint targets.
        targets = []
        n_target = int(self.control_template_size(model))
        for a in range(mjm.nu):
            if mjm.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise RuntimeError(f"actuator {a} does not drive a joint")
            jname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, int(mjm.actuator_trnid[a, 0]))
            (q0, _), (d0, _) = newton_joint[jname]
            targets.append(d0 if n_target == model.joint_dof_count else q0)
        self._check_gains(np.asarray(targets))

        # Sites, in MuJoCo site-id order, to Newton shape indices (template-local).
        site_shape = {name: shape for name, shape in self._recorded_sites}
        site_local = []
        for sid in range(mjm.nsite):
            name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SITE, sid)
            if name not in site_shape:
                raise RuntimeError(f"MuJoCo site {name!r} was not imported by Newton")
            site_local.append(site_shape[name])

        dev = self.device
        self._q_src = wp.array(q_src, dtype=wp.int32, device=dev)
        self._d_src = wp.array(d_src, dtype=wp.int32, device=dev)
        self._q_dst = wp.array(q_dst, dtype=wp.int32, device=dev)
        self._d_dst = wp.array(d_dst, dtype=wp.int32, device=dev)
        self._free = [wp.array(np.asarray(x, dtype=np.int32), dtype=wp.int32, device=dev)
                      for x in (free_mj_q, free_mj_d, free_nw_q, free_nw_d)]
        self._n_free = len(free_mj_q)
        self._targets = wp.array(np.asarray(targets, dtype=np.int32), dtype=wp.int32, device=dev)
        self._site_local = wp.array(np.asarray(site_local, dtype=np.int32), dtype=wp.int32, device=dev)
        self._target_per_dof = n_target == model.joint_dof_count

    @staticmethod
    def control_template_size(model) -> int:
        return int(model.control().joint_target_q.shape[0])

    def _check_gains(self, targets: np.ndarray) -> None:
        """Newton's position-servo gains must be the MJCF actuators' ``kp``/``kv``."""
        mjm = self.mjm
        ke = self.model.joint_target_ke.numpy()[targets]
        kd = self.model.joint_target_kd.numpy()[targets]
        kp, kv = mjm.actuator_gainprm[:, 0], -mjm.actuator_biasprm[:, 2]
        if not (np.allclose(ke, kp) and np.allclose(kd, kv)):
            warnings.warn(f"Newton imported servo gains ke={ke}, kd={kd}; the MJCF has kp={kp}, kv={kv}. "
                          f"Using the MJCF's.", RuntimeWarning, stacklevel=3)
            for arr, values in ((self.model.joint_target_ke, kp), (self.model.joint_target_kd, kv)):
                host = arr.numpy()
                for w in range(self.N):
                    host[w * self._nv_nw + targets] = values
                arr.assign(host)

    def _makeSolver(self, linear: str):
        newton, cfg = self._newton, self.config
        sparse = cfg.dynamics_storage == "sparse"
        dvi = cfg.dynamics_solver == "dvi"
        # Passed to from_model so its solver-dependent defaults (DVI: no
        # preconditioning, a direct bilateral solve) are the ones built.
        sc = newton.solvers.SolverKamino.Config.from_model(
            self.model, dynamics_solver=cfg.dynamics_solver, sparse_jacobian=sparse or dvi,
            sparse_dynamics=sparse, integrator=cfg.integrator)
        sc.use_fk_solver = True
        sc.use_collision_detector = cfg.collision_detection
        sc.collision_detector.default_gap = cfg.contact_gap
        sc.collision_detector.max_contacts_per_world = cfg.max_contacts_per_world
        if sparse and not dvi:
            sc.dynamics.linear_solver_type = linear
        sc.constraints.alpha = cfg.joint_stabilization
        sc.constraints.beta = cfg.limit_stabilization
        sc.constraints.gamma = cfg.contact_stabilization
        sc.constraints.delta = cfg.contact_penetration_margin
        sc.padmm.primal_tolerance = cfg.padmm_tolerance
        sc.padmm.dual_tolerance = cfg.padmm_tolerance
        sc.padmm.compl_tolerance = cfg.padmm_tolerance
        sc.padmm.max_iterations = int(cfg.padmm_max_iterations)
        sc.padmm.rho_0 = cfg.padmm_rho0
        sc.padmm.penalty_update_method = cfg.padmm_penalty_update
        sc.padmm.use_acceleration = cfg.padmm_use_acceleration
        sc.padmm.warmstart_scale = cfg.padmm_warmstart_scale
        sc.dvi.max_alternating_iterations = int(cfg.dvi_max_iterations)
        sc.dvi.inequality_sweeps_per_iteration = int(cfg.dvi_sweeps)
        sc.dvi.tolerance = cfg.dvi_tolerance
        sc.validate()
        try:
            self.solver = newton.solvers.SolverKamino(self.model, config=sc)
        except ValueError as exc:
            if not sparse and "non-negative" in str(exc):
                raise ValueError(f"dense dynamics overflows for N={self.N} worlds of this scene with "
                                 f"max_contacts_per_world={cfg.max_contacts_per_world}; lower the cap "
                                 f"or use dynamics_storage='sparse'") from exc
            raise
        self._reset_q = wp.empty(self.model.joint_coord_count, dtype=wp.float32, device=self.device)
        self._reset_qd = wp.empty(self.model.joint_dof_count, dtype=wp.float32, device=self.device)
        RC = newton.solvers.SolverKamino.ResetConfig
        from_q, from_qd = RC.FromJointQ(self._reset_q), RC.FromJointU(self._reset_qd)
        self._reset_config = RC(body_poses=from_q, body_velocities=from_qd, base_pose=from_q, base_velocity=from_qd)
        return linear if sparse and not dvi else None

    def _probeSolver(self) -> None:
        """One step; under "auto", fall back from fused CR when the GPU lacks shared memory."""
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self._solveStep()
                wp.synchronize_device(self.device)
        except RuntimeError as exc:
            if not (self.config.sparse_linear_solver == "auto" and self.linear_solver == "CRF"):
                raise
            warnings.warn(f"Kamino's fused CR solve does not fit this GPU for this scene ({exc}); "
                          f"using the unfused CR solver, which is much slower.", RuntimeWarning, stacklevel=3)
            with wp.ScopedDevice(self.device):
                self.linear_solver = self._makeSolver("CR")
            self.state_0, self.state_1 = self.model.state(), self.model.state()

    def _allocate(self) -> None:
        dev, N, mjm = self.device, self.N, self.mjm
        self.qpos_wp = wp.zeros((N, mjm.nq), dtype=wp.float32, device=dev)
        self.qvel_wp = wp.zeros((N, mjm.nv), dtype=wp.float32, device=dev)
        self.ctrl_wp = wp.zeros((N, mjm.nu), dtype=wp.float32, device=dev)
        self.site_xpos_wp = wp.zeros((N, mjm.nsite), dtype=wp.vec3, device=dev)
        self._U_wp = wp.zeros((N, self.horizon, mjm.nu), dtype=wp.float32, device=dev)
        self._reset_success = wp.zeros(N, dtype=wp.bool, device=dev)
        self._sequence_active = False
        self._applied_index = -1
        # On the device, advanced by a kernel, so a captured graph keeps it right.
        self._time_wp = wp.zeros(N, dtype=wp.float32, device=dev)
        # Peak active contacts per world since the last Diagnostics() (see _checkContacts).
        self._contact_peak = wp.zeros(N, dtype=wp.int32, device=dev)
        self._warned_contacts = False

    # -- control sequences ---------------------------------------------------
    def SetControlSequence(self, U_n) -> None:
        """Keep the ``(N, H, nu)`` sequence on the device, as ``VectorizedMujoco`` does."""
        if isinstance(U_n, wp.array):
            want = (self.N, self.horizon, self.nu)
            if tuple(U_n.shape) != want:
                raise ValueError(f"U_n warp array must have shape {want}, got {tuple(U_n.shape)}")
            wp.copy(self._U_wp, U_n)
        else:
            self._U_wp.assign(self._validate_control_sequence(U_n).astype(np.float32))
        self._sequence_active = True
        self._sequence_index = 0
        self._applied_index = -1

    def ClearControlSequence(self) -> None:
        self._sequence_active = False
        self._sequence_index = 0
        self._applied_index = -1

    @property
    def _has_control_sequence(self) -> bool:
        return self._sequence_active

    # -- stepping ------------------------------------------------------------
    def _solveStep(self) -> None:
        """One physics step: controls to joint targets, solve, copy the result back into state_0.

        A device copy rather than swapping the two states, so a captured graph
        replays from the same buffers whatever the number of steps.
        """
        wp.launch(_ctrl_to_targets_kernel, dim=(self.N, self.nu),
                  inputs=[self.ctrl_wp, self._targets, self.model.joint_dof_world_start
                          if self._target_per_dof else self.model.joint_coord_world_start],
                  outputs=[self.control.joint_target_q], device=self.device)
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.config.timestep)
        self.state_0.assign(self.state_1)
        contacts = self.solver._contacts_kamino
        if contacts is not None:
            wp.launch(_peak_kernel, dim=self.N, inputs=[contacts.world_active_contacts],
                      outputs=[self._contact_peak], device=self.device)

    def Step_GPU(self, steps: int = 1) -> None:
        """Advance every world ``steps`` timesteps; then refresh the MuJoCo-layout mirror."""
        for _ in range(steps):
            t = self._advance_sequence()
            if t is not None and t != self._applied_index:
                wp.launch(_assign_ctrl_kernel, dim=(self.N, self.nu),
                          inputs=[self._U_wp, t], outputs=[self.ctrl_wp], device=self.device)
                self._applied_index = t
            self._solveStep()
        wp.launch(_advance_time_kernel, dim=self.N, inputs=[float(steps * self.config.timestep)],
                  outputs=[self._time_wp], device=self.device)
        self._refreshMirror()

    def _refreshMirror(self) -> None:
        """Newton state -> MuJoCo-layout ``qpos``/``qvel``/``site_xpos``, on the device."""
        m = self.model
        wp.launch(_newton_to_mujoco_kernel, dim=self.N,
                  inputs=[self.state_0.joint_q, self.state_0.joint_qd, m.joint_coord_world_start,
                          m.joint_dof_world_start, self._q_dst, self._d_dst, *self._free, self._n_free],
                  outputs=[self.qpos_wp, self.qvel_wp], device=self.device)
        wp.launch(_site_kernel, dim=(self.N, self.mjm.nsite),
                  inputs=[self._site_local, m.shape_world_start, m.shape_body, m.shape_transform,
                          self.state_0.body_q],
                  outputs=[self.site_xpos_wp], device=self.device)

    def _resetFromMirror(self) -> None:
        """MuJoCo-layout ``qpos``/``qvel`` -> Newton joint state -> Kamino FK reset."""
        m = self.model
        wp.launch(_mujoco_to_newton_kernel, dim=self.N,
                  inputs=[self.qpos_wp, self.qvel_wp, m.joint_coord_world_start, m.joint_dof_world_start,
                          self._q_src, self._d_src, *self._free, self._n_free],
                  outputs=[self._reset_q, self._reset_qd], device=self.device)
        self.control.clear(self.model)
        self._reset_success.zero_()
        self.solver.reset(self.state_0, config=self._reset_config, success_mask=self._reset_success)
        self._refreshMirror()

    # -- state ---------------------------------------------------------------
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        """Reset every world to a MuJoCo-layout state, through Kamino's FK reset.

        Raises:
            RuntimeError: If the FK reset did not converge in some world.
        """
        self.qpos_wp.assign(self._to_worlds(q, self.nq, "q").astype(np.float32))
        qvel = (np.zeros((self.N, self.nv), dtype=np.float32) if q_dot is None
                else self._to_worlds(q_dot, self.nv, "q_dot").astype(np.float32))
        self.qvel_wp.assign(qvel)
        self._resetFromMirror()
        ok = self._reset_success.numpy()
        if not ok.all():
            raise RuntimeError(f"Kamino's FK reset did not converge in worlds {np.flatnonzero(~ok).tolist()}")

    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        self._checkContacts()
        return self.qpos_wp.numpy(), self.qvel_wp.numpy()

    def BroadcastState(self, q, q_dot=None, u=None) -> None:
        """Seed every world from device ``(nq,)``/``(nv,)``/``(nu,)`` arrays; kernel launches only."""
        wp.launch(_broadcast_kernel, dim=(self.N, self.nq), inputs=[q], outputs=[self.qpos_wp], device=self.device)
        if q_dot is None:
            self.qvel_wp.zero_()
        else:
            wp.launch(_broadcast_kernel, dim=(self.N, self.nv), inputs=[q_dot], outputs=[self.qvel_wp],
                      device=self.device)
        if u is not None:
            wp.launch(_broadcast_kernel, dim=(self.N, self.nu), inputs=[u], outputs=[self.ctrl_wp],
                      device=self.device)
        self._resetFromMirror()

    def DeviceState(self) -> DeviceState:
        return DeviceState(qpos=self.qpos_wp, qvel=self.qvel_wp, ctrl=self.ctrl_wp, site_xpos=self.site_xpos_wp)

    # -- control -------------------------------------------------------------
    def SetControl(self, u) -> None:
        if isinstance(u, wp.array):
            want = (self.N, self.nu)
            if tuple(u.shape) != want:
                raise ValueError(f"u warp array must have shape {want}, got {tuple(u.shape)}")
            wp.copy(self.ctrl_wp, u)
            return
        self.ctrl_wp.assign(self._to_worlds(u, self.nu, "u").astype(np.float32))

    def GetControl(self) -> np.ndarray:
        return self.ctrl_wp.numpy()

    # -- dimensions and diagnostics ----------------------------------------------
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
    def timestep(self) -> float:
        return float(self.config.timestep)

    @property
    def time(self) -> np.ndarray:
        """Simulated time per world, ``(N,)``, counted since construction."""
        return self._time_wp.numpy().astype(float)

    @property
    def contact_capacity(self) -> int:
        """Contacts each world can hold (the cap, or Newton's estimate without one); 0 without collisions."""
        contacts = self.solver._contacts_kamino
        return 0 if contacts is None else int(max(contacts.world_max_contacts_host))

    def _checkContacts(self, reset: bool = False) -> np.ndarray:
        """Peak active contacts per world since the last reset; warns once if a world reached the cap.

        A world at capacity may have dropped contacts. ``Diagnostics`` resets
        the peak, so its report covers the steps since the previous call.
        """
        peak = self._contact_peak.numpy()
        if reset:
            self._contact_peak.zero_()
        cap = self.contact_capacity
        if cap and (peak >= cap).any() and not self._warned_contacts:
            self._warned_contacts = True
            warnings.warn(f"Kamino: {int((peak >= cap).sum())} world(s) reached the contact capacity of {cap} "
                          f"per world, so contacts may have been dropped; raise max_contacts_per_world.",
                          RuntimeWarning, stacklevel=3)
        return peak

    def Diagnostics(self) -> dict:
        """The last step's PADMM status per world, and peak contacts since the last check.

        Returns:
            ``converged``, ``iterations`` and the residuals ``r_p``/``r_d``/``r_c``
            of the last step; ``contacts``, the most active contacts per world
            since the previous ``Diagnostics`` (``(N,)``); and
            ``contact_capacity``.
        """
        s = self.solver.status.numpy()
        return {"converged": s["converged"].astype(bool), "iterations": s["iterations"].astype(int),
                "r_p": s["r_p"].astype(float), "r_d": s["r_d"].astype(float), "r_c": s["r_c"].astype(float),
                "contacts": self._checkContacts(reset=True), "contact_capacity": self.contact_capacity}

    def Close(self) -> None:
        self.solver = None
        self.model = None
        self.state_0 = self.state_1 = self.control = None


# -- kernels --------------------------------------------------------------------
@wp.kernel
def _broadcast_kernel(src: wp.array(dtype=float), dst: wp.array2d(dtype=float)):
    n, i = wp.tid()
    dst[n, i] = src[i]


@wp.kernel
def _peak_kernel(active: wp.array(dtype=wp.int32), peak: wp.array(dtype=wp.int32)):
    n = wp.tid()
    peak[n] = wp.max(peak[n], active[n])


@wp.kernel
def _advance_time_kernel(dt: float, time: wp.array(dtype=float)):
    n = wp.tid()
    time[n] = time[n] + dt


@wp.kernel
def _assign_ctrl_kernel(U: wp.array3d(dtype=float), t: int, ctrl: wp.array2d(dtype=float)):
    n, a = wp.tid()
    ctrl[n, a] = U[n, t, a]


@wp.kernel
def _ctrl_to_targets_kernel(ctrl: wp.array2d(dtype=float), targets: wp.array(dtype=wp.int32),
                            world_start: wp.array(dtype=wp.int32), joint_target_q: wp.array(dtype=float)):
    n, a = wp.tid()
    joint_target_q[world_start[n] + targets[a]] = ctrl[n, a]


@wp.kernel
def _mujoco_to_newton_kernel(
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
    coord_start: wp.array(dtype=wp.int32), dof_start: wp.array(dtype=wp.int32),
    q_src: wp.array(dtype=wp.int32), d_src: wp.array(dtype=wp.int32),
    free_mj_q: wp.array(dtype=wp.int32), free_mj_d: wp.array(dtype=wp.int32),
    free_nw_q: wp.array(dtype=wp.int32), free_nw_d: wp.array(dtype=wp.int32), n_free: int,
    joint_q: wp.array(dtype=float), joint_qd: wp.array(dtype=float),
):
    n = wp.tid()
    cs = coord_start[n]
    ds = dof_start[n]
    for c in range(q_src.shape[0]):
        joint_q[cs + c] = qpos[n, q_src[c]]
    for d in range(d_src.shape[0]):
        joint_qd[ds + d] = qvel[n, d_src[d]]
    for f in range(n_free):
        mq = free_mj_q[f]
        md = free_mj_d[f]
        # MuJoCo angular velocity is in the body frame; Newton's is in the world frame.
        rot = wp.quat(qpos[n, mq + 4], qpos[n, mq + 5], qpos[n, mq + 6], qpos[n, mq + 3])
        rot = wp.normalize(rot)
        w = wp.quat_rotate(rot, wp.vec3(qvel[n, md + 3], qvel[n, md + 4], qvel[n, md + 5]))
        nd = ds + free_nw_d[f]
        joint_qd[nd + 3] = w[0]
        joint_qd[nd + 4] = w[1]
        joint_qd[nd + 5] = w[2]


@wp.kernel
def _newton_to_mujoco_kernel(
    joint_q: wp.array(dtype=float), joint_qd: wp.array(dtype=float),
    coord_start: wp.array(dtype=wp.int32), dof_start: wp.array(dtype=wp.int32),
    q_dst: wp.array(dtype=wp.int32), d_dst: wp.array(dtype=wp.int32),
    free_mj_q: wp.array(dtype=wp.int32), free_mj_d: wp.array(dtype=wp.int32),
    free_nw_q: wp.array(dtype=wp.int32), free_nw_d: wp.array(dtype=wp.int32), n_free: int,
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
):
    n = wp.tid()
    cs = coord_start[n]
    ds = dof_start[n]
    for i in range(q_dst.shape[0]):
        qpos[n, i] = joint_q[cs + q_dst[i]]
    for i in range(d_dst.shape[0]):
        qvel[n, i] = joint_qd[ds + d_dst[i]]
    for f in range(n_free):
        nq = cs + free_nw_q[f]
        nd = ds + free_nw_d[f]
        md = free_mj_d[f]
        rot = wp.normalize(wp.quat(joint_q[nq + 3], joint_q[nq + 4], joint_q[nq + 5], joint_q[nq + 6]))
        w = wp.quat_rotate_inv(rot, wp.vec3(joint_qd[nd + 3], joint_qd[nd + 4], joint_qd[nd + 5]))
        qvel[n, md + 3] = w[0]
        qvel[n, md + 4] = w[1]
        qvel[n, md + 5] = w[2]


@wp.kernel
def _site_kernel(
    site_local: wp.array(dtype=wp.int32), shape_world_start: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32), shape_transform: wp.array(dtype=wp.transformf),
    body_q: wp.array(dtype=wp.transformf), site_xpos: wp.array2d(dtype=wp.vec3),
):
    n, s = wp.tid()
    shape = shape_world_start[n] + site_local[s]
    body = shape_body[shape]
    xf = shape_transform[shape]
    if body >= 0:
        xf = wp.transform_multiply(body_q[body], xf)
    site_xpos[n, s] = wp.transform_get_translation(xf)
