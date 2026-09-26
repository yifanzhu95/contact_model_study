"""Tasks: cost and outcomes against the old code, goal sampling, configs, the three objects."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from conftest import REPO_ROOT
from ContactModelStudy.Tasks.BallReorient import BallReorient
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.DuckReorient import DuckReorient
from ContactModelStudy.Tasks.LeapReorient import COST_WEIGHT_KEYS, SCENES_DIR, LeapReorientConfig
from ContactModelStudy.Tasks.TaskBase import TaskBaseConfig, TaskRole

N = 64


# -- cost and outcomes -------------------------------------------------------
@pytest.fixture(scope="module")
def perturbed_worlds(cube_tasks, cube_initial):
    """64 MJWarp worlds scattered around the initial state."""
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    ro, _ = cube_tasks
    q0, _, u0 = cube_initial
    rng = np.random.default_rng(0)
    vs = VectorizedMujoco(ro.getModelPath(), VectorizedMujocoConfig(horizon=2), N=N)
    Q = np.tile(q0, (N, 1)) + rng.normal(0, 0.06, (N, 23))
    Q[:, 19:23] /= np.linalg.norm(Q[:, 19:23], axis=1, keepdims=True)
    vs.SetState(Q, rng.normal(0, 0.3, (N, 22)))
    vs.SetControl(u0)
    return vs


def _old_cost_kernel():
    import warp as wp
    from contact_study.tasks.grasp_reorient import grasp_reorient_cost_wp

    @wp.kernel
    def old_cost(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                 ctrl: wp.array2d(dtype=float), sx: wp.array2d(dtype=wp.vec3),
                 sm: wp.array2d(dtype=wp.mat33), terminal: bool, goal: wp.array(dtype=float),
                 idx: wp.array(dtype=int), w: wp.array(dtype=float), out: wp.array(dtype=float)):
        n = wp.tid()
        out[n] = grasp_reorient_cost_wp(qpos[n], qvel[n], ctrl[n], sx[n], sm[n], terminal, goal, idx, w)
    return old_cost


@pytest.mark.gpu
@pytest.mark.legacy
@pytest.mark.parametrize("terminal", [False, True])
def test_cost_matches_old_kernel_bit_for_bit(terminal, perturbed_worlds, cube_tasks):
    import warp as wp
    ro, _ = cube_tasks
    vs = perturbed_worlds
    g = ro._gpu_arrays("cuda")
    ref = wp.zeros(N, dtype=wp.float32, device="cuda")
    wp.launch(_old_cost_kernel(), dim=N, inputs=[vs.d.qpos, vs.d.qvel, vs.d.ctrl, vs.d.site_xpos,
              vs.d.site_xmat, terminal, g["goal"], g["indices"], g["weights"]], outputs=[ref])
    new = ro.calcCosts(vs, terminal=terminal)
    wp.synchronize()
    assert np.array_equal(ref.numpy(), new.numpy())


@pytest.mark.gpu
def test_outcomes_vectorized_equal_single_simulator(perturbed_worlds, cube_tasks):
    from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
    ro, ev = cube_tasks
    vs = perturbed_worlds
    succ, fail = ro.isSuccess(vs).numpy(), ro.isFailure(vs).numpy()
    cpu = Mujoco(ev.getModelPath(), MujocoConfig())
    qs, vv = vs.GetState()
    cs, cf = [], []
    for i in range(N):
        cpu.SetState(qs[i], vv[i])
        cs.append(ro.isSuccess(cpu))
        cf.append(ro.isFailure(cpu))
    assert np.array_equal(succ, cs) and np.array_equal(fail, cf)
    assert not np.any(succ & fail)


def test_calc_costs_refuses_a_cpu_simulator(cube_tasks):
    from ContactModelStudy.Simulators.Mujoco import Mujoco
    ro, ev = cube_tasks
    with pytest.raises(TypeError):
        ro.calcCosts(Mujoco(ev.getModelPath()))


# -- goals -------------------------------------------------------------------
def _same_rotation(a, b):
    return min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-9


@pytest.mark.legacy
@pytest.mark.parametrize("level", [0, 1, 2, 5])
def test_goal_levels_match_the_old_sampler(level):
    from contact_study.tasks.grasp_reorient import _TARGET_QUAT, GraspReorientTask

    class OldStub(GraspReorientTask):
        def _update_goal(self, mjd, q):
            self.target_quat = np.asarray(q, float).copy()
            self.seen.append(self.target_quat)

    o = object.__new__(OldStub)
    o.goal_difficulty, o._face_index = level, 0
    o._canonical_quat, o.target_quat, o.seen = _TARGET_QUAT.copy(), _TARGET_QUAT.copy(), []
    r = np.random.default_rng(3)
    for _ in range(15):
        o.sample_new_goal(None, r)
    t = CubeReorient(LeapReorientConfig(goal_difficulty=level, seed=3))
    seq = []
    for _ in range(15):
        x = t.sampleNewGoal()
        t.setGoal(x)
        seq.append(x)
    assert all(_same_rotation(a, b) for a, b in zip(seq, o.seen))


def test_level_8_never_reissues_its_current_goal():
    t = CubeReorient(LeapReorientConfig(goal_difficulty=8, seed=1))
    for _ in range(100):
        before = t._sample_ref.copy()
        x = t.sampleNewGoal()
        assert not _same_rotation(x, before)
        t.setGoal(x)


def test_same_seed_same_goals():
    a, b = CubeReorient(LeapReorientConfig(seed=5)), CubeReorient(LeapReorientConfig(seed=5))
    assert np.allclose(a.sampleNewGoal(), b.sampleNewGoal())


# -- task configs ------------------------------------------------------------
def test_config_fields():
    assert [f for f in TaskBaseConfig.__dataclass_fields__] == ["role", "timestep", "seed"]
    assert list(LeapReorientConfig.__dataclass_fields__)[3:] == [
        "hand_acc", "obj_acc", "scenes_dir", "eval_steps_per_rollout_step", "goal_difficulty",
        "cost_weights"]


def test_defaults_and_role_coercion():
    t = CubeReorient()
    assert t.role is TaskRole.EVAL and t.timestep == 0.002 and t.config.scenes_dir == SCENES_DIR
    assert LeapReorientConfig(role="rollout").role is TaskRole.ROLLOUT


def test_rollout_task_runs_at_the_coarse_timestep():
    kw = dict(timestep=0.0005, eval_steps_per_rollout_step=8)
    r = CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT, **kw))
    e = CubeReorient(LeapReorientConfig(role=TaskRole.EVAL, **kw))
    assert np.isclose(r.timestep, 0.004) and np.isclose(e.timestep, 0.0005)


def test_rollout_path_uses_mesh_fidelity():
    t = CubeReorient(LeapReorientConfig(role="rollout", hand_acc="low", obj_acc="high"))
    assert t.getModelPath().endswith("env_leap_rollout_cube_low_high.xml")


@pytest.mark.parametrize("kwargs", [dict(timestep=0), dict(role="x"), dict(eval_steps_per_rollout_step=0),
                                    dict(eval_steps_per_rollout_step=1.5), dict(goal_difficulty=10)])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        LeapReorientConfig(**kwargs)


def test_wrong_config_type_is_refused():
    with pytest.raises(TypeError, match="LeapReorientConfig"):
        CubeReorient(TaskBaseConfig())


# -- the three objects -------------------------------------------------------
TASKS = [CubeReorient, DuckReorient, BallReorient]

#: Entries deliberately changed from the old table, as (object, key) -> indices.
#: The cube's th_axl start (index 13) was lowered from +1.0 to +0.35 so the
#: thumb starts inside its joint range (2.094) instead of 0.43 rad past it.
DELIBERATE_CHANGES = {("cube", "init_qpos"): {13}}


@pytest.mark.legacy
@pytest.mark.parametrize("cls", TASKS)
def test_object_params_equal_the_old_table(cls):
    from contact_study.tasks.grasp_reorient import _OBJ_PARAMS, _TARGET_QUAT
    old, p = _OBJ_PARAMS[cls.OBJECT], cls().params
    for k in ("init_qpos", "init_ctrl", "target_pos"):
        changed = set(np.nonzero(p[k] != old[k])[0].tolist())
        assert changed <= DELIBERATE_CHANGES.get((cls.OBJECT, k), set()), (k, sorted(changed))
    assert p["fallen_z"] == old["fallen_z"]
    assert np.array_equal(p["target_quat"], _TARGET_QUAT)
    for k in COST_WEIGHT_KEYS:
        assert float(p["cost_weights"][k]) == float(old["cost_weights"][k]), k


@pytest.mark.parametrize("cls", TASKS)
def test_every_scene_variant_compiles(cls):
    name = cls.OBJECT
    prefix = f"env_leap_rollout_{name}_"
    variants = sorted(os.path.basename(f)[len(prefix):-4]
                      for f in glob.glob(str(REPO_ROOT / "scenes" / "leap" / f"{prefix}*.xml")))
    assert variants
    for v in variants:
        h, o = v.split("_", 1)
        t = cls(LeapReorientConfig(role=TaskRole.ROLLOUT, hand_acc=h, obj_acc=o))
        assert (t.nq, t.nu, len(t.indices)) == (23, 16, 9), v
    e = cls(LeapReorientConfig(role=TaskRole.EVAL))
    assert e.nq == 23 and len(e.indices) == 9


@pytest.mark.parametrize("cls", TASKS)
def test_object_stays_held_for_two_seconds(cls):
    from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
    t = cls(LeapReorientConfig(role=TaskRole.EVAL))
    sim = Mujoco(t.getModelPath(), MujocoConfig(timestep=0.002))
    t.setSimToInitialState(sim)
    sim.Step(1000)
    q, _ = sim.GetState()
    assert not t.isFailure(sim) and np.isfinite(q).all() and q[18] > 0.05


def test_renderer_camera_from_the_task():
    from ContactModelStudy.Renderers.RendererBase import VideoRendererBaseConfig
    cfg = VideoRendererBaseConfig()
    CubeReorient().alignRendererConfigWithTask(cfg)
    assert cfg.cam_name == "demo-cam"
    assert np.allclose(cfg.cam_pos, [0.19, 0.01, 0.4])
    assert np.allclose(cfg.cam_quat, [0.68819096, 0.16245985, 0.16245985, 0.68819096])
