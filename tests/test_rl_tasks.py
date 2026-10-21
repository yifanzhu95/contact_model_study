"""Native MPC task/scene choices through RL; no optimizer updates."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.Policy import Actor, export_actor
from ContactModelStudy.RL.RunIO import validate_policy_environment
from ContactModelStudy.Tasks.Registry import TASK_MODULES, available_scenes, make_task


SCENES = available_scenes()


@pytest.mark.parametrize("selection", SCENES, ids=lambda s: "-".join(s.values()))
def test_every_existing_mpc_scene_loads_native_initial_conditions_and_costs(selection):
    config = EnvConfig(num_envs=1, **selection)
    bank = CaseBank.development(count=1, seed=2, difficulty=8, task_name=config.task_name)
    env = LeapVectorEnv(config, bank)
    native = make_task(config.task_name, role=config.scene_role, hand_acc=config.hand_acc,
                       obj_acc=config.obj_acc)
    try:
        assert type(env.task) is type(native)
        assert env.task.getModelPath() == native.getModelPath()
        assert env.task.params["fallen_z"] == native.params["fallen_z"]
        assert env.task.params["cost_weights"] == native.params["cost_weights"]
        np.testing.assert_array_equal(bank.cases[0].qpos, native.getInitialState()[0])
        np.testing.assert_array_equal(bank.cases[0].control, native.getInitialState()[2])
        assert env.reset_all([0]).shape == (1, 73)
        tr = env.step_command(env.current_target)
        assert torch.isfinite(tr.next_obs).all() and torch.isfinite(tr.reward).all()
    finally:
        env.close()


def test_task_catalog_matches_mpc_and_rejects_cross_object_banks(tmp_path):
    from ContactModelStudy.Drivers.run_episodes import TASKS
    assert set(TASKS) == set(TASK_MODULES)
    bank = CaseBank.development(count=1, seed=2, difficulty=8, task_name="duck_reorient")
    path = tmp_path/"duck.json"; bank.save(path)
    assert CaseBank.load(path) == bank
    assert json.loads(path.read_text())["schema"] == "leap_case_bank_v2"
    with pytest.raises(ValueError, match="Case-bank task"):
        LeapVectorEnv(EnvConfig(), bank)
    legacy = CaseBank.development(count=1, seed=2, difficulty=8)
    assert "task_name" not in json.loads(legacy.json_text())
    with pytest.raises(ValueError, match="Case-bank task"):
        LeapVectorEnv(EnvConfig(task_name="duck_reorient"), legacy)


def test_case_creation_cli_records_task_without_physics_and_refuses_overwrite(tmp_path, capsys):
    from ContactModelStudy.Drivers.make_rl_cases import main
    output = tmp_path/"cases"/"ball.json"
    main(["--task", "ball_reorient", "--count", "2", "--seed", "3", "--output", str(output)])
    assert CaseBank.load(output).task_name == "ball_reorient"
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--task", "cube_reorient", "--output", str(output)])
    assert output.read_bytes() == before
    assert main(["--list-scenes"]) == SCENES
    assert len(json.loads(capsys.readouterr().out)) == len(SCENES)


def test_missing_geometry_fails_before_backend_allocation(monkeypatch):
    import ContactModelStudy.RL.LeapVectorEnv as module
    monkeypatch.setattr(module, "make_backend", lambda *a: pytest.fail("Allocated invalid scene"))
    config = EnvConfig(task_name="duck_reorient", scene_role="rollout", obj_acc="not_a_scene")
    bank = CaseBank.development(count=1, seed=2, difficulty=8, task_name="duck_reorient")
    with pytest.raises((ValueError, FileNotFoundError)):
        LeapVectorEnv(config, bank)


@pytest.mark.parametrize("name", tuple(TASK_MODULES))
def test_cost_override_and_goal_difficulty_reuse_native_task(name):
    config = EnvConfig(task_name=name, cost_weights={"w_quat": 13.}, goal_difficulty=3, num_envs=1)
    bank = CaseBank.development(count=2, seed=2, difficulty=3, task_name=name)
    native = make_task(name, seed=2, goal_difficulty=3)
    start = native.getGoal()
    for case in bank.cases:
        np.testing.assert_array_equal(case.goal_quaternion, native.sampleNewGoal(current_goal=start))
    env = LeapVectorEnv(config, bank)
    try:
        assert env.task.params["cost_weights"]["w_quat"] == 13.
        assert env.task.config.goal_difficulty == 3
    finally:
        env.close()


def test_explicit_scene_transfer_preserves_action_and_observation_guards(tmp_path):
    from ContactModelStudy.Drivers.evaluate_rl import run_evaluation
    config = EnvConfig(task_name="duck_reorient", scene_role="rollout", obj_acc="foam4",
                       hand_acc="low", num_envs=1, episode_steps=2)
    bank = CaseBank.development(count=1, seed=2, difficulty=8, task_name=config.task_name)
    env = LeapVectorEnv(config, bank)
    try:
        manifest = env.describe()
        actor_path = tmp_path/"actor.pt"; export_actor(Actor(), manifest, actor_path)
    finally:
        env.close()
    target = replace(config, scene_role="eval").to_dict()
    with pytest.raises(ValueError, match="scene_xml_sha256"):
        run_evaluation(actor_path, bank, tmp_path/"strict", environment=target)
    result = run_evaluation(actor_path, bank, tmp_path/"transfer", environment=target, allow_scene_transfer=True)
    assert result["execution_status"] == "completed" and result["new_PPO_updates"] == 0
    current = dict(manifest); current["action"] = {**manifest["action"], "max_target_change": 100.}
    with pytest.raises(ValueError, match="action"):
        validate_policy_environment(manifest, current, allow_scene_transfer=True)


def test_matrix_uses_requested_object_and_scene_for_each_model(tmp_path):
    from ContactModelStudy.Drivers.evaluate_rl_matrix import main
    config = EnvConfig(num_envs=1, episode_steps=2)
    cube = CaseBank.development(count=1, seed=2, difficulty=8)
    env = LeapVectorEnv(config, cube)
    actor_path = tmp_path/"actor.pt"
    try:
        export_actor(Actor(), env.describe(), actor_path)
    finally:
        env.close()
    bank = CaseBank.development(count=1, seed=2, difficulty=8, task_name="duck_reorient")
    bank_path = tmp_path/"duck.json"; bank.save(bank_path)
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps({"environment": replace(config, task_name="duck_reorient").to_dict()}))
    output = tmp_path/"matrix"
    main(["--policy", str(actor_path), "--case-bank", str(bank_path), "--config", str(config_path),
          "--allow-scene-transfer", "--models", "M1", "M2", "mujoco", "--output", str(output)])
    assert json.loads((output/"summary.json").read_text())["completed_cells"] == 3
    for model in ["M1", "M2", "mujoco"]:
        current = json.loads((output/"policy_000"/model/"manifest.json").read_text())
        assert current["environment"]["config"]["task_name"] == "duck_reorient"
        assert current["environment"]["config"]["contact_preset"] == (model if model != "mujoco" else "native")
        assert current["scene_transfer"]["allowed"]


@pytest.mark.parametrize("name", ["mujoco", "pinocchio", "drake"])
def test_original_cpu_evaluators_use_mpc_factory_and_independent_reset(name):
    if name != "mujoco":
        pytest.importorskip("pydrake" if name == "drake" else "pinocchio")
    from dataclasses import asdict
    from ContactModelStudy.Utils.EvalSimulators import evalSimConfig
    config = EnvConfig(task_name="duck_reorient", physics_backend=name+"_eval", contact_preset="native")
    bank = CaseBank.development(count=2, seed=2, difficulty=8, task_name=config.task_name)
    env = LeapVectorEnv(config, bank)
    try:
        env.reset_all([0, 1])
        assert env.backend.describe()["solver_config"] == asdict(evalSimConfig(name, config.physics_dt)[1])
        other = env.backend.sims[1]
        before = other.GetState()
        env.reset_indices([0], [1])
        assert env.backend.sims[1] is other
        after = other.GetState()
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        assert torch.isfinite(env.step_command(env.current_target).next_obs).all()
    finally:
        env.close()


@pytest.mark.parametrize("name", tuple(TASK_MODULES))
def test_common_mppi_builder_uses_selected_object_and_independent_scene(name, monkeypatch):
    import ContactModelStudy.SamplingBasedPlanners.MPPI as module
    import ContactModelStudy.Utils.ContactModelPresets as presets
    from ContactModelStudy.Controllers.MPPIController import build_mppi_controller
    captured = {}
    def simulator(model, path, **kwargs):
        captured.update(model=model, path=path, kwargs=kwargs)
        return SimpleNamespace(Close=lambda: None)
    def planner(sim, task, config):
        captured.update(task=task, config=config)
        return SimpleNamespace()
    monkeypatch.setattr(presets, "GetContactModelSim", simulator)
    monkeypatch.setattr(module, "MPPI", planner)
    config = EnvConfig(task_name=name, num_envs=1)
    env = SimpleNamespace(config=config, codec=SimpleNamespace(delta=.016))
    ctrl = build_mppi_controller({"rollout_model": "M3", "n_samples": 2, "horizon": 2,
                                 "rollout_scene": {"role": "rollout", "hand_acc": "low", "obj_acc": "high"}}, env)
    assert captured["task"].OBJECT == name.removesuffix("_reorient")
    assert "_low_high.xml" in captured["path"]
    assert not captured["config"].plan_on_eval_scene
    ctrl.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires native CUDA backends")
@pytest.mark.parametrize("name", tuple(TASK_MODULES))
def test_common_mppi_executes_each_native_task_with_independent_rollout_scene(name, tmp_path):
    from ContactModelStudy.Controllers.MPPIController import build_mppi_controller
    from ContactModelStudy.Drivers.ControllerEpisodes import run_controller_cases
    config = EnvConfig(num_envs=1, task_name=name, episode_steps=1)
    bank = CaseBank.development(count=1, seed=2, difficulty=8, task_name=name)
    settings = {"rollout_model": "M2", "n_samples": 2, "horizon": 1,
                "rollout_scene": {"role": "rollout", "hand_acc": "low",
                                  "obj_acc": "foam4" if name == "duck_reorient" else "high"},
                "mppi": {"noise_sigma": .01, "temperature": 1., "use_graph": False}}
    result = run_controller_cases(config, bank, tmp_path/name,
                                  lambda env: build_mppi_controller(settings, env), identity={})
    assert result["execution_status"] == "completed" and result["new_PPO_updates"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires native CUDA backends")
@pytest.mark.parametrize("preset,backend", [("M1", "mjwarp"), ("M2", "mjwarp"),
                                           ("M3", "comfree"), ("M4", "xpbd"), ("M5", "kamino")])
@pytest.mark.parametrize("selection", [dict(task_name=name) for name in TASK_MODULES]+
                         [dict(task_name="duck_reorient", scene_role="rollout", hand_acc="low", obj_acc="foam4"),
                          dict(task_name="duck_reorient", scene_role="rollout", hand_acc="high", obj_acc="foam64")])
def test_native_models_read_all_objects_and_foam_extremes(selection, preset, backend):
    config = EnvConfig(num_envs=1, device="cuda:0", contact_preset=preset, physics_backend=backend, **selection)
    bank = CaseBank.development(count=1, seed=2, difficulty=8, task_name=config.task_name)
    env = LeapVectorEnv(config, bank)
    try:
        assert torch.isfinite(env.reset_all([0])).all()
        env.backend.step(env.current_target, 2)
        frame = env.read_frame()
        assert frame.qpos.shape == (1, 23) and torch.isfinite(frame.object_twist).all()
    finally:
        env.close()
