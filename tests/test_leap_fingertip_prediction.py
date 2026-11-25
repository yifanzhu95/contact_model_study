"""Mean-control rollouts and asynchronous preview without USB, LCM or a GUI."""
import json
import time

import mujoco
import numpy as np
import pytest

import run_leap_hardware as runner
from ContactModelStudy.Hardware.Fingertips import FingertipPrediction, MeanTrajectory
from ContactModelStudy.Hardware.Workflow import build_rollout, build_tasks


def setup(tmp_path, mode='pos_relative', contact='M2'):
    args = runner.parseArgs(['--device', 'cpu', '--control-mode', mode, '--rollout-model', contact,
        '--n-samples', '4', '--horizon', '3', '--substeps', '2', '--contact-param', 'iterations=31'])
    task, _, _ = build_tasks(args, tmp_path)
    q, v, u = task.getInitialState()
    q[18] = .4  # separate the object from contacts for this independent FK/dynamics check
    return args, task, q, v, u


@pytest.mark.parametrize('mode', ['absolute', 'pos_relative', 'ctrl_relative'])
@pytest.mark.parametrize('contact', ['M1', 'M2'])
def test_mean_prediction_matches_direct_selected_physics_and_forward_kinematics(tmp_path, mode, contact):
    args, task, q, v, u = setup(tmp_path, mode, contact)
    args.n_samples = 1
    sim = build_rollout(args, task, *runner.model_limits(task.mjm))
    trajectory = MeanTrajectory(sim, mode)
    mean = np.tile(np.linspace(-.04, .04, 16), (3, 1))
    if mode == 'absolute':
        mean += q[:16]
    try:
        actual = trajectory.predict(mean, q, v, u, u)
        data = mujoco.MjData(sim.mjm)
        data.qpos[:], data.qvel[:], data.act[:] = q.astype(np.float32), v.astype(np.float32), u.astype(np.float32)
        data.ctrl[:] = u.astype(np.float32)
        mujoco.mj_forward(sim.mjm, data)
        expected = [data.site_xpos[trajectory.sites].copy()]
        previous = u.copy()
        for row in mean:
            command = (data.qpos[:16] + row if mode == 'pos_relative' else
                       previous + row if mode == 'ctrl_relative' else row.copy())
            data.ctrl[:] = command
            for _ in range(sim.config.resolved_substeps):
                mujoco.mj_step(sim.mjm, data)
            mujoco.mj_kinematics(sim.mjm, data)
            expected.append(data.site_xpos[trajectory.sites].copy())
            previous = command
        assert actual.shape == (4, 4, 3)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        # Servo dynamics matter: fingertip positions are not FK of raw targets.
        data.qpos[:16] = command
        mujoco.mj_kinematics(sim.mjm, data)
        assert not np.allclose(actual[-1], data.site_xpos[trajectory.sites], atol=1e-5)
    finally:
        sim.Close()


def test_background_prediction_keeps_main_rollout_independent_and_clears_old_jobs(tmp_path):
    args, task, q, v, u = setup(tmp_path)
    main = build_rollout(args, task, *runner.model_limits(task.mjm))
    main.SetState(q, v)
    original = main.GetState()[0].copy()
    path = tmp_path/'paths.json'
    preview = FingertipPrediction(args, task, *runner.model_limits(task.mjm), path)
    mean = np.full((3, 16), .02)
    preview.start(mean, q, v, u, u)
    try:
        preview.submit(mean, q, v, u, u, time.time(), 0, 2)
        deadline = time.monotonic()+5
        while not path.exists() and time.monotonic()<deadline:
            time.sleep(.005)
        record = json.loads(path.read_text())
        assert record['status']=='ok' and record['source']=='mppi_weighted_mean_rollout'
        assert record['step']==2 and record['rollout_model']=='M2'
        assert record['time_offsets_s'] == [0, .008, .016, .024]
        assert np.asarray(record['points']).shape == (4,4,3)
        assert preview.metadata['n_worlds']==1 and args.n_samples==4
        assert preview.metadata['solver_iterations']==31  # explicit contact overrides are retained
        np.testing.assert_array_equal(main.GetState()[0], original)
        preview.last_submit = -float('inf')
        preview.submit(mean, q, v, u, u, time.time(), 0, 3)
        preview.clear()
        time.sleep(.1)
        assert not path.exists()
    finally:
        preview.close()
        main.Close()
    assert not path.exists()  # a completed old job cannot resurrect a cleared phase


def test_bad_mean_is_reported_as_preview_error_without_controller_work(tmp_path):
    args, task, q, v, u = setup(tmp_path)
    path = tmp_path/'paths.json'
    preview = FingertipPrediction(args, task, *runner.model_limits(task.mjm), path)
    preview.start(np.zeros((3,16)), q, v, u, u)
    try:
        preview.submit(np.full((3,16), np.nan), q, v, u, u, time.time(), 0, 0)
        deadline = time.monotonic()+5
        while not path.exists() and time.monotonic()<deadline:
            time.sleep(.005)
        assert json.loads(path.read_text())['status']=='error'
    finally:
        preview.close()


@pytest.mark.parametrize('warmup_failure', [False, True])
def test_preview_cleanup_before_start_or_after_failed_warmup(tmp_path, warmup_failure):
    args, task, q, v, u = setup(tmp_path)
    path = tmp_path/'paths.json'
    preview = FingertipPrediction(args, task, *runner.model_limits(task.mjm), path)
    if warmup_failure:
        with pytest.raises(RuntimeError, match='Weighted mean'):
            preview.start(np.zeros((1,16)), q, v, u, u)
    preview.close()
    preview.close()
    assert not path.exists() and preview.closed
