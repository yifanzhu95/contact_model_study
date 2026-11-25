"""Append/overwrite only matching configurations, retaining raw run provenance."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ContactModelStudy.Hardware.ExperimentLogs import ConfigurationLogs


def batch(base, name, model='M1', temperature=1.0, episodes=10):
    directory = base/name
    directory.mkdir(parents=True)
    report = dict(mode='lcm', status='complete', started_at_utc=f'2026-11-10T00:00:{int(name[-1]):02d}+00:00',
        args=dict(task='cube_reorient', rollout_model=model, hand_acc='high', obj_acc='high',
            n_samples=256, temperature=temperature, results=str(directory/'controller.json'),
            goal_status_file=str(directory/'goal.json'), episode_status_file=str(directory/'episode_status.json'),
            n_episodes=episodes, logging=True, cost_weight=None, contact_param=[], cost_weights=None, contact_params={}),
        summary=dict(requested_episodes=episodes, started_episodes=episodes, completed_episodes=episodes,
            successful_episodes=episodes//2, unsuccessful_episodes=episodes-episodes//2, aborted_episodes=0,
            wall_time_s=5, mppi_time=dict(count=episodes, total_s=episodes*.08, min_s=.08, max_s=.08)),
        weights=dict(w_quat=26.342), planner_config=dict(temperature=temperature),
        rollout_config=dict(control_timestep=.064), hardware_config=dict(ctrl_range=dict(lower=.1,upper=.1)),
        pose_transform=dict(camera_intrinsics=[[600,0,320],[0,600,240],[0,0,1]]),
        mujoco_version='3.14.0', warp_version='1.17.0', python_version='3.12')
    path=directory/'controller.json'
    path.write_text(json.dumps(report))
    for kind, fields, rows in [('episodes', ['episode','success'],
                                [dict(episode=i+1, success=i<episodes//2) for i in range(episodes)]),
                               ('steps', ['episode','step','planning_time_s'],
                                [dict(episode=i+1, step=0, planning_time_s=.08) for i in range(episodes)])]:
        with (directory/(kind+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return path


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_new_configuration_is_created_and_repeat_batches_append_by_default(tmp_path):
    logger=ConfigurationLogs(tmp_path)
    first=logger.record(batch(tmp_path,'run1'), {})
    second=logger.record(batch(tmp_path,'run2'), {})
    assert first['configuration_id']==second['configuration_id']
    directory=Path(second['directory'])
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['runs']==2 and summary['requested_episodes']==20
    assert summary['successful_episodes']==10 and summary['success_rate_completed']==.5
    assert summary['mppi_time']['total_s']==pytest.approx(1.6)
    assert len(rows(directory/'episodes.csv'))==20
    assert len(rows(directory/'steps.csv'))==20
    assert len({row['run_id'] for row in rows(directory/'episodes.csv')})==2
    assert len((directory/'runs.jsonl').read_text().splitlines())==2


def test_overwrite_resets_only_the_matching_configuration(tmp_path):
    logger=ConfigurationLogs(tmp_path)
    first=logger.record(batch(tmp_path,'run1'), {})
    other=logger.record(batch(tmp_path,'run2',model='M2'), {})
    overwritten=logger.record(batch(tmp_path,'run3'), {}, overwrite=True)
    assert first['configuration_id']==overwritten['configuration_id']!=other['configuration_id']
    directory=Path(overwritten['directory'])
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['runs']==1 and summary['requested_episodes']==10
    assert len(rows(directory/'episodes.csv'))==10
    assert len((directory/'runs.jsonl').read_text().splitlines())==1
    other_summary=json.loads((Path(other['directory'])/'summary.json').read_text())
    assert other_summary['runs']==1 and other_summary['requested_episodes']==10
    # Raw records, including any tracker workspace, are never cleared.
    assert (tmp_path/'run1'/'controller.json').exists()


@pytest.mark.parametrize('argument,value', [('rollout_model','M3'), ('hand_acc','low'),
    ('obj_acc','foam16a'), ('temperature',2.0), ('n_samples',512), ('cube_friction',[.8,.005,.0001]),
    ('fingertip_friction',[.4,.005,.0001]), ('palm_friction',[.4,.005,.0001]), ('finger_friction',[.4,.005,.0001]),
    ('plan_blend_alpha', .6)])
def test_different_experimental_parameters_have_separate_logs(tmp_path, argument, value):
    logger=ConfigurationLogs(tmp_path)
    first=logger.record(batch(tmp_path,'run1'), {})
    changed=batch(tmp_path,'run2')
    data=json.loads(changed.read_text())
    data['args'][argument]=value
    changed.write_text(json.dumps(data))
    second=logger.record(changed, {})
    assert first['configuration_id']!=second['configuration_id']
    assert json.loads((Path(first['directory'])/'summary.json').read_text())['runs']==1


def test_omitted_friction_keeps_existing_configuration_log_identity(tmp_path):
    logger = ConfigurationLogs(tmp_path)
    first = logger.record(batch(tmp_path, 'run1'), {})
    path = batch(tmp_path, 'run2')
    data = json.loads(path.read_text())
    data['args']['cube_friction'] = None
    for group in ('fingertip', 'palm', 'finger'):
        data['args'][f'{group}_friction'] = None
    path.write_text(json.dumps(data))
    second = logger.record(path, {})
    assert first['configuration_id'] == second['configuration_id']


def test_default_plan_blending_and_metrics_preserve_existing_configuration_identity(tmp_path):
    logger = ConfigurationLogs(tmp_path)
    first = logger.record(batch(tmp_path, 'run1'), {})
    path = batch(tmp_path, 'run2')
    data = json.loads(path.read_text())
    data['args'].update(plan_blend_alpha=1., plan_blend_metrics=True)
    path.write_text(json.dumps(data))
    second = logger.record(path, {})
    assert first['configuration_id'] == second['configuration_id']


def test_batch_size_does_not_split_configuration_and_reexport_is_idempotent(tmp_path):
    logger=ConfigurationLogs(tmp_path)
    first=logger.record(batch(tmp_path,'run1'), {})
    path=batch(tmp_path,'run2',episodes=20)
    second=logger.record(path, {})
    again=logger.record(path, {})
    assert first['configuration_id']==second['configuration_id']==again['configuration_id']
    assert again['runs']==2
    summary=json.loads((Path(again['directory'])/'summary.json').read_text())
    assert summary['requested_episodes']==30
    assert len(rows(Path(again['directory'])/'episodes.csv'))==30


def test_tracking_session_identity_does_not_split_configuration(tmp_path):
    logger = ConfigurationLogs(tmp_path)
    spec = dict(mesh='cube.obj', mesh_sha256='abc', camera_sha256='def')
    def launch(pid, directory):
        return dict(tracking=dict(reused=pid != 1, session=dict(
            spec=spec, directory=directory, processes=dict(tracker=dict(pid=pid)))))
    first = logger.record(batch(tmp_path, 'run1'), launch(1, '/original'))
    second = logger.record(batch(tmp_path, 'run2'), launch(2, '/recaptured'))
    assert first['configuration_id'] == second['configuration_id']
    assert second['runs'] == 2


@pytest.mark.parametrize('setting', ['camera', 'hardware', 'tracking', 'bridge'])
def test_changed_input_settings_have_separate_logs(tmp_path, setting):
    logger = ConfigurationLogs(tmp_path)
    first = logger.record(batch(tmp_path, 'run1'), {})
    path = batch(tmp_path, 'run2')
    report = json.loads(path.read_text())
    launch = {}
    if setting == 'camera':
        report['pose_transform']['translation_adjust_m'] = [.01, 0, 0]
    elif setting == 'hardware':
        report['hardware_config']['ctrl_range']['upper'] = .2
    elif setting == 'tracking':
        launch = dict(tracking=dict(session=dict(spec=dict(mesh_sha256='different'))))
    else:
        launch = dict(config=dict(bridge_args=['--watchdog', '1.0']))
    path.write_text(json.dumps(report))
    changed = logger.record(path, launch)
    assert first['configuration_id'] != changed['configuration_id']


def test_incomplete_replacement_does_not_erase_existing_configuration(tmp_path):
    logger=ConfigurationLogs(tmp_path)
    first=logger.record(batch(tmp_path,'run1'), {})
    replacement=batch(tmp_path,'run2')
    (replacement.parent/'steps.csv').unlink()
    with pytest.raises(FileNotFoundError):
        logger.record(replacement, {}, overwrite=True)
    assert json.loads((Path(first['directory'])/'summary.json').read_text())['runs']==1
    assert len(rows(Path(first['directory'])/'episodes.csv'))==10


def test_concurrent_appends_preserve_both_batches(tmp_path):
    logger=ConfigurationLogs(tmp_path)
    paths=[batch(tmp_path,'run1'), batch(tmp_path,'run2')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda path: logger.record(path, {}), paths))
    assert results[0]['configuration_id']==results[1]['configuration_id']
    summary=json.loads((Path(results[0]['directory'])/'summary.json').read_text())
    assert summary['runs']==2 and summary['completed_episodes']==20


def test_fixed_goal_configuration_is_separate_from_previous_sampled_goals(tmp_path):
    logger = ConfigurationLogs(tmp_path)
    previous = logger.record(batch(tmp_path, 'run1'), {})
    path = batch(tmp_path, 'run2')
    report = json.loads(path.read_text())
    report.update(goal_policy='fixed', fixed_goal_quat=[1, 0, 0, 0])
    path.write_text(json.dumps(report))
    fixed = logger.record(path, {})
    assert previous['configuration_id'] != fixed['configuration_id']


def test_final_target_clipping_policy_is_separate_from_previous_planner_clipping(tmp_path):
    logger = ConfigurationLogs(tmp_path)
    previous = logger.record(batch(tmp_path, 'run1'), {})
    path = batch(tmp_path, 'run2')
    report = json.loads(path.read_text())
    report['control_policy'] = 'raw_mppi_delta_final_hardware_target_clip'
    path.write_text(json.dumps(report))
    current = logger.record(path, {})
    assert previous['configuration_id'] != current['configuration_id']


def test_per_run_fingertip_ipc_path_does_not_split_configuration_logs(tmp_path):
    paths = [batch(tmp_path, 'run1'), batch(tmp_path, 'run2')]
    logger = ConfigurationLogs(tmp_path)
    results = []
    for path in paths:
        record = json.loads(path.read_text())
        record['args']['fingertip_path_file'] = str(path.parent/'fingertip_paths.json')
        path.write_text(json.dumps(record))
        results.append(logger.record(path, {}))
    assert results[0]['configuration_id'] == results[1]['configuration_id']
    assert results[-1]['runs'] == 2
