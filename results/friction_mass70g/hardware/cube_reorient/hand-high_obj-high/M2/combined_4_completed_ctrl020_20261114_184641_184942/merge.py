"""Combine saved completed episodes only; never import hardware/simulation code."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SOURCES = [BASE / ('leap_experiment_' + name) for name in
           ('20261114_184641_023716', '20261114_184942_221848')]


def main():
    hashes, documents, launches = {}, [], []
    for directory in SOURCES:
        for name in ('controller.json', 'launch.json', 'episodes.csv', 'steps.csv', 'bridge.log', 'controller.log'):
            p = directory / name
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        documents.append(json.loads((directory / 'controller.json').read_text()))
        launches.append(json.loads((directory / 'launch.json').read_text()))
    keys = ['hardware_config', 'weights', 'joint_reference', 'planner_config', 'rollout_config',
        'pose_transform', 'success_thresholds', 'effective_ctrl_range', 'task_contact_params',
        'failure_z', 'goal_policy', 'fixed_goal_quat', 'scenes', 'execution', 'action_rate_cost']
    checks = {key: documents[0].get(key) == documents[1].get(key) for key in keys}
    assert all(checks.values()), checks
    identifiers = [d['configuration_log']['configuration_id'] for d in launches]
    assert identifiers[0] == identifiers[1]
    assert documents[0]['hardware_config']['ctrl_range'] == dict(lower=.2, upper=.2)
    assert documents[0]['args']['palm_friction'] == [.4,.005,.0001]
    assert documents[0]['args']['finger_friction'] == [.4,.005,.0001]
    for d in documents:
        for role in ('rollout', 'eval'):
            root = ET.fromstring(d['scenes'][role]['scene_xml'])
            assert float(root.find('.//body[@name="obj"]/geom[@name="obj"]').get('mass')) == .07
    selected, source_rows, excluded, episode_rows, step_rows = [], [], [], [], []
    expected_fields = {}
    for directory, d, launch in zip(SOURCES, documents, launches):
        mapping = {}
        for e in d['episodes']:
            source_episode = e['episode'] + 1
            if not e.get('completed'):
                excluded.append(dict(source_run=directory.name, source_episode=source_episode,
                    finish_reason=e.get('finish_reason'), n_steps=e.get('n_steps')))
                continue
            assert e['finish_reason'] in ('success', 'failure', 'timeout')
            assert len(e['steps']) == e['n_steps'] and not e.get('failed_step')
            new = copy.deepcopy(e)
            new.update(episode=len(selected), source_run=directory.name, source_episode=source_episode)
            selected.append(new)
            mapping[source_episode] = len(selected)
        source_rows.append(dict(run=directory.name, controller_json=str(directory/'controller.json'),
            controller_sha256=hashes[str(directory/'controller.json')], configuration_id=identifiers[0],
            source_status=d['status'], source_error=d.get('error'),
            selected_source_episode_numbers=list(mapping), source_wall_time_s=d.get('wall_time_s'),
            stopped_during='reference hold before next episode; MPPI not started',
            bridge_error=(directory/'bridge.log').read_text().strip().splitlines()[-1]))
        for filename, destination in [('episodes.csv',episode_rows), ('steps.csv',step_rows)]:
            with (directory/filename).open(newline='') as f:
                reader = csv.DictReader(f)
                if filename in expected_fields:
                    assert reader.fieldnames == expected_fields[filename]
                else:
                    expected_fields[filename] = reader.fieldnames
                for row in reader:
                    old = int(row['episode'])
                    if old in mapping:
                        destination.append(dict(source_run=directory.name, source_episode=old,
                            **dict(row, episode=mapping[old])))
    assert [len(r['selected_source_episode_numbers']) for r in source_rows] == [3,1]
    assert len(selected) == len(episode_rows) == 4
    assert len(step_rows) == sum(e['n_steps'] for e in selected) == 667
    assert len({(s['source_run'],s['source_episode'],s['step']) for s in step_rows}) == 667
    times = [s['planning_time'] for e in selected for s in e['steps']]
    assert math.isclose(sum(times),sum(float(s['planning_time_s']) for s in step_rows),abs_tol=1e-9)
    outcomes = [e['finish_reason'] for e in selected]
    summary = dict(record_type='completed_episode_aggregate', source_runs=2, completed_episodes=4,
        successful_episodes=outcomes.count('success'), failure_episodes=outcomes.count('failure'),
        timeout_episodes=outcomes.count('timeout'), success_rate_completed=outcomes.count('success')/4,
        excluded_aborted_episodes=len(excluded), completed_steps=len(step_rows),
        manipulation_wall_time_s=sum(e['manipulation_wall_time_s'] for e in selected),
        source_wall_time_s=sum(d['wall_time_s'] for d in documents),
        source_requested_episodes=sum(d['args']['n_episodes'] for d in documents),
        mppi_time=dict(count=len(times), total_s=sum(times), mean_s=sum(times)/len(times),
                       min_s=min(times), max_s=max(times)),
        configuration_id=identifiers[0], model_mass_kg=.07,
        ctrl_range=dict(lower=.2,upper=.2),
        scope='Completed successes, failures and timeouts retained. Source crashes after completed episodes remain run-level diagnostics, not extra episode outcomes.',
        episodes=[dict(episode=e['episode']+1,source_run=e['source_run'],source_episode=e['source_episode'],
            outcome=e['finish_reason'],steps=e['n_steps'],duration_s=e['manipulation_wall_time_s'],
            initial_position_m=e['steps'][0]['q'][16:19],initial_quaternion_wxyz=e['steps'][0]['q'][19:23],
            final_position_error_m=e['goal_errors']['pos'],final_quaternion_error=e['goal_errors']['quat'],
            final_orientation_error_deg=math.degrees(e['quaternion_error_rad'])) for e in selected])
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), sources=source_rows,
        excluded_episodes=excluded, configuration_equivalence_checks=checks, source_file_sha256=hashes,
        note='Derived view of two raw runs. Do not count this aggregate in addition to its source episodes. Raw logs and cumulative configuration logs are unchanged.')
    def write(name, data):
        (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    for name,rows in [('episodes.csv',episode_rows),('steps.csv',step_rows)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['source_run','source_episode']+expected_fields[name])
            writer.writeheader();writer.writerows(rows)
    parameters=Path(launches[0]['configuration_log']['directory'])/'parameters.json'
    write('parameters.json',json.loads(parameters.read_text()))
    write('summary.json',summary)
    write('sources.json',provenance)
    write('completed_episodes.json',dict(record_type='completed_episode_aggregate',
        configuration_id=identifiers[0],episodes=selected,summary=summary))
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    assert [r['finish_reason'] for r in episode_rows] == outcomes == ['failure','success','failure','failure']
    write('validation.json',dict(passed=True,raw_files_unchanged=True,duplicate_episodes=False,
        configuration_match=True,selected_episodes=4,step_rows=667,excluded_aborted_episodes=len(excluded),
        simulations_run=False,hardware_run=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
