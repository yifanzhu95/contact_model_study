"""Retain the requested M6 success, excluding two watchdog-aborted manipulations."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ContactModelStudy.Hardware.ExperimentLogs import configuration_parameters

BASE = ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high'
NAMES = ['leap_experiment_20261114_000214_299972', 'leap_experiment_20261114_000548_284896']
OUT = BASE/'M6/combined_1_completed_20261114_000214_299972'
DIAG = Path(__file__).parent


def write_json(path, data):
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def main():
    assert not OUT.exists(), 'Selection already exists; do not overwrite it automatically'
    now = datetime.now(timezone.utc).isoformat()
    loaded, hashes = [], {}
    for name in NAMES:
        path = BASE/'M6'/name
        controller = path/'controller.json'
        hashes[str(controller)] = hashlib.sha256(controller.read_bytes()).hexdigest()
        loaded.append((path, json.loads(controller.read_text()), json.loads((path/'launch.json').read_text())))
    source_path, source, launch = loaded[0]
    episode = source['episodes'][0]
    assert episode['completed'] and episode['finish_reason'] == 'success'
    assert [e.get('completed', False) for _, d, _ in loaded for e in d['episodes']] == [True, False, False]
    assert all(d['args']['rollout_model'] == 'M6' for _, d, _ in loaded)
    assert all(l['reason'] == 'Supervision display stopped receiving fresh frames' for _, _, l in loaded)
    assert all('Torque disabled and verified on all 16 motors.' in (p/'bridge.log').read_text() for p, _, _ in loaded)
    parameters = configuration_parameters(source, launch)
    assert parameters == configuration_parameters(loaded[1][1], loaded[1][2])
    identifier = launch['configuration_log']['configuration_id']
    assert loaded[1][2]['configuration_log']['configuration_id'] == identifier
    sources, excluded, diagnostics = [], [], []
    for path, controller, launcher in loaded:
        selected = [1] if path == source_path else []
        sources.append(dict(run=path.name, controller_json=str(path/'controller.json'),
            controller_sha256=hashes[str(path/'controller.json')], configuration_id=identifier,
            source_status=controller['status'], source_error=controller.get('error'),
            launcher_reason=launcher['reason'], selected_source_episode_numbers=selected,
            source_wall_time_s=controller['wall_time_s'],
            tracking_gpu=launcher['config']['tracking_gpu_assignment'], mppi_gpu=launcher['config']['gpu_assignment']))
        timings = []
        for ep in controller['episodes']:
            values = [step['planning_time'] for step in ep['steps']]
            timings.append(dict(source_episode=ep['episode']+1, completed=ep['completed'], outcome=ep['finish_reason'],
                published_steps=ep['n_steps'], manipulation_wall_time_s=ep['manipulation_wall_time_s'],
                planning_mean_ms=statistics.mean(values)*1000, planning_max_ms=max(values)*1000,
                failed_step=ep.get('failed_step')))
            if not ep['completed']:
                excluded.append(dict(source_run=path.name, source_episode=ep['episode']+1,
                    outcome=ep['finish_reason'], n_steps=ep['n_steps'], duration_s=ep['manipulation_wall_time_s'],
                    error=ep.get('error'), exclusion_reason='Software abort, excluded by user request'))
        diagnostics.append(dict(run=str(path), launcher_reason=launcher['reason'], controller_error=controller.get('error'),
            supervision_timeout_s=launcher['config']['supervision_timeout'],
            exit_codes=launcher['exit_codes'], torque_disabled_verified=True, episodes=timings))
    OUT.mkdir()
    for kind in ('episodes', 'steps'):
        with (source_path/(kind+'.csv')).open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row['episode']) == 1]
        assert len(rows) == (1 if kind == 'episodes' else episode['n_steps'])
        with (OUT/(kind+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source_run', 'source_episode', 'max_robot_age_s', *fields])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, source_run=source_path.name, source_episode=1,
                    max_robot_age_s=source['args']['max_robot_age']))
    q = episode['steps'][0]['q']
    errors = episode['goal_errors']
    detail = dict(episode=1, source_run=source_path.name, source_episode=1,
        max_robot_age_s=source['args']['max_robot_age'], outcome='success', steps=episode['n_steps'],
        duration_s=episode['manipulation_wall_time_s'], initial_position_m=q[16:19],
        initial_quaternion_wxyz=q[19:23], final_position_error_m=errors['pos'],
        final_quaternion_error=errors['quat'], final_orientation_error_deg=math.degrees(episode['quaternion_error_rad']),
        final_velocity_norm=errors['vel'])
    summary = dict(record_type='completed_episode_aggregate', source_runs=1, inspected_runs=2,
        completed_episodes=1, successful_episodes=1, failure_episodes=0, timeout_episodes=0,
        success_rate_completed=1.0, excluded_aborted_episodes=2, completed_steps=episode['n_steps'],
        manipulation_wall_time_s=episode['manipulation_wall_time_s'], mppi_time=episode['mppi_time'],
        source_configuration_ids=[identifier],
        scope='Only episode 1 of 000214 is selected. Episode 2 of 000214 and episode 1 of 000548 are excluded from all comparison metrics and CSVs.',
        configuration_notes=['Raw logs and cumulative configuration logs remain unchanged for diagnosis; use this selected dataset for the comparison.'],
        episodes=[detail])
    write_json(OUT/'sources.json', dict(created_at_utc=now, sources=sources, excluded_episodes=excluded,
        reference_configuration=parameters))
    write_json(OUT/'summary.json', summary)
    # Update the explicit comparison selection without altering other models.
    for name in ('comparison_index.json', 'comparison.csv'):
        shutil.copy2(BASE/name, DIAG/(name+'.before'))
    index = json.loads((BASE/'comparison_index.json').read_text())
    assert not any(d['model'] == 'M6' for d in index['datasets']), 'Existing M6 selection needs an explicit merge'
    relative = str(OUT.relative_to(BASE))
    index['datasets'].append(dict(model='M6', completed_episodes=1, successful_episodes=1, drops=0, timeouts=0,
        success_rate=1.0, dataset=relative, summary=relative+'/summary.json', evaluated_episodes=1,
        planner_error_episodes=0, inclusion_policy='completed_episodes_only'))
    index.update(completed_episodes=sum(d['completed_episodes'] for d in index['datasets']),
        evaluated_episodes=sum(d['evaluated_episodes'] for d in index['datasets']), updated_at_utc=now,
        dataset_selection='Use only the listed datasets for this comparison. Software-aborted episodes are excluded except for the explicitly requested M4 failed-attempt dataset.')
    index['configuration_notes'].append('M6 currently contains one completed success from 000214; two supervision-display aborts from the last two runs are excluded. This is a partial dataset, not a ten-episode result.')
    write_json(BASE/'comparison_index.json', index)
    with (DIAG/'comparison.csv.before').open(newline='') as stream:
        fields = csv.DictReader(stream).fieldnames
    with (BASE/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index['datasets'])
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'Raw source changed'
    with (OUT/'episodes.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    with (OUT/'steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 844 and all(row['episode'] == '1' and not row['error'] for row in rows)
    write_json(DIAG/'analysis.json', dict(created_at_utc=now, runs=diagnostics, selected_dataset=str(OUT),
        retained_completed=1, excluded_aborted=2, raw_controller_hashes=hashes, raw_sources_verified_unchanged=True,
        diagnosis='Both launchers terminated the controller on the 2-second supervision-display freshness gate. The final controller feedback was fresh and MPPI timings were normal. Receiver logs contain no timestamped stall diagnostics, so the cause of the display gap cannot be distinguished between GUI blocking and transport/frame delivery.',
        tracker_log=str(ROOT/'results/diagnostics/gpu_split_tracking_restart/vision_20261114_000001_694650/tracker.log')))
    print(json.dumps(dict(dataset=str(OUT), completed=1, successes=1, excluded_aborts=2,
        completed_steps=844, raw_sources_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
