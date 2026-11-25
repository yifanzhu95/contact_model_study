"""Configuration-specific cumulative results; raw run folders remain independent."""
import csv
from contextlib import closing
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3


# These describe a batch/output location, not the manipulation configuration.
BATCH_ARGUMENTS = {'results', 'episode_status_file', 'goal_status_file', 'logging',
                   'fingertip_path_file', 'plan_blend_metrics',
                   'n_episodes', 'save_stride', 'hardware_config', 'pose_transform',
                   'calibrated_hand', 'model', 'cost_weight', 'contact_param', 'joint_reference_file'}
FRICTION_ARGUMENTS = {'cube_friction', 'fingertip_friction', 'palm_friction', 'finger_friction'}


def configuration_parameters(controller, launch):
    inputs = launch.get('input_snapshots', {})
    launch_config = launch.get('config', {})
    tracking = (launch.get('tracking', {}).get('session') or {}).get('spec')
    scenes = {role: dict(source_scene=scene.get('source_scene'),
                         xml_sha256=hashlib.sha256(scene['scene_xml'].encode()).hexdigest(),
                         task_config=scene.get('task_config'))
              for role, scene in controller.get('scenes', {}).items()}
    return dict(schema_version=1, mode=controller['mode'],
        arguments={key: value for key, value in controller['args'].items() if key not in BATCH_ARGUMENTS
                   and not (key in FRICTION_ARGUMENTS and value is None)
                   and not (key == 'camera_calibration' and value is None)
                   and not (key == 'plan_blend_alpha' and value == 1.0)},
        hardware_config=controller.get('hardware_config', inputs.get('hardware_config')),
        camera_config=controller.get('pose_transform', inputs.get('camera_config')),
        tracking=tracking, bridge_args=launch_config.get('bridge_args', []), scenes=scenes,
        success_thresholds=controller.get('success_thresholds'), failure_z=controller.get('failure_z'),
        goal_policy=controller.get('goal_policy'), fixed_goal_quat=controller.get('fixed_goal_quat'),
        control_policy=controller.get('control_policy'),
        weights=controller.get('weights'), rollout_config=controller.get('rollout_config'),
        joint_reference=controller.get('joint_reference'),
        planner_config=controller.get('planner_config'),
        calibrated_hand_sha256=inputs.get('calibrated_hand_sha256'),
        software={key: controller.get(key) for key in ('mujoco_version', 'warp_version', 'python_version')})


def _json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.writelines(json.JSONEncoder(indent=2, allow_nan=False, default=str).iterencode(value))
        stream.write('\n')
    temporary.replace(path)


def _summary(records, identifier):
    summaries = [entry['summary'] for entry in records]
    counts = {key: sum(row.get(key, 0) for row in summaries) for key in
              ('requested_episodes', 'started_episodes', 'completed_episodes', 'successful_episodes',
               'unsuccessful_episodes', 'aborted_episodes')}
    times = [row.get('mppi_time', {}) for row in summaries]
    calls = sum(row.get('count', 0) for row in times)
    total = sum(row.get('total_s', 0) for row in times)
    mins = [row['min_s'] for row in times if row.get('min_s') is not None]
    maxes = [row['max_s'] for row in times if row.get('max_s') is not None]
    return dict(configuration_id=identifier, runs=len(records), **counts,
        success_rate_completed=counts['successful_episodes']/counts['completed_episodes'] if counts['completed_episodes'] else None,
        success_rate_requested=counts['successful_episodes']/counts['requested_episodes'] if counts['requested_episodes'] else None,
        mppi_time=dict(count=calls, total_s=total, mean_s=total/calls if calls else None,
                       min_s=min(mins) if mins else None, max_s=max(maxes) if maxes else None),
        wall_time_s=sum(row.get('wall_time_s') or 0 for row in summaries))


class ConfigurationLogs:
    def __init__(self, base):
        self.base = Path(base)

    def record(self, results, launch, overwrite=False):
        results = Path(results).resolve()
        controller = json.loads(results.read_text())
        parameters = configuration_parameters(controller, launch)
        encoded = json.dumps(parameters, sort_keys=True, separators=(',', ':'), allow_nan=False)
        identifier = hashlib.sha256(encoded.encode()).hexdigest()
        directory = self.base/'configurations'/identifier[:16]
        directory.mkdir(parents=True, exist_ok=True)
        # Serializes update plus exports, including separate camera sessions
        # that happen to produce results for the same configuration.
        with (directory/'.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest = directory/'parameters.json'
            if manifest.exists() and json.loads(manifest.read_text())['configuration_id'] != identifier:
                raise RuntimeError('Configuration log identifier collision; existing results were not modified')
            _json(manifest, dict(configuration_id=identifier, parameters=parameters))
            run_id = hashlib.sha256((str(results)+'\0'+controller['started_at_utc']).encode()).hexdigest()[:16]
            fields, metrics = {}, []
            for kind in ('episodes', 'steps'):
                with (results.parent/(kind+'.csv')).open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields[kind] = reader.fieldnames or []
                    for index, row in enumerate(reader):
                        metrics.append((run_id, kind, index, json.dumps(row, allow_nan=False)))
            entry = dict(run_id=run_id, run_directory=str(results.parent), controller_json=str(results),
                started_at_utc=controller['started_at_utc'], finished_at_utc=controller.get('finished_at_utc'),
                status=controller['status'], args=controller['args'], argv=controller.get('argv'),
                launch_argv=launch.get('argv'), summary=controller['summary'],
                error=controller.get('error'), csv_fields=fields)
            # The transaction is the authoritative log. CSV/JSON exports can
            # always be regenerated from it without needing old raw run files.
            with closing(sqlite3.connect(directory/'results.sqlite3', timeout=30)) as database:
                database.execute('PRAGMA foreign_keys=ON')
                database.execute('CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started TEXT NOT NULL, record TEXT NOT NULL)')
                database.execute('CREATE TABLE IF NOT EXISTS metrics (run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE, kind TEXT, row_number INTEGER, record TEXT, PRIMARY KEY(run_id, kind, row_number))')
                with database:
                    if overwrite:
                        database.execute('DELETE FROM runs')
                    # Re-exporting a run is idempotent; new invocations append.
                    database.execute('DELETE FROM runs WHERE run_id=?', (run_id,))
                    database.execute('INSERT INTO runs VALUES (?, ?, ?)',
                                     (run_id, entry['started_at_utc'], json.dumps(entry, allow_nan=False)))
                    database.executemany('INSERT INTO metrics VALUES (?, ?, ?, ?)', metrics)
                records = [json.loads(row[0]) for row in database.execute('SELECT record FROM runs ORDER BY started, run_id')]
                self._export(directory, identifier, database, records)
            return dict(configuration_id=identifier, directory=str(directory),
                        mode='overwrite' if overwrite else 'append', runs=len(records))

    @staticmethod
    def _export(directory, identifier, database, records):
        paths = {entry['run_id']: entry['run_directory'] for entry in records}
        pending = directory/'runs.jsonl.tmp'
        with pending.open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        pending.replace(directory/'runs.jsonl')
        for kind in ('episodes', 'steps'):
            fields = ['run_id', 'run_directory']
            for entry in records:
                for field in entry['csv_fields'][kind]:
                    if field not in fields:
                        fields.append(field)
            pending = directory/(kind+'.csv.tmp')
            with pending.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for run_id, value in database.execute(
                    'SELECT runs.run_id, metrics.record FROM metrics JOIN runs USING(run_id) WHERE kind=? ORDER BY runs.started, runs.run_id, row_number', (kind,)):
                    row = json.loads(value)
                    row.update(run_id=run_id, run_directory=paths[run_id])
                    writer.writerow(row)
            pending.replace(directory/(kind+'.csv'))
        _json(directory/'summary.json', _summary(records, identifier))
