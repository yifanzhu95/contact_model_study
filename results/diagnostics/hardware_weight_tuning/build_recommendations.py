"""Summarize the completed offline study; never starts experiment services."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import shlex

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def records(name):
    path = BASE / (name + '.jsonl')
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def counts(record):
    episodes = record.get('episodes', [])
    return {
        'episodes': len(episodes),
        'goal_reached': sum(e['success'] for e in episodes),
        'goal_held_0p5s': sum(e['hold_success'] for e in episodes),
    }


def main():
    source = (ROOT / 'ContactModelStudy/Tasks/CubeReorient.py').read_text()
    baseline = json.loads((BASE / 'protected_baseline.json').read_text())
    assert baseline['protected_text'] in source, 'Protected Bayesian weights changed'
    weights = next(ast.literal_eval(node) for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Dict)
                   and any(isinstance(key, ast.Constant) and key.value == 'w_quat'
                           for key in node.keys))
    extra = records('xpbd_temperature_refinement')
    timing = {r['row']['index']: r for r in records('serial_timing_N512')}
    heldout = {r['row']['index']: r for r in records('heldout_nominal')}
    rows = []
    commands = [
        '# Offline candidate settings, NOT hardware-qualified experiment presets.',
        '# M4 failed every screened temperature. Inspect recommendations.json and summary.txt.',
        '# These commands start physical experiments when executed; none were executed by this study.',
        '# Activate mj_env first. Logs append for matching configurations by default.',
        '',
    ]
    for row in json.loads((BASE / 'matrix.json').read_text()):
        index = row['index']
        trials = records(f'train_row_{index:02d}') + [r for r in extra if r['row']['index'] == index]
        # Select using training data only; unused starts never affect this ranking.
        best = min(trials, key=lambda r: (-counts(r)['goal_held_0p5s'],
                                         -counts(r)['goal_reached'], r['mean_score']))
        assert best['weights'] == weights
        ts = sorted(timing[index]['steady_plan_seconds'])
        median = (ts[len(ts)//2 - 1] + ts[len(ts)//2]) / 2
        train = counts(best)
        entry = {
            **row, 'temperature': best['temperature'], 'n_samples': 512,
            'noise_sigma': 0.05, 'n_iterations': 1, 'ctrl_time_step_s': 0.064,
            'training': train, 'training_mean_score': best['mean_score'],
            'temperatures_screened': [r['temperature'] for r in trials],
            'isolated_plan_median_ms': median * 1000,
            'isolated_plan_max_ms': max(ts) * 1000,
            'median_plan_within_64ms': median <= 0.064,
            'hardware_validated': False,
            'status': ('no_nominal_success' if not train['goal_reached'] else
                       'no_nominal_hold' if not train['goal_held_0p5s'] else
                       'offline_candidate'),
            'rollout_configuration': best['rollout_config'],
        }
        if index in heldout:
            entry['unused_start_validation'] = counts(heldout[index])
            entry['unused_start_indices'] = [e['state_index'] for e in heldout[index]['episodes']]
            if not entry['unused_start_validation']['goal_reached']:
                entry['status'] = 'no_unused_start_success'
            elif not entry['unused_start_validation']['goal_held_0p5s']:
                entry['status'] = 'no_unused_start_hold'
        arguments = [
            'python', 'launch_leap_experiment.py', '--logging', '--log-dir',
            f"results/planned/hardware/cube_reorient/hand-{row['hand']}_obj-high/{row['model']}",
            '--n-episodes', '10', '--', '--task', row['task'],
            '--rollout-model', row['model'], '--hand-acc', row['hand'], '--obj-acc', 'high',
            '--n-samples', '512', '--n-iterations', '1', '--ctrl-time-step', '0.064',
            '--time-horizon', str(row['horizon']), '--noise-sigma', '0.05',
            '--temperature', str(best['temperature']),
            '--action-rate-weight', '0',  # Preserve this historical profile.
        ]
        entry['command_argv'] = arguments
        commands += [f"# Row {index}: {row['model']}, hand {row['hand']}; {entry['status']}",
                     shlex.join(arguments), '']
        rows.append(entry)
    profile = {
        'status': 'provisional_offline_candidate_not_hardware_qualified',
        'task': 'cube_reorient', 'shared_cost_weights': weights,
        'shared_n_samples': 512, 'shared_noise_sigma': 0.05,
        'shared_action_rate_weight': 0.0,
        'fallen_z_m': 0.065, 'goal_z_m': 0.070,
        'fallen_note': '5 mm below goal; both fallen weights are zero, so this threshold currently contributes no cost. Failure and success thresholds were not changed.',
        'temperature_selection': 'Most 0.5 s holds, then most goal reaches, then lowest mean study score on states 0 and 1. No held-out results used for selection.',
        'protected_weights_sha256': hashlib.sha256(baseline['protected_text'].encode()).hexdigest(),
        'physical_trials_started': False, 'rows': rows,
        'limitations': [
            'Nominal calibration uses two recorded starting states per row and one deterministic sampling seed per state. This is screening, not a statistical success-rate estimate.',
            'M4 produced no nominal successes for any tested temperature (0.1, 0.3, 1, 3, 10). Its listed temperature is only a starting candidate.',
            'M6/low reached neither goal from the two unused starting poses; successful training trajectories did not generalize to those placements.',
            'Recorded MCP feedback reaches about -0.35 rad, while the calibrated model has a zero lower joint limit. Coordinate calibration or model limits need verification before claiming transfer.',
            'The planning/evaluation model limits absolute actuator targets; the hardware manipulation bridge only enforces its configured delta bounds. Offline range/target stress reduces holding success.',
            'Increasing the posture weight to 5, 20 or 50 did not solve that transfer stress. The calibrated model and hardware command policy were not modified.',
            'Sensor/gain/friction/mass stress is synthetic. Estimated activation-state errors, true noisy success gating, occlusions, unmodeled contact and thermal/current dynamics are not fully simulated.',
            'Success-hold evaluation lasts only 0.5 s; it does not establish indefinite stable holding.',
            'Six isolated timing samples per row were measured on GPU 2, not the deployment GPU. Some exceed the 64 ms model control interval; actual closed-loop latency includes sensing, communication and compute.',
            'The surrogate advances at a prescribed period and holds the previous command for half a cycle. It is not a timing-exact replica of the asynchronous hardware system.',
            'The current hardware config uses measured_position delta_reference and +/-3.14 rad delta bounds. This study did not change these or add motion smoothing.',
            'A 0.04 quaternion error threshold means 1-dot(q,goal)^2 < 0.04, approximately 23.1 degrees; it is not a 0.04 rad angular threshold.',
            'No hardware validation was performed. Other objects, protected Bayesian weights, controller defaults and planning kernels were not changed.',
        ],
        'snapshots': {},
    }
    for name in ['configs/leap_hardware.json', 'configs/leap_experiment.json', 'configs/leap_pose_transform.json']:
        text = (ROOT / name).read_text()
        profile['snapshots'][name] = {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'value': json.loads(text)}
    (BASE / 'recommendations.json').write_text(json.dumps(profile, indent=2) + '\n')
    columns = ['index','model','hand','obj','horizon','n_samples','noise_sigma','n_iterations',
               'ctrl_time_step_s','temperature','train_goal','train_hold','train_n',
               'unused_start_goal','unused_start_hold','unused_start_n',
               'isolated_plan_median_ms','status']
    with (BASE / 'recommendations.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = {k: row[k] for k in columns if k in row}
            for prefix, key in [('train','training'), ('unused_start','unused_start_validation')]:
                count = row.get(key, {})
                output.update({prefix+'_goal':count.get('goal_reached',''),
                               prefix+'_hold':count.get('goal_held_0p5s',''),
                               prefix+'_n':count.get('episodes','')})
            writer.writerow(output)
    (BASE / 'commands.txt').write_text('\n'.join(commands))
    pilot = rows[1]['command_argv'].copy()
    pilot[pilot.index('--n-episodes')+1] = '2'
    pilot[pilot.index('--log-dir')+1] = 'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2'
    (BASE / 'pilot_command.txt').write_text('# Supervised hardware pilot; not executed by this study.\n'+shlex.join(pilot)+'\n')
    protocol = json.loads((BASE / 'protocol.json').read_text())
    protocol['selection_basis'] = profile['temperature_selection']
    protocol['limitations'] = profile['limitations']
    protocol['held_out_indices_evaluated'] = sorted({e['state_index'] for r in heldout.values() for e in r['episodes']})
    protocol['held_out_indices_unused'] = sorted(set(protocol['held_out_indices']) - set(protocol['held_out_indices_evaluated']))
    (BASE / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for row in rows:
        print(row['index'], row['temperature'], row['training'], row.get('unused_start_validation'), row['status'])


if __name__ == '__main__':
    main()
