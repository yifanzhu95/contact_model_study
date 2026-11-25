"""Compare saved joint-velocity-weight pilots without starting hardware services."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT.parent / 'hardware_pilot_temperature_analysis'))
from analyze import angle_degrees, stats, window_metrics


def main():
    groups = {0.05: [], 0.2: [], 1.0: []}
    base = ROOT / 'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2'
    for path in base.glob('*/controller.json'):
        log = json.loads(path.read_text())
        a = log['args']
        weight = log['weights']['w_joint_velo']
        if (a['temperature'] == .2 and a['n_samples'] == 512
                and a['noise_sigma'] == .05 and a['time_horizon'] == .32
                and weight in groups and len(log['episodes']) == 2):
            groups[weight].append((path, log))
    selected = []
    for weight, limit in [(0.05, 1), (0.2, 2), (1.0, 1)]:
        matches = sorted(groups[weight], key=lambda pair: pair[0].stat().st_mtime)
        assert len(matches) >= limit, (weight, len(matches))
        selected.extend(matches[-limit:])
    reference = selected[0][1]
    result = {
        'settings': {'temperature': .2, 'n_samples': 512, 'noise_sigma': .05,
                     'horizon_s': .32, 'control_timestep_s': .064},
        'only_joint_velocity_weight_changed_verified': True,
        'notes': [
            'Target changes use the controller estimate of the bridge target, not motor target-register readback.',
            'First-2s statistics use the available prefix for episodes shorter than 2s.',
            'Instantaneous weighted terms below are reconstructed from measured states, not logged predicted trajectory costs or their gradients.',
            'Independent placements and small trial counts prevent causal claims or reliable success-rate estimation.',
            'Controller logging stops at the success gate; these logs do not independently establish post-success holding stability.',
        ],
        'runs': [],
    }
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    for column, (path, log) in enumerate(selected):
        weight = log['weights']['w_joint_velo']
        for key in ['hardware_config', 'rollout_config', 'planner_config', 'pose_transform']:
            assert log[key] == reference[key], key
        assert {k:v for k,v in log['weights'].items() if k != 'w_joint_velo'} == {
            k:v for k,v in reference['weights'].items() if k != 'w_joint_velo'}
        differing_args = [k for k in set(log['args']) | set(reference['args'])
                          if log['args'].get(k) != reference['args'].get(k)]
        assert set(differing_args) <= {'cost_weights','cost_weight','results','episode_status_file','goal_status_file'}
        launch = json.loads((path.parent/'launch.json').read_text())
        run = {'source': str(path.relative_to(ROOT)), 'w_joint_velo': weight,
               'controller_status': log['status'], 'launcher_status': launch['status'],
               'launcher_reason': launch['reason'], 'changed_arguments': differing_args,
               'torque_disabled_verified': 'Torque disabled and verified on all 16 motors.' in (path.parent/'bridge.log').read_text(),
               'episodes': []}
        title = f'w_joint_velo={weight}'
        if weight == .2:
            title += ' (run A)' if column == 1 else ' (run B)'
        axes[0, column].set_title(title)
        for e, color in zip(log['episodes'], ['#2171b5','#d95f02']):
            ss = e['steps']
            best = min(ss, key=lambda s:s['goal_errors']['quat'])
            eligible = [s for s in ss if s['goal_errors']['quat'] < .04]
            position_eligible = [s for s in ss if s['goal_errors']['pos'] < .02]
            vel = np.array([s['q_dot'][:16] for s in ss])
            episode = {
                'episode': e['episode']+1, 'finish': e['finish_reason'], 'n_steps': e['n_steps'],
                'manipulation_seconds': e['manipulation_wall_time_s'],
                'actual_control_frequency_hz': e['actual_control_frequency_hz'],
                'initial_errors': ss[0]['goal_errors'],
                'final_errors': e['goal_errors'], 'final_object_z_m': e['q_end'][18],
                'final_angle_deg': angle_degrees(e['goal_errors']['quat']),
                'best_logged_orientation': {'elapsed_s': best['elapsed_s'],
                    'angle_deg': angle_degrees(best['goal_errors']['quat']),
                    'position_error_m': best['goal_errors']['pos'],
                    'position_offset_xyz_m': (np.array(best['q'][16:19])-e['target_pos']).tolist()},
                'quat_eligible_logged_steps': len(eligible),
                'minimum_position_error_at_eligible_quat_m': min((s['goal_errors']['pos'] for s in eligible), default=None),
                'best_angle_at_eligible_position_deg': min((angle_degrees(s['goal_errors']['quat']) for s in position_eligible), default=None),
                'motion_all': window_metrics(ss), 'motion_first_2s': window_metrics(ss, 2),
                'first_window_available_seconds': min(2, ss[-1]['elapsed_s']),
                'planning_time_ms': stats([1000*s['planning_time'] for s in ss]),
                'measured_instantaneous_joint_velocity_cost': stats(weight*np.sum(vel*vel, axis=1)),
                'measured_instantaneous_quaternion_cost': stats([log['weights']['w_quat']*s['goal_errors']['quat'] for s in ss]),
            }
            run['episodes'].append(episode)
            t = [s['elapsed_s'] for s in ss] + [e['manipulation_wall_time_s']]
            qerr = [angle_degrees(s['goal_errors']['quat']) for s in ss]+[episode['final_angle_deg']]
            perr = [1000*s['goal_errors']['pos'] for s in ss]+[1000*e['goal_errors']['pos']]
            axes[0,column].plot(t,qerr,color=color,linewidth=.9,label=f"Episode {e['episode']+1}: {e['finish_reason']}")
            axes[0,column].plot(t[-1],qerr[-1],'o',color=color,markersize=4)
            axes[1,column].plot(t,perr,color=color,linewidth=.9)
            targets = np.array([s['u'] for s in ss])
            axes[2,column].plot(t[1:-1],np.max(np.abs(np.diff(targets,axis=0)),axis=1),color=color,linewidth=1)
        axes[0,column].axhline(angle_degrees(.04),color='black',linestyle='--',linewidth=.8,label='Success threshold')
        axes[0,column].set_ylim(0,180)
        axes[0,column].legend(fontsize=8)
        axes[1,column].axhline(20,color='black',linestyle='--',linewidth=.8)
        axes[1,column].set_ylim(0,210)
        axes[2,column].set_xlim(0,2)
        axes[2,column].set_ylim(0,.65)
        for row in range(3):
            axes[row,column].grid(alpha=.2)
            axes[row,column].set_xlabel('Manipulation time (s)' if row < 2 else 'Manipulation time (s), first 2 s')
        result['runs'].append(run)
    axes[0,0].set_ylabel('Object angular error (degrees)')
    axes[1,0].set_ylabel('Object position error (mm)')
    axes[2,0].set_ylabel('Largest estimated joint target change\n(rad per controller update)')
    fig.suptitle('Hardware M2/high: joint-velocity cost did not reliably smooth target updates or retain the cube\nTemperature=0.2, N=512, noise=0.05, horizon=0.32 s; independent placements, small trial counts',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=160)
    plt.close(fig)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for run in result['runs']:
        print(run['w_joint_velo'],[(e['finish'],round(e['manipulation_seconds'],2)) for e in run['episodes']])


if __name__ == '__main__':
    main()
