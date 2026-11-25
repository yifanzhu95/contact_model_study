"""Read saved hardware pilot logs; produces metrics and a plot, never moves hardware."""
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PILOTS = ROOT / 'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2'


def angle_degrees(error):
    return math.degrees(2 * math.asin(math.sqrt(np.clip(error, 0, 1))))


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(['median', 'p95', 'maximum'], map(float, np.percentile(values, [50, 95, 100]))))


def window_metrics(steps, seconds=None):
    t = np.array([s['elapsed_s'] for s in steps])
    selected = np.ones(len(t), dtype=bool) if seconds is None else t < seconds
    target = np.array([s['u'] for s in steps])
    delta = np.array([s['action_delta'] for s in steps])
    joint_velocity = np.array([s['q_dot'][:16] for s in steps])
    # u is the controller's estimate of the bridge target, not a motor target readback.
    target_jump = np.max(np.abs(np.diff(target, axis=0)), axis=1)
    return {
        'n_steps': int(selected.sum()),
        'estimated_target_jump_max_joint_rad': stats(target_jump[selected[1:]]),
        'action_delta_max_joint_rad': stats(np.max(np.abs(delta[selected]), axis=1)),
        'measured_joint_velocity_l2_rad_s': stats(np.linalg.norm(joint_velocity[selected], axis=1)),
    }


def main():
    chosen = {}
    for path in PILOTS.glob('*/controller.json'):
        log = json.loads(path.read_text())
        temp = log['args']['temperature']
        if temp not in [0.1, 0.2, 0.5] or len(log.get('episodes', [])) != 2:
            continue
        if temp not in chosen or path.stat().st_mtime > chosen[temp][0].stat().st_mtime:
            chosen[temp] = (path, log)
    assert sorted(chosen) == [0.1, 0.2, 0.5]
    reference = chosen[0.1][1]
    result = {
        'comparison': 'Three completed two-episode M2/high hardware pilots',
        'temperature_only_parameter_change_verified': True,
        'metric_notes': {
            'quaternion': '1-dot(q,goal)^2; angular error = 2 asin(sqrt(error))',
            'estimated_target_jump': 'max_j |u[t,j]-u[t-1,j]|, the controller estimate of the bridge command, not verified motor target readback',
            'delta': 'MPPI requested offset from the measured planning joints; distinct from target-to-target change',
            'joint_velocity_norm': 'L2 norm across 16 joint velocities, not individual joint velocity',
            'hold': 'Controller step logging stops at success. Continued successful holding is the operator observation, not independently measured by these metrics.',
            'scope': 'Two episodes per setting, with different placements and contact states, do not establish a statistically reliable success rate or temperature causality.',
        },
        'runs': [],
    }
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    colors = ['#2171b5', '#d95f02']
    for col, temp in enumerate(sorted(chosen)):
        path, log = chosen[temp]
        for key in ['weights', 'hardware_config', 'rollout_config', 'pose_transform']:
            assert log[key] == reference[key], f'{key} changed'
        changed_args = [k for k in set(log['args']) | set(reference['args'])
                        if log['args'].get(k) != reference['args'].get(k)]
        assert set(changed_args) <= {'temperature','results','episode_status_file','goal_status_file'}
        assert {k:v for k,v in log['planner_config'].items() if k != 'temperature'} == {
            k:v for k,v in reference['planner_config'].items() if k != 'temperature'}
        launch = json.loads((path.parent / 'launch.json').read_text())
        run = {
            'source': str(path.relative_to(ROOT)), 'temperature': temp,
            'started_phoenix': datetime.fromisoformat(log['started_at_utc']).astimezone(ZoneInfo('America/Phoenix')).isoformat(),
            'controller_status': log['status'], 'launcher_status': launch['status'],
            'launcher_reason': launch['reason'], 'exit_codes': launch['exit_codes'],
            'bridge_action_timeout_logged': 'Action timeout' in (path.parent/'bridge.log').read_text(),
            'torque_disabled_verified_in_bridge_log': 'Torque disabled and verified on all 16 motors.' in (path.parent/'bridge.log').read_text(),
            'weights': log['weights'], 'changed_args_vs_temperature_0p1': changed_args,
            'episodes': [],
        }
        axes[0, col].set_title(f'Temperature {temp}')
        for e in log['episodes']:
            steps = e['steps']
            end = e['goal_errors']
            best = min(steps, key=lambda s:s['goal_errors']['quat'])
            best_error = min(end['quat'], best['goal_errors']['quat'])
            tail = [s for s in steps if s['elapsed_s'] >= steps[-1]['elapsed_s'] - 10]
            quat_eligible = [s for s in steps if s['goal_errors']['quat'] < log['success_thresholds']['quat']]
            episode = {
                'episode': e['episode'] + 1, 'finish': e['finish_reason'],
                'n_steps': e['n_steps'], 'manipulation_seconds': e['manipulation_wall_time_s'],
                'actual_control_frequency_hz': e['actual_control_frequency_hz'],
                'initial_errors': steps[0]['goal_errors'],
                'initial_angle_deg': angle_degrees(steps[0]['goal_errors']['quat']),
                'final_errors': end, 'final_angle_deg': angle_degrees(end['quat']),
                'best_orientation_error_including_endpoint': best_error,
                'best_angle_deg_including_endpoint': angle_degrees(best_error),
                'quaternion_eligible_logged_steps': len(quat_eligible),
                'tail_10s_mean_errors': {k:float(np.mean([s['goal_errors'][k] for s in tail])) for k in ['pos','quat','vel']},
                'motion_all': window_metrics(steps), 'motion_first_2s': window_metrics(steps, 2),
                'motion_first_10s': window_metrics(steps, 10),
                'planning_time_ms': stats([1000*s['planning_time'] for s in steps]),
                'inter_plan_start_ms': stats(1000*np.diff([s['elapsed_s'] for s in steps])),
                'robot_age_at_planning_ms': stats([1000*s['feedback']['robot_age'] for s in steps]),
                'object_age_at_planning_ms': stats([1000*s['feedback']['pose_age'] for s in steps]),
                'raw_delta_clipped_steps_at_3p14': sum(max(map(abs,s['action_delta'])) > 3.14 for s in steps),
            }
            run['episodes'].append(episode)
            color = colors[e['episode']]
            t = [s['elapsed_s'] for s in steps] + [e['manipulation_wall_time_s']]
            orientation = [angle_degrees(s['goal_errors']['quat']) for s in steps]+[episode['final_angle_deg']]
            position = [1000*s['goal_errors']['pos'] for s in steps]+[1000*end['pos']]
            label = f"Episode {e['episode']+1}: {e['finish_reason']}"
            axes[0,col].plot(t, orientation, color=color, linewidth=1, label=label)
            axes[0,col].plot(t[-1],orientation[-1],'o',color=color,markersize=4)
            axes[1,col].plot(t, position, color=color, linewidth=1)
            u = np.array([s['u'] for s in steps])
            axes[2,col].plot(t[1:-1],np.max(np.abs(np.diff(u,axis=0)),axis=1),color=color,linewidth=1)
        axes[0,col].axhline(angle_degrees(.04), color='black', linestyle='--', linewidth=.8, label='Success threshold')
        axes[0,col].legend(fontsize=8)
        axes[1,col].axhline(20, color='black', linestyle='--', linewidth=.8)
        axes[0,col].set_ylim(0,180)
        axes[1,col].set_ylim(0,210)
        axes[2,col].set_xlim(0,3)
        axes[2,col].set_ylim(0,.65)
        axes[2,col].set_xlabel('Time from manipulation start (s); first 3 s')
        for row in [0,1]:
            axes[row,col].set_xlabel('Time from manipulation start (s)')
        for row in range(3):
            axes[row,col].grid(alpha=.2)
        result['runs'].append(run)
    axes[0,0].set_ylabel('Object angular error (degrees)')
    axes[1,0].set_ylabel('Object position error (mm)')
    axes[2,0].set_ylabel('Largest estimated target change\n(rad per controller update)')
    fig.suptitle('Hardware M2/high pilots: completion, retention and early command changes\nN=512, noise=0.05, horizon=0.32 s; controller target estimates, no target-register readback',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=170)
    plt.close(fig)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    for run in result['runs']:
        print('temperature',run['temperature'])
        for e in run['episodes']:
            print(e['episode'],e['finish'],round(e['manipulation_seconds'],2),round(e['final_angle_deg'],1),round(e['final_errors']['pos']*1000,1),
                  'early_target_jump_p95',round(e['motion_first_2s']['estimated_target_jump_max_joint_rad']['p95'],3))


if __name__ == '__main__':
    main()
