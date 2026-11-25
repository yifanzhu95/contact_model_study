"""Compare the failed action-rate pilots with the earlier matched baseline."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0,str(OUT.parent/'hardware_pilot_temperature_analysis'))
from analyze import angle_degrees, stats


def main():
    base = ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2'
    names = ['leap_experiment_20261112_234408_731938',
             'leap_experiment_20261113_004857_186406',
             'leap_experiment_20261113_005144_729212']
    output = {'comparison':'old no-rate baseline versus new rate weights 1 and 5',
              'notes':['Commands are controller estimates, not motor target-register readback.',
                       'Only one manipulation episode was started at weight 5.',
                       'The new feature also initialized activation/first-command reference from robot_qref; the earlier baseline initialized from measured joints.',
                       'Small trial counts and independent placements do not isolate causality.'],
              'runs':[]}
    fig, axes = plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    baseline = json.loads((base/names[0]/'controller.json').read_text())
    for col,name in enumerate(names):
        path = base/name/'controller.json'
        log = json.loads(path.read_text())
        weight = log['planner_config'].get('action_rate_weight',0)
        for key in ['weights','hardware_config','rollout_config','pose_transform']:
            assert log[key] == baseline[key], key
        assert {k:v for k,v in log['planner_config'].items() if k!='action_rate_weight'} == {
            k:v for k,v in baseline['planner_config'].items() if k!='action_rate_weight'}
        run = {'source':str(path.relative_to(ROOT)),'action_rate_weight':weight,
               'controller_status':log['status'],'episodes':[]}
        axes[0,col].set_title(f'Action-rate weight {weight:g}'+(' (one episode)' if weight==5 else ''))
        for e,color in zip(log['episodes'],['#2171b5','#d95f02']):
            ss = e['steps']
            t = np.array([s['elapsed_s'] for s in ss])
            q = np.array([s['q'][:16] for s in ss])
            command_q = np.array([s['feedback']['command_qpos'] for s in ss])
            planned = np.array([s['planned_u'] for s in ss])
            u = np.array([s['u'] for s in ss])
            jump = np.max(np.abs(np.diff(u,axis=0)),axis=1)
            delta = np.array([s['action_delta'] for s in ss])
            drift = command_q-q
            # This identity follows from the existing measured-position bridge policy.
            np.testing.assert_allclose(u-planned,drift,atol=1e-7)
            best = min(ss,key=lambda s:s['goal_errors']['quat'])
            result = {'episode':e['episode']+1,'finish':e['finish_reason'],
                'n_steps':e['n_steps'],'manipulation_seconds':e['manipulation_wall_time_s'],
                'final_object_z_m':e['q_end'][18],'final_errors':e['goal_errors'],
                'best_orientation':{'angle_deg':angle_degrees(best['goal_errors']['quat']),
                                    'position_error_m':best['goal_errors']['pos'],'elapsed_s':best['elapsed_s']},
                'target_jump_rad':stats(jump),
                'first_1s_target_jump_rad':stats(jump[t[1:]<1]),
                'equivalent_target_change_rate_rad_s':stats(jump/np.diff(t)),
                'raw_action_delta_max_joint_rad':stats(np.max(np.abs(delta),axis=1)),
                'rebase_shift_max_joint_rad':stats(np.max(np.abs(drift),axis=1)),
                'planning_time_ms':stats([1000*s['planning_time'] for s in ss]),
                'measured_joint_velocity_l2_rad_s':stats(np.linalg.norm([s['q_dot'][:16] for s in ss],axis=1))}
            if weight:
                reference = np.array([s['action_rate_reference'] for s in ss])
                np.testing.assert_allclose(reference[0],log['hardware_config']['robot_qref'])
                np.testing.assert_allclose(reference[1:],u[:-1])
                i,j = np.unravel_index(np.argmax(np.abs(u-reference)),u.shape)
                result['largest_jump_decomposition'] = {
                    'step':int(i),'joint':log['joint_order'][j],'elapsed_s':float(t[i]),
                    'predicted_target_change_rad':float(planned[i,j]-reference[i,j]),
                    'rebase_shift_rad':float(drift[i,j]),
                    'estimated_applied_target_change_rad':float(u[i,j]-reference[i,j])}
                result['planned_first_rate_cost'] = stats([s['planned_action_rate_cost'] for s in ss])
                result['estimated_first_rate_cost'] = stats([s['estimated_action_rate_cost'] for s in ss])
            run['episodes'].append(result)
            times = list(t)+[e['manipulation_wall_time_s']]
            angles = [angle_degrees(s['goal_errors']['quat']) for s in ss]+[angle_degrees(e['goal_errors']['quat'])]
            positions = [1000*s['goal_errors']['pos'] for s in ss]+[1000*e['goal_errors']['pos']]
            axes[0,col].plot(times,angles,color=color,linewidth=1,label=f"Episode {e['episode']+1}: {e['finish_reason']}")
            axes[1,col].plot(times,positions,color=color,linewidth=1)
            axes[2,col].plot(t[1:],jump,color=color,linewidth=1)
        for row in range(3):
            axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('Manipulation time (s)'+('; first second' if row==2 else ''))
        axes[0,col].axhline(angle_degrees(.04),color='black',linestyle='--',linewidth=.8)
        axes[0,col].set_ylim(0,180)
        axes[0,col].legend(fontsize=8)
        axes[1,col].axhline(20,color='black',linestyle='--',linewidth=.8)
        axes[1,col].set_ylim(0,210)
        axes[2,col].set_xlim(0,1)
        axes[2,col].set_ylim(0,.75)
        output['runs'].append(run)
    axes[0,0].set_ylabel('Object angular error (degrees)')
    axes[1,0].set_ylabel('Object position error (mm)')
    axes[2,0].set_ylabel('Largest estimated joint target change\n(rad per controller update)')
    fig.suptitle('M2/high hardware: action-rate pilots failed without reducing target jumps\nTemperature=0.2, N=512, noise=0.05, w_joint_velo=0.05; independent placements, small trial counts',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=170)
    plt.close(fig)
    support = json.loads((OUT/'support.json').read_text())
    qualified = []
    for run in support['runs']:
        if not run['scene_matches_snapshot']:
            continue
        for episode in run['episodes']:
            # Use only the contiguous prefix that reconstructs to <.001 rad.
            for s in episode['steps']:
                if s['reconstruction_max_error_rad']>=.001:
                    break
                qualified.append({'weight':run['weight'],'episode':episode['episode'],**s})
    output['matched_reconstruction_prefixes'] = qualified
    (OUT/'metrics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Saved hardware comparison and',len(qualified),'closely reconstructed prefix steps.')


if __name__=='__main__':main()
