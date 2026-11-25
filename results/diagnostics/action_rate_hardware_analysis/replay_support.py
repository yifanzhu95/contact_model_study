"""Reconstruct short recorded plans and inspect candidate support; offline only."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import warp as wp

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0,str(ROOT))
import run_leap_hardware as runner
from ContactModelStudy.Hardware.Workflow import build_tasks, build_rollout
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config


def main():
    names = ['leap_experiment_20261113_004857_186406','leap_experiment_20261113_005144_729212']
    output = {'type':'offline reconstruction; filtered activation timing approximated from step logs','runs':[]}
    with wp.ScopedDevice('cuda:0'), tempfile.TemporaryDirectory(prefix='leap_support_') as directory:
        for name in names:
            path = ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2'/name/'controller.json'
            log = json.loads(path.read_text())
            args = runner.parseArgs(log['argv'])
            args.device = 'cuda:0'
            cell = Path(directory)/name
            cell.mkdir()
            task,evaluation,scenes = build_tasks(args,cell,hardware_config=log['hardware_config'])
            task.params['init_ctrl'] = np.array(log['hardware_config']['robot_qref'])
            task.setCostWeights(log['weights'])
            sim = build_rollout(args,task,*runner.model_limits(evaluation.mjm))
            result = {'source':str(path.relative_to(ROOT)),
                'scene_matches_snapshot':scenes['rollout']['scene_xml']==log['scenes']['rollout']['scene_xml'],
                'weight':log['planner_config']['action_rate_weight'],'episodes':[]}
            try:
                tau = evaluation.mjm.actuator_dynprm[:,0]
                # A planner is shared between episodes; warmup advanced noise once.
                p = MPPI(sim,task,MPPI_Config(**log['planner_config']))
                p._resample_count = 1
                for episode in log['episodes']:
                    task.params['init_ctrl'] = np.array(episode['home'])
                    task.setTargetPosition(episode['target_pos'])
                    task.setGoal(episode['goal'])
                    p.Reset()
                    activation = np.array(log['hardware_config']['robot_qref'])
                    previous_step = None
                    rows = []
                    for s in episode['steps']:
                        q,v,reference = np.array(s['q']),np.array(s['q_dot']),np.array(s['action_rate_reference'])
                        if previous_step is not None:
                            old_reference = np.array(previous_step['action_rate_reference'])
                            plan_interval = previous_step['wall_time_s']
                            activation = old_reference+(activation-old_reference)*np.exp(-plan_interval/tau)
                            gap = max(0,s['elapsed_s']-previous_step['elapsed_s']-plan_interval)
                            activation = reference+(activation-reference)*np.exp(-gap/tau)
                        sim.activation.assign(activation.astype(np.float32))
                        action = p.Plan(q,v,u=reference)
                        assert p.last_plan_ok
                        candidates = q[:16]+p.V_wp.numpy()[:,0,:]
                        low,high = candidates.min(axis=0),candidates.max(axis=0)
                        unavoidable = np.maximum(np.maximum(low-reference,reference-high),0)
                        gap = np.abs(action-np.array(s['planned_u']))
                        rows.append({'step':s['step'],'elapsed_s':s['elapsed_s'],
                            'reconstruction_max_error_rad':float(gap.max()),
                            'previous_target_outside_sample_range_joints':[log['joint_order'][j] for j in np.flatnonzero(unavoidable>1e-6)],
                            'minimum_jump_from_componentwise_support_rad':float(unavoidable.max()),
                            'mean_first_target_change_max_joint_rad':float(np.max(np.abs(candidates.mean(axis=0)-reference))),
                            'planned_change_max_joint_rad':float(np.max(np.abs(action-reference))),
                            'first_action_weight_ess':float(1/np.sum(p.w_wp.numpy().astype(float)**2))})
                        previous_step = s
                    result['episodes'].append({'episode':episode['episode']+1,'steps':rows,
                        'median_reconstruction_error_rad':float(np.median([r['reconstruction_max_error_rad'] for r in rows])),
                        'max_reconstruction_error_rad':float(max(r['reconstruction_max_error_rad'] for r in rows)),
                        'max_unavoidable_jump_rad':float(max(r['minimum_jump_from_componentwise_support_rad'] for r in rows)),
                        'steps_with_unavoidable_jump_over_0p1':sum(r['minimum_jump_from_componentwise_support_rad']>.1 for r in rows)})
                    print(json.dumps({'weight':result['weight'],**{k:v for k,v in result['episodes'][-1].items() if k!='steps'}}),flush=True)
            finally:
                sim.Close()
            output['runs'].append(result)
    (OUT/'support.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
