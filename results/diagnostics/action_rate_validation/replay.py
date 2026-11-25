"""Counterfactual planning on recorded states; never connects to hardware."""
from dataclasses import replace, asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0,str(ROOT))
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI


def main():
    driver = BASE.parent/'hardware_weight_tuning/offline_tuning.py'
    spec = importlib.util.spec_from_file_location('offline_study',driver)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    source = ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261112_234408_731938/controller.json'
    record = json.loads(source.read_text())
    steps = record['episodes'][1]['steps'][:60]
    output = {'source':str(source.relative_to(ROOT)), 'episode':2,
              'type':'recorded-state replay, not a closed-loop hardware or simulation outcome',
              'notes':['The observed state and previous logged target are reused across candidates.',
                       'No success, retention or physical smoothness claim follows from this replay.',
                       'The estimated applied target includes the recorded planning-to-command joint displacement.'],
              'candidates':[]}
    with wp.ScopedDevice('cuda:0'):
        cell = study.Cell(study.MATRIX[1],512,.05)
        try:
            cell.prepare(record['weights'])
            tau = cell.eval_model.actuator_dynprm[:,0]
            for weight in [0.,1.,5.]:
                cfg = replace(cell.planner.config,temperature=.2,seed=0,action_rate_weight=weight)
                planner = MPPI(cell.sim,cell.task,cfg)
                # The hardware runner performed one warmup before Reset.
                planner._resample_count = 1
                activation = np.array(steps[0]['q'][:16])
                previous = activation.copy()
                previous_time = 0.
                changes, deltas, planning, baseline_error = [],[],[],[]
                for step in steps:
                    q,v = np.array(step['q']),np.array(step['q_dot'])
                    dt = step['elapsed_s']-previous_time
                    activation = previous+(activation-previous)*np.exp(-dt/tau)
                    cell.sim.activation.assign(activation.astype(np.float32))
                    started = time.perf_counter()
                    planned = planner.Plan(q,v,u=previous)
                    planning.append(time.perf_counter()-started)
                    assert planner.last_plan_ok and np.isfinite(planned).all()
                    command = np.array(step['feedback']['command_qpos']) + planned-q[:16]
                    changes.append(command-previous)
                    deltas.append(planned-q[:16])
                    baseline_error.append(np.max(np.abs(planned-np.array(step['planned_u']))))
                    previous = np.array(step['u'])
                    previous_time = step['elapsed_s']
                max_change = np.max(np.abs(changes),axis=1)
                result = {'weight':weight,'planner_config':asdict(cfg),'n_steps':len(steps),
                    'estimated_target_change_max_joint_median_rad':float(np.median(max_change)),
                    'estimated_target_change_max_joint_p95_rad':float(np.percentile(max_change,95)),
                    'estimated_target_change_l2_mean_rad':float(np.mean(np.linalg.norm(changes,axis=1))),
                    'mean_squared_target_change':float(np.mean(np.sum(np.array(changes)**2,axis=1))),
                    'max_delta_rad':float(np.max(np.abs(deltas))),
                    'median_hot_plan_ms':float(np.median(planning[3:])*1000),
                    'max_difference_from_recorded_planned_target_rad':float(max(baseline_error))}
                output['candidates'].append(result)
                print(json.dumps(result),flush=True)
        finally:
            cell.close()
    difference = output['candidates'][0]['max_difference_from_recorded_planned_target_rad']
    output['notes'].append(
        f'Reconstruction is not exact: the weight-0 target differs from the saved plan by up to '
        f'{difference:.3f} rad. Activation and complete planner state were not logged, '
        'so these comparisons cannot demonstrate a hardware improvement.')
    (BASE/'replay.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
