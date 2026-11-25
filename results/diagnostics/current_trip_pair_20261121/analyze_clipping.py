"""Audit both clipping stages from frozen controller logs; no hardware access."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/baseline_trial314_shift512_h0256_paced_measured_limits'
RUNS = ['leap_experiment_20261121_012849_115802', 'leap_experiment_20261121_013714_689713']
EPS = 1e-7


def describe(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), p95=float(np.percentile(a, 95)), max=float(a.max()))


def summarize(rows, names):
    raw = np.array([r['raw'] for r in rows])
    stage1 = np.array([r['stage1'] for r in rows])
    final = np.array([r['final'] for r in rows])
    base = np.array([r['base'] for r in rows])
    q = np.array([r['q'] for r in rows])
    low = np.array([r['low'] for r in rows])
    high = np.array([r['high'] for r in rows])
    down = np.array([r['down'] for r in rows])
    up = np.array([r['up'] for r in rows])
    first, second = abs(stage1-raw)>EPS, abs(final-stage1)>EPS
    exceeds_delta = (final-base > up+EPS) | (final-base < -down-EPS)
    actual_out = (q < low-EPS) | (q > high+EPS)
    raw_plan = np.array([r['planned'] for r in rows])
    stats = dict(actions=len(rows), delta_clipped_actions=int(first.any(axis=1).sum()),
        delta_clipped_action_percent=float(100*first.any(axis=1).mean()),
        delta_clipped_joint_percent=float(100*first.mean()),
        absolute_clipped_actions=int(second.any(axis=1).sum()),
        absolute_clipped_action_percent=float(100*second.any(axis=1).mean()),
        absolute_clipped_joint_percent=float(100*second.mean()),
        either_clipped_actions=int((first | second).any(axis=1).sum()),
        either_clipped_action_percent=float(100*(first | second).any(axis=1).mean()),
        both_stages_clipped_actions=int((first.any(axis=1) & second.any(axis=1)).sum()),
        final_delta_bound_exceeded_actions=int(exceeds_delta.any(axis=1).sum()),
        final_delta_bound_exceeded_joint_samples=int(exceeds_delta.sum()),
        final_targets_outside_absolute_bounds=int(((final<low-EPS)|(final>high+EPS)).sum()),
        raw_planner_target_outside_actions=int(((raw_plan<low-EPS)|(raw_plan>high+EPS)).any(axis=1).sum()),
        measured_outside_actions=int(actual_out.any(axis=1).sum()),
        delta_clip_l2_rad=describe(np.linalg.norm(stage1-raw,axis=1)),
        absolute_clip_l2_rad=describe(np.linalg.norm(final-stage1,axis=1)),
        maximum_final_joint_step_rad=float(abs(final-base).max()),
        joints=[])
    for j, name in enumerate(names):
        stats['joints'].append(dict(joint=name,
            delta_clipped_count=int(first[:,j].sum()), delta_clipped_percent=float(first[:,j].mean()*100),
            absolute_clipped_count=int(second[:,j].sum()), absolute_clipped_percent=float(second[:,j].mean()*100),
            absolute_lower_count=int((second[:,j] & (stage1[:,j]<low[:,j])).sum()),
            absolute_upper_count=int((second[:,j] & (stage1[:,j]>high[:,j])).sum()),
            absolute_clip_max_rad=float(abs(final[:,j]-stage1[:,j]).max()),
            measured_outside_count=int(actual_out[:,j].sum()),
            measured_outside_percent=float(actual_out[:,j].mean()*100)))
    return stats


def main():
    all_rows, runs, sources = [], {}, []
    names = None
    for run in RUNS:
        path=BASE/run/'controller.json';payload=path.read_bytes();d=json.loads(payload)
        names=d['joint_order'];bounds=d['joint_limits'];margins=d['effective_ctrl_range']
        lo, hi = np.array(bounds['lower']), np.array(bounds['upper'])
        down, up = np.array(margins['lower']), np.array(margins['upper'])
        run_rows, episodes = [], []
        for e in d['episodes']:
            rows=[]
            for s in e['steps']:
                if not s.get('action_published'):continue
                raw=np.array(s['requested_target']);base=np.array(s['feedback']['command_qpos'])
                stage1=np.array(s['joint_limit_clipping']['delta_clipped_target']);final=np.array(s['u'])
                np.testing.assert_allclose(stage1,np.clip(raw,base-down,base+up),rtol=0,atol=1e-12)
                np.testing.assert_allclose(final,np.clip(stage1,lo,hi),rtol=0,atol=1e-12)
                assert np.count_nonzero(abs(final-stage1)>EPS)==s['joint_limit_clipping']['estimated_clipped_joints']
                rows.append(dict(run=run,episode=e['episode']+1,completed=e['completed'],step=s['step'],
                    raw=raw,stage1=stage1,final=final,base=base,q=np.array(s['q'][:16]),
                    planned=np.array(s['planned_u']),low=lo,high=hi,down=down,up=up))
            if rows:
                episodes.append(dict(episode=e['episode']+1,completed=e['completed'],outcome=e['finish_reason'],**summarize(rows,names)))
            run_rows.extend(rows)
        runs[run]=dict(**summarize(run_rows,names),episodes=episodes)
        all_rows.extend(run_rows)
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(payload).hexdigest(),
                            measured_limits_sha256=bounds['model_sha256']))
    result=dict(sources=sources,epsilon_rad=EPS,
        interpretation='Controller estimates after rebasing on its latest feedback; exact bridge targets unavailable because bridge telemetry was not enabled.',
        validation='Every published target matches delta clamp followed by frozen measured-range clamp; failed/unpublished plans excluded.',
        all_published=summarize(all_rows,names),
        completed_only=summarize([r for r in all_rows if r['completed']],names),
        aborted_only=summarize([r for r in all_rows if not r['completed']],names),runs=runs)
    (OUT/'clipping_report.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'clipping_joints.csv').open('w',newline='') as f:
        rows=[dict(group=group,actions=data['actions'],**joint) for group,data in
              [('all_published',result['all_published']),('completed_only',result['completed_only']),*runs.items()]
              for joint in data['joints']]
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for group,data in [('all_published',result['all_published']),('completed_only',result['completed_only']),*runs.items()]:
        print(group,json.dumps({k:v for k,v in data.items() if k not in ['joints','episodes']}))
        if 'episodes' in data:
            for e in data['episodes']:
                print('EP',json.dumps({k:e[k] for k in ['episode','completed','outcome','actions','delta_clipped_actions',
                      'delta_clipped_action_percent','absolute_clipped_actions','absolute_clipped_action_percent','either_clipped_action_percent']}))
        print('Joint breakdown:',json.dumps(sorted(data['joints'],key=lambda x:-x['absolute_clipped_count'])))


if __name__=='__main__':main()
