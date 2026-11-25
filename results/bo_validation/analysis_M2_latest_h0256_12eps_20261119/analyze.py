"""Offline analysis of the 7+5 completed M2/latest-calibration/H4 episodes."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'camera_hand_calibration/cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial314_shift512_latest_h0256_unpaced'
RUNS=['leap_experiment_20261119_011314_943307','leap_experiment_20261119_011946_059801']
HIST=ROOT/'results/diagnostics/usb_latency_comparison_20261117/variation_review_20261117'
spec=importlib.util.spec_from_file_location('m2_features',HIST/'analyze.py')
features=importlib.util.module_from_spec(spec);spec.loader.exec_module(features)


def dist(values):
    a=np.asarray(values,float).ravel();a=a[np.isfinite(a)]
    return dict(n=int(a.size),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),
        p5=float(np.percentile(a,5)),median=float(np.median(a)),p95=float(np.percentile(a,95)),
        p99=float(np.percentile(a,99)),max=float(a.max())) if len(a) else None


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def timing(rows,arrays):
    planning=np.concatenate([a['planning'] for a in arrays]);gaps=np.concatenate([a['gaps'] for a in arrays])
    assert np.all(gaps>0)
    actions=sum(r['steps'] for r in rows);seconds=sum(r['duration_s'] for r in rows)
    return dict(actions=actions,total_manipulation_s=seconds,effective_hz=actions/seconds,
        equivalent_mean_cycle_ms=seconds/actions*1000,planning_ms=dist(planning),publication_interval_ms=dist(gaps),
        instantaneous_hz=dist(1000/gaps),episode_hz=dist([r['steps']/r['duration_s'] for r in rows]),
        planning_over64_count=int((planning>64).sum()),planning_over64_fraction=float((planning>64).mean()),
        publication_interval_over64_count=int((gaps>64).sum()),publication_interval_over64_fraction=float((gaps>64).mean()))


def main():
    rows=[];arrays=[];runs=[];excluded=[];documents=[]
    for run in RUNS:
        path=BASE/run/'controller.json';data=json.loads(path.read_text());documents.append(data)
        assert data['args']['time_horizon']==.256 and data['args']['ctrl_time_step']==.064
        assert data['args']['n_samples']==512 and not data['args']['pace_control']
        assert data['hardware_config']['ctrl_range']==dict(lower=.2,upper=.2)
        assert data['pose_transform']['selected_calibration']['name']=='latest'
        rr=[];aa=[]
        for ep in data['episodes']:
            if not ep.get('completed'):
                excluded.append(dict(source=str(path.relative_to(ROOT)),episode=ep['episode']+1,steps=ep['n_steps'],reason='aborted/incomplete'))
                continue
            row,series=features.feature(path,data,ep,'latest_H4')
            row.update(analysis_episode=len(rows)+1,control_hz=row['steps']/row['duration_s'])
            tt=timing([row],[series]);row.update(min_instantaneous_hz=tt['instantaneous_hz']['min'],
                max_instantaneous_hz=tt['instantaneous_hz']['max'],min_planning_ms=tt['planning_ms']['min'],
                max_planning_ms=tt['planning_ms']['max'],planning_over64_percent=tt['planning_over64_fraction']*100)
            rows.append(row);arrays.append(series);rr.append(row);aa.append(series)
        runs.append(dict(source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            episodes=len(rr),successes=sum(r['success'] for r in rr),outcomes=dict(Counter(r['outcome'] for r in rr)),
            status=data['status'],error=data.get('error'),timing=timing(rr,aa)))
    assert len(rows)==12
    for key in ['weights','hardware_config','joint_reference','success_thresholds','failure_z','rollout_config','planner_config','pose_transform']:
        assert documents[0][key]==documents[1][key],key
    aggregate=features.summarize(rows,arrays);aggregate['timing']=timing(rows,arrays)
    for key in ['initial_pos_mm','initial_angle_deg','final_pos_mm','final_angle_deg','best_angle_deg']:
        aggregate[key]=dist([r[key] for r in rows])
    aggregate['success_final_position_mm']=dist([r['final_pos_mm'] for r in rows if r['success']])
    aggregate['success_final_orientation_deg']=dist([r['final_angle_deg'] for r in rows if r['success']])
    aggregate['timeout_details']=[{k:r[k] for k in ['analysis_episode','duration_s','best_angle_deg','final_pos_mm','final_angle_deg','best_pos_with_ori_ok_mm','orientation_ever']} for r in rows if r['timeout']]
    previous=json.loads((HIST/'report.json').read_text())['cohorts']
    selection=json.loads((HIST/'selection.json').read_text())['included']
    reference=next(row for row in selection if row['cohort']=='usb1_c0.2_T0.1_U')
    reference_data=json.loads((ROOT/reference['source']).read_text())
    unchanged={key:documents[0][key]==reference_data[key] for key in
        ['weights','hardware_config','joint_reference','success_thresholds','failure_z','task_contact_params',
         'planner_config','control_policy','fixed_goal_quat']}
    validation=dict(reference_source=reference['source'],unchanged_fields=unchanged)
    assert all(unchanged.values()),unchanged
    (OUT/'baseline_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    comparisons=[]
    for name,key in [('Old calibration, H5, USB before change','historical_c0.2_T0.1_U'),
                     ('Old calibration, H5, USB 1ms','usb1_c0.2_T0.1_U')]:
        g=previous[key];comparisons.append(dict(group=name,episodes=g['episodes'],successes=g['successes'],
            drops=g['drops'],timeouts=g['timeouts'],success_percent=g['success_rate']*100,
            planning_mean_ms=g['planning_ms']['mean'],planning_p95_ms=g['planning_ms']['p95'],
            effective_hz=g['effective_hz'],clipped_action_percent=g['clipped_action_fraction']*100))
    t=aggregate['timing'];comparisons.append(dict(group='Latest calibration, H4, USB 1ms',episodes=12,
        successes=aggregate['successes'],drops=aggregate['drops'],timeouts=sum(r['timeout'] for r in rows),
        success_percent=aggregate['success_rate']*100,planning_mean_ms=t['planning_ms']['mean'],
        planning_p95_ms=t['planning_ms']['p95'],effective_hz=t['effective_hz'],
        clipped_action_percent=aggregate['clipped_action_fraction']*100))
    d=documents[0]
    report=dict(sources=runs,excluded=excluded,selected_episodes=[dict(source=r['source'],source_episode=r['source_episode'],analysis_episode=r['analysis_episode']) for r in rows],
        baseline_validation=validation,settings={k:d['args'].get(k) for k in ['rollout_model','hand_acc','obj_acc','n_samples','n_iterations','ctrl_time_step','time_horizon','temperature','noise_sigma','warm_start','pace_control','steps']},
        success_thresholds=d['success_thresholds'],failure_z=d['failure_z'],effective_ctrl_range=d['effective_ctrl_range'],
        selected_calibration=d['pose_transform']['selected_calibration'],weights=d['weights'],aggregate=aggregate,comparison=comparisons,
        definitions=dict(average_control_frequency='Total published actions / total manipulation time; excludes resets, operator waits, warmup and aborted episodes.',
            min_max_control_frequency='Reciprocal of consecutive action_timestamp_us intervals, within each completed episode. No cross-episode intervals.',
            planning_distribution='All completed-episode published-action planner calls, pooled by step; excludes warmup and incomplete episode 8.',
            clipping='Controller action_delta exceeding +/-0.2 rad; inferred from logged deltas, not servo error.',
            endpoint_errors='Final tracked state including drop endpoints. Near-threshold statistics include terminal state.'),
        caveats=['Horizon and calibration both changed; comparisons do not isolate either cause.',
            'Twelve new episodes versus 24 per baseline, with manual initial placement and run-order/GPU-load differences.',
            'A faster unpaced loop shortens target hold time as well as feedback delay; faster frequency alone does not ensure higher task success.',
            'The tracker rotation-cache normalization fix preceded these runs; production calibration files remain unchanged.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');write_csv('episodes.csv',rows);write_csv('comparison.csv',comparisons)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    planning=np.concatenate([a['planning'] for a in arrays]);gaps=np.concatenate([a['gaps'] for a in arrays])
    ax[0].hist(planning,bins=45,color='#2877a7');ax[0].axvline(64,color='black',linestyle='--',label='64 ms control period')
    ax[0].set(xlabel='Planning time (ms)',ylabel='Published actions',title='M2, four steps: planning distribution');ax[0].legend(fontsize=8)
    ax[1].hist(gaps,bins=45,color='#df8f44');ax[1].axvline(64,color='black',linestyle='--',label='64 ms = 15.625 Hz')
    ax[1].set(xlabel='Consecutive command interval (ms)',ylabel='Intervals',title=f'Effective frequency: {t["effective_hz"]:.2f} Hz');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'timing.png',dpi=180);fig.savefig(OUT/'timing.pdf');plt.close(fig)
    print(json.dumps(dict(aggregate=aggregate,comparison=comparisons,per_run=runs),indent=2))


if __name__=='__main__':main()
