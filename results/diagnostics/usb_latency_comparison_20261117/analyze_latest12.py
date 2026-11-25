"""Reproduce the latest 9+3 completed M2 episode analysis, without hardware."""
import csv
from datetime import datetime
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('m2_prior_metrics',HERE/'analyze_m2_manipulation.py')
metrics_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics_module)
stats=metrics_module.stats
NAMES=['leap_experiment_20261117_013433_789897','leap_experiment_20261117_014022_979856']
OUT=HERE/'m2_latest12_013433_014022'


def actual_bridge(path,d):
    completed=[e for e in d['episodes'] if e.get('completed')]
    published={s['action_timestamp_us']:s for e in completed for s in e['steps'] if s.get('action_published')}
    owners={s['action_timestamp_us']:e['episode']+1 for e in completed for s in e['steps'] if s.get('action_published')}
    ends={s['action_timestamp_us']:datetime.fromisoformat(e['finished_at_utc']).timestamp()*1e6
          for e in completed for s in e['steps'] if s.get('action_published')}
    writes={};tracking=[];intervals=[];reads=[];previous=None;fault=None
    with path.with_name('bridge_telemetry.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r.get('event')!='cycle':continue
            if r.get('error'):fault=r
            if r.get('write_succeeded') and r.get('new_action') and r.get('written_action_timestamp_us') in published:
                writes[r['written_action_timestamp_us']]=r
            identity=r.get('held_action_timestamp_us')
            if (r.get('held_command_type')=='delta' and identity in ends
                    and r.get('state_timestamp_us') is not None and r['read_finished_us']<=ends[identity]):
                tracking.append(np.degrees(np.abs(np.asarray(r['q'])-r['held_target_before_read'])))
                reads.append(r['read_duration_s']*1000)
                if previous is not None:intervals.append((r['state_timestamp_us']-previous)*.001)
                previous=r['state_timestamp_us']
            else:previous=None
    a=np.asarray(tracking)
    clipped=np.array([abs(np.asarray(r['written_target'])-r['requested_target'])>1e-7 for r in writes.values()])
    overlap=[]
    if fault is not None:
        for e in completed:
            started=datetime.fromisoformat(e['started_at_utc']).timestamp()*1e6
            ended=datetime.fromisoformat(e['finished_at_utc']).timestamp()*1e6
            if started<=fault['read_finished_us']<=ended:
                overlap.append(dict(source_episode=e['episode']+1,fault_relative_to_episode_end_ms=(fault['read_finished_us']-ended)*.001))
    report=dict(published_actions=len(published),matched_written_actions=len(writes),
        unmatched_actions=[dict(source_episode=owners[k],step=published[k]['step'],action_timestamp_us=k) for k in published.keys()-writes.keys()],
        current_fault_overlaps_episodes=overlap,
        publish_to_write_ms=stats([(writes[k]['write_started_us']-k)*.001 for k in writes]),
        serial_read_ms=stats(reads),feedback_interval_ms=stats(intervals),
        tracking_error_deg=stats(a),tracking_joint_fraction_over_5deg=float(np.mean(a>5)),
        clipped_actions=int(np.any(clipped,axis=1).sum()),clipped_joint_values=int(clipped.sum()),
        recorder=json.loads(path.with_name('bridge_telemetry.status.json').read_text()),
        torque_off_verified='Torque disabled and verified on all 16 motors.' in path.with_name('bridge.log').read_text(),
        fault=None if fault is None else {k:fault.get(k) for k in ['error','read_finished_us','held_command_type','held_target_before_read','q','current_raw_native']})
    return report


def main():
    OUT.mkdir(exist_ok=True)
    runs=[(ROOT/'results/diagnostics/m2_usb_latency_1ms'/name/'controller.json') for name in NAMES]
    runs=[(p,json.loads(p.read_text())) for p in runs]
    profile=json.loads((ROOT/'results/bo_validation/M2_profile_20261115/profile.json').read_text())
    baseline_sources=[ROOT/r['file'] for r in profile['sources'] if r['group']=='Unpaced']
    baseline_runs=[(p,json.loads(p.read_text())) for p in baseline_sources]
    baseline,_,_,_=metrics_module.metrics(baseline_runs)
    latest,episodes,curves,steps=metrics_module.metrics(runs)
    ref=baseline_runs[0][1]
    checks={}
    for p,d in runs:
        same={k:d[k]==ref[k] for k in ['weights','hardware_config','success_thresholds','joint_reference','pose_transform',
                                     'failure_z','task_contact_params','planner_config','rollout_config']}
        same.update({f'{role}_scene':metrics_module.normalized_xml(d['scenes'][role]['scene_xml'])==metrics_module.normalized_xml(ref['scenes'][role]['scene_xml']) for role in ['rollout','eval']})
        assert all(same.values()),same
        checks[p.parent.name]=same
    threshold=np.degrees(2*np.arcsin(np.sqrt(ref['success_thresholds']['quat'])))
    combined=0
    for p,d in runs:
        for e in d['episodes']:
            if not e.get('completed'):continue
            row=episodes[combined];row['source_episode']=row['episode'];row['episode']=combined+1
            curve=curves[combined];curve['episode']=combined+1
            q=np.array([s['q'] for s in e['steps']]+[e['q_end']]);angles=np.degrees([s['quaternion_error_rad'] for s in e['steps']]+[e['quaternion_error_rad']])
            pos=np.linalg.norm(q[:,16:19]-e['target_pos'],axis=1)*1000
            i=int(np.argmin(angles));row['position_at_best_orientation_mm']=float(pos[i])
            row['position_axes_at_best_orientation_mm']=((q[i,16:19]-e['target_pos'])*1000).tolist()
            row['final_position_axes_mm']=((q[-1,16:19]-e['target_pos'])*1000).tolist()
            a_ok=angles<threshold
            row['best_position_when_orientation_ok_mm']=float(pos[a_ok].min()) if a_ok.any() else None
            row['off_90_degree_protocol']=not 75<=row['initial_orientation_error_deg']<=105
            combined+=1
    assert combined==12
    bridges={p.parent.name:actual_bridge(p,d) for p,d in runs}
    excluded={(name,e['source_episode']) for name,b in bridges.items() for e in b['current_fault_overlaps_episodes']}
    clean_runs=[(p,dict(d,episodes=[e for e in d['episodes'] if (p.parent.name,e['episode']+1) not in excluded])) for p,d in runs]
    clean,_,_,_=metrics_module.metrics(clean_runs)
    for e in episodes:e['current_fault_overlap']=(Path(e['source']).parent.name,e['source_episode']) in excluded
    prior=json.loads((HERE/'m2_manipulation/report.json').read_text())['usb1_completed']
    report=dict(completed_only=True,source_runs=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,d in runs],
        configuration_matches=checks,latest12=latest,clean11=clean,previous7=prior,baseline24=baseline,episodes=episodes,
        thresholds=dict(position_mm=20,orientation_deg=threshold),bridges=bridges,
        launches={p.parent.name:{k:v for k,v in json.loads(p.with_name('launch.json').read_text()).items() if k in ['reason','exit_codes','completed_services','argv']} for p,d in runs},
        limitations=['All 12 episodes are marked completed in raw logs. Episode 9 overlaps a current cutoff 35.5 ms before its logged ending; a separate clean-11 comparison excludes it.',
            'One episode starts near 172 degrees; it is retained and flagged. Initial direction counts are not balanced.',
            'Historical comparison is unpaired and shared GPU/camera conditions differ; lower success cannot be attributed uniquely to USB latency.',
            'Step-pooled clipping/tracking statistics weight the 1000-step timeout heavily. First-100-action clipping is also provided.',
            'A small orientation error during an object drop is not a stable grasp or a near-success; concurrent position errors are reported.',
            'No hardware or dynamics simulation was run by this analysis.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'episodes.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(episodes[0]));w.writeheader();w.writerows(episodes)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,constrained_layout=True)
    colors=plt.get_cmap('tab20').colors
    for i,c in enumerate(curves):
        label=f"{c['episode']}: {c['outcome']}"
        axes[0].plot(c['time_s'],c['position_mm'],label=label,color=colors[i]);axes[1].plot(c['time_s'],c['angle_deg'],color=colors[i])
    axes[0].axhline(20,color='black',ls='--');axes[1].axhline(threshold,color='black',ls='--')
    axes[0].set_ylabel('Position error (mm)');axes[1].set(ylabel='Orientation error (deg)',xlabel='Manipulation time (s)')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8);fig.suptitle('Latest 12 completed M2 episodes: 9 + 3, SAM released after acceptance')
    for ext in ['png','pdf']:fig.savefig(OUT/f'episode_errors.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    groups=[baseline,prior,latest];x=np.arange(3)
    axes[0].bar(x,[100*g['success_rate'] for g in groups],color=['#2166ac','#888888','#d6604d'])
    axes[0].set_xticks(x,['Baseline\n11/24','Previous USB 1 ms\n1/7','Latest USB 1 ms\n1/12']);axes[0].set(ylabel='Success rate (%)',ylim=(0,100))
    agekeys=['joint_age_at_command_ms','object_age_at_command_ms'];a=np.arange(2)
    for i,g in enumerate(groups):axes[1].bar(a+(i-1)*.25,[g[k]['mean'] for k in agekeys],.25,label=['Baseline','Previous 7','Latest 12'][i])
    axes[1].set_xticks(a,['Joint age at command','Object age at command']);axes[1].set_ylabel('Mean measurement age (ms)');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    for ext in ['png','pdf']:fig.savefig(OUT/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>Latest 12 M2 episodes</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}img{max-width:100%}</style>',
        '<h1>Latest 12 logged-completed M2 episodes</h1><p>Latest two source runs mark 9 and 3 episodes completed: one success, ten drop/failure outcomes and one timeout. Episode 9 overlaps a current cutoff, despite its completed flag. Excluding this infrastructure-affected episode leaves 11 clean episodes: one success, nine drops and one timeout (9.1% success). All saved physics, planner, cost, reference and transformation settings match the historical baseline.</p>',
        '<table><tr><th>Metric</th><th>Historical 24</th><th>Previous USB 1 ms: 7</th><th>Latest USB 1 ms: 12</th></tr>']
    table=[('Success rate (%)',lambda g:g['success_rate']*100),('Successes',lambda g:g['successes']),('Drops/failures',lambda g:g['drops']),
        ('Timeouts',lambda g:g['timeouts']),('Planning mean (ms)',lambda g:g['planning_ms']['mean']),('Effective control rate (Hz)',lambda g:g['effective_hz']),
        ('Joint age at command (ms)',lambda g:g['joint_age_at_command_ms']['mean']),('Object age at command (ms)',lambda g:g['object_age_at_command_ms']['mean']),
        ('Clipped actions (%)',lambda g:g['clipped_action_percent']),('First 100 actions per episode clipped (%)',lambda g:g['first100_clipped_action_percent'])]
    for label,get in table:body.append('<tr><td>'+label+'</td>'+''.join(f'<td>{get(g):.2f}</td>' for g in groups)+'</tr>')
    body+=['</table><h2>Episodes</h2><table><tr><th>Combined episode</th><th>Source episode</th><th>Outcome</th><th>Start angle</th><th>Steps</th><th>Duration</th><th>Best orientation</th><th>Position at best orientation</th><th>Final position</th><th>Final orientation</th><th>Clipped actions</th></tr>']
    for e in episodes:
        body.append(f"<tr><td>{e['episode']}</td><td>{Path(e['source']).parent.name.split('_')[3]} / {e['source_episode']}</td><td>{e['outcome']}</td><td>{e['initial_orientation_error_deg']:.1f}°</td><td>{e['steps']}</td><td>{e['duration_s']:.2f} s</td><td>{e['best_orientation_error_deg']:.1f}°</td><td>{e['position_at_best_orientation_mm']:.1f} mm</td><td>{e['final_position_error_mm']:.1f} mm</td><td>{e['final_orientation_error_deg']:.1f}°</td><td>{e['clipped_action_percent']:.1f}%</td></tr>")
    body+=['</table><p>Combined episodes 1–9 are from 013433; 10–12 are from 014022. Episode 4 starts at 171.9 degrees. Position threshold: 20 mm; orientation threshold: 23.07 degrees. Five episodes ever reach the orientation threshold, but only one meets both pose thresholds together.</p>',
        '<p>Run 013433 stopped with motor 12 current 561 raw against a 550 raw limit. The cutoff was 35.5 ms before the logged end of episode 9 and its final published action was not written; treat that episode as infrastructure-affected. Run 014022 ended on operator interruption; zero return had not converged (mf_mcp error 0.0675 rad versus 0.06 rad tolerance). Torque-off was verified in both.</p>',
        '<img src="comparison.png"><img src="episode_errors.png"><h2>Limitations</h2><ul>']
    body+=['<li>'+html.escape(x)+'</li>' for x in report['limitations']]
    body+=['</ul><p><a href="report.json">Full report</a> · <a href="episodes.csv">Episode CSV</a></p>']
    (OUT/'index.html').write_text('\n'.join(body))
    print(json.dumps(dict(latest=latest,bridge_summaries={name:{k:v for k,v in b.items() if k not in ['fault']} for name,b in bridges.items()},episodes=episodes),indent=2))


if __name__=='__main__':main()
