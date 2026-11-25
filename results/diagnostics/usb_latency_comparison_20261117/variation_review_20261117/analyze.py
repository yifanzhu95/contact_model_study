"""Offline M2 parameter review. Raw logs and production configuration stay unchanged."""
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PROFILE=ROOT/'results/bo_validation/M2_profile_20261115/profile.json'
ANGLES=['-X','+X','-Y','+Y','-Z','+Z']
PT=20.;AT=float(np.degrees(2*np.arcsin(np.sqrt(.04))));VT=10.


def stats(a):
    a=np.asarray(a,dtype=float).ravel();a=a[np.isfinite(a)]
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if a.size else None


def wilson(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(c-r),float(c+r)]


def xml(text):
    root=ET.fromstring(text)
    for m in root.findall('.//mesh'):m.attrib.pop('content_type',None)
    return ET.tostring(root)


def stamp(text):return datetime.fromisoformat(text).timestamp()*1e6


def cohort(d,era):
    b=d['hardware_config']['ctrl_range'];assert b['lower']==b['upper']
    return f"{era}_c{b['lower']:g}_T{d['args']['temperature']:g}_{'P' if d['args'].get('pace_control',False) else 'U'}"


def label(key):
    era,c,t,p=key.split('_')
    return f"{'USB 1 ms' if era=='usb1' else 'Historical'} | ±{c[1:]} | {t} | {'paced' if p=='P' else 'unpaced'}"


def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def feature(path,d,e,key):
    ss=[s for s in e['steps'] if s.get('action_published')];assert len(ss)==e['n_steps'] and ss
    q=np.asarray([s['q'] for s in ss]+[e['q_end']]);v=np.asarray([s['q_dot'] for s in ss]+[e['q_dot_end']])
    t=np.asarray([s['elapsed_s'] for s in ss]+[e['manipulation_wall_time_s']]);assert np.all(np.diff(t)>=0)
    xyz=(q[:,16:19]-e['target_pos'])*1000;pos=np.linalg.norm(xyz,axis=1)
    angle=np.degrees([s['quaternion_error_rad'] for s in ss]+[e['quaternion_error_rad']])
    speed=np.array([s['goal_errors']['vel'] for s in ss]+[e['goal_errors']['vel']])
    po=pos<PT;ao=(angle<AT)&(speed<VT);both=po&ao
    delta=np.array([s['action_delta'] for s in ss]);b=d['hardware_config']['ctrl_range']
    clipped=np.clip(delta,-b['lower'],b['upper']);hits=abs(delta-clipped)>1e-7
    u=np.array([s['u'] for s in ss]);change=np.degrees(abs(np.diff(u,axis=0)))
    wait=np.array([s.get('pacing_wait_s',0.) for s in ss])*1000
    publish=np.array([s.get('action_timestamp_us',stamp(s['action_published_at_utc'])) if s.get('action_published_at_utc') else
                      stamp(s['timestamp_utc'])+s['planning_time']*1e6 for s in ss]);gaps=np.diff(publish)*.001
    # One joint state and one tracked pose may have different acquisition times.
    planned_age=np.array([s['feedback']['robot_age'] for s in ss])*1000
    pose_age=np.array([s['feedback']['pose_age'] for s in ss])*1000
    command_pose_age=np.array([s['feedback']['command_state']['pose_age'] for s in ss])*1000
    command_joint_age=np.array([s['feedback']['command_state']['robot_age'] for s in ss])*1000
    planning=np.array([s['planning_time'] for s in ss])*1000
    quat=q[0,19:23].copy();quat/=np.linalg.norm(quat)
    if quat[0]<0:quat=-quat
    axis=int(np.argmax(abs(quat[1:])));direction=('+' if quat[axis+1]>=0 else '-')+'XYZ'[axis]
    score=np.maximum.reduce([pos/PT,angle/AT,speed/VT]);best=int(score.argmin())
    oi=int(np.flatnonzero(ao)[np.argmin(pos[ao])]) if ao.any() else None
    window=min(1.,float(t[-1]));m=max(1,int(np.searchsorted(t[:-1],window,side='right')))
    early_angles=np.interp(window,t,angle);early_pos=pos[:m]
    early_changes=change[:max(m-1,1)]
    final_step_lag=[]
    for a,z in zip(ss,ss[1:]):
        pub=a.get('action_timestamp_us');state=z['feedback']['command_state']['robot_timestamp']
        if pub is not None and state>pub:
            final_step_lag.extend(np.degrees(abs(np.asarray(a['u'])-z['feedback']['command_qpos'])))
    row=dict(cohort=key,source=str(path.relative_to(ROOT)),source_episode=e['episode']+1,
        outcome=e['finish_reason'],success=e['finish_reason']=='success',drop=e['finish_reason']=='failure',
        timeout=e['finish_reason']=='timeout',steps=len(ss),duration_s=e['manipulation_wall_time_s'],
        direction=direction,initial_angle_deg=float(angle[0]),off90=not 75<=angle[0]<=105,
        initial_pos_mm=float(pos[0]),initial_x_mm=float(xyz[0,0]),initial_y_mm=float(xyz[0,1]),initial_z_mm=float(xyz[0,2]),
        final_pos_mm=float(pos[-1]),final_angle_deg=float(angle[-1]),best_angle_deg=float(angle.min()),
        best_angle_with_pos_ok_deg=float(angle[po].min()) if po.any() else None,
        orientation_ever=bool(ao.any()),both_pose_ever=bool(both.any()),
        best_pos_with_ori_ok_mm=float(pos[oi]) if oi is not None else None,
        ori_ok_closest_x_mm=float(xyz[oi,0]) if oi is not None else None,
        ori_ok_closest_y_mm=float(xyz[oi,1]) if oi is not None else None,
        ori_ok_closest_z_mm=float(xyz[oi,2]) if oi is not None else None,
        ori_ok_closest_time_s=float(t[oi]) if oi is not None else None,
        best_joint_score=float(score[best]),best_joint_pos_mm=float(pos[best]),best_joint_angle_deg=float(angle[best]),
        pose_reached_at25mm=bool(np.any((pos<25)&ao)),pose_reached_at30mm=bool(np.any((pos<30)&ao)),
        pos_ok_angle_under30=bool(np.any(po&(angle<30)&(speed<VT))),
        final_x_mm=float(xyz[-1,0]),final_y_mm=float(xyz[-1,1]),final_z_mm=float(xyz[-1,2]),
        clipped_actions=int(np.any(hits,axis=1).sum()),clipped_joint_values=int(hits.sum()),
        clipped_action_fraction=float(np.any(hits,axis=1).mean()),
        prefix100_clipped_action_fraction=float(np.any(hits[:100],axis=1).mean()),
        mean_planning_ms=float(planning.mean()),p95_planning_ms=float(np.percentile(planning,95)),
        command_joint_age_ms=float(command_joint_age.mean()),command_object_age_ms=float(command_pose_age.mean()),
        planned_joint_age_ms=float(planned_age.mean()),planned_object_age_ms=float(pose_age.mean()),
        original_object_age_at_plan_end_ms=float((pose_age+planning).mean()),
        first1s_complete_window=bool(t[-1]>=1.),first1s_angle_improvement_deg=float(angle[0]-early_angles),
        first1s_pos_peak_mm=float(early_pos.max()),first1s_x_drift_mm=float(np.interp(window,t,xyz[:,0])-xyz[0,0]),
        first1s_hand_speed_rms_rad_s=float(np.sqrt(np.mean(v[:m,:16]**2))),
        first1s_target_step_mean_deg=float(early_changes.mean()),first1s_target_step_p95_deg=float(np.percentile(early_changes,95)),
        first1s_clipped_joint_fraction=float(hits[:m].mean()),
        wait_steps=int((wait>0).sum()),total_wait_s=float(wait.sum()*.001),
        max_wait_ms=float(wait.max()),positive_wait_mean_ms=float(wait[wait>0].mean()) if (wait>0).any() else 0.,
        estimated_next_command_joint_gap_mean_deg=float(np.mean(final_step_lag)) if final_step_lag else None)
    arrays=dict(t=t,pos=pos,angle=angle,xyz=xyz,planning=planning,gaps=gaps,wait=wait,hits=hits,delta=delta)
    return row,arrays


def summarize(rows,arrays):
    n=len(rows);k=sum(r['success'] for r in rows);total_steps=sum(r['steps'] for r in rows)
    cat=lambda key:np.concatenate([a[key].ravel() for a in arrays])
    return dict(episodes=n,successes=k,drops=sum(r['drop'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
        success_rate=k/n,wilson95=wilson(k,n),steps=total_steps,
        non_drop_finish_rate=sum(not r['drop'] for r in rows)/n,
        total_manipulation_s=sum(r['duration_s'] for r in rows),
        effective_hz=total_steps/sum(r['duration_s'] for r in rows),planning_ms=stats(cat('planning')),
        publication_interval_ms=stats(cat('gaps')),duration_s=stats([r['duration_s'] for r in rows]),
        success_duration_s=stats([r['duration_s'] for r in rows if r['success']]),
        clipped_action_fraction=sum(r['clipped_actions'] for r in rows)/total_steps,
        clipped_joint_fraction=sum(r['clipped_joint_values'] for r in rows)/(total_steps*16),
        median_episode_clipped_fraction=float(np.median([r['clipped_action_fraction'] for r in rows])),
        median_episode_prefix100_clipped_fraction=float(np.median([r['prefix100_clipped_action_fraction'] for r in rows])),
        orientation_ever=sum(r['orientation_ever'] for r in rows),both_pose_ever=sum(r['both_pose_ever'] for r in rows),
        near_position_misses=sum(not r['success'] and r['best_pos_with_ori_ok_mm'] is not None and 20<=r['best_pos_with_ori_ok_mm']<30 for r in rows),
        counterfactual_pose25mm=sum(r['pose_reached_at25mm'] for r in rows),counterfactual_pose30mm=sum(r['pose_reached_at30mm'] for r in rows),
        position_ok_orientation30deg=sum(r['pos_ok_angle_under30'] for r in rows),
        command_joint_age_ms=stats([r['command_joint_age_ms'] for r in rows]),
        command_object_age_ms=stats([r['command_object_age_ms'] for r in rows]),
        original_object_age_at_plan_end_ms=stats([r['original_object_age_at_plan_end_ms'] for r in rows]),
        first1s_angle_improvement_deg=stats([r['first1s_angle_improvement_deg'] for r in rows]),
        first1s_target_step_mean_deg=stats([r['first1s_target_step_mean_deg'] for r in rows]),
        first1s_hand_speed_rms_rad_s=stats([r['first1s_hand_speed_rms_rad_s'] for r in rows]),
        waited_steps=sum(r['wait_steps'] for r in rows),waited_fraction=sum(r['wait_steps'] for r in rows)/total_steps,
        total_wait_s=sum(r['total_wait_s'] for r in rows),
        pacing_fraction_of_manipulation=sum(r['total_wait_s'] for r in rows)/sum(r['duration_s'] for r in rows),
        positive_wait_ms=stats(cat('wait')[cat('wait')>0]),
        off90=sum(r['off90'] for r in rows),
        by_direction={direction:dict(n=sum(r['direction']==direction and not r['off90'] for r in rows),
            successes=sum(r['direction']==direction and not r['off90'] and r['success'] for r in rows)) for direction in ANGLES})


def main():
    profile=json.loads(PROFILE.read_text());manifest=json.loads((OUT/'exclusions.json').read_text())
    manual={(x['source'],x['source_episode']):x['reason'] for x in manifest['user_exclusions']+manifest['infrastructure_exclusions']}
    paths={ROOT/s['file']:'historical' for s in profile['sources']}
    inventory=json.loads((ROOT/'results/diagnostics/hardware_physical_parameter_review_20261116/inventory.json').read_text())
    for g in inventory['groups']:
        if (g['model']=='M2' and g['weight_family']=='trial314' and g['mass_kg']==.07 and g['hand_acc']==g['obj_acc']=='high'
                and g['horizon']==.32 and g['n_samples']==512 and g['noise']==.05 and g['warm_start']=='shift'
                and g['delta_reference']=='measured_position' and [g['cube_mu'],g['tip_mu'],g['palm_mu'],g['finger_mu']]==[.3,.5,.4,.4]):
            # Preserve the established curated 24+24 baseline. Older additional
            # baseline/video/retest attempts are not silently added to it.
            if json.loads(g['clip'])=={'lower':.2,'upper':.2} and g['temperature']==.1:continue
            for name in g['sources']:paths[ROOT/name]='historical'
    for folder in (ROOT/'results/diagnostics').glob('m2_usb_latency*'):
        for p in folder.glob('leap_*/controller.json'):paths[p]='usb1'
    reference=json.loads((ROOT/profile['sources'][0]['file']).read_text())
    rows=[];series=[];sources=[];excluded=[];skipped=[];seen=set();groups=defaultdict(list);run_data={}
    for path,era in sorted(paths.items(),key=lambda x:str(x[0])):
        d=json.loads(path.read_text());rel=str(path.relative_to(ROOT));mismatch=[]
        for key in ['weights','success_thresholds','joint_reference','pose_transform','failure_z','task_contact_params','rollout_config']:
            if d[key]!=reference[key]:mismatch.append(key)
        for key,value in reference['hardware_config'].items():
            if key!='ctrl_range' and d['hardware_config'].get(key)!=value:mismatch.append('hardware.'+key)
        for key,value in reference['planner_config'].items():
            if key!='temperature' and d['planner_config'].get(key)!=value:mismatch.append('planner.'+key)
        for role in ['rollout','eval']:
            if xml(d['scenes'][role]['scene_xml'])!=xml(reference['scenes'][role]['scene_xml']):mismatch.append(role+'_scene')
        if mismatch:skipped.append(dict(source=rel,reason='Unmatched settings',fields=mismatch));continue
        key=cohort(d,era);faults=[]
        telemetry=path.with_name('bridge_telemetry.jsonl')
        if era=='usb1' and telemetry.exists():
            for line in telemetry.open():
                if '"error"' not in line:continue
                try:r=json.loads(line)
                except ValueError:continue
                if r.get('event')=='cycle' and r.get('error'):faults.append(r)
        source_rows=[]
        for e in d['episodes']:
            episode_key=(rel,e['episode']+1);reason=manual.get(episode_key)
            if not e.get('completed') or e['finish_reason'] not in ['success','failure','timeout']:reason=reason or 'Aborted/incomplete episode'
            for fault in faults:
                if (e.get('started_at_utc') and e.get('finished_at_utc') and
                        stamp(e['started_at_utc'])<=fault['read_finished_us']<=stamp(e['finished_at_utc'])):
                    reason=reason or 'Device fault overlaps manipulation: '+fault['error']
            if reason:
                excluded.append(dict(source=rel,source_episode=e['episode']+1,outcome=e['finish_reason'],reason=reason));continue
            identity=(e.get('started_at_utc'),tuple(e['steps'][0]['q']))
            if identity in seen:excluded.append(dict(source=rel,source_episode=e['episode']+1,outcome=e['finish_reason'],reason='Duplicate episode identity'));continue
            seen.add(identity)
            row,arr=feature(path,d,e,key);rows.append(row);series.append(arr);groups[key].append(len(rows)-1);source_rows.append(len(rows)-1)
        text=path.with_name('controller.log').read_text() if path.with_name('controller.log').exists() else ''
        known={e['episode']+1 for e in d['episodes']}
        for ep in set(map(int,re.findall(r'Episode (\d+): MPPI starting\.',text)))-known:
            excluded.append(dict(source=rel,source_episode=ep,outcome='uncheckpointed',reason='Started episode absent from saved completed checkpoint'))
        sources.append(dict(source=rel,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cohort=key,status=d['status'],included=len(source_rows),
                            sam_released=json.loads(path.with_name('launch.json').read_text()).get('completed_services') if path.with_name('launch.json').exists() else None))
        if era=='usb1' and source_rows:run_data[path]=dict(document=d,index=source_rows)
    summaries={key:summarize([rows[i] for i in indices],[series[i] for i in indices]) for key,indices in sorted(groups.items())}
    print('Controller metrics collected:',len(rows),'episodes;',sum(r['cohort'].startswith('usb1') for r in rows),'USB 1 ms episodes',flush=True)
    # Exact bridge-held-target diagnostics for the selected current episodes.
    bridge_groups=defaultdict(lambda:dict(errors=[],latency=[],gaps=[],written=0,published=0,recorder=[]))
    for path,run in run_data.items():
        selected={rows[i]['source_episode'] for i in run['index']};d=run['document'];key=cohort(d,'usb1');bag=bridge_groups[key]
        episodes=[e for e in d['episodes'] if e['episode']+1 in selected]
        ends={s['action_timestamp_us']:stamp(e['finished_at_utc']) for e in episodes for s in e['steps'] if s.get('action_timestamp_us')}
        writes=set();previous=None
        for line in path.with_name('bridge_telemetry.jsonl').open():
            try:r=json.loads(line)
            except ValueError:continue  # An interrupted writer can leave one partial final line.
            if r.get('event')!='cycle':continue
            action=r.get('written_action_timestamp_us')
            if r.get('new_action') and r.get('write_succeeded') and action in ends:
                writes.add(action);bag['latency'].append((r['write_started_us']-action)*.001)
            action=r.get('held_action_timestamp_us')
            if (r.get('held_command_type')=='delta' and action in ends and r.get('state_timestamp_us') is not None
                    and r['read_finished_us']<=ends[action]):
                bag['errors'].append(np.degrees(abs(np.asarray(r['q'])-r['held_target_before_read'])))
                if previous is not None:bag['gaps'].append((r['state_timestamp_us']-previous)*.001)
                previous=r['state_timestamp_us']
            else:previous=None
        bag['written']+=len(writes);bag['published']+=len(ends)
        status=path.with_name('bridge_telemetry.status.json');bag['recorder'].append(json.loads(status.read_text()) if status.exists() else {'status':'not finalized'})
        print('Bridge diagnostics read:',path.parent.name,flush=True)
    bridge={}
    for key,b in bridge_groups.items():
        a=np.asarray(b['errors']);bridge[key]=dict(tracking_error_deg=stats(a),fraction_joint_samples_over5deg=float(np.mean(a>5)),
            publish_to_write_ms=stats(b['latency']),feedback_interval_ms=stats(b['gaps']),published=b['published'],matched_writes=b['written'],recorder=b['recorder'])
    current=[r for r in rows if r['cohort'].startswith('usb1')]
    correlations=[]
    features=['initial_pos_mm','initial_angle_deg','initial_x_mm','initial_z_mm','first1s_angle_improvement_deg','first1s_pos_peak_mm',
        'first1s_hand_speed_rms_rad_s','first1s_target_step_mean_deg','first1s_clipped_joint_fraction','command_object_age_ms']
    for outcome in ['success','drop']:
        for name in features:
            rr=[r for r in current if r['first1s_complete_window']]
            x=np.array([r[name] for r in rr]);y=np.array([r[outcome] for r in rr],float)
            with warnings.catch_warnings():warnings.simplefilter('ignore');rho=float(spearmanr(x,y).statistic)
            xr=[];yr=[]
            for key in sorted({r['cohort'] for r in rr}):
                subset=[r for r in rr if r['cohort']==key]
                if len(subset)<4 or len({r[outcome] for r in subset})<2:continue
                xx=rankdata([r[name] for r in subset]);yy=rankdata([r[outcome] for r in subset])
                xr.extend(xx-xx.mean());yr.extend(yy-yy.mean())
            centered=float(np.corrcoef(xr,yr)[0,1]) if len(xr)>2 and np.std(xr)>0 else None
            correlations.append(dict(feature=name,outcome=outcome,n=len(rr),pooled_spearman=rho if np.isfinite(rho) else None,
                within_config_centered_rank_r=centered,within_config_n=len(xr),
                positive_outcome_median=float(np.median(x[y==1])),negative_outcome_median=float(np.median(x[y==0]))))
    near=[r for r in current if not r['success'] and r['best_pos_with_ori_ok_mm'] is not None and 20<=r['best_pos_with_ori_ok_mm']<=35]
    for r in near:
        values=[r[f'ori_ok_closest_{axis}_mm'] for axis in 'xyz'];i=int(np.argmax(abs(np.array(values))))
        r['dominant_near_miss_axis']=('+' if values[i]>=0 else '-')+'XYZ'[i]
    near_summary=dict(episodes=len(near),dominant_axes=dict(Counter(r['dominant_near_miss_axis'] for r in near)),
                      signed_xyz_mm={axis:stats([r[f'ori_ok_closest_{axis}_mm'] for r in near]) for axis in 'xyz'})
    # Optional near-miss fields must be present for every CSV row.
    for r in rows:r.setdefault('dominant_near_miss_axis',None)
    report=dict(scope='Matched M2 / trial314 / P011 / 70 g / friction .3,.5,.4,.4 / high-high / N512 / H.32 / noise.05 / shift',
        thresholds=dict(position_mm=PT,orientation_deg=AT,velocity=VT),cohorts=summaries,bridge=bridge,
        included_episodes=len(rows),usb1_episodes=len(current),excluded=excluded,skipped=skipped,sources=sources,
        correlations=correlations,near_misses=near_summary,
        caveats=['The final episode of the latest run is excluded per user request even though it was logged successful; raw logs are unchanged.',
            'The known current-cutoff episode and aborted/uncheckpointed episodes are excluded. Historical curated 24+24 baseline selection is preserved.',
            'Historical and USB-1ms cohorts are separate; GPU load, tracking history, operator placement, run order and device conditions are uncontrolled.',
            'Success counts are small and initial rotation counts differ. Confidence intervals are descriptive binomial Wilson intervals.',
            'Final-error comparisons are outcome-censored: successes stop early and drop endpoints can be far away. Joint-time near-miss metrics are used instead.',
            'Clipping is defined by action deltas exceeding the hardware bounds, not by a servo following error. Smaller bounds mechanically reduce measured/target error.',
            'Correlations are per episode, not per timestep. Within-configuration centered ranks remove group mean ranks but not initial-orientation or run-order confounding.',
            'Whole-episode clipping and timing can be consequences of a drop or timeout; early one-second features reduce, but do not eliminate, reverse-causation concerns.',
            '25/30 mm counts are post-hoc threshold crossings on existing trajectories, not new tests or recommendations to relax success thresholds.',
            'Logs do not contain cost-term variation across every sampled rollout; observed-state correlations cannot identify optimal cost weights.',
            'Historical USB latency was not embedded in each recording; the historical label denotes experiments before the verified 16-to-1 ms setting change.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUT/'selection.json').write_text(json.dumps(dict(included=[{k:r[k] for k in ['source','source_episode','cohort']} for r in rows],excluded=excluded),indent=2)+'\n')
    write_csv('episodes.csv',rows);write_csv('correlations.csv',correlations);write_csv('near_misses.csv',near)
    write_csv('cohorts.csv',[dict(cohort=k,episodes=g['episodes'],successes=g['successes'],drops=g['drops'],timeouts=g['timeouts'],
        success_rate=g['success_rate'],ci_low=g['wilson95'][0],ci_high=g['wilson95'][1],clipped_action_fraction=g['clipped_action_fraction'],
        waited_fraction=g['waited_fraction'],planning_mean_ms=g['planning_ms']['mean'],effective_hz=g['effective_hz']) for k,g in summaries.items()])
    make_plots(rows,series,groups,summaries,near)
    make_html(report,rows)
    print('Saved review:',OUT,flush=True)
    for k,g in summaries.items():print(k,{f:g[f] for f in ['episodes','successes','drops','timeouts','orientation_ever','near_position_misses','waited_fraction','effective_hz']},flush=True)
    print('Near misses:',near_summary,flush=True)
    print('Early-feature correlations:',json.dumps(correlations,indent=2),flush=True)


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def make_plots(rows,series,groups,summary,near):
    keys=[k for k in summary if k.startswith('usb1')];colors=dict(zip(keys,plt.get_cmap('tab10').colors))
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True);x=np.arange(len(keys));bottom=np.zeros(len(keys))
    for outcome,color in [('successes','#2b8c6b'),('drops','#d95f4b'),('timeouts','#e5b54b')]:
        vals=np.array([summary[k][outcome]/summary[k]['episodes']*100 for k in keys]);ax.bar(x,vals,bottom=bottom,label=outcome,color=color);bottom+=vals
    ax.set_xticks(x,[f"±{k.split('_')[1][1:]} / {k.split('_')[2]}\n{k.split('_')[3]}: {summary[k]['successes']}/{summary[k]['episodes']} successes" for k in keys]);ax.set(ylabel='Episodes (%)',ylim=(0,100),title='USB 1 ms variants; final episode excluded as requested');ax.legend(ncol=3);save(fig,'outcomes')
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for era,marker,offset in [('historical','o',-.1),('usb1','s',.1)]:
        kk=[k for k in summary if k.startswith(era)];labels=sorted({k.removeprefix(era+'_') for k in kk})
        for k in kk:
            g=summary[k];all_labels=sorted({s.split('_',1)[1] for s in summary});index=all_labels.index(k.split('_',1)[1])+offset
            p=g['success_rate']*100;lo,hi=np.array(g['wilson95'])*100
            ax.errorbar(index,p,yerr=[[p-lo],[hi-p]],fmt=marker,color='#2166ac' if era=='historical' else '#d6604d',capsize=4)
            ax.annotate(f"{g['successes']}/{g['episodes']}",(index,p),xytext=(5,4),textcoords='offset points',fontsize=8)
    ax.plot([],[],'o',color='#2166ac',label='Historical');ax.plot([],[],'s',color='#d6604d',label='USB 1 ms')
    all_labels=sorted({s.split('_',1)[1] for s in summary});ax.set_xticks(np.arange(len(all_labels)),[s.replace('_','\n') for s in all_labels]);ax.set(ylabel='Success (%) with 95% Wilson interval',ylim=(-3,105),title='Matched settings, separate historical and current cohorts');ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'historical_comparison')
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    for k in keys:
        rr=[r for r in rows if r['cohort']==k]
        for outcome,marker in [('success','o'),('failure','x'),('timeout','s')]:
            sub=[r for r in rr if r['outcome']==outcome]
            ax.scatter([r['best_joint_pos_mm'] for r in sub],[r['best_joint_angle_deg'] for r in sub],marker=marker,color=colors[k],alpha=.8)
        ax.scatter([],[],color=colors[k],label=label(k).removeprefix('USB 1 ms | '))
    ax.axvline(PT,color='black',ls='--');ax.axhline(AT,color='black',ls='--');ax.set(xlabel='Position error (mm)',ylabel='Orientation error (deg)',title='Best simultaneous approach to the success region\nCircles: success; crosses: drop; squares: timeout');ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'joint_pose_errors')
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for j,axis in enumerate('xyz'):
        vals=np.array([r[f'ori_ok_closest_{axis}_mm'] for r in near]);ax.scatter(np.full(len(vals),j)+np.linspace(-.13,.13,len(vals)),vals,alpha=.8)
    ax.axhline(0,color='black',lw=1);ax.set_xticks(range(3),['X','Y','Z']);ax.set(ylabel='Signed position error (mm)',title='Non-success near-misses with orientation already passing\nClosest position between 20 and 35 mm; one point per episode');ax.grid(axis='y',alpha=.2);save(fig,'near_miss_axes')
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    for outcome,marker,color in [('success','o','#2b8c6b'),('failure','x','#d95f4b'),('timeout','s','#e5b54b')]:
        rr=[r for r in rows if r['cohort'].startswith('usb1') and r['outcome']==outcome and r['first1s_complete_window']]
        ax.scatter([r['first1s_angle_improvement_deg'] for r in rr],[r['first1s_pos_peak_mm'] for r in rr],marker=marker,color=color,label=outcome,alpha=.8)
    ax.axhline(PT,color='black',ls='--');ax.set(xlabel='Orientation improvement in first second (deg)',ylabel='Peak position error in first second (mm)',title='Early rotation progress versus displacement from the goal');ax.grid(alpha=.2);ax.legend();save(fig,'early_progress')
    paced=[k for k in keys if k.endswith('_P')];fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for k in paced:
        data=np.concatenate([series[i]['planning'] for i in groups[k]]);a=np.sort(data);axes[0].plot(a,np.arange(1,len(a)+1)/len(a),label=label(k).removeprefix('USB 1 ms | '))
    axes[0].axvline(64,color='black',ls='--');axes[0].set(xlabel='Planning time (ms)',ylabel='Fraction of steps',title='Pacing only waits when the loop is early');axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(len(paced)),[summary[k]['pacing_fraction_of_manipulation']*100 for k in paced]);axes[1].set_xticks(np.arange(len(paced)),[k.split('_')[1] for k in paced]);axes[1].set(ylabel='Manipulation time spent pacing (%)',title='Measured pacing intervention')
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'pacing')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);values=np.full((len(keys),6),np.nan)
    for i,k in enumerate(keys):
        for j,direction in enumerate(ANGLES):
            d=summary[k]['by_direction'][direction]
            if d['n']:values[i,j]=d['successes']/d['n']
    ax.imshow(values,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
    for i,k in enumerate(keys):
        for j,direction in enumerate(ANGLES):
            d=summary[k]['by_direction'][direction];ax.text(j,i,f"{d['successes']}/{d['n']}" if d['n'] else '—',ha='center',va='center')
    ax.set_xticks(range(6),ANGLES);ax.set_yticks(range(len(keys)),[label(k).removeprefix('USB 1 ms | ') for k in keys]);ax.set_title('Success by starting rotation; starts outside 75–105° omitted');save(fig,'starting_rotations')


def make_html(report,rows):
    body=['<!doctype html><meta charset="utf-8"><title>M2 variation review</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}img{max-width:100%}</style>',
        '<h1>M2 clipping, temperature and pacing review</h1>',
        '<p>Matched trial-314 weights, P011 reference, 70 g cube, sliding friction .3/.5/.4/.4, calibrated high/high hand, 512 samples, .32 s horizon, .064 s control timestep, noise .05 and shifted warm start. The final episode of the latest run is explicitly excluded even though it was logged successful. Raw logs are unchanged.</p>',
        f"<p>Selected {report['usb1_episodes']} current USB-1ms episodes and {report['included_episodes']-report['usb1_episodes']} historical episodes. The historical curated 24+24 comparison is preserved; other historical parameter combinations are included separately.</p>",
        '<table><tr><th>Cohort</th><th>Success</th><th>Drops</th><th>Timeouts</th><th>Success % (95% CI)</th><th>Orientation ever passing</th><th>Near position misses</th><th>Planning ms</th><th>Control Hz</th><th>Clipped actions %</th><th>Pacing steps %</th></tr>']
    for k,g in report['cohorts'].items():
        lo,hi=np.array(g['wilson95'])*100
        body.append(f"<tr><td>{html.escape(label(k))}</td><td>{g['successes']}/{g['episodes']}</td><td>{g['drops']}</td><td>{g['timeouts']}</td><td>{g['success_rate']*100:.1f} ({lo:.1f}–{hi:.1f})</td><td>{g['orientation_ever']}</td><td>{g['near_position_misses']}</td><td>{g['planning_ms']['mean']:.2f}</td><td>{g['effective_hz']:.2f}</td><td>{g['clipped_action_fraction']*100:.1f}</td><td>{g['waited_fraction']*100:.1f}</td></tr>")
    body+=['</table><p>Near position misses: non-success episodes with orientation and velocity passing and closest simultaneous position between 20 and 30 mm. Position/orientation minima from different times are never combined.</p>']
    for name in ['outcomes','historical_comparison','starting_rotations','joint_pose_errors','near_miss_axes','early_progress','pacing']:body.append(f'<h2>{name.replace("_"," ").title()}</h2><img src="{name}.png">')
    body+=['<h2>Early correlations (exploratory; not causal)</h2><table><tr><th>Feature</th><th>Outcome</th><th>N</th><th>Spearman</th><th>Within-config centered rank r</th><th>Outcome-positive median</th><th>Outcome-negative median</th></tr>']
    for c in report['correlations']:
        body.append('<tr>'+''.join('<td>'+html.escape(str(c[k]) if c[k] is None or isinstance(c[k],str) else f'{c[k]:.3f}')+'</td>' for k in ['feature','outcome','n','pooled_spearman','within_config_centered_rank_r','positive_outcome_median','negative_outcome_median'])+'</tr>')
    body+=['</table><h2>Exclusions and interpretation limits</h2><ul>']+['<li>'+html.escape(c)+'</li>' for c in report['caveats']]
    body+=['</ul><p><a href="report.json">Full report</a> · <a href="episodes.csv">Every selected episode</a> · <a href="cohorts.csv">Cohort metrics</a> · <a href="correlations.csv">Correlations</a> · <a href="near_misses.csv">Near-miss details</a> · <a href="selection.json">Exact selection/exclusions</a></p>']
    (OUT/'index.html').write_text('\n'.join(body))


if __name__=='__main__':main()
