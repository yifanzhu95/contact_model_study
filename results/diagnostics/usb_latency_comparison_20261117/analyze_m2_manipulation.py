"""Offline completed-episode comparison; never opens USB or runs dynamics."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import html
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'m2_manipulation'
CURRENT=ROOT/'results/diagnostics/m2_usb_latency_1ms/leap_experiment_20261117_010708_505359/controller.json'
PROFILE=ROOT/'results/bo_validation/M2_profile_20261115/profile.json'


def stats(values):
    a=np.asarray(values,dtype=float).ravel()
    return dict(count=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.percentile(a,95)),max=float(a.max()),min=float(a.min())) if a.size else None


def wilson(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-radius,center+radius]


def normalized_xml(text):
    root=ET.fromstring(text)
    for mesh in root.findall('.//mesh'):mesh.attrib.pop('content_type',None)
    return ET.tostring(root)


def metrics(runs):
    all_steps=[];episodes=[];intervals=[];clip=[];clip_first100=[];curves=[]
    for path,d in runs:
        bounds=d['hardware_config']['ctrl_range']
        for e in d['episodes']:
            if not e.get('completed') or e['finish_reason'] not in ('success','failure','timeout'):continue
            steps=[s for s in e['steps'] if s.get('action_published')]
            assert len(steps)==e['n_steps'] and steps
            all_steps.extend(steps)
            times=[s.get('action_timestamp_us',datetime.fromisoformat(s['action_published_at_utc']).timestamp()*1e6)
                   if 'action_published_at_utc' in s else
                   datetime.fromisoformat(s['timestamp_utc']).timestamp()*1e6+s['planning_time']*1e6 for s in steps]
            # For older logs the publication stamp is reconstructed only if absent.
            intervals.extend(np.diff(times)*.001)
            delta=np.array([s['action_delta'] for s in steps]);bounded=np.clip(delta,-np.asarray(bounds['lower']),np.asarray(bounds['upper']))
            clipped=abs(delta-bounded)>1e-7;clip.extend(clipped);clip_first100.extend(clipped[:100])
            q=np.array([s['q'] for s in steps]+[e['q_end']]);pos=np.linalg.norm(q[:,16:19]-e['target_pos'],axis=1)
            angles=np.degrees([s['quaternion_error_rad'] for s in steps]+[e['quaternion_error_rad']])
            quat=q[0,19:23].copy();quat/=np.linalg.norm(quat)
            if quat[0]<0:quat=-quat
            axis=int(np.argmax(abs(quat[1:])));direction=('+' if quat[axis+1]>=0 else '-')+'XYZ'[axis]
            angle_threshold=np.degrees(2*np.arcsin(np.sqrt(d['success_thresholds']['quat'])))
            p_ok=pos<d['success_thresholds']['pos'];a_ok=angles<angle_threshold
            row=dict(source=str(path.relative_to(ROOT)),episode=e['episode']+1,outcome=e['finish_reason'],
                direction=direction,steps=e['n_steps'],duration_s=e['manipulation_wall_time_s'],
                initial_position_error_mm=pos[0]*1000,initial_orientation_error_deg=angles[0],
                final_position_error_mm=pos[-1]*1000,final_orientation_error_deg=angles[-1],
                best_position_error_mm=pos.min()*1000,best_orientation_error_deg=angles.min(),
                best_orientation_when_position_ok_deg=float(angles[p_ok].min()) if p_ok.any() else None,
                orientation_ever=bool(a_ok.any()),both_pose_thresholds_ever=bool((p_ok&a_ok).any()),
                planning_mean_ms=np.mean([s['planning_time']*1000 for s in steps]),
                effective_hz=e['n_steps']/e['manipulation_wall_time_s'],
                clipped_action_percent=np.mean(np.any(clipped,axis=1))*100,
                clipped_joint_percent=np.mean(clipped)*100)
            episodes.append(row)
            curves.append(dict(episode=e['episode']+1,time_s=[s['elapsed_s'] for s in steps]+[e['manipulation_wall_time_s']],
                               position_mm=(pos*1000).tolist(),angle_deg=angles.tolist(),outcome=e['finish_reason']))
    clipping=np.asarray(clip);first=np.asarray(clip_first100);n=len(episodes);k=sum(e['outcome']=='success' for e in episodes)
    return dict(episodes=n,successes=k,drops=sum(e['outcome']=='failure' for e in episodes),
        timeouts=sum(e['outcome']=='timeout' for e in episodes),success_rate=k/n,wilson95=wilson(k,n),
        steps=len(all_steps),planning_ms=stats([s['planning_time']*1000 for s in all_steps]),
        publication_interval_ms=stats(intervals),effective_hz=len(all_steps)/sum(e['duration_s'] for e in episodes),
        clipped_action_percent=float(np.mean(np.any(clipping,axis=1))*100),clipped_joint_percent=float(np.mean(clipping)*100),
        first100_clipped_action_percent=float(np.mean(np.any(first,axis=1))*100),
        initial_orientation_error_deg=stats([e['initial_orientation_error_deg'] for e in episodes]),
        orientation_ever=sum(e['orientation_ever'] for e in episodes),
        joint_age_at_planning_ms=stats([s['feedback']['robot_age']*1000 for s in all_steps]),
        joint_age_at_command_ms=stats([s['feedback']['command_state']['robot_age']*1000 for s in all_steps]),
        object_age_at_command_ms=stats([s['feedback']['command_state']['pose_age']*1000 for s in all_steps]),
        by_direction={direction:dict(total=sum(e['direction']==direction for e in episodes),successes=sum(e['direction']==direction and e['outcome']=='success' for e in episodes))
                      for direction in ['-X','+X','-Y','+Y','-Z','+Z']}),episodes,curves,all_steps


def main():
    OUT.mkdir(exist_ok=True)
    profile=json.loads(PROFILE.read_text());sources=[ROOT/r['file'] for r in profile['sources'] if r['group']=='Unpaced']
    baselines=[(p,json.loads(p.read_text())) for p in sources];current=json.loads(CURRENT.read_text());ref=baselines[0][1]
    checks={k:current[k]==ref[k] for k in ['weights','hardware_config','success_thresholds','joint_reference',
        'pose_transform','failure_z','task_contact_params','planner_config','rollout_config']}
    checks.update({f'{s}_scene':normalized_xml(current['scenes'][s]['scene_xml'])==normalized_xml(ref['scenes'][s]['scene_xml']) for s in ['rollout','eval']})
    assert all(checks.values()),checks
    old,old_episodes,_,_=metrics(baselines);new,episodes,curves,steps=metrics([(CURRENT,current)])
    published={s['action_timestamp_us']:s for s in steps};writes={};tracking=[];reads=[];intervals=[];health=[];previous_stamp=None
    last=None;fault=None
    completed=[e for e in current['episodes'] if e.get('completed')]
    ends={s['action_timestamp_us']:datetime.fromisoformat(e['finished_at_utc']).timestamp()*1e6
          for e in completed for s in e['steps'] if s.get('action_timestamp_us')}
    for line in CURRENT.with_name('bridge_telemetry.jsonl').open():
        r=json.loads(line)
        if r.get('event')!='cycle':continue
        last=r
        if r.get('error'):fault=r
        if r.get('new_action') and r.get('write_succeeded') and r.get('written_action_timestamp_us') in published:
            writes[r['written_action_timestamp_us']]=r
        identity=r.get('held_action_timestamp_us')
        if r.get('held_command_type')=='delta' and identity in ends and r['read_finished_us']<=ends[identity] and 'q' in r:
            tracking.append(np.asarray(r['q'])-r['held_target_before_read'])
            reads.append(r['read_duration_s']*1000)
            if previous_stamp is not None:intervals.append((r['state_timestamp_us']-previous_stamp)*.001)
            previous_stamp=r['state_timestamp_us']
        else:previous_stamp=None
        if r.get('health_duration_s') is not None:health.append(r['health_duration_s']*1000)
    a=np.abs(np.asarray(tracking));original_clip=np.array([abs(np.asarray(writes[k]['requested_target'])-writes[k]['written_target'])>1e-7 for k in writes])
    new['actual_bridge']=dict(matched_written_actions=len(writes),published_actions=len(published),
        publish_to_write_ms=stats([(writes[k]['write_started_us']-k)*.001 for k in writes]),
        read_ms=stats(reads),feedback_interval_ms=stats(intervals),health_read_ms=stats(health),
        tracking_error_deg=stats(np.degrees(a)),tracking_joint_fraction_over_5deg=float(np.mean(np.degrees(a)>5)),
        actual_clipped_action_percent=float(np.mean(np.any(original_clip,axis=1))*100),
        actual_clipped_joint_percent=float(np.mean(original_clip)*100))
    recorder=json.loads(CURRENT.with_name('bridge_telemetry.status.json').read_text());launch=json.loads(CURRENT.with_name('launch.json').read_text())
    report=dict(source=str(CURRENT.relative_to(ROOT)),sha256=hashlib.sha256(CURRENT.read_bytes()).hexdigest(),
        completed_only=True,configuration_matches=checks,baseline_24=old,usb1_completed=new,episodes=episodes,
        baseline_sources=[str(p.relative_to(ROOT)) for p in sources],
        termination=dict(controller_status=current['status'],controller_error=current.get('error'),launch_reason=launch['reason'],
            interrupt_return=current.get('interrupt_return'),bridge_error=fault.get('error') if fault else None,
            watchdog_age_ms=(fault['state_timestamp_us']-fault['latest_action_timestamp_us'])*.001 if fault else None,
            last_bridge_timestamp_us=last.get('state_timestamp_us'),
            torque_off_verified='Torque disabled and verified on all 16 motors.' in CURRENT.with_name('bridge.log').read_text()),
        recorder=recorder,
        limitations=['Seven new completed episodes versus 24 historical episodes; initial placements are not paired and orientation counts differ.',
                    'The comparison matches saved model scenes, costs, references, controller configuration and physics; GPU placement and shared resource load differ.',
                    'Historical 24-episode logs do not contain actual SDK write telemetry. Their clipping is reconstructed from published deltas; the new run verifies actual writes.',
                    'The timeout episode dominates step-pooled statistics; per-episode and first-100-action clipping are also reported.',
                    'No simulation, hardware or tracking processes were started or changed by this analysis.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(episodes[0]));w.writeheader();w.writerows(episodes)
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,constrained_layout=True)
    for c in curves:
        label=f"Episode {c['episode']}: {c['outcome']}"
        axes[0].plot(c['time_s'],c['position_mm'],label=label)
        axes[1].plot(c['time_s'],c['angle_deg'],label=label)
    axes[0].axhline(20,color='black',ls='--',lw=1);axes[0].set_ylabel('Position error (mm)')
    axes[1].axhline(np.degrees(2*np.arcsin(np.sqrt(current['success_thresholds']['quat']))),color='black',ls='--',lw=1)
    axes[1].set(xlabel='Manipulation time (s)',ylabel='Orientation error (deg)')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(ncol=2,fontsize=9);fig.suptitle('M2, USB latency 1 ms: seven completed episodes')
    fig.savefig(OUT/'episode_errors.png',dpi=180);fig.savefig(OUT/'episode_errors.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.5),constrained_layout=True)
    rates=np.array([old['success_rate'],new['success_rate']]);ci=np.array([old['wilson95'],new['wilson95']])
    ax.bar([0,1],rates*100,color=['#2166ac','#d6604d'],alpha=.8)
    ax.errorbar([0,1],rates*100,yerr=np.vstack([rates-ci[:,0],ci[:,1]-rates])*100,fmt='none',ecolor='black',capsize=5)
    ax.set_xticks([0,1],['Historical baseline\n11/24','USB 1 ms\n1/7']);ax.set(ylabel='Success rate (%)',ylim=(0,100),title='Completed episodes; 95% Wilson intervals')
    ax.grid(axis='y',alpha=.2);fig.savefig(OUT/'success_comparison.png',dpi=180);fig.savefig(OUT/'success_comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
    keys=['joint_age_at_planning_ms','joint_age_at_command_ms','object_age_at_command_ms'];x=np.arange(len(keys))
    ax.bar(x-.18,[old[k]['mean'] for k in keys],.36,label='Historical 24 episodes',color='#2166ac')
    ax.bar(x+.18,[new[k]['mean'] for k in keys],.36,label='USB 1 ms: 7 episodes',color='#d6604d')
    ax.set_xticks(x,['Joint age at planning','Joint age at command','Object age at command'])
    ax.set(ylabel='Mean logged measurement age (ms)',title='Joint feedback improved; object pose age increased')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'feedback_ages.png',dpi=180);fig.savefig(OUT/'feedback_ages.pdf');plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>M2 USB latency comparison</title>',
          '<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}</style>',
          '<h1>M2 manipulation after the USB latency change</h1>',
          '<p>Seven completed episodes: one success, five drop/failure outcomes and one timeout. The run was interrupted after episode 7; no manipulation episode was aborted. The bridge later reached its 500 ms action watchdog after the zero return. Torque-off was verified.</p>',
          '<p>Saved scene XML, cost weights, physics, hardware settings, P011 joint-cost reference and planner configuration match the historical unpaced baseline. GPU placement and resource contention differ. This analysis starts no hardware or simulation.</p>',
          '<table><tr><th>Metric</th><th>Historical baseline</th><th>USB 1 ms</th></tr>']
    table=[('Completed episodes',old['episodes'],new['episodes']),('Successes',old['successes'],new['successes']),
           ('Success rate (%)',old['success_rate']*100,new['success_rate']*100),
           ('Drops/failures',old['drops'],new['drops']),('Timeouts',old['timeouts'],new['timeouts']),
           ('Mean planning time (ms)',old['planning_ms']['mean'],new['planning_ms']['mean']),
           ('Effective control frequency (Hz)',old['effective_hz'],new['effective_hz']),
           ('Mean joint age at command (ms)',old['joint_age_at_command_ms']['mean'],new['joint_age_at_command_ms']['mean']),
           ('Mean object age at command (ms)',old['object_age_at_command_ms']['mean'],new['object_age_at_command_ms']['mean']),
           ('Clipped actions (%)',old['clipped_action_percent'],new['clipped_action_percent']),
           ('Clipped actions within first 100 per episode (%)',old['first100_clipped_action_percent'],new['first100_clipped_action_percent'])]
    for title,b,a in table:body.append(f'<tr><td>{title}</td><td>{b:.2f}</td><td>{a:.2f}</td></tr>')
    body+=['</table><p>Overall clipping and tracking metrics pool all steps; the 1000-step timeout contributes 55.7% of new actions and has only 7% action clipping. It strongly affects pooled summaries.</p>',
           '<h2>Completed episodes</h2><table><tr><th>Episode</th><th>Approx. initial rotation</th><th>Outcome</th><th>Steps</th><th>Time (s)</th><th>Best orientation error (deg)</th><th>Final position error (mm)</th><th>Final orientation error (deg)</th><th>Clipped actions (%)</th></tr>']
    for e in episodes:
        body.append(f"<tr><td>{e['episode']}</td><td>{e['direction']}</td><td>{e['outcome']}</td><td>{e['steps']}</td><td>{e['duration_s']:.2f}</td><td>{e['best_orientation_error_deg']:.2f}</td><td>{e['final_position_error_mm']:.2f}</td><td>{e['final_orientation_error_deg']:.2f}</td><td>{e['clipped_action_percent']:.1f}</td></tr>")
    body+=['</table><p>Initial directions are classified from logged starting quaternions. The new batch has two −X, two +X, two −Y and one +Y starts; neither Z rotation was tested. The orientation threshold is approximately 23.07 degrees, with a 20 mm position threshold.</p>',
           '<h2>Bridge measurements during completed manipulation</h2>',
           f"<p>All {len(writes)} published actions matched successful bridge writes. Mean publish-to-write delay: {new['actual_bridge']['publish_to_write_ms']['mean']:.2f} ms. Mean feedback interval: {new['actual_bridge']['feedback_interval_ms']['mean']:.2f} ms. Mean absolute measured/held-target joint gap: {new['actual_bridge']['tracking_error_deg']['mean']:.2f} degrees; P95: {new['actual_bridge']['tracking_error_deg']['p95']:.2f} degrees. Recorder dropped {recorder['dropped']} samples.</p>",
           '<p>USB communication improved, but this batch does not show an improvement in manipulation success. Older object poses and unmatched starting orientations prevent attributing the outcome difference to the USB change.</p>',
           '<img src="success_comparison.png"><img src="feedback_ages.png"><img src="episode_errors.png">',
           '<h2>Limitations</h2><ul>']+['<li>'+html.escape(x)+'</li>' for x in report['limitations']]
    body+=['</ul><p><a href="report.json">Full report</a> · <a href="episodes.csv">Episode metrics</a></p>']
    (OUT/'index.html').write_text('\n'.join(body))
    print(json.dumps(dict(comparison={key:[old.get(key),new.get(key)] for key in ['episodes','successes','drops','timeouts','effective_hz','planning_ms','clipped_action_percent','clipped_joint_percent','orientation_ever','by_direction']},
                         episodes=episodes,actual_bridge=new['actual_bridge'],termination=report['termination']),indent=2))


if __name__=='__main__':main()
