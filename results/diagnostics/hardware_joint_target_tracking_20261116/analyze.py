"""Offline estimated target-following analysis; no hardware/controller imports."""
from pathlib import Path
from datetime import datetime
import json,csv,hashlib,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
files={'clip010':ROOT/'results/bo_validation/analysis_original_P011_clip010_paced_20261116/combined_controller.json',
       'clip020':ROOT/'results/bo_validation/analysis_original_P011_clip020_paced_repeat_20261116/combined24_controller.json'}
lags=[0,25,50,75,100,125,150,175,200,250]
thresholds=[.02,.05,.1,.15,.2]

def stats(values):
 a=np.asarray(values,float).ravel()
 return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()),rms=float(np.sqrt(np.mean(a*a))))

report={};arrays={};per_episode=[];per_joint=[]
for label,path in files.items():
 d=json.loads(path.read_text());err=[];lead=[];ages=[];old_ages=[];locations=[];lagged={lag:[] for lag in lags};spacing=[];skipped=0;total=0
 for ei,e in enumerate(d['episodes'],1):
  ss=[s for s in e['steps'] if s.get('action_published')];episode_errors=[];states={}
  for s in ss:
   f=s['feedback'];states[f['robot_timestamp']*1e-6]=np.asarray(s['q'][:16]);states[f['command_state']['robot_timestamp']*1e-6]=np.asarray(f['command_qpos'])
  times=np.array(sorted(states));q=np.array([states[t] for t in times]);spacing.extend(np.diff(times)*1000)
  pub=np.array([datetime.fromisoformat(s['action_published_at_utc']).timestamp() for s in ss]);u=np.array([s['u'] for s in ss])
  for lag in lags:
   at=pub+lag*.001;valid=(at>=times[0])&(at<=times[-1]);measured=np.stack([np.interp(at[valid],times,q[:,j]) for j in range(16)],axis=1)
   lagged[lag].append(measured-u[valid])
  for i,s in enumerate(ss[:-1]):
   n=ss[i+1];total+=1;old_ages.append((n['feedback']['robot_timestamp']*1e-6-pub[i])*1000)
   age=(n['feedback']['command_state']['robot_timestamp']*1e-6-pub[i])*1000
   if age<=0:skipped+=1;continue
   target=np.asarray(s['u']);nextq=np.asarray(n['feedback']['command_qpos']);base=np.asarray(s['feedback']['command_qpos'])
   residual=nextq-target;err.append(residual);episode_errors.append(residual);lead.append(target-base);ages.append(age)
   locations.append(dict(episode=ei,source=e.get('source'),source_episode=e.get('source_episode',e['episode'])+1,step=i))
  a=abs(np.array(episode_errors));per_episode.append(dict(cohort=label,episode=ei,source=e.get('source'),source_episode=e.get('source_episode',e['episode'])+1,outcome=e['finish_reason'],pairs=len(a),mean_abs_rad=float(a.mean()),p95_abs_rad=float(np.percentile(a,95)),joint_fraction_over_005=float((a>.05).mean()),joint_fraction_over_010=float((a>.1).mean())))
 a=abs(np.array(err));l=abs(np.array(lead));maximum=a.max(axis=1);worst=np.unravel_index(a.argmax(),a.shape)
 aligned={str(lag):stats(abs(np.concatenate(values))) for lag,values in lagged.items()}
 best=min(aligned,key=lambda x:aligned[x]['mean'])
 report[label]=dict(source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),episodes=len(d['episodes']),valid_pairs=len(a),skipped_pairs=skipped,
  next_command_measurement_offset_after_previous_publication_ms=stats(ages),estimated_target_residual_rad=stats(a),estimated_target_residual_deg=stats(np.degrees(a)),
  maximum_joint_residual_per_action_rad=stats(maximum),maximum_joint_residual_per_action_deg=stats(np.degrees(maximum)),issued_target_lead_rad=stats(l),
  thresholds={str(t):dict(joint_fraction=float((a>t).mean()),action_with_any_joint_fraction=float((maximum>t).mean()),joint_count=int((a>t).sum()),action_count=int((maximum>t).sum())) for t in thresholds},
  next_planning_input_predates_previous_publication_fraction=float((np.asarray(old_ages)<0).mean()),next_planning_input_relative_publication_ms=stats(old_ages),
  alignment_scan_rad=aligned,lowest_mean_error_tested_lag_ms=int(best),observed_distinct_packet_spacing_ms=stats(spacing),
  worst=dict(**locations[worst[0]],joint=d['joint_order'][worst[1]],estimated_gap_rad=float(a[worst]),estimated_gap_deg=float(np.degrees(a[worst]))))
 for j,name in enumerate(d['joint_order']):per_joint.append(dict(cohort=label,joint=name,mean_abs_rad=float(a[:,j].mean()),p95_abs_rad=float(np.percentile(a[:,j],95)),max_abs_rad=float(a[:,j].max()),fraction_over_005=float((a[:,j]>.05).mean()),fraction_over_010=float((a[:,j]>.1).mean())))
 arrays[label]=a
refpath=ROOT/'configs/leap_joint_references/cube.json';ref=json.loads(refpath.read_text());diff=abs(np.array(ref['joint_reference'])-np.array(ref['hold_targets']))
report['static_capture']=dict(source=str(refpath.relative_to(ROOT)),captured_at=ref.get('captured_at_utc'),absolute_gap_rad=stats(diff),absolute_gap_deg=stats(np.degrees(diff)),worst_joint=ref['joint_order'][int(diff.argmax())],scope='One saved settled capture, not a general static accuracy specification.')
report['method']=dict(comparison='For issued command k, compare logged estimated absolute target u[k] with fresh joint sample in command_state of k+1, retaining only timestamps later than publication k. Never compare the new target against the same step preplanning measurement.',
 caveats=['The bridge actual applied target and action-to-write timestamps are not logged in measured_position mode; u is an estimate rebased on controller-side feedback.',
 'The next measurement can be before, just before, or after the bridge applies that target. State publication occurs before the motor write. These are communication+motion+model-base proxy gaps, not isolated servo errors.',
 'Commands keep changing; 57 ms measurement offset is not a settled-response accuracy test.',
 'Per-joint sample statistics pool 16 joints and all retained steps; long episodes receive more weight.',
 'The lag scan linearly interpolates observed joint feedback and is descriptive temporal alignment. Its best lag is not an identified motor time constant or a uniquely identified transport delay.',
 'Observed packet intervals are from controller-sampled feedback; intermediate bridge packets may be absent.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for filename,rows in [('episodes.csv',per_episode),('joints.csv',per_joint)]:
 with (OUT/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for label,a in arrays.items():
 sorted_a=np.sort(np.degrees(a.ravel()));axs[0].plot(sorted_a,np.arange(1,sorted_a.size+1)/sorted_a.size,label=label)
 vals=report[label]['alignment_scan_rad'];axs[1].plot(lags,[np.degrees(vals[str(lag)]['mean']) for lag in lags],marker='o',label=label)
axs[0].set(xlim=(0,20),ylim=(0,1.01),xlabel='Estimated target / next sampled joint gap (deg)',ylabel='Fraction of joint samples',title='Dynamic command-following proxy; not settled servo error')
axs[1].set(xlabel='Measurement offset after publication (ms)',ylabel='Mean absolute discrepancy (deg)',title='Descriptive target / feedback alignment scan')
for ax in axs:ax.grid(alpha=.2);ax.legend()
for ext in ['png','pdf']:fig.savefig(OUT/f'estimated_tracking.{ext}',dpi=180)
plt.close(fig)
text=['<!doctype html><meta charset="utf-8"><title>LEAP commanded / measured joint gap</title><style>body{font:16px system-ui;margin:30px auto;max-width:1150px;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}</style>',
 '<h1>Commanded versus measured LEAP joints</h1><p><strong>Dynamic gaps below use estimated controller targets, not acknowledged bridge-applied targets.</strong> They include transport, sampling, rebasing uncertainty, motor motion and contact loading. They cannot isolate hardware servo accuracy.</p>',
 '<p>For each action, compare its estimated absolute target with the next iteration’s fresh pre-publication measurement. This sample is timestamped about 57 ms after the previous action publication. The bridge publishes a read before writing its next target; some such measurements may not yet reflect the prior action being applied. The input at the start of the next planning call actually predates the previous publication in about 99.6% of transitions.</p>',
 '<table><tr><th>Metric</th><th>+/-0.1, 12 episodes</th><th>+/-0.2, 24 episodes</th></tr>']
for title,field in [('Mean per-joint absolute gap (deg)','mean'),('Median per-joint absolute gap (deg)','median'),('P95 per-joint absolute gap (deg)','p95'),('Maximum estimated gap (deg)','max')]:text.append('<tr><td>'+title+'</td>'+''.join(f'<td>{report[g]["estimated_target_residual_deg"][field]:.3f}</td>' for g in files)+'</tr>')
for t in [.05,.1,.2]:
 for kind,title in [('joint_fraction','Joint samples'),('action_with_any_joint_fraction','Actions with any joint')]:text.append(f'<tr><td>{title} exceeding {t} rad (%)</td>'+''.join(f'<td>{100*report[g]["thresholds"][str(t)][kind]:.3f}</td>' for g in files)+'</tr>')
text.extend(['</table><h2>Static capture</h2>',f"<p>In the saved P010 commanded hold / P011 settled measured reference capture, the RMS gap was {report['static_capture']['absolute_gap_deg']['rms']:.3f} degrees and maximum gap {report['static_capture']['absolute_gap_deg']['max']:.3f} degrees. This demonstrates a close settled match for that one grasp; it is not a guarantee under all loads.</p>",
 '<h2>Interpretation</h2><ul><li>During manipulation the hand is not reaching every rapidly changing estimated target before the next target is issued.</li><li>Timing contributes: aligning measurements roughly 125 ms after publication reduces discrepancy in these records. That value is descriptive and includes pipeline effects; it is not an isolated motor delay.</li><li>Contact requires position error to generate proportional holding effort. A nonzero target/measurement gap does not itself imply a faulty or physically incapable hand.</li><li>To measure actual tracking, log the bridge’s applied target, action timestamp and write time alongside each joint read, including in measured_position mode. This can be added as telemetry without changing the control law; no such code change is made by this analysis.</li></ul>',
 '<img src="estimated_tracking.png" alt="Estimated target gap distribution and timing alignment"><p><a href="report.json">Full statistics and caveats</a> · <a href="joints.csv">Per-joint metrics</a> · <a href="episodes.csv">Per-episode metrics</a> · <a href="analyze.py">Reproduce analysis</a></p>'])
(OUT/'index.html').write_text('\n'.join(text))
for label in files:assert hashlib.sha256(files[label].read_bytes()).hexdigest()==report[label]['source_sha256']
print('Saved offline analysis:',OUT)
for label in files:print(label,report[label]['estimated_target_residual_deg'],'valid_pairs',report[label]['valid_pairs'],'best tested alignment',report[label]['lowest_mean_error_tested_lag_ms'])
