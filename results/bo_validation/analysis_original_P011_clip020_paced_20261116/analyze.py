"""Offline completed-episode analysis. Does not import or run any controller."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv,hashlib,json,math,copy
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2'
NEW=BASE/'original_P011_shift512_clip020_paced'
new_paths=[NEW/name/'controller.json' for name in ['leap_experiment_20261116_191608_648042','leap_experiment_20261116_192515_381817']]
profile=json.loads((ROOT/'results/bo_validation/M2_profile_20261115/profile.json').read_text())
old_report=json.loads((ROOT/'results/bo_validation/analysis_trial622_vs314_20261115/report.json').read_text())
groups={'Original paced':new_paths,
 '314 paced':[ROOT/s['file'] for s in profile['sources'] if s['group']=='Paced'],
 '314 unpaced':[ROOT/s['file'] for s in profile['sources'] if s['group']=='Unpaced'],
 '622 paced':[ROOT/p for p in old_report['sources']['622_paced']]}
directions=['-X','+X','-Y','+Y','-Z','+Z']
faces=['O','B','blue H','purple H','R (-yaw)','R (+yaw)']

def stats(values):
 a=np.asarray(values,float).ravel();a=a[np.isfinite(a)]
 if not len(a):return dict(mean=None,median=None,p95=None,max=None,min=None)
 return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()),min=float(a.min()))

def wilson(s,n):
 z=1.959963984540054;den=1+z*z/n;p=s/n
 c=(p+z*z/(2*n))/den;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [c-r,c+r]

def normal_xml(text):
 root=ET.fromstring(text)
 for mesh in root.findall('.//mesh'):mesh.attrib.pop('content_type',None)
 return ET.tostring(root)

runs={g:[json.loads(p.read_text()) for p in paths] for g,paths in groups.items()}
reference=runs['Original paced'][0]
matched={}
for g,ds in runs.items():
 checks={k:all(d.get(k)==reference.get(k) for d in ds) for k in ['hardware_config','joint_reference','pose_transform','success_thresholds','failure_z','planner_config','rollout_config','control_policy','task_contact_params','control_period']}
 checks['scene_xml']=all(normal_xml(d['scenes'][role]['scene_xml'])==normal_xml(reference['scenes'][role]['scene_xml']) for d in ds for role in ['rollout','eval'])
 assert all(checks.values()),(g,checks)
 matched[g]=checks
assert runs['Original paced'][1]['weights']==reference['weights']
assert reference['hardware_config']['ctrl_range']=={'lower':.2,'upper':.2}
assert reference['args']['pace_control'] and reference['args']['n_samples']==512
rows=[];series={g:[] for g in groups};excluded=[];sources=[];health=[];selected=[]
for group,paths in groups.items():
 for path,d in zip(paths,runs[group]):
  sources.append(dict(group=group,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  if group=='Original paced':
   launch=json.loads((path.parent/'launch.json').read_text())
   health.append(dict(run=path.parent.name,status=d['status'],error=d.get('error'),launch_reason=launch.get('reason'),exit_codes=launch.get('exit_codes')))
  for e in d['episodes']:
   if not e.get('completed') or e['finish_reason'] not in ['success','failure','timeout']:
    if group=='Original paced':excluded.append(dict(source=str(path.relative_to(ROOT)),source_episode=e['episode']+1,reason=e.get('finish_reason'),completed=e.get('completed'),steps=e.get('n_steps'),wall_s=e.get('manipulation_wall_time_s'),run_error=d.get('error')))
    continue
   ss=[s for s in e['steps'] if s.get('action_published')]
   assert ss and len(ss)==e['n_steps']
   q=np.asarray([s['q'] for s in ss]+[e['q_end']]);pos=np.linalg.norm(q[:,16:19]-e['target_pos'],axis=1)*1000
   angle=np.degrees([s['quaternion_error_rad'] for s in ss]+[e['quaternion_error_rad']])
   quat=q[0,19:23].copy();quat/=np.linalg.norm(quat)
   if quat[0]<0:quat=-quat
   axis=np.argmax(np.abs(quat[1:]));direction=('+' if quat[axis+1]>=0 else '-')+'XYZ'[axis]
   raw=np.asarray([s['action_delta'] for s in ss]);bounds=d['hardware_config']['ctrl_range']
   clipped=np.clip(raw,-np.asarray(bounds['lower']),np.asarray(bounds['upper']));hits=np.abs(raw-clipped)>1e-6
   requested=np.asarray([s['requested_target'] for s in ss]);command=np.asarray([s['u'] for s in ss]);base=np.asarray([s['feedback'].get('command_target',s['feedback']['command_qpos']) for s in ss])
   np.testing.assert_allclose(requested,base+raw,atol=1e-6)
   command_clipped=np.clip(requested,base-np.asarray(bounds['lower']),base+np.asarray(bounds['upper']))
   np.testing.assert_allclose(command,command_clipped,atol=1e-6)
   xyz=np.abs((q[:-1,16:19]-e['target_pos'])*1000)
   waits=np.array([s.get('pacing_wait_s',0) for s in ss]);plans=np.array([s['planning_time'] for s in ss])
   together=(pos<1000*d['success_thresholds']['pos'])&(angle<math.degrees(2*math.asin(math.sqrt(d['success_thresholds']['quat']))))
   row=dict(group=group,episode=len(series[group])+1,source=str(path.relative_to(ROOT)),source_episode=e['episode']+1,direction=direction,outcome=e['finish_reason'],steps=len(ss),wall_s=e['manipulation_wall_time_s'],
    start_angle_deg=float(angle[0]),start_position_mm=(q[0,16:19]*1000).tolist(),start_position_error_mm=float(pos[0]),
    final_angle_deg=float(angle[-1]),final_position_error_mm=float(pos[-1]),final_height_mm=float(q[-1,18]*1000),
    best_angle_deg=float(angle.min()),position_error_at_best_angle_mm=float(pos[angle.argmin()]),orientation_threshold_ever=bool((angle<23.0739180656).any()),simultaneous_pose_threshold_ever=bool(together.any()),
    clipped_steps=int(hits.any(axis=1).sum()),clipped_step_percent=float(hits.any(axis=1).mean()*100),clipped_joint_percent=float(hits.mean()*100),
    removed_delta_percent=float(np.abs(raw-clipped).sum()/np.abs(raw).sum()*100),planning_mean_ms=float(plans.mean()*1000),wait_steps=int((waits>0).sum()),wait_s=float(waits.sum()))
   rows.append(row)
   series[group].append(dict(row=row,steps=ss,q=q,pos=pos,angle=angle,xyz=xyz,raw=raw,clipped=clipped,hits=hits,waits=waits,plans=plans,command=command,
    t=np.r_[[s['elapsed_s'] for s in ss],e['manipulation_wall_time_s']]))
   if group=='Original paced':
    item=copy.deepcopy(e);item['source']=str(path.relative_to(ROOT));item['source_episode']=e['episode'];item['episode']=len(selected);selected.append(item)
assert len(selected)==12 and len(excluded)==1 and excluded[0]['source_episode']==9
summary={}
for group,es in series.items():
 rr=[e['row'] for e in es];ss=[s for e in es for s in e['steps']];n=len(ss)
 raw=np.concatenate([e['raw'] for e in es]);clipped=np.concatenate([e['clipped'] for e in es]);hits=np.concatenate([e['hits'] for e in es]);waits=np.concatenate([e['waits'] for e in es]);plans=np.concatenate([e['plans'] for e in es]);changes=np.concatenate([np.abs(np.diff(e['command'],axis=0)).ravel() for e in es])
 intervals=[]
 for e in es:
  if all(s.get('action_published_at_utc') for s in e['steps']):intervals.extend(np.diff([datetime.fromisoformat(s['action_published_at_utc']).timestamp() for s in e['steps']]))
 success=sum(r['outcome']=='success' for r in rr);wall=sum(r['wall_s'] for r in rr)
 summary[group]=dict(episodes=len(rr),successes=success,success_rate=success/len(rr),wilson95=wilson(success,len(rr)),drops=sum(r['outcome']=='failure' for r in rr),timeouts=sum(r['outcome']=='timeout' for r in rr),steps=n,wall_s=wall,effective_hz=n/wall,
  planning_ms=stats(plans*1000),step_wall_ms=stats([s['wall_time_s']*1000 for s in ss]),publication_interval_ms=stats(np.asarray(intervals)*1000),
  plans_below64ms=int((plans<.064).sum()),plans_over64ms=int((plans>.064).sum()),wait_steps=int((waits>0).sum()),wait_percent=float((waits>0).mean()*100),wait_s=float(waits.sum()),wait_share_wall_percent=float(waits.sum()/wall*100),positive_wait_ms=stats(waits[waits>0]*1000),
  clipped_steps=int(hits.any(axis=1).sum()),clipped_step_percent=float(hits.any(axis=1).mean()*100),clipped_joint_percent=float(hits.mean()*100),removed_delta_percent=float(np.abs(raw-clipped).sum()/np.abs(raw).sum()*100),
  mean_abs_xyz_error_mm=np.concatenate([e['xyz'] for e in es]).mean(axis=0).tolist(),clipping_by_joint_percent=(100*hits.mean(axis=0)).tolist(),raw_delta_rad=stats(np.abs(raw)),target_change_rad=stats(changes),
  start_angle_deg=stats([r['start_angle_deg'] for r in rr]),start_position_error_mm=stats([r['start_position_error_mm'] for r in rr]),
  final_position_error_mm=stats([r['final_position_error_mm'] for r in rr]),final_angle_deg=stats([r['final_angle_deg'] for r in rr]),
  success_time_s=stats([r['wall_s'] for r in rr if r['outcome']=='success']),drop_time_s=stats([r['wall_s'] for r in rr if r['outcome']=='failure']),
  robot_age_ms=stats([s['feedback']['robot_age']*1000 for s in ss]),pose_age_ms=stats([s['feedback']['pose_age']*1000 for s in ss]),
  orientation_threshold_ever=sum(r['orientation_threshold_ever'] for r in rr),simultaneous_pose_threshold_ever=sum(r['simultaneous_pose_threshold_ever'] for r in rr),
  by_direction={a:dict(attempts=sum(r['direction']==a for r in rr),successes=sum(r['direction']==a and r['outcome']=='success' for r in rr),outcomes=dict(Counter(r['outcome'] for r in rr if r['direction']==a))) for a in directions})
report=dict(summary=summary,episodes=rows,excluded=excluded,sources=sources,health=health,matched_configuration=matched,
 caveats=['Raw source logs preserved; aborted episode excluded only from this aggregate.','Starting placements are manual and not randomized/paired across cohorts.','The signed-axis labels approximate initial rotations from the dominant quaternion component.','Clipping is reconstructed from controller records; verified against logged commands, not independently measured motor-side application.','Pooling steps gives longer episodes more weight.','The earlier simulation is synchronous, uncalibrated, has no software clipping and starts from overlapping contacts; its success rate is not a hardware prediction.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(OUT/'sources.json').write_text(json.dumps(dict(included=[r for r in rows if r['group']=='Original paced'],excluded=excluded,source_files=[s for s in sources if s['group']=='Original paced']),indent=2)+'\n')
aggregate={k:v for k,v in reference.items() if k not in ['episodes','error','status','started_at_utc','finished_at_utc','argv']}
aggregate.update(mode='offline_completed_episode_aggregate',status='complete',episodes=selected,source_runs=[str(p.relative_to(ROOT)) for p in new_paths],excluded_episodes=excluded,summary=summary['Original paced'])
(OUT/'combined_controller.json').write_text(json.dumps(aggregate,separators=(',',':'),allow_nan=False)+'\n')
for filename,records in [('episodes.csv',[r for r in rows if r['group']=='Original paced']),('comparison_episodes.csv',rows)]:
 with (OUT/filename).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
for i,e in enumerate(series['Original paced']):
 color=plt.get_cmap('tab20')(i);r=e['row'];label=f"E{r['episode']} {r['direction']}: {r['outcome']}"
 axs[0].plot(e['t'],e['angle'],color=color,label=label);axs[1].plot(e['t'],e['pos'],color=color);axs[2].plot(e['t'],e['q'][:,18]*1000,color=color)
axs[0].axhline(23.073918,color='black',ls='--');axs[0].set(ylabel='Orientation error (deg)',title='Original weights, P011, clipping +/-0.2 rad, paced M2: completed episodes');axs[0].legend(ncol=3,fontsize=8)
axs[1].axhline(20,color='black',ls='--');axs[1].set(ylabel='Position error (mm)')
axs[2].axhline(65,color='#b33',ls='--',label='Fallen-cost threshold');axs[2].axhline(0,color='black',ls='--',label='Episode drop threshold');axs[2].set(ylabel='Cube height (mm)',xlabel='Manipulation wall time (s)');axs[2].legend()
for ax in axs:ax.grid(alpha=.2)
for ext in ['png','pdf']:fig.savefig(OUT/f'trajectories.{ext}',dpi=170)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
labels=list(groups);rates=[summary[g]['success_rate'] for g in labels]
axs[0].bar(range(len(labels)),rates,color=['#2166ac','#d6604d','#999999','#4a976a'])
for i,g in enumerate(labels):
 m=summary[g];axs[0].text(i,rates[i]+.025,f"{m['successes']}/{m['episodes']}",ha='center')
axs[0].set(xticks=range(len(labels)),xticklabels=labels,ylim=(0,1),ylabel='Success fraction',title='Observed hardware success (different manual placements)')
for j,g in enumerate(['Original paced','314 paced','622 paced']):
 values=[summary[g]['by_direction'][a] for a in directions];x=np.arange(6)+(j-1)*.25
 axs[1].bar(x,[v['successes']/v['attempts'] if v['attempts'] else 0 for v in values],width=.24,label=g)
 for xx,v in zip(x,values):axs[1].text(xx,v['successes']/v['attempts']+.02 if v['attempts'] else .02,f"{v['successes']}/{v['attempts']}",ha='center',fontsize=7)
axs[1].set(xticks=range(6),xticklabels=directions,ylim=(0,1.2),title='Paced results by approximate starting rotation');axs[1].legend(fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for ext in ['png','pdf']:fig.savefig(OUT/f'comparison.{ext}',dpi=170)
plt.close(fig)
print('Saved:',OUT)
for g,m in summary.items():print(g,{k:m[k] for k in ['episodes','successes','drops','timeouts','effective_hz','planning_ms','clipped_step_percent','clipped_joint_percent','removed_delta_percent','wait_steps','wait_s','start_angle_deg','start_position_error_mm','success_time_s','orientation_threshold_ever','simultaneous_pose_threshold_ever']})
print('NEW EPISODES')
for r in rows:
 if r['group']=='Original paced':print({k:r[k] for k in ['episode','source_episode','direction','outcome','steps','wall_s','start_angle_deg','best_angle_deg','position_error_at_best_angle_mm','final_angle_deg','final_position_error_mm','clipped_step_percent']})

# Human-readable, standalone report and a flat comparison table.
import html
new=summary['Original paced']
comparison_rows=[('Completed episodes','episodes',1),('Successes','successes',1),('Drops','drops',1),('Timeouts','timeouts',1),('Success rate (%)','success_rate',100),('Effective control frequency (Hz)','effective_hz',1),('Actions with clipping (%)','clipped_step_percent',1),('Joint-action entries clipped (%)','clipped_joint_percent',1),('Requested absolute delta magnitude removed (%)','removed_delta_percent',1),('Pacing wait steps','wait_steps',1),('Total deliberate wait (s)','wait_s',1)]
with (OUT/'metrics.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['metric',*groups])
 for label,key,scale in comparison_rows:writer.writerow([label,*[summary[g][key]*scale for g in groups]])

def esc(value):return html.escape(str(value))
body=['<!doctype html><html><head><meta charset="utf-8"><title>M2 original weights: 12 completed hardware episodes</title>',
 '<style>body{font:16px system-ui;margin:32px auto;max-width:1200px;padding:0 20px;color:#222}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;border:1px solid #ccc;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}code{background:#eee;padding:2px 5px}.note{background:#f3f5f7;padding:16px}</style></head><body>',
 '<h1>M2 original weights: 12 completed hardware episodes</h1>',
 '<p>Combined the first run’s 8 completed episodes and the second run’s 4 completed episodes. Excluded the first run’s episode 9 (821 actions), which aborted on stale object feedback. All raw source logs remain unchanged.</p>',
 '<p>Original weights; P011 joint-cost reference; calibrated high/high hand; cube mass 70 g; sliding friction cube/tips/palm/fingers 0.3/0.5/0.4/0.4; 512 samples; temperature 0.1; noise 0.05; shifted warm start; horizon 0.32 s; nominal period 0.064 s; paced; hardware clipping +/-0.2 rad around current measured joints.</p>',
 '<h2>Comparison</h2><p>The saved hardware configuration, physical scene XML, reference, transform, planner configuration and rollout configuration match the historical cohorts. The 314-unpaced cohort also changes pacing. Manual placements and session order are not paired/randomized.</p>',
 '<table><tr><th>Metric</th>'+''.join('<th>'+esc(g)+'</th>' for g in groups)+'</tr>']
for label,key,scale in comparison_rows:
 body.append('<tr><td>'+esc(label)+'</td>'+''.join(f'<td>{summary[g][key]*scale:.3f}</td>' for g in groups)+'</tr>')
body.extend(['</table><h2>Interpretation</h2><ul>',
 '<li>Observed success was 3/12 (25%; 95% Wilson interval 8.9–53.2%), with 8 drops and 1 timeout. This matches the observed paced trial-314 rate (6/24); it does not establish equal true performance.</li>',
 '<li>Each of the six signed-axis starting rotations has two completed attempts. Initial angle was 90.8 degrees on average; starting position error was 4.47 mm, similar to the paced trial-314 cohort.</li>',
 '<li>Successes were combined episodes 5, 8 and 10 (first-run episodes 5 and 8; second-run episode 2). Their mean completion time was 7.51 s. Successful-only time comparisons have survivor and starting-orientation selection effects.</li>',
 '<li>Seven of the eight drops never reached the orientation threshold; the remaining drop reached about 19.5 degrees while position error was about 34 mm. These were not predominantly near-complete rotations rejected only by a tight success threshold.</li>',
 '<li>The timeout ended at 37.4 degrees orientation error and 26.1 mm position error; its best orientation was 33.5 degrees. It was a manipulation timeout, not evidence of a motor stall.</li>',
 '<li>Mean planning time was 73.78 ms (P95 81.49 ms), yielding 13.42 Hz. Pacing waited on 67/3319 actions for 0.327 s in total: 0.13% of active manipulation time. There is no sustained planning slowdown in these completed episodes.</li>',
 '<li>Clipping affected 49.26% of actions and 5.01% of joint-action entries, removing 3.00% of requested absolute delta magnitude. Median per-episode clipping was 64.97%; the long timeout reduced the pooled figure. Drops and successes clipped at similar rates (66.47% and 65.67% of actions), whereas the timeout clipped only 9.60%. This does not identify clipping as the sole limiting factor or justify widening it from these data alone.</li>',
 '<li>The original weights’ strong simulation ranking did not transfer to this hardware cohort. The simulation used an uncalibrated hand, no software clipping and synchronous execution; these differences, contact/actuator model errors, and manual initial contact states remain plausible contributors. No single cause is isolated here.</li>',
 '</ul><h2>Completed episodes</h2><table><tr><th>Combined episode</th><th>Run / source episode</th><th>Initial rotation</th><th>Outcome</th><th>Duration (s)</th><th>Final angle (deg)</th><th>Final position error (mm)</th><th>Actions clipped (%)</th></tr>'])
for row in rows:
 if row['group']!='Original paced':continue
 source_run=1 if '191608' in row['source'] else 2
 body.append(f"<tr><td>{row['episode']}</td><td>{source_run} / {row['source_episode']}</td><td>{esc(row['direction'])}</td><td>{esc(row['outcome'])}</td><td>{row['wall_s']:.2f}</td><td>{row['final_angle_deg']:.1f}</td><td>{row['final_position_error_mm']:.1f}</td><td>{row['clipped_step_percent']:.1f}</td></tr>")
body.extend(['</table><p>Large final errors on drop episodes describe the dropped object, not its last stable grasp.</p>',
 '<h2>Process health</h2><p>First run: excluded episode 9 stopped on object-feedback age reaching 300 ms. The bridge subsequently reported an action timeout and verified torque off. Second run: Ctrl-C occurred during preparation for its next episode, after all four retained episodes completed; the controller returned to zero. The bridge also logged an action timeout, while confirming torque disabled on all motors. These shutdown events are kept distinct from the 12 completed manipulation outcomes.</p>',
 '<h2>Plots</h2><img src="comparison.png" alt="Hardware outcome comparison"><img src="trajectories.png" alt="Errors and height during the 12 completed episodes">',
 '<h2>Data</h2><p><a href="combined_controller.json">Combined completed-episode log</a> · <a href="episodes.csv">Episode metrics</a> · <a href="metrics.csv">Comparison metrics</a> · <a href="sources.json">Included/excluded source manifest</a> · <a href="report.json">Full analysis</a> · <a href="analyze.py">Reproduce analysis</a></p></body></html>'])
(OUT/'index.html').write_text('\n'.join(body))
for source in sources:
 assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
assert len(aggregate['episodes'])==12 and all(e['completed'] for e in aggregate['episodes'])
assert len({(e['source'],e['source_episode']) for e in aggregate['episodes']})==12
print('Verified: 12 unique completed episodes, excluded abort, unchanged source hashes.')
