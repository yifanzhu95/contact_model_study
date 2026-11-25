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
NEW=BASE/'original_P011_shift512_clip010_paced'
new_paths=[NEW/name/'controller.json' for name in ['leap_experiment_20261116_200122_025376','leap_experiment_20261116_201054_680583']]
profile=json.loads((ROOT/'results/bo_validation/M2_profile_20261115/profile.json').read_text())
old_report=json.loads((ROOT/'results/bo_validation/analysis_trial622_vs314_20261115/report.json').read_text())
groups={'Original paced':new_paths,
 'Original clip020 paced':[BASE/'original_P011_shift512_clip020_paced'/n/'controller.json' for n in ['leap_experiment_20261116_191608_648042','leap_experiment_20261116_192515_381817','leap_experiment_20261116_193833_284125','leap_experiment_20261116_194450_763550','leap_experiment_20261116_194848_714604']],
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
 checks={k:all(d.get(k)==reference.get(k) for d in ds) for k in ['joint_reference','pose_transform','success_thresholds','failure_z','planner_config','rollout_config','control_policy','task_contact_params','control_period']}
 checks['hardware_config_except_ctrl_range']=all({k:v for k,v in d['hardware_config'].items() if k!='ctrl_range'}=={k:v for k,v in reference['hardware_config'].items() if k!='ctrl_range'} for d in ds)
 checks['scene_xml']=all(normal_xml(d['scenes'][role]['scene_xml'])==normal_xml(reference['scenes'][role]['scene_xml']) for d in ds for role in ['rollout','eval'])
 assert all(checks.values()),(g,checks)
 matched[g]=checks
assert all(d['weights']==reference['weights'] for d in runs['Original paced']+runs['Original clip020 paced'])
assert reference['hardware_config']['ctrl_range']=={'lower':.1,'upper':.1}
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
assert len(selected)==12 and len(excluded)==1 and excluded[0]['source_episode']==7
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
axs[0].axhline(23.073918,color='black',ls='--');axs[0].set(ylabel='Orientation error (deg)',title='Original weights, P011, clipping +/-0.1 rad, paced M2: completed episodes');axs[0].legend(ncol=3,fontsize=8)
axs[1].axhline(20,color='black',ls='--');axs[1].set(ylabel='Position error (mm)')
axs[2].axhline(65,color='#b33',ls='--',label='Fallen-cost threshold');axs[2].axhline(0,color='black',ls='--',label='Episode drop threshold');axs[2].set(ylabel='Cube height (mm)',xlabel='Manipulation wall time (s)');axs[2].legend()
for ax in axs:ax.grid(alpha=.2)
for ext in ['png','pdf']:fig.savefig(OUT/f'trajectories.{ext}',dpi=170)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
labels=list(groups);rates=[summary[g]['success_rate'] for g in labels]
axs[0].bar(range(len(labels)),rates,color=['#2166ac','#9ecae1','#d6604d','#999999','#4a976a'])
for i,g in enumerate(labels):
 m=summary[g];axs[0].text(i,rates[i]+.025,f"{m['successes']}/{m['episodes']}",ha='center')
axs[0].set(xticks=range(len(labels)),xticklabels=[g.replace(' ','\n',1) for g in labels],ylim=(0,1),ylabel='Success fraction',title='Observed hardware success (different manual placements)')
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



# Diagnose near-misses using logged poses and static forward kinematics only.
# No physics stepping, planning, LCM or hardware communication occurs here.
import tempfile,html
import mujoco
latest=summary['Original paced'];baseline=summary['Original clip020 paced']
near=[];geometry=[];threshold_counts={str(a):0 for a in [23.0739180656,25,30,35]}
with tempfile.TemporaryDirectory(prefix='m2_clip010_static_geometry_') as tmp:
 modelpath=Path(tmp)/'scene.xml';modelpath.write_text(reference['scenes']['rollout']['scene_xml'])
 model=mujoco.MjModel.from_xml_path(str(modelpath));data=mujoco.MjData(model)
 obj=model.geom('obj').id;tips=[model.geom(n).id for n in ['if_tip','mf_tip','rf_tip','th_tip']]
 for entry,e in zip(series['Original paced'],selected):
  row=entry['row'];q=entry['q'];pos=entry['pos'];angle=entry['angle'];t=entry['t'];valid=pos<20
  best=int(np.argmin(np.where(valid,angle,np.inf)));end10=t>=t[-1]-10
  speed=np.r_[[s['goal_errors']['vel'] for s in entry['steps']],e['goal_errors']['vel']]
  for a in threshold_counts:threshold_counts[a]+=int(np.any((angle<float(a))&valid&(speed<10)&(q[:,18]>=0)))
  data.qpos[:]=q[best];mujoco.mj_kinematics(model,data)
  gaps=[float(mujoco.mj_geomDistance(model,data,tip,obj,.3,None))*1000 for tip in tips]
  result=dict(episode=row['episode'],source=row['source'],source_episode=row['source_episode'],outcome=row['outcome'],direction=row['direction'],
   best_angle_with_position_valid_deg=float(angle[best]),position_error_mm=float(pos[best]),height_mm=float(q[best,18]*1000),time_s=float(t[best]),
   max_height_above_initial_mm=float((q[:,18].max()-q[0,18])*1000),last10_angle_percentiles_deg=np.percentile(angle[end10],[10,50,90]).tolist(),last10_height_percentiles_mm=np.percentile(q[end10,18]*1000,[10,50,90]).tolist(),
   static_tip_gap_mm=gaps,static_tip_gap_order=['index','middle','ring','thumb'],
   geometry_caveat='Signed mesh/box distance from measured joint angles and tracked object pose using the saved model. Negative is model overlap; this is not evidence of real contact force or stable load-bearing support.')
  geometry.append(result)
  if angle[best]<=35 and row['outcome']!='success':near.append(result)
assert len(near)==4
report.update(near_misses=near,static_geometry=geometry,retrospective_orientation_threshold_counts=threshold_counts,
 matched_comparison_note='Saved hardware configuration matches clip020 except ctrl_range; task/model/weights/pose reference and remaining planner settings match. Placements remain manual, not paired.',
 interpretation='Clip010 trades fewer drops for many timeouts. Near-miss episodes lack a clear upward lift; model fingertip proximity is not a grasp-force objective. Neither force sufficiency nor which pads physically support the cube is measured.',
 recommendations=['Keep the success threshold unchanged for comparison.','Prioritize stable grasp/lift support and planning/execution consistency rather than increasing orientation weight alone.','The current goal already targets z=80 mm, above the 67-75 mm near-miss heights. A lift toward that height is already rewarded; merely raising the goal does not ensure grip.','A grip/lift phase or grasp-quality term would need to assess opposing support, not only sum of fingertip distances.','Current-reference clipping limits both target lead and the proportional position-error component that generates holding force; increasing costs cannot restore a command removed by clipping.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(OUT/'static_fingertip_gaps.json').write_text(json.dumps(geometry,indent=2)+'\n')
metric_rows=[('Completed episodes','episodes',1),('Successes','successes',1),('Drops','drops',1),('Timeouts','timeouts',1),('Success rate (%)','success_rate',100),('Effective frequency (Hz)','effective_hz',1),('Actions with clipping (%)','clipped_step_percent',1),('Joint-action entries clipped (%)','clipped_joint_percent',1),('Requested delta magnitude removed (%)','removed_delta_percent',1),('Total deliberate wait (s)','wait_s',1)]
with (OUT/'metrics.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['metric','clip010 latest12','clip020 previous24'])
 for label,key,scale in metric_rows:w.writerow([label,latest[key]*scale,baseline[key]*scale])
text=['<!doctype html><meta charset="utf-8"><title>M2 +/-0.1: near-miss and grasp analysis</title><style>body{font:16px system-ui;margin:30px auto;max-width:1200px;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}</style>',
 '<h1>M2 original weights, P011, clipping +/-0.1: 12 completed episodes</h1>',
 '<p>Sources: November 16 runs 20:01:22 and 20:10:54 America/Phoenix, six completed episodes each. Excluded the first run’s seventh episode, aborted after six published actions on stale joint feedback (167 ms). Original weights, P011, calibrated high/high hand, 70 g cube, sliding friction 0.3/0.5/0.4/0.4, 512 samples, shifted warm start, temperature 0.1, noise 0.05, horizon 0.32 s and paced control.</p>',
 '<h2>Clipping comparison</h2><table><tr><th>Metric</th><th>+/-0.1, latest 12</th><th>+/-0.2, previous 24</th></tr>']
for label,key,scale in metric_rows:text.append(f'<tr><td>{html.escape(label)}</td><td>{latest[key]*scale:.3f}</td><td>{baseline[key]*scale:.3f}</td></tr>')
text.extend(['</table><p>Saved scene geometry, physical values, weights, reference, thresholds, transform and planner settings match; only clipping differs among these paced original-weight cohorts. Placements are manually selected and not paired/randomized.</p>',
 '<h2>What changed</h2><p>With +/-0.1, success was 1/12, drops 3/12 and timeouts 8/12. With +/-0.2, success was 7/24, drops 15/24 and timeouts 2/24. The tighter bound produced a retention-versus-completion tradeoff in these observed cohorts. It did not improve success.</p>',
 '<p>Mean planning times are nearly equal: 76.81 vs 76.64 ms, and effective frequencies 12.933 vs 12.931 Hz. This comparison has no meaningful average speed difference. At +/-0.1, clipping removes 21.90% of requested absolute delta magnitude, versus 3.20% at +/-0.2.</p>',
 '<h2>Near misses</h2><p>The orientation threshold is about 23.07 degrees and position threshold 20 mm. The following failed episodes reached &lt;35 degrees while position remained valid. Values are from that episode’s best orientation with position below threshold, not necessarily its terminal sample.</p>',
 '<table><tr><th>Combined / source episode</th><th>Outcome</th><th>Best angle (deg)</th><th>Position error (mm)</th><th>Height (mm)</th><th>Time (s)</th><th>Max height above initial (mm)</th></tr>'])
for row in near:
 run=1 if '200122' in row['source'] else 2
 text.append(f"<tr><td>E{row['episode']} / run {run} E{row['source_episode']}</td><td>{row['outcome']}</td><td>{row['best_angle_with_position_valid_deg']:.2f}</td><td>{row['position_error_mm']:.2f}</td><td>{row['height_mm']:.2f}</td><td>{row['time_s']:.2f}</td><td>{row['max_height_above_initial_mm']:.2f}</td></tr>")
text.extend(['</table>',
 '<p>The four near-miss episodes never rose more than 1 mm above their initial tracked height. They reached their closest valid-position orientations with cube centers at 67–75 mm, below the 80 mm goal. This supports investigating support/lift quality. It does not prove that adding a lift will complete the rotation.</p>',
 '<p>Other failures are farther away: first-run E2 got to 34.37 degrees only with position error 29.86 mm; its best orientation while position was valid was 80.47 degrees. The timeout endpoints therefore should not all be described as almost successful.</p>',
 '<p>Retrospective threshold check (holding position/velocity requirements): 23.07 degrees yields 1/12; 25 or 30 degrees yields 2/12; 35 degrees yields 5/12. This is descriptive only and does not recommend relaxing the success criterion.</p>',
 '<h2>Finger contact and lifting</h2><p>The contact cost sums distances from four fingertip sites to the cube center. It does not measure opposing contact forces, slip margin, or whether two fingers can support a lift. Static geometry at near-miss E11 puts all four pads overlapping the nominal cube by roughly 0.7–2.8 mm; E10 includes about 9.5 mm of nominal thumb overlap. These reconstructed distances are not contact-force measurements. They can reflect geometry, tracking/frame error or real surface compliance, and cannot establish that the physical fingers carry the cube.</p>',
 '<p>Because clipping is applied around the current measured joints, +/-0.1 limits the target’s lead over the measured pose. For position servos it can limit the proportional contribution to gripping force as well as the commanded displacement. This is a plausible contributor to inadequate lift support, not an identified force measurement.</p>',
 '<p>The running Z-position weight is only 0.0503, while terminal position weight is 1000 and target height already 80 mm. There is little direct running-height shaping, but moving upward toward 80 mm already reduces terminal position cost when other coordinates are fixed. Raising target height or orientation weight alone does not create a stable grasp.</p>',
 '<h2>Recommendation</h2><p>Prioritize grip/lift support or a grasp-then-reorient phase, with opposing support or slip resistance assessed explicitly. Do not infer grip from fingertip count or proximity alone. Keep the success threshold and physical parameters fixed during the next diagnostic; inspect prediction versus measured loaded joint/object motion. No production cost weights, thresholds, models or hardware settings were modified by this analysis.</p>',
 '<h2>Plots</h2><img src="trajectories.png" alt="Episode error and height trajectories"><img src="comparison.png" alt="Hardware cohort comparison">',
 '<p><a href="combined_controller.json">Combined 12 completed-episode log</a> · <a href="episodes.csv">Episode metrics</a> · <a href="report.json">Full report</a> · <a href="static_fingertip_gaps.json">Static geometry diagnostics</a> · <a href="sources.json">Selection manifest</a></p>'])
(OUT/'index.html').write_text('\n'.join(text))
for source in sources:assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
assert len(selected)==12 and all(e['completed'] for e in selected)
print('Verified source hashes, completed-episode aggregate, and four near-miss diagnostics; no physics stepping or hardware operations.')
