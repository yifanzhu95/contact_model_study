"""Offline trial-622 hardware comparison; no controller/simulator execution."""
from pathlib import Path
from collections import Counter
import csv
import json
import math
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
profile=json.loads((ROOT/'results/bo_validation/M2_profile_20261115/profile.json').read_text())
BASE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2'
groups={
 '314_unpaced':[ROOT/s['file'] for s in profile['sources'] if s['group']=='Unpaced'],
 '314_paced':[ROOT/s['file'] for s in profile['sources'] if s['group']=='Paced'],
 '622_unpaced':[BASE/'trial622_shift512/leap_experiment_20261115_231917_832648/controller.json'],
 '622_paced':[BASE/'trial622_shift512/leap_experiment_20261115_232411_974280/controller.json'],
}
directions=['-X','+X','-Y','+Y','-Z','+Z']
faces=['O','B','blue H','purple H','R (−yaw)','R (+yaw)']
data={g:[json.loads(p.read_text()) for p in paths] for g,paths in groups.items()}
def stat(v):
 a=np.asarray(v,float).ravel();a=a[np.isfinite(a)]
 return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if a.size else None
def normxml(x):
 root=ET.fromstring(x)
 for m in root.findall('.//mesh'):m.attrib.pop('content_type',None)
 return ET.tostring(root)
baseline=data['314_unpaced'][0]
checks={k:all(d.get(k)==baseline.get(k) for runs in data.values() for d in runs)
 for k in ['hardware_config','joint_reference','pose_transform','success_thresholds','failure_z','planner_config','rollout_config','control_policy','task_contact_params']}
checks['physical_scenes']=all(normxml(d['scenes'][role]['scene_xml'])==normxml(baseline['scenes'][role]['scene_xml'])
 for runs in data.values() for d in runs for role in ['rollout','eval'])
assert all(checks.values()),checks
rows=[];series={g:[] for g in groups};summary={};health=[]
for g,runs in data.items():
 for path,d in zip(groups[g],runs):
  assert bool(d['args'].get('pace_control',False))==g.endswith('_paced')
  launcher=json.loads((path.parent/'launch.json').read_text())
  health.append(dict(group=g,path=str(path.relative_to(ROOT)),controller_status=d['status'],
   launch_reason=launcher.get('reason'),exit_codes=launcher.get('exit_codes')))
  for e in d['episodes']:
   if not e.get('completed') or e['finish_reason'] not in ['success','failure','timeout']:continue
   ss=[s for s in e['steps'] if s.get('action_published')]
   if not ss:continue
   q=np.asarray([s['q'] for s in ss]+[e['q_end']]);xyz=(q[:,16:19]-e['target_pos'])*1000
   pos=np.linalg.norm(xyz,axis=1);ang=np.degrees([s['quaternion_error_rad'] for s in ss]+[e['quaternion_error_rad']])
   quat=q[0,19:23].copy();quat/=np.linalg.norm(quat)
   if quat[0]<0:quat=-quat
   axis=int(np.argmax(np.abs(quat[1:])))
   direction=('+' if quat[axis+1]>0 else '-')+'XYZ'[axis]
   raw=np.asarray([s['action_delta'] for s in ss]);bound=d['hardware_config']['ctrl_range']
   clipped=np.clip(raw,-np.asarray(bound['lower']),np.asarray(bound['upper']));hits=np.abs(raw-clipped)>1e-6
   change=np.abs(np.diff(np.asarray([s['u'] for s in ss]),axis=0))
   t=np.r_[[s['elapsed_s'] for s in ss],e['manipulation_wall_time_s']]
   retained=q[:,18]>=e['fallen_z'];abovezero=q[:,18]>=d['failure_z']
   number=len(series[g])+1
   entry=dict(group=g,episode=number,source_episode=e['episode']+1,run=path.parent.name,direction=direction,
    outcome=e['finish_reason'],steps=len(ss),wall_s=e['manipulation_wall_time_s'],
    start_angle_deg=float(ang[0]),start_position_mm=q[0,16:19].tolist(),start_position_error_mm=float(pos[0]),
    final_angle_deg=float(ang[-1]),final_position_error_mm=float(pos[-1]),
    best_angle_deg=float(ang.min()),position_error_at_best_angle_mm=float(pos[ang.argmin()]),
    best_angle_above_cost_height_deg=float(ang[retained].min()) if retained.any() else None,
    angle_threshold_reached_above_zero=bool(np.any((ang<23.07391806563097)&abovezero)),
    mean_abs_xyz_error_mm=np.abs(xyz[:-1]).mean(axis=0).tolist(),
    clipped_step_fraction=float(hits.any(axis=1).mean()),clipped_joint_fraction=float(hits.mean()),
    removed_delta_fraction=float(np.abs(raw-clipped).sum()/np.abs(raw).sum()),
    mean_abs_target_change_rad=float(change.mean()),mean_planning_ms=float(np.mean([s['planning_time'] for s in ss])*1000))
   rows.append(entry)
   series[g].append(dict(row=entry,steps=ss,q=q,xyz=xyz,pos=pos,ang=ang,t=t,hits=hits,raw=raw,clipped=clipped,change=change))
for g,episodes in series.items():
 ss=[s for e in episodes for s in e['steps']];rr=[e['row'] for e in episodes]
 raw=np.concatenate([e['raw'] for e in episodes]);clipped=np.concatenate([e['clipped'] for e in episodes]);hits=np.concatenate([e['hits'] for e in episodes])
 plans=np.array([s['planning_time'] for s in ss]);waits=np.array([s.get('pacing_wait_s',0) for s in ss])
 prefixraw=np.concatenate([e['raw'][:100] for e in episodes]);down=data[g][0]['hardware_config']['ctrl_range']['lower'];up=data[g][0]['hardware_config']['ctrl_range']['upper']
 summary[g]=dict(episodes=len(rr),successes=sum(r['outcome']=='success' for r in rr),drops=sum(r['outcome']=='failure' for r in rr),
  timeouts=sum(r['outcome']=='timeout' for r in rr),published_steps=len(ss),
  success_rate=sum(r['outcome']=='success' for r in rr)/len(rr),
  balanced_first6_successes=sum(r['outcome']=='success' for r in rr[:6]),
  balanced_first6_directions=[r['direction'] for r in rr[:6]],
  actual_hz=len(ss)/sum(r['wall_s'] for r in rr),planning_ms=stat(plans*1000),
  step_wall_ms=stat([s['wall_time_s']*1000 for s in ss]),
  planning_over_64ms_fraction=float((plans>.064).mean()),
  clipped_step_fraction=float(hits.any(axis=1).mean()),clipped_joint_fraction=float(hits.mean()),
  removed_delta_fraction=float(np.abs(raw-clipped).sum()/np.abs(raw).sum()),
  raw_delta_rad=stat(np.abs(raw)),target_change_rad=stat(np.concatenate([e['change'].ravel() for e in episodes])),
  first100_target_change_rad=stat(np.concatenate([e['change'][:99].ravel() for e in episodes])),
  first100_clipped_step_fraction=float((np.abs(prefixraw-np.clip(prefixraw,-down,up))>1e-6).any(axis=1).mean()),
  mean_abs_xyz_error_mm=np.abs(np.concatenate([e['xyz'][:-1] for e in episodes])).mean(axis=0).tolist(),
  start_angle_deg=stat([r['start_angle_deg'] for r in rr]),start_pos_error_mm=stat([r['start_position_error_mm'] for r in rr]),
  published_state_position_error_mm=stat(np.concatenate([e['pos'][:-1] for e in episodes])),
  published_state_orientation_error_deg=stat(np.concatenate([e['ang'][:-1] for e in episodes])),
  success_time_s=stat([r['wall_s'] for r in rr if r['outcome']=='success']),
  drop_time_s=stat([r['wall_s'] for r in rr if r['outcome']=='failure']),
  robot_age_ms=stat([s['feedback']['robot_age']*1000 for s in ss]),pose_age_ms=stat([s['feedback']['pose_age']*1000 for s in ss]),
  wait_steps=int((waits>0).sum()),total_wait_s=float(waits.sum()),
  angles_reached_threshold=sum(r['angle_threshold_reached_above_zero'] for r in rr),
  by_direction={a:dict(attempts=sum(r['direction']==a for r in rr),successes=sum(r['direction']==a and r['outcome']=='success' for r in rr)) for a in directions})
weight_difference={k:dict(trial314=v,trial622=data['622_unpaced'][0]['weights'][k],
 ratio=data['622_unpaced'][0]['weights'][k]/v if v else None) for k,v in baseline['weights'].items() if v!=data['622_unpaced'][0]['weights'][k]}
report=dict(summary=summary,episodes=rows,matched_configuration=checks,changed_weights=weight_difference,sources={g:[str(p.relative_to(ROOT)) for p in ps] for g,ps in groups.items()},health=health,
 caveats=['Small, manually placed starting-pose samples; no paired or randomized A/B comparison.',
 'Trial 622 unpaced has seven episodes, with an extra -X attempt. First six span each signed axis once.',
 'Clipping inferred from raw controller deltas and logged bounds; not independent motor-side instrumentation.',
 'Step-pooled errors/motion statistics weight longer episodes more heavily.',
 'Original BO used uncalibrated simulation, 256 samples and no warm start. Hardware uses the calibrated model, 512 samples and shifted warm start.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
with (OUT/'episodes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'314':'#2166ac','622':'#d6604d'}
for mode in ['unpaced','paced']:
 fig,ax=plt.subplots(figsize=(10,4.7),constrained_layout=True)
 for j,trial in enumerate(['314','622']):
  by=summary[f'{trial}_{mode}']['by_direction'];x=np.arange(6)+(j-.5)*.36
  values=[by[a]['successes']/by[a]['attempts'] for a in directions]
  ax.bar(x,values,width=.34,color=colors[trial],label=f'Trial {trial}')
  for xx,v,a in zip(x,values,directions):ax.text(xx,v+.025,f"{by[a]['successes']}/{by[a]['attempts']}",ha='center',fontsize=9)
 ax.set(xticks=range(6),xticklabels=[f'{a}\n{f}' for a,f in zip(directions,faces)],ylim=(0,1.15),ylabel='Success fraction',
  title=f'M2 {mode}: trial 622 vs trial 314 by initial rotation')
 ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 for ext in ['png','pdf']:fig.savefig(OUT/f'success_{mode}.{ext}',dpi=180)
 plt.close(fig)
 fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
 for trial in ['314','622']:
  for i,e in enumerate(series[f'{trial}_{mode}']):
   new=trial=='622';color=plt.get_cmap('tab10')(i) if new else '#999999'
   label=f"622 E{i+1} {e['row']['direction']}: {e['row']['outcome']}" if new else ('314 baseline' if i==0 else None)
   kwargs=dict(color=color,alpha=.95 if new else .3,lw=1.4 if new else .8,ls='-' if new else ':')
   axs[0].plot(e['t'],e['ang'],label=label,**kwargs);axs[1].plot(e['t'],e['pos'],**kwargs);axs[2].plot(e['t'],e['q'][:,18]*1000,**kwargs)
 axs[0].set(ylabel='Orientation error (deg)',ylim=(0,185),title=f'Trial 622 M2 {mode}: individual trajectories vs trial 314');axs[0].axhline(23.073918,color='black',ls='--',lw=1);axs[0].legend(ncol=2,fontsize=8)
 axs[1].set(ylabel='3D position error (mm)');axs[1].axhline(20,color='black',ls='--',lw=1)
 axs[2].set(ylabel='Cube height (mm)',xlabel='Manipulation wall time (s)');axs[2].axhline(80,color='black',ls='--',lw=1);axs[2].axhline(65,color='#b33c3c',ls='--',lw=1)
 for ax in axs:ax.grid(alpha=.2)
 for ext in ['png','pdf']:fig.savefig(OUT/f'errors_{mode}.{ext}',dpi=180)
 plt.close(fig)
print('Saved',OUT)
for g,m in summary.items():
 print(g,{k:m[k] for k in ['episodes','successes','drops','timeouts','actual_hz','clipped_step_fraction','removed_delta_fraction','wait_steps','total_wait_s']})
