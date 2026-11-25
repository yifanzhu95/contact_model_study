"""Read-only analysis of the ten completed manipulations; outputs local artifacts."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261113_213615_713571'
OUT=Path(__file__).resolve().parent
log=json.loads((RUN/'controller.json').read_text())
launch=json.loads((RUN/'launch.json').read_text())
threshold=log['success_thresholds']
angle_limit=float(np.degrees(2*np.arcsin(np.sqrt(threshold['quat']))))
limits=log['effective_ctrl_range']
lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper'])
def angle(q):
 return np.degrees(2*np.arccos(np.clip(np.abs(np.asarray(q)[...,0]),0,1)))
def stats(x):
 a=np.asarray(x).ravel()
 return dict(mean=float(np.mean(a)),median=float(np.median(a)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=float(np.max(a))) if a.size else None
rows=[];extras=[];planning=[];vels=[];changes=[];deltas=[];ages=[]
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
colors=plt.get_cmap('tab10').colors
for e in log['episodes']:
 steps=e['steps']; q=np.asarray([s['q'] for s in steps]);qend=np.asarray(e['q_end'])
 t=np.asarray([s['elapsed_s'] for s in steps]);endtime=e['manipulation_wall_time_s']
 ang=np.r_[angle(q[:,19:23]),angle(qend[19:23])]
 pos=np.r_[[s['goal_errors']['pos']*1000 for s in steps],e['goal_errors']['pos']*1000]
 times=np.r_[t,endtime]
 plan=np.array([s['planning_time'] for s in steps]); planning.extend(plan)
 delta=np.asarray([s['action_delta'] for s in steps]);deltas.extend(delta)
 u=np.asarray([s['u'] for s in steps]);du=np.diff(u,axis=0);changes.extend(du)
 dt=np.diff(t)
 saturated=(delta < -lo-1e-6)|(delta>hi+1e-6)
 gv=np.array([s['goal_errors']['vel'] for s in steps]);vels.extend(gv)
 startpos=q[0,16:19]
 first_orientation=np.where(ang<angle_limit)[0]
 best_angle_index=int(np.argmin(ang))
 robot_age=np.array([s['feedback']['robot_age'] for s in steps]);pose_age=np.array([s['feedback']['pose_age'] for s in steps]);ages.extend(np.c_[robot_age,pose_age])
 row=dict(episode=e['episode']+1,outcome=e['finish_reason'],duration_s=endtime,steps=e['n_steps'],
  start_angle_deg=float(ang[0]),final_angle_deg=float(ang[-1]),best_angle_deg=float(ang.min()),
  start_position_error_mm=float(pos[0]),final_position_error_mm=float(pos[-1]),
  final_z_mm=float(qend[18]*1000),mean_plan_ms=float(plan.mean()*1000),
  fraction_plans_above_64ms=float(np.mean(plan>.064)),
  effective_hz=e['n_steps']/endtime,
  fraction_steps_clipped=float(np.mean(saturated.any(axis=1))),fraction_components_clipped=float(saturated.mean()),
  estimated_target_change_p95_deg=float(np.degrees(np.percentile(np.abs(du),95))),
  estimated_target_change_max_deg=float(np.degrees(np.abs(du).max())),
  first_orientation_pass_s=None if not len(first_orientation) else float(times[first_orientation[0]]),
  position_error_at_best_orientation_mm=float(pos[best_angle_index]),
  fraction_position_pass=float(np.mean(pos<20)),fraction_orientation_pass=float(np.mean(ang<angle_limit)))
 rows.append(row)
 extras.append(dict(episode=row['episode'],initial_position_m=startpos.tolist(),initial_quaternion_wxyz=q[0,19:23].tolist(),
   final_velocity_norm=e['goal_errors']['vel'], raw_delta_abs=stats(np.abs(delta)),
   estimated_target_change_abs=stats(np.abs(du)),estimated_target_rate_rad_s=stats(np.abs(du)/dt[:,None]),
   measured_joint_velocity_rad_s=stats(np.abs(np.asarray([s['q_dot'][:16] for s in steps]))),
   minimum_position_error_mm=float(pos.min()),minimum_z_mm=float(min(q[:,18].min(),qend[18])*1000)))
 color=colors[e['episode']%10];style='-' if e['finish_reason']=='success' else '--'
 label=f"{row['episode']}: {row['outcome']}"
 axes[0,0].plot(times,ang,label=label,color=color,linestyle=style,lw=1.1,alpha=.85)
 axes[0,1].plot(times,pos,color=color,linestyle=style,lw=1.1,alpha=.85)
axes[0,0].axhline(np.degrees(2*np.arcsin(.2)),color='black',linestyle=':',label='Success threshold')
axes[0,1].axhline(20,color='black',linestyle=':')
axes[0,0].set(xlabel='Manipulation time (s)',ylabel='Orientation error (degrees)',title='Orientation convergence')
axes[0,1].set(xlabel='Manipulation time (s)',ylabel='Position error (mm)',title='Position retention',ylim=(0,210))
axes[0,0].legend(fontsize=8,ncol=2)
planning=np.asarray(planning);deltas=np.asarray(deltas);changes=np.asarray(changes);ages=np.asarray(ages)
axes[1,0].hist(planning*1000,bins=35,color='#4063a3',alpha=.85)
axes[1,0].axvline(64,color='#a32a2a',linestyle='--',label='Requested 64 ms')
axes[1,0].set(xlabel='Planning time (ms)',ylabel='Number of plans',title='Planning time')
axes[1,0].legend()
x=np.arange(1,len(rows)+1)
axes[1,1].bar(x,[r['fraction_steps_clipped']*100 for r in rows],label='Steps with any joint clipped',color='#4063a3')
axes[1,1].plot(x,[r['fraction_components_clipped']*100 for r in rows],'o-',label='Joint components clipped',color='#c17918')
axes[1,1].set(xlabel='Episode',ylabel='Percent',title='Final hardware target clipping at ±0.1 rad',xticks=x,ylim=(0,105))
axes[1,1].legend(fontsize=9)
for ax in axes.flat: ax.grid(alpha=.2)
fig.suptitle('Cube M2, high/high · 512 samples · noise 0.05 · temperature 0.1',fontsize=14)
fig.savefig(OUT/'analysis.png',dpi=160)
plt.close(fig)
success=[r for r in rows if r['outcome']=='success']
summary=dict(run_directory=str(RUN),controller_status=log['status'],launcher_status=launch['status'],launcher_reason=launch['reason'],
 requested_episodes=10,completed_episodes=len(rows),successes=len(success),failures=sum(r['outcome']=='failure' for r in rows),
 timeouts=sum(r['outcome']=='timeout' for r in rows),success_rate=len(success)/len(rows),
 success_time_s=stats([r['duration_s'] for r in success]),
 total_manipulation_s=sum(r['duration_s'] for r in rows),recorded_controller_wall_s=log['wall_time_s'],launcher_wall_s=launch['wall_time_s'],
 planning_ms=stats(planning*1000),fraction_plans_above_64ms=float(np.mean(planning>.064)),
 actual_hz=len(planning)/sum(r['duration_s'] for r in rows),
 fraction_steps_clipped=float(np.mean(((deltas < -lo-1e-6)|(deltas>hi+1e-6)).any(axis=1))),
 fraction_components_clipped=float(np.mean((deltas < -lo-1e-6)|(deltas>hi+1e-6))),
 estimated_target_change_deg=stats(np.degrees(np.abs(changes))),
 max_joint_target_change_per_update_deg=stats(np.degrees(np.abs(changes).max(axis=1))),
 robot_age_ms=stats(ages[:,0]*1000),pose_age_ms=stats(ages[:,1]*1000),
 logged_ctrl_range=limits,weights=log['weights'],params=log['planner_config'])
(OUT/'analysis.json').write_text(json.dumps(dict(summary=summary,episodes=rows,extra=extras),indent=2)+'\n')
with (OUT/'episodes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
for row in rows:print('EP',json.dumps(row))
