"""Offline analysis only: compare six hardware episodes with prior M2 baselines."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
NEW=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial314_shift512'
PREVIOUS=NEW.parent/'trial314'
OLD=ROOT/'results/friction_mass70g/hardware/cube_reorient/hand-high_obj-high/M2'
GROUPS={
    'old_010':[OLD/'leap_experiment_20261114_182513_421747/controller.json'],
    'old_020':[OLD/'leap_experiment_20261114_184641_023716/controller.json',
               OLD/'leap_experiment_20261114_184942_221848/controller.json'],
    'trial314_256_none':sorted(PREVIOUS.glob('leap_experiment_*/controller.json'))[-3:],
    'trial314_512_shift':sorted(NEW.glob('leap_experiment_*/controller.json'))[-3:],
}
assert len(GROUPS['trial314_512_shift'])==3
episodes=[];steps_all={};configs={};source_data={}


def finite_stats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not a.size:return None
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),
                maximum=float(a.max()),minimum=float(a.min()))


for group,paths in GROUPS.items():
    steps_all[group]=[];configs[group]=[];source_data[group]=[]
    for path in paths:
        d=json.loads(path.read_text());source_data[group].append(d)
        launch=json.loads((path.parent/'launch.json').read_text()) if (path.parent/'launch.json').exists() else {}
        configs[group].append(dict(path=str(path),run_status=d['status'],args=d['args'],
            weights=d['weights'],planner=d['planner_config'],rollout=d['rollout_config'],
            hardware=d['hardware_config'],joint_reference=d['joint_reference'],
            launch_status=launch.get('status'),launch_reason=launch.get('reason'),exit_codes=launch.get('exit_codes')))
        for e in d['episodes']:
            if not e.get('completed') or e['finish_reason'] not in ['success','failure','timeout']:continue
            steps=[s for s in e['steps'] if s.get('action_published')]
            steps_all[group].extend(steps)
            q=np.asarray([s['q'] for s in steps]+[e['q_end']],float)
            t=np.asarray([s['elapsed_s'] for s in steps]+[e['manipulation_wall_time_s']],float)
            target=np.asarray(e['target_pos']);pos=q[:,16:19];diff=pos-target
            angles=np.degrees([s['quaternion_error_rad'] for s in steps]+[e['quaternion_error_rad']])
            mask=pos[:,2]>=e['fallen_z'];mask40=pos[:,2]>=.04
            raw=np.asarray([s['action_delta'] for s in steps],float)
            bound=d['hardware_config']['ctrl_range'];down=np.asarray(bound['lower']);up=np.asarray(bound['upper'])
            clipped=(raw < -down-1e-6)|(raw > up+1e-6)
            clipped_steps=clipped.any(axis=1)
            plans=[s['planning_time'] for s in steps]
            crosses=np.flatnonzero(~mask)
            start=steps[0]
            row=dict(group=group,run=path.parent.name,episode=e['episode']+1,
                outcome=e['finish_reason'],steps=e['n_steps'],wall_s=e['manipulation_wall_time_s'],
                loop_hz=e['actual_control_frequency_hz'],start_position_mm=(pos[0]*1000).tolist(),
                start_angle_deg=float(angles[0]),start_position_error_mm=float(np.linalg.norm(diff[0])*1000),
                final_angle_deg=math.degrees(e['quaternion_error_rad']),
                final_position_error_mm=e['goal_errors']['pos']*1000,
                best_angle_above_drop_deg=float(angles[mask].min()) if mask.any() else None,
                best_angle_above_40mm_deg=float(angles[mask40].min()) if mask40.any() else None,
                angle_threshold_ever_above_drop=bool(np.any((angles < math.degrees(2*math.asin(.2))) & mask)),
                median_abs_xyz_error_above_drop_mm=(np.median(np.abs(diff[mask]),axis=0)*1000).tolist() if mask.any() else None,
                max_lateral_error_above_drop_mm=float(np.linalg.norm(diff[mask,:2],axis=1).max()*1000) if mask.any() else None,
                median_lateral_error_above_drop_mm=float(np.median(np.linalg.norm(diff[mask,:2],axis=1))*1000) if mask.any() else None,
                first_below_drop_s=float(t[crosses[0]]) if crosses.size else None,
                position_at_first_below_drop_mm=(diff[crosses[0]]*1000).tolist() if crosses.size else None,
                published_steps=len(steps),clipped_steps=int(clipped_steps.sum()),clipped_joint_entries=int(clipped.sum()),
                clipped_step_fraction=float(clipped_steps.mean()),clipped_joint_fraction=float(clipped.mean()),
                mean_planning_ms=float(np.mean(plans)*1000),p95_planning_ms=float(np.percentile(plans,95)*1000),
                max_planning_ms=float(max(plans)*1000),
                raw_delta_max_abs=float(np.abs(raw).max()),raw_delta_p95_abs=float(np.percentile(np.abs(raw),95)))
            episodes.append(row)

summary={}
for group in GROUPS:
    eps=[e for e in episodes if e['group']==group];ss=steps_all[group]
    raw=np.asarray([s['action_delta'] for s in ss]);bound=configs[group][0]['hardware']['ctrl_range']
    hits=(raw < -np.asarray(bound['lower'])-1e-6)|(raw > np.asarray(bound['upper'])+1e-6)
    qerr=np.asarray([np.asarray(s['u'])-s['q'][:16] for s in ss])
    fresh_offset=np.asarray([np.asarray(s['u'])-s['feedback']['command_qpos'] for s in ss])
    requested=[np.asarray(s['requested_target']) for s in ss]
    commands=[np.asarray(s['u']) for s in ss]
    actual_changes=np.concatenate([np.abs(np.diff(np.asarray([s['u'] for s in e['steps']]),axis=0)).ravel()
        for d in source_data[group] for e in d['episodes'] if e.get('completed') and len(e['steps'])>1])
    startangles=[e['start_angle_deg'] for e in eps]
    summary[group]=dict(completed=len(eps),successes=sum(e['outcome']=='success' for e in eps),
        drops=sum(e['outcome']=='failure' for e in eps),timeouts=sum(e['outcome']=='timeout' for e in eps),
        published_steps=len(ss),start_angle_deg=finite_stats(startangles),
        best_angle_above_drop_deg=finite_stats([e['best_angle_above_drop_deg'] for e in eps]),
        episodes_orientation_threshold_reached_above_drop=sum(e['angle_threshold_ever_above_drop'] for e in eps),
        dropped_episode_lifetime_s=finite_stats([e['wall_s'] for e in eps if e['outcome']=='failure']),
        median_lateral_error_above_drop_mm=finite_stats([e['median_lateral_error_above_drop_mm'] for e in eps]),
        clipped_steps=int(hits.any(axis=1).sum()),clipped_joint_entries=int(hits.sum()),
        clipped_step_fraction=float(hits.any(axis=1).mean()),clipped_joint_fraction=float(hits.mean()),
        raw_delta_abs=finite_stats(np.abs(raw).ravel()),
        clipped_excess_rad=finite_stats((np.abs(raw)-np.minimum(np.abs(raw),np.asarray(bound['upper'])))[hits]),
        clipping_by_joint=[dict(index=i,count=int(hits[:,i].sum()),fraction=float(hits[:,i].mean())) for i in range(16)],
        command_change_abs_rad=finite_stats(actual_changes),
        command_minus_measured_abs=finite_stats(np.abs(qerr).ravel()),
        estimated_target_offset_from_fresh_joint_state_rad=finite_stats(np.abs(fresh_offset).ravel()),
        planning_ms=finite_stats([s['planning_time']*1000 for s in ss]),
        step_wall_ms=finite_stats([s['wall_time_s']*1000 for s in ss]),
        aggregate_frequency_hz=sum(e['steps'] for e in eps)/sum(e['wall_s'] for e in eps),
        robot_age_ms=finite_stats([s['feedback']['robot_age']*1000 for s in ss]),
        pose_age_ms=finite_stats([s['feedback']['pose_age']*1000 for s in ss]),
        command_robot_age_ms=finite_stats([s['feedback']['command_state']['robot_age']*1000 for s in ss]),
        command_pose_age_ms=finite_stats([s['feedback']['command_state']['pose_age']*1000 for s in ss]),
        estimated_clipped_command_entries=int(sum(np.count_nonzero(np.abs(a-b)>1e-6) for a,b in zip(requested,commands))))
    summary[group]['success_rate']=summary[group]['successes']/len(eps)

new=source_data['trial314_512_shift'][0];old=source_data['old_020'][0]
same={k:new.get(k)==old.get(k) for k in ['joint_reference','success_thresholds','failure_z','task_contact_params','control_policy','action_representation','plan_base_source']}
scene_same={k:new['scenes'][k]['scene_xml']==old['scenes'][k]['scene_xml'] for k in ['rollout','eval']}
changed_args={k:[old['args'].get(k),new['args'].get(k)] for k in ['n_samples','warm_start','time_horizon','noise_sigma','temperature','ctrl_time_step','action_rate_weight'] if old['args'].get(k)!=new['args'].get(k)}
report=dict(summary=summary,episodes=episodes,configs=configs,matched_fields=same,
    identical_generated_scene_xml=scene_same,changed_planner_args=changed_args,
    caveats=['Descriptive comparison with small sample counts and manually placed starting poses.',
        'Compared with the previous trial314 group, sample count and warm start changed together; effective horizon is unchanged.',
        'Clipping is inferred from requested controller deltas and logged bounds; motor-side clipping was not independently measured.',
        'Fresh tracking timestamps do not independently validate tracking accuracy.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
with (OUT/'episodes.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(episodes[0]));writer.writeheader();writer.writerows(episodes)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
colors=plt.get_cmap('tab10').colors
for group,paths in GROUPS.items():
    if group not in ['trial314_256_none','trial314_512_shift']:continue
    n=0
    for path in paths:
        d=json.loads(path.read_text())
        for e in d['episodes']:
            if not e.get('completed'):continue
            ss=e['steps'];t=np.array([s['elapsed_s'] for s in ss]+[e['manipulation_wall_time_s']]);q=np.array([s['q'][16:19] for s in ss]+[e['q_end'][16:19]]);a=np.degrees([s['quaternion_error_rad'] for s in ss]+[e['quaternion_error_rad']]);target=np.asarray(e['target_pos'])
            n+=1;isnew=group=='trial314_512_shift';color=colors[n-1] if isnew else '#999999'
            label=f'512 / shift E{n}: {e["finish_reason"]}' if isnew else ('Previous 256 / none (6 episodes)' if n==1 else None)
            kw=dict(color=color,lw=1.4 if isnew else .9,alpha=.9 if isnew else .45,ls='-' if isnew else ':')
            axs[0].plot(t,a,label=label,**kw);axs[1].plot(t,q[:,2]*1000,**kw);axs[2].plot(t,np.linalg.norm(q[:,:2]-target[:2],axis=1)*1000,**kw)
axs[0].axhline(math.degrees(2*math.asin(.2)),color='black',ls='--',lw=.8)
axs[0].set(ylabel='Orientation error (deg)',ylim=(0,185),title='Trial 314: 512 samples / shift vs 256 samples / none (both ±0.2 rad)')
axs[0].legend(ncol=2,fontsize=8)
axs[1].axhline(80,color='black',ls='--',lw=.8,label='Target: 80 mm');axs[1].axhline(65,color='#b22222',ls='--',lw=.8,label='Drop-cost threshold: 65 mm');axs[1].axhline(0,color='black',lw=.8)
axs[1].set(ylabel='Object height (mm)');axs[1].legend(fontsize=8)
axs[2].axhline(20,color='black',ls='--',lw=.8);axs[2].set(ylabel='Horizontal position error (mm)',xlabel='Manipulation wall time (s)')
for ax in axs:ax.grid(alpha=.2)
for ext in ['png','pdf']:fig.savefig(OUT/f'trajectories.{ext}',dpi=180)
plt.close(fig)
print('Saved', OUT/'report.json')
