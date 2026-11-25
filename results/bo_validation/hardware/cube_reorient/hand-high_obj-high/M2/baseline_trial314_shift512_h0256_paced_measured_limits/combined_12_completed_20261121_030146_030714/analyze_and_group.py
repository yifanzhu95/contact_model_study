"""Reproduce this completed-only aggregate from immutable recorded data.

No hardware, tracking, simulator, planner, or video encoder is imported/run.
Videos are hard-linked (copy fallback), without reencoding. No files are deleted.
"""
import copy
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=next(p for p in OUT.parents if (p/'run_leap_hardware.py').is_file())
NAMES=['leap_experiment_20261121_030146_687194','leap_experiment_20261121_030714_425660']
ANALYSIS=OUT/'analysis'
EPS=1e-7


def stats(values):
    a=np.asarray(values,dtype=float).reshape(-1)
    if not a.size:return {'n':0}
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
                min=float(a.min()),p90=float(np.percentile(a,90)),
                p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=float(a.max()))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def normalized_xml(text):
    root=ET.fromstring(text)
    for mesh in root.findall('.//mesh'):mesh.attrib.pop('content_type',None)
    return ET.tostring(root)


def orientation_group(q,goal):
    current=Rotation.from_quat(np.asarray(q)[[1,2,3,0]])
    desired=Rotation.from_quat(np.asarray(goal)[[1,2,3,0]])
    vector=(current*desired.inv()).as_rotvec()
    axis=int(np.argmax(abs(vector)))
    return ('+' if vector[axis]>=0 else '-')+'XYZ'[axis]


def clip_stats(steps,doc):
    raw=np.array([s['requested_target'] for s in steps])
    stage1=np.array([s['joint_limit_clipping']['delta_clipped_target'] for s in steps])
    final=np.array([s['u'] for s in steps])
    base=np.array([s['feedback']['command_qpos'] for s in steps])
    low,high=np.array(doc['joint_limits']['lower']),np.array(doc['joint_limits']['upper'])
    down,up=np.array(doc['effective_ctrl_range']['lower']),np.array(doc['effective_ctrl_range']['upper'])
    np.testing.assert_allclose(stage1,np.clip(raw,base-down,base+up),rtol=0,atol=1e-12)
    np.testing.assert_allclose(final,np.clip(stage1,low,high),rtol=0,atol=1e-12)
    c1,c2=abs(stage1-raw)>EPS,abs(final-stage1)>EPS
    q=np.array([s['q'][:16] for s in steps]);command_q=np.array([s['feedback']['command_qpos'] for s in steps])
    violation=np.maximum(low-q,0)+np.maximum(q-high,0)
    return dict(actions=len(steps),delta_clipped_actions=int(c1.any(axis=1).sum()),
        delta_clipped_percent=float(c1.any(axis=1).mean()*100),
        absolute_clipped_actions=int(c2.any(axis=1).sum()),absolute_clipped_percent=float(c2.any(axis=1).mean()*100),
        either_clipped_actions=int((c1|c2).any(axis=1).sum()),either_clipped_percent=float((c1|c2).any(axis=1).mean()*100),
        delta_clipped_joint_percent=float(c1.mean()*100),absolute_clipped_joint_percent=float(c2.mean()*100),
        final_targets_outside_absolute_limits=int(((final<low-EPS)|(final>high+EPS)).sum()),
        final_changes_exceed_delta_bound=int(((final-base>up+EPS)|(final-base < -down-EPS)).any(axis=1).sum()),
        absolute_correction_l2_rad=stats(np.linalg.norm(final-stage1,axis=1)),
        measured_joint_violation_percent=float((violation>EPS).mean()*100),
        measured_samples_any_violation_percent=float((violation>EPS).any(axis=1).mean()*100),
        joint_motion_during_planning_max_per_action_deg=stats(np.degrees(abs(command_q-q).max(axis=1))),
        joints=[dict(joint=name,delta_clipped_percent=float(c1[:,j].mean()*100),
            absolute_clipped_percent=float(c2[:,j].mean()*100),absolute_clipped_count=int(c2[:,j].sum()),
            lower_clip_count=int((c2[:,j] & (stage1[:,j]<low[j])).sum()),
            upper_clip_count=int((c2[:,j] & (stage1[:,j]>high[j])).sum()),
            measured_violation_percent=float((violation[:,j]>EPS).mean()*100),
            measured_violation_over_0p05rad_percent=float((violation[:,j]>.05).mean()*100),
            measured_violation_max_rad=float(violation[:,j].max())) for j,name in enumerate(doc['joint_order'])])


def metrics(episodes,doc):
    steps=[s for e in episodes for s in e['steps'] if s.get('action_published')]
    plans=np.array([s['planning_time'] for s in steps]);wait=np.array([s.get('pacing_wait_s',0) for s in steps])
    intervals=[];target_changes=[];delta_changes=[];est_tracking=[]
    for e in episodes:
        ss=[s for s in e['steps'] if s.get('action_published')]
        intervals.extend(np.diff([s['action_timestamp_us'] for s in ss])*1e-6)
        target_changes.extend(np.max(abs(np.diff([s['u'] for s in ss],axis=0)),axis=1))
        delta_changes.extend(np.linalg.norm(np.diff([s['action_delta'] for s in ss],axis=0),axis=1))
        if len(ss)>1:
            est_tracking.extend(abs(np.array([s['q'][:16] for s in ss[1:]])-np.array([s['u'] for s in ss[:-1]])))
    intervals=np.array(intervals)
    duration=sum(e['manipulation_wall_time_s'] for e in episodes)
    blends=[s['plan_blending'] for s in steps];overlaps=[b for b in blends if b['status']=='overlap']
    for b in overlaps:
        np.testing.assert_allclose(b['blended_delta'],b['alpha']*np.array(b['unblended_delta'])+
                                   (1-b['alpha'])*np.array(b['old_aligned_delta']),rtol=0,atol=1e-10)
    outcomes=Counter(e['finish_reason'] for e in episodes)
    return dict(episodes=len(episodes),successes=outcomes['success'],drops=outcomes['failure'],timeouts=outcomes['timeout'],
        success_rate=outcomes['success']/len(episodes),actions=len(steps),manipulation_duration_s=duration,
        successful_episode_duration_s=stats([e['manipulation_wall_time_s'] for e in episodes if e['success']]),
        planning_ms=stats(plans*1000),planning_exceeded_64ms=int((plans>.064).sum()),
        planning_exceeded_64ms_percent=float((plans>.064).mean()*100),
        pacing_wait_steps=int((wait>0).sum()),pacing_wait_ms_all=stats(wait*1000),
        pacing_wait_ms_when_waiting=stats(wait[wait>0]*1000),pacing_wait_total_s=float(wait.sum()),
        publish_intervals_ms=stats(intervals*1000),instantaneous_frequency_hz=stats(1/intervals),
        publication_rate_hz=float(len(intervals)/intervals.sum()),
        episode_actions_per_second=float(len(steps)/duration),
        cycle_wall_time_ms=stats([s['wall_time_s']*1000 for s in steps]),
        command_joint_age_ms=stats([s['feedback']['command_state']['robot_age']*1000 for s in steps]),
        command_object_age_ms=stats([s['feedback']['command_state']['pose_age']*1000 for s in steps]),
        plan_start_joint_age_ms=stats([s['feedback']['robot_age']*1000 for s in steps]),
        plan_start_object_age_ms=stats([s['feedback']['pose_age']*1000 for s in steps]),
        original_object_age_at_plan_end_ms=stats([(s['feedback']['pose_age']+s['planning_time'])*1000 for s in steps]),
        estimated_target_change_max_joint_deg=stats(np.degrees(target_changes)),
        consecutive_transmitted_delta_change_l2_rad=stats(delta_changes),
        estimated_previous_target_tracking_gap_deg=stats(np.degrees(est_tracking)),
        clipping=clip_stats(steps,doc),
        blending=dict(status_counts=dict(Counter(b['status'] for b in blends)),
            old_slot_counts=dict(Counter(b['old_slot'] for b in overlaps)),
            compute_ms=stats([b['compute_s']*1000 for b in blends]),
            aligned_disagreement_l2_rad=stats([np.linalg.norm(b['first_aligned_disagreement']) for b in overlaps]),
            adjustment_l2_rad=stats([b['blend_adjustment_l2'] for b in overlaps]),
            within_plan_difference_l2_rad=stats([v for b in blends for v in b['within_plan_diff_l2']]),
            raw_delta_l2_rad=stats([np.linalg.norm(b['unblended_delta']) for b in blends]),
            blended_delta_l2_rad=stats([np.linalg.norm(b['blended_delta']) for b in blends]),
            same_state_unblended_any_clipping_percent=float(np.mean([b['estimated_unblended_clipped_joints']>0 for b in blends])*100),
            same_state_blended_any_clipping_percent=float(np.mean([b['estimated_blended_clipped_joints']>0 for b in blends])*100),
            same_state_unblended_removed_l2_rad=stats([b['estimated_unblended_clip_removed_l2'] for b in blends]),
            same_state_blended_removed_l2_rad=stats([b['estimated_blended_clip_removed_l2'] for b in blends])))


def episode_analysis(e,doc):
    q=np.array([s['q'] for s in e['steps']]+[e['q_end']]);pos=q[:,16:19]-e['target_pos']
    times=np.array([s['elapsed_s'] for s in e['steps']]+[e['manipulation_wall_time_s']])
    angle=np.degrees([s['quaternion_error_rad'] for s in e['steps']]+[e['quaternion_error_rad']])
    errors=[s['goal_errors'] for s in e['steps']]+[e['goal_errors']]
    position=np.linalg.norm(pos,axis=1)*1000
    in_hand=q[:,18]>=e['fallen_z']
    aok=np.array([r['quat']<doc['success_thresholds']['quat'] for r in errors])
    pok=position<doc['success_thresholds']['pos']*1000
    valid=np.flatnonzero(in_hand);best=valid[np.argmin(angle[valid])] if valid.size else int(np.argmin(angle))
    cs=clip_stats(e['steps'],doc)
    row=dict(episode=e['episode']+1,source_run=e['source_run'],source_episode=e['source_episode'],outcome=e['finish_reason'],
        steps=e['n_steps'],duration_s=e['manipulation_wall_time_s'],
        rotation=orientation_group(q[0,19:23],e['goal']),initial_angle_deg=float(angle[0]),initial_position_error_mm=float(position[0]),
        final_angle_deg=float(angle[-1]),final_position_error_mm=float(position[-1]),
        best_angle_before_below_cost_drop_height_deg=float(angle[best]),
        position_at_best_supported_orientation_mm=float(position[best]),
        x_at_best_supported_orientation_mm=float(pos[best,0]*1000),
        y_at_best_supported_orientation_mm=float(pos[best,1]*1000),
        z_at_best_supported_orientation_mm=float(pos[best,2]*1000),
        time_at_best_supported_orientation_s=float(times[best]),
        ever_orientation_ok_while_above_drop_height=bool(np.any(aok & in_hand)),
        best_position_when_orientation_ok_mm=float(position[aok & in_hand].min()) if np.any(aok & in_hand) else None,
        both_pose_thresholds_reached=bool(np.any(aok & pok)),
        delta_clipped_percent=cs['delta_clipped_percent'],absolute_clipped_percent=cs['absolute_clipped_percent'],
        either_clipped_percent=cs['either_clipped_percent'],
        mean_plan_ms=float(np.mean([s['planning_time'] for s in e['steps']])*1000))
    return row,dict(time_s=times.tolist(),angle_deg=angle.tolist(),position_mm=position.tolist(),
                    position_axes_mm=(pos*1000).tolist())


def main():
    ANALYSIS.mkdir(exist_ok=True);(OUT/'video').mkdir(exist_ok=True);(OUT/'source_runs').mkdir(exist_ok=True)
    docs=[json.loads((BASE/name/'controller.json').read_text()) for name in NAMES]
    keys=['weights','hardware_config','joint_reference','planner_config','rollout_config','pose_transform',
          'success_thresholds','effective_ctrl_range','joint_limits','temporal_plan_blending','failure_z','fixed_goal_quat']
    checks={key:docs[0][key]==docs[1][key] for key in keys}
    checks.update({role+'_scene_equal_except_mesh_mime':normalized_xml(docs[0]['scenes'][role]['scene_xml'])==
        normalized_xml(docs[1]['scenes'][role]['scene_xml']) for role in ['rollout','eval']})
    assert all(checks.values()),checks
    episodes=[];sources=[];videos=[];excluded=[]
    for source_index,(name,doc) in enumerate(zip(NAMES,docs),1):
        source=BASE/name;target=OUT/'source_runs'/f'source_{source_index:02d}';target.mkdir(exist_ok=True)
        log_hashes={}
        for filename in ['controller.json','launch.json','controller.log','bridge.log','mujoco.log','video.log',
                         'episodes.csv','steps.csv','episode_status.json','bridge_status.json','goal.json',
                         'preparation_trace.jsonl','summary.json']:
            p=source/filename
            if p.is_file():shutil.copy2(p,target/filename);log_hashes[filename]=sha(p)
        recording=json.loads((source/'video/recording.json').read_text())
        write_json(target/'recording.original.json',recording)
        mapping={}
        for episode in doc['episodes']:
            if not episode.get('completed'):
                excluded.append(dict(source_run=name,source_episode=episode['episode']+1,reason=episode['finish_reason']));continue
            assert len(episode['steps'])==episode['n_steps'] and episode['finish_reason'] in ['success','failure','timeout']
            new=copy.deepcopy(episode);number=len(episodes)+1
            new.update(episode=number-1,source_run=name,source_episode=episode['episode']+1)
            assert new['success']==(new['finish_reason']=='success')
            episodes.append(new);mapping[new['source_episode']]=number
            records=[r for r in recording['episodes'] if r['episode']==new['source_episode']]
            assert len(records)==1
            record=records[0];assert record['outcome']==new['finish_reason'] and record['error'] is None
            item=dict(episode=number,source_run=name,source_episode=new['source_episode'],outcome=new['finish_reason'],
                      recording=copy.deepcopy(record),files=[])
            for suffix in ['mkv','frames.csv','ffmpeg.log']:
                old=source/'video'/f"episode_{new['source_episode']:02d}.{suffix}"
                new_path=OUT/'video'/f'episode_{number:02d}.{suffix}'
                if not new_path.exists():
                    try:os.link(old,new_path)
                    except OSError:shutil.copy2(old,new_path)
                digest=sha(old);assert sha(new_path)==digest
                item['files'].append(dict(source=str(old.relative_to(ROOT)),path=str(new_path.relative_to(OUT)),
                    bytes=new_path.stat().st_size,sha256=digest,hardlinked=os.path.samefile(old,new_path)))
            item['recording'].update(episode=number,source_episode=new['source_episode'],
                video=f'episode_{number:02d}.mkv',timestamps=f'episode_{number:02d}.frames.csv')
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height',
                '-of','json',str(OUT/'video'/f'episode_{number:02d}.mkv')],capture_output=True,text=True,check=True)
            item['probe']=json.loads(probe.stdout);assert item['probe']['streams']
            assert abs(float(item['probe']['format']['duration'])-record['video_duration_s'])<.15
            videos.append(item)
        for record in recording['episodes']:
            if record['episode'] not in mapping:
                excluded.append(dict(source_run=name,source_episode=record['episode'],reason=record['outcome'],
                    source='video/recording.json',note='No completed controller episode; video excluded from aggregate.'))
        launch=json.loads((source/'launch.json').read_text())
        sources.append(dict(run=name,archived_logs=str(target.relative_to(OUT)),selected_episodes=mapping,source_hashes=log_hashes,
            controller_status=doc['status'],launch_reason=launch['reason'],exit_codes=launch['exit_codes'],
            torque_off_verified='Torque disabled and verified on all 16 motors.' in (source/'bridge.log').read_text()))
    assert len(episodes)==12 and [len(s['selected_episodes']) for s in sources]==[9,3]
    aggregate=copy.deepcopy(docs[0]);aggregate.pop('summary',None)
    aggregate.update(status='complete',record_type='completed_episode_aggregate',episodes=episodes,
        source_runs=[s['run'] for s in sources],started_at_utc=episodes[0]['started_at_utc'],
        finished_at_utc=episodes[-1]['finished_at_utc'],
        wall_time_s=sum(e['manipulation_wall_time_s'] for e in episodes),
        aggregation_note='Derived completed-only view; do not count again alongside source runs. Wall time is completed manipulation time, excluding gates, preparation, resets, and aborted attempts.')
    aggregate['args'].update(n_episodes=12,results=str(OUT/'controller.json'))
    all_metrics=metrics(episodes,aggregate)
    rows=[];curves=[]
    for e in episodes:
        row,curve=episode_analysis(e,aggregate);rows.append(row);curves.append(curve)
    aggregate['summary']=dict(requested_episodes=12,completed_episodes=12,successful_episodes=all_metrics['successes'],
        unsuccessful_episodes=12-all_metrics['successes'],aborted_episodes=0,
        success_rate_completed=all_metrics['success_rate'],wall_time_s=aggregate['wall_time_s'],
        completed_steps=all_metrics['actions'])
    write_json(OUT/'controller.json',aggregate);write_json(OUT/'summary.json',aggregate['summary'])
    write_csv(OUT/'episodes.csv',rows)
    write_csv(OUT/'steps.csv',[dict(episode=e['episode']+1,source_run=e['source_run'],source_episode=e['source_episode'],
        step=s['step'],action_timestamp_us=s['action_timestamp_us'],planning_time_s=s['planning_time'],
        pacing_wait_s=s.get('pacing_wait_s',0),position_error_m=s['goal_errors']['pos'],
        quaternion_error=s['goal_errors']['quat'],orientation_error_deg=float(np.degrees(s['quaternion_error_rad'])),
        estimated_joint_limit_clipped_joints=s['joint_limit_clipping']['estimated_clipped_joints'])
        for e in episodes for s in e['steps']])
    write_json(OUT/'video/recording.json',dict(status='complete',mode='completed_episode_aggregate',episodes=[v['recording'] for v in videos],
        total_video_duration_s=sum(v['recording']['video_duration_s'] for v in videos),
        total_source_frames=sum(v['recording']['source_frames_written'] for v in videos),
        total_video_frames=sum(v['recording']['video_frames'] for v in videos),
        queue_dropped_frames=sum(v['recording']['queue_dropped_frames'] for v in videos),
        codec='mjpeg',container='matroska',fps=30,reencoding=False))
    provenance=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        equivalence_checks=checks,excluded_episodes=excluded,videos=videos,
        note='Raw source logs preserved, videos linked without reencoding; source episode IDs retained. Never sum this aggregate and its sources.')
    write_json(OUT/'sources.json',provenance)
    comparison=[]
    for path in sorted(BASE.glob('leap_experiment_*/controller.json')):
        if path.parent.name>=NAMES[0]:continue
        d=json.loads(path.read_text());es=[e for e in d.get('episodes',[]) if e.get('completed')]
        if es:
            comparison.append(dict(run=path.parent.name,completed=len(es),successes=sum(e['success'] for e in es),
                outcomes=dict(Counter(e['finish_reason'] for e in es)),rate_weight=d['args']['action_rate_weight'],
                blend_alpha=d['args']['plan_blend_alpha'],w_pos_y=d['weights']['w_pos_y'],w_contact=d['weights']['w_contact'],
                measured_limits_sha256=d['joint_limits']['model_sha256']))
    supported_misses=[r['episode'] for r in rows if r['outcome']!='success' and r['ever_orientation_ok_while_above_drop_height']]
    successes=metrics([e for e in episodes if e['success']],aggregate)
    failures=metrics([e for e in episodes if not e['success']],aggregate)
    rotation_counts=[]
    for rotation in sorted(set(r['rotation'] for r in rows)):
        subset=[r for r in rows if r['rotation']==rotation]
        rotation_counts.append(dict(rotation=rotation,episodes=len(subset),successes=sum(r['outcome']=='success' for r in subset)))
    usb_observation=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),latency_timers_ms={
        str(p):int(p.read_text().strip()) for p in Path('/sys/bus/usb-serial/devices').glob('*/latency_timer')},
        note='Read-only current inspection, not a saved observation during these episodes. No setting changed.')
    report=dict(summary=all_metrics,successes=successes,failures=failures,episodes=rows,
        current_usb_observation=usb_observation,
        successful_final_angle_deg=stats([r['final_angle_deg'] for r in rows if r['outcome']=='success']),
        successful_final_position_mm=stats([r['final_position_error_mm'] for r in rows if r['outcome']=='success']),
        source_runs=sources,configuration_equivalence=checks,previous_measured_limit_runs=comparison,
        starting_rotation_groups=rotation_counts,failed_episodes_that_reached_orientation_while_supported=supported_misses,
        configuration=dict(rollout_model=aggregate['rollout_model'],weights=aggregate['weights'],
            planner=aggregate['planner_config'],action_rate=aggregate['action_rate_cost'],
            temporal_blending=aggregate['temporal_plan_blending'],joint_reference=aggregate['joint_reference'],
            joint_limits=aggregate['joint_limits'],ctrl_range=aggregate['hardware_config']['ctrl_range'],
            goal_pos_m=episodes[0]['target_pos'],cost_drop_height_m=episodes[0]['fallen_z'],
            failure_height_m=aggregate['failure_z'],success_thresholds=aggregate['success_thresholds']),
        definitions=dict(control_frequency='Inverse of within-episode publication intervals; mean rate is interval count / total interval time.',
            clipping='Controller estimates, validated as ctrl_range then saved absolute joint limits. Exact bridge targets/current unavailable: telemetry was off.',
            supported='Tracked object center at/above the cost drop height 0.065m; this is a geometric filter, not proof of physical contact.',
            joint_reference='P011 hardware joint-cost reference; preparation remains robot_qref.',
            angle='2 acos(abs(dot(q,goal))), degrees; quaternion metric 0.04 equals 23.0739 deg.'),
        limitations=['Small unpaired sample, with six starting rotation directions repeated twice. Historical differences do not establish causation.',
            'Action-rate 10, blending .75, w_pos_y=1 and w_contact=80 were all active, so this is not pure trial314 baseline.',
            'No exact bridge target or current trace; clipping and target continuity are controller estimates.',
            'Drop/failure is based on tracked object crossing z=0; videos are included for physical verification.',
            'First source had a later uncompleted episode 10 recorded on video and in controller.log, but absent from last controller.json checkpoint; it is excluded.',
            'No dynamics simulation, hardware motion, tracking restart or production parameter changes were performed.'])
    write_json(ANALYSIS/'report.json',report)
    write_csv(ANALYSIS/'joint_clipping.csv',all_metrics['clipping']['joints'])
    write_csv(ANALYSIS/'previous_runs.csv',comparison)
    threshold=np.degrees(2*np.arcsin(np.sqrt(aggregate['success_thresholds']['quat'])))
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
    colors=plt.get_cmap('tab20').colors
    for row,curve in zip(rows,curves):
        label=f"{row['episode']}: {row['outcome']}"
        axes[0].plot(curve['time_s'],curve['angle_deg'],color=colors[row['episode']-1],label=label)
        axes[1].plot(curve['time_s'],curve['position_mm'],color=colors[row['episode']-1])
    axes[0].axhline(threshold,color='black',ls='--',label=f'{threshold:.1f} deg success threshold')
    axes[1].axhline(20,color='black',ls='--');axes[0].set_ylabel('Orientation error (deg)')
    axes[1].set(ylabel='Position error (mm)',xlabel='Manipulation time (s)',ylim=(0,100))
    axes[0].legend(ncol=4,fontsize=8);axes[0].set_title('Latest 12 completed M2 episodes — failures exceed position plot scale after falling')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(ANALYSIS/'episode_errors.png',dpi=170);fig.savefig(ANALYSIS/'episode_errors.pdf');plt.close(fig)
    steps=[s for e in episodes for s in e['steps']]
    plans=np.array([s['planning_time'] for s in steps])*1000
    intervals=np.concatenate([np.diff([s['action_timestamp_us'] for s in e['steps']])*.001 for e in episodes])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].hist(plans,bins=40,color='#367ca0');axes[0].axvline(64,color='black',ls='--')
    axes[1].hist(intervals,bins=40,color='#418a66');axes[1].axvline(64,color='black',ls='--')
    axes[0].set(xlabel='MPPI planning time (ms)',ylabel='Actions',title='Planning time')
    axes[1].set(xlabel='Target publication interval (ms)',ylabel='Intervals',title='Paced execution, nominal 64 ms')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(ANALYSIS/'timing.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    joints=all_metrics['clipping']['joints'];x=np.arange(16)
    axes[0].barh(x,[j['absolute_clipped_percent'] for j in joints],color='#b76b46')
    axes[0].set_yticks(x,aggregate['joint_order']);axes[0].set(xlabel='Actions clipped at this joint (%)',title='Absolute joint-limit clipping')
    axes[1].barh(x,[j['measured_violation_percent'] for j in joints],color='#6b78a8')
    axes[1].set_yticks(x,aggregate['joint_order']);axes[1].set(xlabel='Planning measurements outside saved bounds (%)',title='Measured joint limits (zero tolerance)')
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.savefig(ANALYSIS/'joint_limits.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    labels=[r['rotation'] for r in rotation_counts];success=np.array([r['successes'] for r in rotation_counts])
    ax.bar(labels,success,color='#418a66',label='Success');ax.bar(labels,2-success,bottom=success,color='#be6958',label='Logged drop')
    ax.set(ylabel='Episodes',yticks=[0,1,2],title='Starting rotation direction — two approximately 90-degree trials each')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(ANALYSIS/'starting_rotation.png',dpi=170);plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>M2 latest 12 completed episodes</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}video{width:480px;max-width:100%}</style>',
        '<h1>M2 — 12 completed episodes (9 + 3)</h1>',
        f"<p>{all_metrics['successes']}/12 successes ({100*all_metrics['success_rate']:.1f}%), {all_metrics['drops']} logged drops, {all_metrics['timeouts']} timeouts. Aborted attempts excluded.</p>",
        '<p>Paced, 512 samples, 0.256 s horizon, 0.064 s control step, temperature 0.1, noise 0.05, shifted warm start. Action-rate weight 10, blending alpha 0.75, joint-limit penalty 100, hardware clipping +/-0.2 followed by measured bounds. Trial314-derived weights with w_pos_y=1.0 and w_contact=80.0.</p>',
        '<p><a href="controller.json">Combined full controller data</a> · <a href="summary.json">Summary</a> · <a href="episodes.csv">Episode metrics CSV</a> · <a href="sources.json">Provenance/video hashes</a> · <a href="analysis/report.json">Full analysis</a></p>',
        f"<p>Planning mean/median/p95/max: {all_metrics['planning_ms']['mean']:.2f} / {all_metrics['planning_ms']['median']:.2f} / {all_metrics['planning_ms']['p95']:.2f} / {all_metrics['planning_ms']['max']:.2f} ms. Effective publication rate {all_metrics['publication_rate_hz']:.2f} Hz; instantaneous range {all_metrics['instantaneous_frequency_hz']['min']:.2f}–{all_metrics['instantaneous_frequency_hz']['max']:.2f} Hz. {all_metrics['planning_exceeded_64ms']}/{all_metrics['actions']} plans exceeded 64 ms. Delta clipping {all_metrics['clipping']['delta_clipped_percent']:.1f}%; absolute clipping {all_metrics['clipping']['absolute_clipped_percent']:.1f}%.</p>",
        '<p>Both +Z starts failed: they reached the orientation threshold while drifting in +Y. At their best supported orientations the lateral offsets were +75.3 and +43.7 mm, with Z errors +7.2 and +1.4 mm. This repeated pattern points to lateral retention, rather than height alone. Other failed starts did not reach the orientation threshold.</p>',
        '<p>The USB latency_timer is currently 16 ms. Logged joint-feedback age at command averaged 16.3 ms (max 32.1 ms), compared with about 1.5 ms in the earlier 1-ms-latency analysis. The current timer is an observation after these runs; its exact setting during recording was not saved. No setting changed.</p>',
        '<table><tr><th>Episode</th><th>Starting rotation</th><th>Result</th><th>Steps</th><th>Duration</th><th>Final orientation</th><th>Final position</th><th>Delta clipped</th><th>Absolute clipped</th><th>Video</th></tr>']
    for r in rows:
        body.append(f"<tr><td>{r['episode']}</td><td>{r['rotation']} ({r['initial_angle_deg']:.1f} deg)</td><td>{r['outcome']}</td><td>{r['steps']}</td><td>{r['duration_s']:.2f} s</td><td>{r['final_angle_deg']:.1f} deg</td><td>{r['final_position_error_mm']:.1f} mm</td><td>{r['delta_clipped_percent']:.1f}%</td><td>{r['absolute_clipped_percent']:.1f}%</td><td><a href=\"video/episode_{r['episode']:02d}.mkv\">episode_{r['episode']:02d}.mkv</a></td></tr>")
    body+=['</table>','<h2>Starting rotations</h2><img src="analysis/starting_rotation.png">','<h2>Errors</h2><img src="analysis/episode_errors.png">',
           '<h2>Timing</h2><img src="analysis/timing.png">','<h2>Joint constraints</h2><img src="analysis/joint_limits.png">',
           '<h2>Interpretation limits</h2><ul>']
    body+=['<li>'+html.escape(s)+'</li>' for s in report['limitations']]
    body+=['</ul>'];(OUT/'index.html').write_text('\n'.join(body))
    print(json.dumps(dict(out=str(OUT),summary=all_metrics,episodes=rows),indent=2))


if __name__=='__main__':main()
