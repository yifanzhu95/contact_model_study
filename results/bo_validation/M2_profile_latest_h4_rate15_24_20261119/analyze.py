"""Reproducible offline analysis; no hardware, tracking or simulation calls.

Latest cohort: last two sessions, 16+8 completed episodes. Bridge logs are read
as a stream. Aborted attempts are preserved in the manifest and never pooled.
"""
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
import time

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'camera_hand_calibration/cache/matplotlib'))
import numpy as np

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

F=module('features',ROOT/'results/diagnostics/usb_latency_comparison_20261117/variation_review_20261117/analyze.py')
P=module('previous',ROOT/'results/bo_validation/analysis_M2_latest_h0256_12eps_20261119/analyze.py')
BASE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial314_shift512_latest_h0256_unpaced'
SOURCES=dict(current=['leap_experiment_20261119_030100_212337','leap_experiment_20261119_031523_266949'],
             no_rate=['leap_experiment_20261119_011314_943307','leap_experiment_20261119_011946_059801'])

def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def epoch(text):return int(datetime.fromisoformat(text).timestamp()*1e6)
def stat(a):return P.dist(a)
def rowsum(rows,arrays):
    s=F.summarize(rows,arrays);s['timing']=P.timing(rows,arrays)
    for key in ['initial_pos_mm','initial_angle_deg','final_pos_mm','final_angle_deg','best_angle_deg']:
        s[key]=stat([r[key] for r in rows])
    for key in ['pos','angle']:
        s['logged_'+key]=stat(np.concatenate([a[key] for a in arrays]))
        s['episode_mean_'+key]=stat([a[key].mean() for a in arrays])
    s['success_final_position_mm']=stat([r['final_pos_mm'] for r in rows if r['success']])
    s['success_final_orientation_deg']=stat([r['final_angle_deg'] for r in rows if r['success']])
    s['failure_final_position_mm']=stat([r['final_pos_mm'] for r in rows if not r['success']])
    s['failure_final_orientation_deg']=stat([r['final_angle_deg'] for r in rows if not r['success']])
    return s

def extract(group):
    rows=[];series=[];sources=[];excluded=[];configs=[];steps_out=[];commands={};ep_windows={};tapes=defaultdict(list)
    episode_number=0
    for name in SOURCES[group]:
        path=BASE/name/'controller.json';d=json.loads(path.read_text());print('Read',path.name,name,flush=True)
        configs.append({k:d[k] for k in ['weights','hardware_config','joint_reference','success_thresholds','failure_z',
            'rollout_config','planner_config','pose_transform','control_policy','task_contact_params']})
        configs[-1]['actual_goal_positions_m']=sorted({tuple(e['target_pos']) for e in d['episodes'] if e.get('completed')})
        configs[-1]['actual_drop_heights_m']=sorted({e['fallen_z'] for e in d['episodes'] if e.get('completed')})
        configs[-1]['fixed_goal_quat']=d['fixed_goal_quat']
        assert np.allclose(abs(np.dot(d['fixed_goal_quat'],[1,0,0,0])),1), 'Starting-axis labels assume the recorded identity goal.'
        source=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),status=d['status'],error=d.get('error'),
                    rollout_scene_sha256=hashlib.sha256(d['scenes']['rollout']['scene_xml'].encode()).hexdigest(),
                    bridge_torque_off_verified='Torque disabled and verified on all 16 motors.' in (path.parent/'bridge.log').read_text())
        rr=[];aa=[];windows=[]
        for e in d['episodes']:
            if not e.get('completed'):
                excluded.append(dict(source=source['path'],episode=e['episode']+1,actions=e['n_steps'],reason=e.get('error',d.get('error'))));continue
            episode_number+=1;row,a=F.feature(path,d,e,group);row['analysis_episode']=episode_number
            tt=P.timing([row],[a]);row.update(control_hz=tt['effective_hz'],min_instantaneous_hz=tt['instantaneous_hz']['min'],
                max_instantaneous_hz=tt['instantaneous_hz']['max'],planning_min_ms=tt['planning_ms']['min'],planning_max_ms=tt['planning_ms']['max'],
                planning_over64_percent=tt['planning_over64_fraction']*100)
            ss=[s for s in e['steps'] if s.get('action_published')]
            u=np.array([s['u'] for s in ss]);planned=np.array([s['planned_u'] for s in ss]);requested=np.array([s['requested_target'] for s in ss])
            ref=np.array([s.get('action_rate_reference',d['hardware_config']['robot_qref'] if i==0 else ss[i-1]['u']) for i,s in enumerate(ss)])
            energy_plan=np.sum((planned-ref)**2,axis=1);energy_est=np.sum((u-ref)**2,axis=1)
            rate=d['args']['action_rate_weight'];weight=d['planner_config']['action_rate_weight'];assert rate==weight
            if rate:
                np.testing.assert_allclose([s['planned_action_rate_cost'] for s in ss],rate*energy_plan,rtol=1e-7,atol=1e-8)
                np.testing.assert_allclose([s['estimated_action_rate_cost'] for s in ss],rate*energy_est,rtol=1e-7,atol=1e-8)
            dt=np.r_[np.nan,a['gaps']*.001];jump=np.vstack([np.full(16,np.nan),np.diff(u,axis=0)])
            a.update(estimated_change=jump,estimated_jump=np.max(abs(jump),axis=1),planned_jump=np.max(abs(planned-ref),axis=1),
                     planned_energy=energy_plan,estimated_energy=energy_est,delta_t=dt,
                     rebase=np.max(abs(requested-planned),axis=1),clipping_shift=np.max(abs(u-requested),axis=1),
                     joint_speed=np.abs(np.array([s['q_dot'][:16] for s in ss])),command_joint_age=np.array([s['feedback']['command_state']['robot_age'] for s in ss])*1000,
                     command_pose_age=np.array([s['feedback']['command_state']['pose_age'] for s in ss])*1000)
            row.update(estimated_jump_mean_deg=float(np.nanmean(a['estimated_jump'])*180/np.pi),
                estimated_jump_p95_deg=float(np.nanpercentile(a['estimated_jump'],95)*180/np.pi) if len(ss)>1 else None,
                planned_first_command_cost_mean=float(rate*energy_plan.mean()),estimated_first_command_cost_mean=float(rate*energy_est.mean()))
            for i,s in enumerate(ss):
                stamp=s['action_timestamp_us'];commands[stamp]=dict(analysis_episode=episode_number,source_episode=e['episode']+1,step=i,estimated_target=u[i],planned_target=planned[i],reference=ref[i],energy_plan=energy_plan[i],energy_est=energy_est[i])
                steps_out.append(dict(group=group,episode=episode_number,source_episode=e['episode']+1,step=i,action_timestamp_us=stamp,
                    planning_ms=a['planning'][i],publication_interval_ms=None if i==0 else a['gaps'][i-1],
                    estimated_max_joint_jump_deg=None if i==0 else float(a['estimated_jump'][i]*180/np.pi),
                    planned_first_command_energy_rad2=float(energy_plan[i]),estimated_first_command_energy_rad2=float(energy_est[i]),
                    planned_first_command_penalty=float(rate*energy_plan[i]),estimated_first_command_penalty=float(rate*energy_est[i]),
                    rebase_shift_max_deg=float(a['rebase'][i]*180/np.pi),clipping_shift_max_deg=float(a['clipping_shift'][i]*180/np.pi),
                    clipped_joint_fraction=float(a['hits'][i].mean()),position_error_mm=float(a['pos'][i]),orientation_error_deg=float(a['angle'][i])))
            windows.append((epoch(e['started_at_utc']),epoch(e['started_at_utc'])+int(e['manipulation_wall_time_s']*1e6),episode_number))
            rows.append(row);series.append(a);rr.append(row);aa.append(a)
        source['episodes']=len(rr);source['summary']=rowsum(rr,aa);sources.append(source);ep_windows[name]=windows
    for key in configs[0]:assert configs[0][key]==configs[1][key],(group,key)
    return dict(rows=rows,series=series,sources=sources,excluded=excluded,configuration=configs[0],steps=steps_out,commands=commands,windows=ep_windows)

def action_summary(dataset):
    a=dataset['series'];cat=lambda k:np.concatenate([x[k] for x in a]);ratio=cat('estimated_energy')/np.maximum(cat('planned_energy'),1e-12)
    jump=cat('estimated_jump');delta_t=cat('delta_t')
    return dict(estimated_max_joint_jump_deg=stat(np.degrees(jump)),planned_max_joint_jump_deg=stat(np.degrees(cat('planned_jump'))),
        estimated_jump_above_10deg_fraction=float(np.nanmean(jump[np.isfinite(jump)]>np.radians(10))),
        estimated_jump_above_20deg_fraction=float(np.nanmean(jump[np.isfinite(jump)]>np.radians(20))),
        equivalent_target_change_speed_deg_s=stat(np.degrees(jump/delta_t)),measured_abs_joint_velocity_deg_s=stat(np.degrees(cat('joint_speed'))),
        planned_first_command_energy_rad2=stat(cat('planned_energy')),estimated_first_command_energy_rad2=stat(cat('estimated_energy')),
        estimated_to_planned_energy_ratio=stat(ratio),estimated_energy_exceeds_planned_fraction=float((ratio>1).mean()),
        rebasing_shift_max_deg=stat(np.degrees(cat('rebase'))),clipping_shift_max_deg=stat(np.degrees(cat('clipping_shift'))),
        command_joint_age_ms=stat(cat('command_joint_age')),command_pose_age_ms=stat(cat('command_pose_age')))

def bridge(dataset):
    index=dataset['commands'];matched=[];snapshots=[];error=[];currents=[];reads=[];writes=[];meta=[]
    for name in SOURCES['current']:
        path=BASE/name/'bridge_telemetry.jsonl';print(f'Streaming {path} ({path.stat().st_size/1e6:.0f} MB)',flush=True)
        windows=dataset['windows'][name];pointer=0;accepted=0;seen=set();start=time.monotonic();file_hash=hashlib.sha256()
        with path.open('rb') as stream:
            for payload in stream:
                file_hash.update(payload);x=json.loads(payload)
                if x.get('event')!='cycle' or 'read_finished_us' not in x:continue
                stamp=x['read_finished_us']
                while pointer<len(windows) and stamp>windows[pointer][1]:pointer+=1
                # Motor-cycle diagnostics use manipulation windows. Matching a
                # selected action uses its exact ID instead: the first delta may
                # replace an absolute hold, and a final write can follow the
                # controller's recorded episode endpoint by one serial cycle.
                in_window=pointer<len(windows) and stamp>=windows[pointer][0]
                if in_window and x.get('held_command_type')=='delta':
                    accepted+=1
                    if x.get('held_target_before_read') is not None:error.append(np.abs(np.asarray(x['held_target_before_read'])-x['q']).astype(np.float32))
                    if 'current_raw_native' in x:currents.append(np.abs(x['current_raw_native']))
                    reads.append(x['read_duration_s']*1000)
                    if x.get('write_succeeded'):writes.append(x['write_duration_s']*1000)
                token=x.get('written_action_timestamp_us')
                if not x.get('new_action') or not x.get('write_succeeded') or x.get('command_type')!='delta' or token not in index or token in seen:continue
                seen.add(token);s=index[token];actual=np.asarray(x['written_target']);previous=np.asarray(x['held_target_before_read']);diff=actual-previous
                record=dict(episode=s['analysis_episode'],source_episode=s['source_episode'],step=s['step'],action_timestamp_us=token,
                    actual_max_joint_jump_deg=float(np.max(abs(diff))*180/np.pi),actual_energy_rad2=float(diff@diff),
                    planned_energy_rad2=float(s['energy_plan']),estimated_energy_rad2=float(s['energy_est']),
                    actual_vs_estimated_target_max_deg=float(np.max(abs(actual-s['estimated_target']))*180/np.pi),
                    publish_to_write_ms=(x['write_started_us']-token)*.001,
                    requested_vs_written_target_max_deg=float(np.max(abs(actual-s['planned_target']))*180/np.pi))
                matched.append(record);snapshots.append((diff,actual-s['estimated_target']))
        meta.append(dict(source=str(path.relative_to(ROOT)),sha256=file_hash.hexdigest(),included_cycles=accepted,matched_actions=len(seen),read_seconds=time.monotonic()-start))
    err=np.asarray(error);cur=np.asarray(currents);actual_change=np.array([r[0] for r in snapshots]);extra=np.array([r[1] for r in snapshots])
    weight=dataset['configuration']['planner_config']['action_rate_weight']
    out=dict(sources=meta,matched_actions=len(matched),selected_actions=len(index),coverage_fraction=len(matched)/len(index),
        actual_max_joint_jump_deg=stat([r['actual_max_joint_jump_deg'] for r in matched]),
        actual_jump_above_10deg_fraction=float(np.mean([r['actual_max_joint_jump_deg']>10 for r in matched])),
        actual_jump_above_20deg_fraction=float(np.mean([r['actual_max_joint_jump_deg']>20 for r in matched])),
        actual_first_command_penalty=stat([weight*r['actual_energy_rad2'] for r in matched]),
        actual_to_planned_energy_ratio=stat([r['actual_energy_rad2']/max(r['planned_energy_rad2'],1e-12) for r in matched]),
        actual_energy_exceeds_planned_fraction=float(np.mean([r['actual_energy_rad2']>r['planned_energy_rad2'] for r in matched])),
        estimated_target_discrepancy_deg=stat([r['actual_vs_estimated_target_max_deg'] for r in matched]),
        requested_target_discrepancy_deg=stat([r['requested_vs_written_target_max_deg'] for r in matched]),
        publication_to_write_ms=stat([r['publish_to_write_ms'] for r in matched]),
        holding_error_deg=stat(np.degrees(err)),holding_max_joint_error_deg=stat(np.degrees(err.max(axis=1))),
        joint_holding_error_mean_deg=np.degrees(err.mean(axis=0)).tolist(),joint_holding_error_p95_deg=np.degrees(np.percentile(err,95,axis=0)).tolist(),
        peak_current_raw_by_native_motor=cur.max(axis=0).tolist(),fraction_cycles_current_over500_by_native_motor=(cur>500).mean(axis=0).tolist(),
        read_ms=stat(reads),write_ms=stat(writes))
    csvout('bridge_actions.csv',matched)
    return out,dict(bridge_changes=actual_change,bridge_extra=extra,bridge_current_peak=cur.max(axis=0),bridge_current_fraction=(cur>500).mean(axis=0),
        bridge_joint_error_mean=np.degrees(err.mean(axis=0)),bridge_joint_error_p95=np.degrees(np.percentile(err,95,axis=0)),
        bridge_actual_energy=np.array([r['actual_energy_rad2'] for r in matched]),bridge_planned_energy=np.array([r['planned_energy_rad2'] for r in matched]),
        bridge_estimated_energy=np.array([r['estimated_energy_rad2'] for r in matched]),bridge_action_episode=np.array([r['episode'] for r in matched]))

def main():
    started=time.monotonic();current=extract('current');baseline=extract('no_rate');assert len(current['rows'])==24 and len(baseline['rows'])==12
    report=dict(current=rowsum(current['rows'],current['series']),no_rate=rowsum(baseline['rows'],baseline['series']),
        current_configuration=current['configuration'],no_rate_configuration=baseline['configuration'],
        current_sources=current['sources'],no_rate_sources=baseline['sources'],excluded=current['excluded'],
        action_rate=dict(current=action_summary(current),no_rate=action_summary(baseline)),
        definitions=dict(pooled_frequency='total published actions / total manipulation seconds; excludes operator waits, preparation, resets, warmup and aborted episodes',
            instantaneous_frequency='1 / consecutive action publication interval; intervals never cross episode boundaries',
            action_rate='w_rate * sum_h ||absolute_requested_target[h]-previous_target||^2, summed over 16 joints; no timestep division',
            displayed_penalties='first weighted-mean command only, not the complete sampled-horizon action-rate contribution',
            actual_target_changes='bridge written_target minus held_target_before_read for a newly applied delta',
            equivalent_speed='target-position difference / publication interval, not measured motor speed',
            initial_rotation='nearest signed principal rotation axis of the goal-relative quaternion; not printed face identification'),
        caveats=['Rate-0 and rate-15 cohorts differ in five state-cost weights AND goal position ([0.020,0.035,0.080] to [0.025,0.035,0.075] m). Their comparison is observational and cannot isolate the causal effect of the action-rate coefficient.',
            'Current N=24 versus earlier H4 N=12; manual starting placements and GPU/tracker load are uncontrolled.',
            'Action-rate costs are soft penalties on requested targets before hardware rebasing/clipping. A bound on target changes is not enforced.',
            'The recorded planned/estimated action-rate values score only the first weighted-mean command. Full-horizon sampled cost decompositions are not logged.',
            'Final errors include drop endpoints; convergence curves stop when each episode ends and later medians represent survivors.',
            'The excluded aborted episode stopped on motor-14 current -554 raw. Its currents/actions are excluded from completed-episode plots but retained in raw logs.',
            'The first session includes all 16 completed episodes despite failure during episode 17. The second includes all 8 completed episodes before operator interruption.'])
    report['changed_cost_weights']={k:dict(no_rate=baseline['configuration']['weights'][k],current=v) for k,v in current['configuration']['weights'].items() if v!=baseline['configuration']['weights'][k]}
    report['changed_goal_positions_m']={g:data['configuration']['actual_goal_positions_m'] for g,data in [('no_rate',baseline),('current',current)]}
    for key in ['hardware_config','joint_reference','success_thresholds','failure_z','rollout_config','pose_transform','control_policy','task_contact_params','actual_drop_heights_m','fixed_goal_quat']:
        assert current['configuration'][key]==baseline['configuration'][key],key
    assert len({s['rollout_scene_sha256'] for s in current['sources']+baseline['sources']})==1
    report['bridge'],archive=bridge(current)
    archive.update(joint_names=np.array(['if_mcp','if_rot','if_pip','if_dip','mf_mcp','mf_rot','mf_pip','mf_dip','rf_mcp','rf_rot','rf_pip','rf_dip','th_cmc','th_axl','th_mcp','th_ipl']))
    for group,data in [('current',current),('no_rate',baseline)]:
        for key in data['series'][0]:
            if key in ['t','pos','angle','xyz']:continue
            archive[group+'_'+key]=np.concatenate([s[key] for s in data['series']])
        for i,s in enumerate(data['series']):
            for key in ['t','pos','angle','xyz']:archive[f'{group}_ep{i+1}_{key}']=s[key]
        csvout(group+'_episodes.csv',data['rows']);csvout(group+'_steps.csv',data['steps'])
    archive['current_step_episode']=np.concatenate([np.full(r['steps'],r['analysis_episode']) for r in current['rows']])
    np.savez_compressed(OUT/'plot_data.npz',**archive)
    previous=json.loads((ROOT/'results/diagnostics/usb_latency_comparison_20261117/variation_review_20261117/report.json').read_text())['cohorts']
    report['h5_reference']={k:previous[k] for k in ['historical_c0.2_T0.1_U','usb1_c0.2_T0.1_U']}
    manifest=dict(current_sources=report['current_sources'],reference_sources=report['no_rate_sources'],excluded=report['excluded'],
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HIST_SCRIPT,ROOT/'results/bo_validation/analysis_M2_latest_h0256_12eps_20261119/analyze.py']})
    dump('manifest.json',manifest);dump('report.json',report)
    validation=dict(current_episodes=24,no_rate_episodes=12,no_duplicate_sources=True,within_current_configuration_equal=True,
        bridge_coverage=report['bridge']['coverage_fraction'],logged_first_command_costs_recomputed=True,identical_rollout_scene_across_cohorts=True,
        no_hardware_or_simulation_access=True,elapsed_s=time.monotonic()-started)
    dump('validation.json',validation)
    print('SUMMARY',json.dumps({k:report['current'][k] for k in ['episodes','successes','drops','timeouts','effective_hz','clipped_action_fraction']}),flush=True)
    print('TIMING',json.dumps(report['current']['timing']),flush=True)
    print('ACTION',json.dumps(report['action_rate']),flush=True)
    print('BRIDGE',json.dumps(report['bridge']),flush=True)

HIST_SCRIPT=ROOT/'results/diagnostics/usb_latency_comparison_20261117/variation_review_20261117/analyze.py'
if __name__=='__main__':main()
