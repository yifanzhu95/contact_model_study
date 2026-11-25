"""Offline analysis of the latest 12 completed temporal-blending episodes.

No simulator, LCM connection, camera, or hardware API is instantiated.
Conditional no-blend commands reuse recorded feedback, NOT a replayed physical
trajectory; their success or physical smoothness cannot be inferred.
"""
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import importlib.util
import json
import os
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
PREV=ROOT/'results/bo_validation/M2_profile_latest_h4_rate15_24_20261119'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=module('prior_analysis',PREV/'analyze.py');F=M.F;P=M.P
SOURCE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial314_shift512_latest_h0256_unpaced/leap_experiment_20261119_034105_014853'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for block in iter(lambda:s.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as s:
        w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def stat(x):return P.dist(x)
def maxdeg(x):return np.degrees(np.max(np.abs(x),axis=-1))
def l2(x):return np.linalg.norm(x,axis=-1)

def main():
    d=json.loads((SOURCE/'controller.json').read_text());old=json.loads((PREV/'report.json').read_text())
    assert d['args']['plan_blend_alpha']==.6 and d['args']['action_rate_weight']==15
    completed=[e for e in d['episodes'] if e.get('completed')];assert len(completed)==12
    rows=[];series=[];steps=[];commands={};archive={};computed=defaultdict(list);windows=[];max_error=0.;slots=Counter();statuses=Counter()
    down=np.asarray(d['effective_ctrl_range']['lower']);up=np.asarray(d['effective_ctrl_range']['upper'])
    for ei,e in enumerate(completed,1):
        row,a=F.feature(SOURCE/'controller.json',d,e,'blend');row['analysis_episode']=ei
        row['control_hz']=P.timing([row],[a])['effective_hz'];ss=[s for s in e['steps'] if s.get('action_published')]
        raw=[];blend=[];target=[];raw_target=[];ages=[];compute=[];adjust=[];prev_target=[]
        previous_mean=None;window_start=M.epoch(e['started_at_utc']);windows.append((window_start,window_start+int(e['manipulation_wall_time_s']*1e6)))
        for si,s in enumerate(ss):
            b=s['plan_blending'];new=np.asarray(b['new_mean']);r=np.asarray(b['unblended_delta']);z=np.asarray(b['blended_delta']);base=np.asarray(s['feedback']['command_qpos']);held=np.asarray(s['action_rate_reference'])
            np.testing.assert_allclose(new[0],r,atol=1e-12);np.testing.assert_allclose(s['action_delta'],z,atol=1e-12)
            statuses[b['status']]+=1
            expected=r.copy();old_delta=None
            if b['status']=='overlap':
                slot=b['old_slot'];slots[slot]+=1;old_delta=np.asarray(b['old_aligned_delta'])
                assert previous_mean is not None
                np.testing.assert_allclose(old_delta,previous_mean[slot],rtol=0,atol=1e-12)
                expected=.6*r+.4*old_delta
                np.testing.assert_allclose(np.asarray(b['first_aligned_disagreement']),r-old_delta,atol=1e-12)
                computed['aligned_raw_l2'].append(l2(r-old_delta));computed['aligned_blend_l2'].append(l2(z-old_delta))
            elif si==0:assert b['status']=='first_plan'
            max_error=max(max_error,float(np.max(abs(expected-z))))
            assert max_error<1e-11
            previous_mean=new
            u=np.asarray(s['u']);cf=base+np.clip(r,-down,up)
            np.testing.assert_allclose(u,base+np.clip(z,-down,up),atol=1e-12)
            np.testing.assert_allclose(s['planned_action_rate_cost'],15*np.sum((np.asarray(s['planned_u'])-held)**2),rtol=1e-7,atol=1e-8)
            np.testing.assert_allclose(s['estimated_action_rate_cost'],15*np.sum((u-held)**2),rtol=1e-7,atol=1e-8)
            raw_hits=np.abs(r-np.clip(r,-down,up))>1e-7;blend_hits=np.abs(z-np.clip(z,-down,up))>1e-7
            assert int(raw_hits.sum())==b['estimated_unblended_clipped_joints']
            assert int(blend_hits.sum())==b['estimated_blended_clipped_joints']
            computed['raw_hits'].append(raw_hits);computed['blend_hits'].append(blend_hits)
            computed['within_l2'].extend(b['within_plan_diff_l2'])
            compute.append(b['compute_s']*1e6);adjust.append(np.degrees(l2(z-r)))
            age=np.nan if b['age_s'] is None else b['age_s']*1000;ages.append(age)
            stamp=s['action_timestamp_us']
            commands[stamp]=dict(episode=ei,step=si,raw=r,blended=z,estimated_target=u,previous_estimated=held,planned=np.asarray(s['planned_u']))
            step=dict(episode=ei,step=si,elapsed_s=s['elapsed_s'],action_timestamp_us=stamp,status=b['status'],old_slot=b['old_slot'],old_plan_age_ms=None if not np.isfinite(age) else age,
                blend_compute_us=b['compute_s']*1e6,blend_adjustment_l2_deg=np.degrees(l2(z-r)),
                unblended_max_delta_deg=float(maxdeg(r)),blended_max_delta_deg=float(maxdeg(z)),
                conditional_raw_target_jump_max_deg=float(maxdeg(cf-held)),blended_target_jump_max_deg=float(maxdeg(u-held)),
                conditional_raw_target_energy_rad2=float(np.sum((cf-held)**2)),blended_target_energy_rad2=float(np.sum((u-held)**2)),
                raw_clipped_joints=int(raw_hits.sum()),blended_clipped_joints=int(blend_hits.sum()),
                planning_ms=s['planning_time']*1000,
                aligned_raw_l2_deg=None if old_delta is None else float(np.degrees(l2(r-old_delta))),
                aligned_blended_l2_deg=None if old_delta is None else float(np.degrees(l2(z-old_delta))))
            steps.append(step);raw.append(r);blend.append(z);target.append(u);raw_target.append(cf);prev_target.append(held)
        raw=np.array(raw);blend=np.array(blend);target=np.array(target);raw_target=np.array(raw_target)
        for key,v in [('t',a['t']),('position',a['pos']),('angle',a['angle']),('raw',raw),('blend',blend),('target',target),('raw_target',raw_target),('measured',np.array([s['q'][:16] for s in ss])),('sample_t',a['t'][:-1]),('age_ms',np.array(ages))]:archive[f'ep{ei}_{key}']=v
        computed['raw_delta_jump_deg'].extend(maxdeg(np.diff(raw,axis=0)));computed['blend_delta_jump_deg'].extend(maxdeg(np.diff(blend,axis=0)))
        computed['estimated_target_jump_deg'].extend(maxdeg(np.diff(target,axis=0)))
        computed['raw_target_sequence_jump_deg'].extend(maxdeg(np.diff(raw_target,axis=0)))
        computed['measured_joint_speed_deg_s'].extend(np.degrees(np.abs([s['q_dot'][:16] for s in ss])).ravel())
        computed['blend_compute_us'].extend(compute);computed['blend_adjustment_l2_deg'].extend(adjust);computed['age_ms'].extend(ages)
        computed['planning_ms'].extend(a['planning']);computed['gaps_ms'].extend(a['gaps'])
        row.update(raw_delta_jump_p95_deg=float(np.percentile(maxdeg(np.diff(raw,axis=0)),95)),
            blend_delta_jump_p95_deg=float(np.percentile(maxdeg(np.diff(blend,axis=0)),95)),
            estimated_target_jump_p95_deg=float(np.percentile(maxdeg(np.diff(target,axis=0)),95)),
            blend_compute_mean_us=float(np.mean(compute)),blend_compute_p95_us=float(np.percentile(compute,95)),
            raw_clipped_action_fraction=float(np.any(np.asarray(computed['raw_hits'][-len(ss):]),axis=1).mean()))
        rows.append(row);series.append(a)
    print('Loaded and validated 12 episodes /',len(steps),'commands',flush=True)
    raw_bridge=[];blend_bridge=[];bridge_ep=[];bridge_rows=[];seen=set();write_latency=[];errors=[]
    path=SOURCE/'bridge_telemetry.jsonl';print('Streaming bridge telemetry:',path.stat().st_size/1e6,'MB',flush=True)
    digest=hashlib.sha256();wi=0
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line);x=json.loads(line)
            if x.get('event')!='cycle':continue
            tm=x.get('read_finished_us',0)
            while wi<len(windows) and tm>windows[wi][1]:wi+=1
            if wi<len(windows) and tm>=windows[wi][0] and x.get('held_command_type')=='delta':
                errors.append(np.degrees(abs(np.asarray(x['held_target_before_read'])-x['q'])).astype(np.float32))
            token=x.get('written_action_timestamp_us')
            if not x.get('new_action') or not x.get('write_succeeded') or x.get('command_type')!='delta' or token not in commands or token in seen:continue
            seen.add(token);s=commands[token];base=np.asarray(x['application_base']);prior=np.asarray(x['held_target_before_read']);written=np.asarray(x['written_target'])
            # LCM joint-delta payloads are float32; controller diagnostics are
            # float64. Apply the same wire conversion for both command options.
            wire_blend=s['blended'].astype(np.float32).astype(float)
            wire_raw=s['raw'].astype(np.float32).astype(float)
            np.testing.assert_allclose(written,base+np.clip(wire_blend,-down,up),rtol=0,atol=1e-11)
            cf=base+np.clip(wire_raw,-down,up);rd=cf-prior;bd=written-prior
            raw_bridge.append(rd);blend_bridge.append(bd);bridge_ep.append(s['episode']);latency=(x['write_started_us']-token)*.001;write_latency.append(latency)
            bridge_rows.append(dict(episode=s['episode'],step=s['step'],action_timestamp_us=token,
                conditional_raw_jump_max_deg=float(maxdeg(rd)),actual_jump_max_deg=float(maxdeg(bd)),
                conditional_raw_energy_rad2=float(rd@rd),actual_energy_rad2=float(bd@bd),
                publish_to_write_ms=latency,actual_vs_estimated_target_max_deg=float(maxdeg(written-s['estimated_target']))))
    assert len(seen)==len(commands)
    raw_bridge=np.asarray(raw_bridge);blend_bridge=np.asarray(blend_bridge);errors=np.asarray(errors)
    for k,v in computed.items():archive[k]=np.asarray(v)
    archive.update(raw_bridge_changes=raw_bridge,actual_bridge_changes=blend_bridge,bridge_episode=np.array(bridge_ep),joint_names=np.array(d['joint_order']))
    np.savez_compressed(OUT/'plot_data.npz',**archive)
    summary=M.rowsum(rows,series)
    smooth={k:stat(v) for k,v in computed.items() if k not in ['raw_hits','blend_hits']}
    smooth.update(conditional_raw_bridge_jump_deg=stat(maxdeg(raw_bridge)),actual_bridge_jump_deg=stat(maxdeg(blend_bridge)),
        conditional_raw_bridge_energy_rad2=stat(np.sum(raw_bridge**2,axis=1)),actual_bridge_energy_rad2=stat(np.sum(blend_bridge**2,axis=1)),
        paired_bridge_smaller_fraction=float(np.mean(maxdeg(blend_bridge)<maxdeg(raw_bridge))),
        paired_bridge_larger_fraction=float(np.mean(maxdeg(blend_bridge)>maxdeg(raw_bridge))),
        actual_jump_above10_fraction=float(np.mean(maxdeg(blend_bridge)>10)),conditional_raw_jump_above10_fraction=float(np.mean(maxdeg(raw_bridge)>10)),
        raw_clipped_action_fraction=float(np.any(computed['raw_hits'],axis=1).mean()),blended_clipped_action_fraction=float(np.any(computed['blend_hits'],axis=1).mean()),
        raw_clipped_joint_fraction=float(np.mean(computed['raw_hits'])),blended_clipped_joint_fraction=float(np.mean(computed['blend_hits'])),
        publish_to_write_ms=stat(write_latency),holding_error_deg=stat(errors),holding_max_joint_error_deg=stat(errors.max(axis=1)))
    changes={k:dict(previous=old['current_configuration']['weights'][k],current=v) for k,v in d['weights'].items() if v!=old['current_configuration']['weights'][k]}
    # The comparison has matching scene physics, preparation, calibration, goal
    # and hardware clipping. Its two changed weights prevent causal attribution.
    assert sha(SOURCE/'controller.json')
    scene_hash=hashlib.sha256(d['scenes']['rollout']['scene_xml'].encode()).hexdigest()
    assert all(s['rollout_scene_sha256']==scene_hash for s in old['current_sources'])
    for k in ['hardware_config','pose_transform','joint_reference']:assert d[k]==old['current_configuration'][k],k
    assert sorted({tuple(e['target_pos']) for e in completed})==[tuple(v) for v in old['current_configuration']['actual_goal_positions_m']]
    assert sorted({e['fallen_z'] for e in completed})==old['current_configuration']['actual_drop_heights_m']
    report=dict(source=str(SOURCE.relative_to(ROOT)),source_status=d['status'],source_error=d.get('error'),summary=summary,
        smoothness=smooth,blend_status_counts=dict(statuses),old_slot_counts=dict(slots),changed_cost_weights_vs_previous24=changes,
        previous_unblended24=old['current'],previous_smoothness=old['action_rate']['current'],previous_bridge=old['bridge'],
        configuration={k:d[k] for k in ['args','weights','joint_reference','hardware_config','rollout_config','planner_config','pose_transform','temporal_plan_blending','effective_ctrl_range']},
        actual_goal_positions_m=sorted({tuple(e['target_pos']) for e in completed}),actual_fallen_z_m=sorted({e['fallen_z'] for e in completed}),
        bridge=dict(matched_actions=len(seen),published_actions=len(commands),coverage=1.,sha256=digest.hexdigest(),
            torque_off_verified='Torque disabled and verified on all 16 motors.' in (SOURCE/'bridge.log').read_text(),
            cleanup_action_watchdog='Action timeout; disabling torque' in (SOURCE/'bridge.log').read_text()),
        definitions=dict(alpha='0.6 new raw delta + 0.4 old time-aligned raw delta, before ±0.2 hardware clipping; only first command is blended',
            slot='floor(elapsed seconds since previous publication / 0.064); old raw plan retained; no recursive blend of blended history',
            raw_vs_blend='same recorded state/history; diagnostic conditional commands, not unblended closed-loop physical outcomes',
            jump='largest absolute change over the 16 joints; target jumps differ from incremental-delta jumps',
            bridge_raw_counterfactual='bridge application base + clip(raw delta) compared with the same actual prior held target',
            costs='Action-rate penalty remains 15; temporal blending is post-planning and does not alter MPPI warm starting or sampled costs'),
        caveats=['Previous unblended24 differs in w_joint (0.10→0.15) and w_pos_term (100→120), so success differences cannot be attributed to blending alone.',
            'The conditional raw trace uses recorded states. It cannot predict object trajectories, success, or motor velocity without blending.',
            'A 40% reduction in distance from the old aligned action follows algebraically from alpha=0.6; it is not itself proof of physical smoothness.',
            'At replanning intervals below 64ms, old slot 0 is used. This often blends the previous raw first action rather than its next horizon row.',
            'Compute timing covers blending.prepare only; per-step diagnostic formatting, JSON logging and serialization overhead are not separately measured.'])
    dump('report.json',report);csvout('episodes.csv',rows);csvout('steps.csv',steps);csvout('bridge_actions.csv',bridge_rows)
    dump('manifest.json',dict(source_controller=str((SOURCE/'controller.json').relative_to(ROOT)),source_controller_sha256=sha(SOURCE/'controller.json'),
        source_bridge=str(path.relative_to(ROOT)),source_bridge_sha256=digest.hexdigest(),rollout_scene_sha256=scene_hash,
        completed_source_episodes=[e['episode']+1 for e in completed],excluded_episodes=[e['episode']+1 for e in d['episodes'] if not e.get('completed')],
        prior_report=str((PREV/'report.json').relative_to(ROOT)),prior_report_sha256=sha(PREV/'report.json'),analysis_sha256=sha(Path(__file__)),
        plan_blending_implementation_sha256=sha(ROOT/'ContactModelStudy/Hardware/PlanBlending.py')))
    dump('validation.json',dict(completed_episodes=12,bridge_matching_fraction=1.,blending_formula_max_error_rad=max_error,all_previous_plan_slots_verified=True,
        logged_clipping_counts_recomputed=True,logged_action_rate_diagnostics_recomputed=True,no_duplicate_action_ids=len(commands)==len(steps),
        identical_scene_physics_to_previous24=True,no_hardware_camera_tracking_or_simulation_calls=True))
    print('SUMMARY',json.dumps({k:summary[k] for k in ['episodes','successes','drops','timeouts','effective_hz','clipped_action_fraction']}))
    print('BLENDING',dict(slots),dict(statuses));print('SMOOTHNESS',json.dumps(smooth));print('CHANGES',changes)

if __name__=='__main__':main()
