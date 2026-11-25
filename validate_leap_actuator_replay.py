#!/usr/bin/env python3
"""Offline validation of calibrated joint dynamics against bridge recordings.

No hardware/LCM clients are imported or created. Replay changes no calibration,
controller configuration or production XML. Loaded recordings are compared to
the hand-only model as explicitly labeled disturbance baselines because their
object pose/contact states were not recorded.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from ContactModelStudy.Hardware.TargetResponseAnalysis import harmonic_lag

ROOT=Path(__file__).resolve().parent
DEFAULT_RUNS=[ROOT/'results/diagnostics'/name for name in (
    'target_response_20261116_empty_hand','target_response_20261116_cube_003',
    'target_response_20261116_cube_sine008')]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(values):
    values=np.asarray(values,dtype=float)
    return dict(count=int(values.size),bias=float(values.mean()),mae=float(abs(values).mean()),
        rmse=float(np.sqrt(np.mean(values**2))),p95_abs=float(np.percentile(abs(values),95)),
        max_abs=float(abs(values).max())) if values.size else None


def sample_time(row,alignment):
    if alignment=='start':return row['read_started_us']
    if alignment=='end':return row['read_finished_us']
    if alignment=='midpoint':return (row['read_started_us']+row['read_finished_us'])/2
    raise ValueError(f'Unknown alignment: {alignment}')


def changed_writes(cycles,end_us):
    """Keep every changed SDK-written target; heartbeats hold the same input."""
    events=[];previous=None
    for row in cycles:
        if not row.get('write_succeeded') or row['write_finished_us']>end_us:continue
        value=np.asarray(row['written_target'],dtype=float)
        if previous is None or not np.array_equal(previous,value):
            events.append((row['write_finished_us'],value.copy()))
            previous=value
    return events


def filter_state(initial,time_us,events,at_us,tau):
    """Infer the model's activation from recorded targets; no parameter fitting."""
    activation=np.asarray(initial,dtype=float).copy();target=activation.copy()
    for stamp,value in events:
        if stamp>at_us:break
        elapsed=max(0.,(stamp-time_us)*1e-6)
        activation=target+(activation-target)*np.exp(-elapsed/tau)
        time_us=stamp;target=value
    elapsed=max(0.,(at_us-time_us)*1e-6)
    return target+(activation-target)*np.exp(-elapsed/tau),target.copy()


def replay(model,q0,v0,activation0,initial_target,sample_times,events,max_step=.002):
    """Integrate exactly to read/write event times without resetting feedback.

    sample_times and event times are relative seconds. A sample at the same
    instant as a write observes the state before that new target acts.
    """
    if max_step<=0 or np.any(np.diff(sample_times)<=0):raise ValueError('Invalid replay times/step')
    data=mujoco.MjData(model);data.qpos[:]=q0;data.qvel[:]=v0
    data.ctrl[:]=initial_target;data.act[:]=activation0
    mujoco.mj_forward(model,data)
    result=np.empty((len(sample_times),model.nq));velocity=np.empty((len(sample_times),model.nv))
    schedule=[(float(t),0,i,None) for i,t in enumerate(sample_times)]
    schedule.extend((float(t),1,None,u) for t,u in events if t>=0)
    schedule.sort(key=lambda item:(item[0],item[1]))
    original_step=model.opt.timestep;step_count=0
    try:
        for goal,kind,index,target in schedule:
            while goal-data.time>1e-11:
                model.opt.timestep=min(max_step,goal-data.time)
                mujoco.mj_step(model,data);step_count+=1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError('Nonfinite simulated state')
            if kind==0:
                result[index]=data.qpos;velocity[index]=data.qvel
            else:data.ctrl[:]=target
    finally:model.opt.timestep=original_step
    return result,velocity,step_count


def load_recording(directory):
    directory=Path(directory);experiment=json.loads((directory/'experiment.json').read_text())
    raw=[json.loads(line) for line in (directory/'bridge_telemetry.jsonl').read_text().splitlines()]
    cycles=[r for r in raw if r.get('event')=='cycle' and 'q' in r]
    if any(r.get('error') for r in cycles):raise ValueError('Faulting recording requires separate replay selection')
    start_limit=experiment['phases'][0]['started_us']-400000
    start_index=max(i for i,r in enumerate(cycles) if r['read_finished_us']<=start_limit)
    end=experiment['phases'][-1]['ended_us']+750000
    selected=[r for r in cycles[start_index:] if r['read_finished_us']<=end]
    if not selected or not selected[0].get('held_target_since_us'):
        raise ValueError('Replay needs a held target before the first phase')
    # Verify that held targets actually track preceding writes, rather than
    # accidentally comparing feedback with targets first written after a read.
    previous=None;verified=0
    for row in cycles:
        if previous is not None:
            np.testing.assert_array_equal(row['held_target_before_read'],previous['written_target'])
            verified+=1
        if row.get('write_succeeded'):previous=row
    return experiment,raw[0],cycles,selected,end,verified


def phase_metrics(experiment,selected,simulated,alignment):
    measured=np.array([r['q'] for r in selected]);rows=[]
    for phase in experiment['phases']:
        mask=np.array([phase['started_us']<=r['read_started_us'] and r['read_finished_us']<=phase['ended_us'] for r in selected])
        if not mask.any():continue
        joint=phase['joint_index'];target=np.array([r['held_target_before_read'] for r in selected])[mask,joint]
        hw=measured[mask,joint];predicted=simulated[mask,joint]
        item=dict(index=phase['index'],mode=phase['mode'],joint=phase['joint'],joint_index=joint,
            frames=int(mask.sum()),prediction_error_deg=statistics(np.degrees(predicted-hw)),
            hardware_target_error_deg=statistics(np.degrees(hw-target)),
            model_target_error_deg=statistics(np.degrees(predicted-target)))
        if phase['mode']=='fast':
            times=np.array([sample_time(r,alignment) for r in selected])[mask]*1e-6
            commands=[c for c in experiment['commands'] if c.get('phase')==phase['index'] and c['kind']=='new_target']
            period=np.mean(np.diff([c['timestamp_us'] for c in commands]))*1e-6
            samples=phase.get('samples_per_cycle',4)
            item['hardware_response']=harmonic_lag(times,target,hw,period,samples)
            item['model_response']=harmonic_lag(times,target,predicted,period,samples)
            if item['hardware_response'] and item['model_response']:
                item['hardware_minus_model_lag_ms']=item['hardware_response']['lag_ms']-item['model_response']['lag_ms']
        else:
            hw_offsets=[];sim_offsets=[];residuals=[]
            commands=[c for c in experiment['commands'] if c.get('phase')==phase['index'] and c['kind']=='new_target']
            for i,c in enumerate(commands):
                end=commands[i+1]['timestamp_us'] if i+1<len(commands) else phase['ended_us']
                tail=np.array([end-250000<=r['read_started_us'] and r['read_finished_us']<=end
                    and np.allclose(r['held_target_before_read'],c['target'],rtol=0,atol=2e-7) for r in selected])
                if not tail.any():continue
                actual=measured[tail,joint].mean();prediction=simulated[tail,joint].mean();u=c['target'][joint]
                hw_offsets.append(actual-u);sim_offsets.append(prediction-u);residuals.append(prediction-actual)
            item['settled_hardware_minus_target_deg']=statistics(np.degrees(hw_offsets))
            item['settled_model_minus_target_deg']=statistics(np.degrees(sim_offsets))
            item['settled_model_minus_hardware_deg']=statistics(np.degrees(residuals))
        rows.append(item)
    return rows


def validate_run(directory,output):
    experiment,header,cycles,selected,end,verified=load_recording(directory)
    source_model=Path(experiment['model']);model=mujoco.MjModel.from_xml_path(str(source_model))
    if (model.nq,model.nv,model.nu,model.na)!=(16,16,16,16):raise ValueError('Use hand-only calibrated model')
    report_dir=output/Path(directory).name;report_dir.mkdir()
    mujoco.mj_saveModel(model,str(report_dir/'calibrated_model.mjb'))
    (report_dir/'model_source.xml').write_bytes(source_model.read_bytes())
    # The copied XML is a source snapshot. The MJB snapshot contains all assets
    # and can be loaded directly, independent of relative mesh paths.
    measured=np.array([r['q'] for r in selected]);velocity=np.array([r['qvel'] for r in selected])
    joint_order=header['metadata']['joint_order'];events=changed_writes(cycles,end)
    q0,v0=measured[0].copy(),velocity[0].copy();alignments={};arrays={}
    tau=model.actuator_dynprm[:,0]
    for alignment in ['midpoint','start','end']:
        sample_us=np.array([sample_time(r,alignment) for r in selected]);origin=sample_us[0]
        times=(sample_us-origin)*1e-6
        activation,target=filter_state(cycles[0]['held_target_before_read'],cycles[0]['read_started_us'],events,origin,tau)
        upcoming=[((t-origin)*1e-6,u) for t,u in events if origin<t<=sample_us[-1]]
        started=time.monotonic()
        simulated,sim_v,steps=replay(model,q0,v0,activation,target,times,upcoming)
        masks=np.array([any(p['started_us']<=r['read_started_us'] and r['read_finished_us']<=p['ended_us'] for p in experiment['phases']) for r in selected])
        metrics=dict(all_joints_prediction_error_deg=statistics(np.degrees(simulated[masks]-measured[masks])),
            phases=phase_metrics(experiment,selected,simulated,alignment),
            simulation_steps=steps,wall_s=time.monotonic()-started)
        alignments[alignment]=metrics
        arrays[alignment]=(times,simulated,sim_v)
        print(Path(directory).name,alignment,json.dumps(metrics['all_joints_prediction_error_deg']),flush=True)
    primary=arrays['midpoint'];times,simulated,sim_v=primary
    sample_us=np.array([sample_time(r,'midpoint') for r in selected]);origin=sample_us[0]
    activation,target=filter_state(cycles[0]['held_target_before_read'],cycles[0]['read_started_us'],events,origin,tau)
    upcoming=[((t-origin)*1e-6,u) for t,u in events if origin<t<=sample_us[-1]]
    # A finer integration check changes only numerical resolution, not inputs.
    fine,fine_v,steps=replay(model,q0,v0,activation,target,times,upcoming,max_step=.001)
    violations=np.maximum(model.jnt_range[:,0]-measured,0)+np.maximum(measured-model.jnt_range[:,1],0)
    mask=np.array([any(p['started_us']<=r['read_started_us'] and r['read_finished_us']<=p['ended_us'] for p in experiment['phases']) for r in selected])
    joints=[]
    for j,name in enumerate(joint_order):
        joints.append(dict(joint=name,**statistics(np.degrees(simulated[mask,j]-measured[mask,j])),
            measured_outside_model_limit_fraction=float(np.mean(violations[mask,j]>1e-6)),
            maximum_measured_limit_violation_deg=float(np.degrees(violations[mask,j].max()))))
    row=dict(run=Path(directory).name,condition=experiment['object_condition'],
        scope='Matched empty-hand validation' if experiment['object_condition']=='empty_hand' else
              'Loaded hardware versus hand-only model; contact disturbances unmodeled, not a matched loaded validation',
        sources={name:dict(path=str(Path(directory)/name),sha256=digest(Path(directory)/name))
                 for name in ['experiment.json','bridge_telemetry.jsonl']},
        model=dict(path=str(source_model),sha256=digest(source_model),mujoco_version=mujoco.__version__,
            gravity=model.opt.gravity.tolist(),original_timestep_s=float(model.opt.timestep),
            backend='Native CPU MuJoCo; no MPPI or GPU rollout approximation',
            solver=int(model.opt.solver),iterations=int(model.opt.iterations),tolerance=float(model.opt.tolerance),
            integrator=int(model.opt.integrator),cone=int(model.opt.cone),
            actuator_time_constants_s=tau.tolist()),
        recorded_hardware_controller=header['metadata']['hardware_config']['expected_registers'],
        initialization=dict(qpos=q0.tolist(),qvel=v0.tolist(),activation=activation.tolist(),
            initial_target=target.tolist(),start_us=float(origin),end_us=float(sample_us[-1]),
            target_unchanged_before_start_s=(selected[0]['read_started_us']-selected[0]['held_target_since_us'])*1e-6,
            state_resets='One initial measured state per recording; no phase/step resets afterward'),
        verified_held_target_transitions=verified,changed_targets_replayed=len(upcoming),samples=len(selected),
        alignments=alignments,joints=joints,
        numerical_resolution_deg=statistics(np.degrees(fine-simulated)),
        caveats=['Actual SDK write-completion times define input changes; no extra fitted transport delay is inserted.',
            'Encoder samples are acquired somewhere within each ~16 ms read; midpoint is primary, with start/end sensitivity shown.',
            'There is no fitted offset, gain, time shift or calibration change in primary results.',
            'Loaded recordings lack object pose and contact state; hand-only residuals cannot isolate contact-model accuracy.',
            'Joint-limit disagreements are reported and not hidden by clipping recorded states.',
            'The compiled MJB is a complete model snapshot; model_source.xml is a verbatim source snapshot whose relative asset paths refer to the original model directory.'])
    (report_dir/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(report_dir/'trajectories.npz',times_s=times,read_started_us=np.array([r['read_started_us'] for r in selected]),
        read_finished_us=np.array([r['read_finished_us'] for r in selected]),q_hardware=measured,q_model=simulated,
        q_model_1ms=fine,qvel_hardware=velocity,qvel_model=sim_v,
        held_target=np.array([r['held_target_before_read'] for r in selected]),joint_order=np.array(joint_order))
    with (report_dir/'joints.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(joints[0]));writer.writeheader();writer.writerows(joints)
    plots(report_dir,experiment,selected,measured,simulated,origin)
    return row


def diagnose_mcp_limit(recording,output):
    """One offline constraint ablation; never writes a production model."""
    experiment,header,cycles,selected,end,verified=load_recording(recording)
    source=output/Path(recording).name
    saved=np.load(source/'trajectories.npz')
    model=mujoco.MjModel.from_binary_path(str(source/'calibrated_model.mjb'))
    joint=model.joint('mf_mcp').id
    model.jnt_limited[joint]=False
    sample_us=np.array([sample_time(r,'midpoint') for r in selected]);origin=sample_us[0]
    events=changed_writes(cycles,end)
    activation,target=filter_state(cycles[0]['held_target_before_read'],cycles[0]['read_started_us'],events,origin,model.actuator_dynprm[:,0])
    future=[((t-origin)*1e-6,u) for t,u in events if origin<t<=sample_us[-1]]
    predicted,velocity,steps=replay(model,saved['q_hardware'][0],saved['qvel_hardware'][0],activation,target,saved['times_s'],future)
    mask=np.array([any(p['started_us']<=r['read_started_us'] and r['read_finished_us']<=p['ended_us'] for p in experiment['phases']) for r in selected])
    result=dict(scope='Offline diagnostic ablation, not a replacement model or a fitted calibration',
        change='Only mf_mcp jnt_limited disabled in memory',original_lower_bound_rad=float(model.jnt_range[joint,0]),
        measured_range_rad=[float(saved['q_hardware'][mask,joint].min()),float(saved['q_hardware'][mask,joint].max())],
        baseline_mf_mcp_prediction_error_deg=statistics(np.degrees(saved['q_model'][mask,joint]-saved['q_hardware'][mask,joint])),
        diagnostic_mf_mcp_prediction_error_deg=statistics(np.degrees(predicted[mask,joint]-saved['q_hardware'][mask,joint])),
        baseline_all_joints_deg=statistics(np.degrees(saved['q_model'][mask]-saved['q_hardware'][mask])),
        diagnostic_all_joints_deg=statistics(np.degrees(predicted[mask]-saved['q_hardware'][mask])),
        model_changed_on_disk=False,
        caveat='This establishes a constraint-driven discrepancy in this pose. It does not distinguish an incorrect joint-range model from a joint-zero convention error, or validate an unconstrained physical joint.')
    (output/'mf_mcp_limit_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    t=saved['times_s'][mask]
    ax.plot(t,np.degrees(saved['q_hardware'][mask,joint]),label='Measured mf_mcp')
    ax.plot(t,np.degrees(saved['q_model'][mask,joint]),label='Original calibrated model')
    ax.plot(t,np.degrees(predicted[mask,joint]),label='Diagnostic: only this joint limit disabled',linestyle='--')
    ax.axhline(0,color='black',linestyle=':',label='Modeled lower joint limit')
    ax.set(xlabel='Time from replay start (s)',ylabel='mf_mcp angle (deg)',title='Joint-range sensitivity; production model unchanged')
    ax.grid(alpha=.2);ax.legend();fig.savefig(output/'mf_mcp_limit_sensitivity.png',dpi=180);plt.close(fig)
    return result


def plots(directory,experiment,selected,measured,simulated,origin):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([sample_time(r,'midpoint')-origin for r in selected])*1e-6
    for joint in dict.fromkeys(p['joint'] for p in experiment['phases']):
        phases=[p for p in experiment['phases'] if p['joint']==joint];j=phases[0]['joint_index']
        fig,axs=plt.subplots(2,2,figsize=(12,6),constrained_layout=True)
        for column,phase in enumerate(phases):
            mask=np.array([phase['started_us']<=r['read_started_us'] and r['read_finished_us']<=phase['ended_us'] for r in selected])
            x=t[mask]-(phase['started_us']-origin)*1e-6
            u=np.array([r['held_target_before_read'][j] for r in selected])[mask]
            axs[0,column].step(x,np.degrees(u),where='mid',label='Written target',alpha=.6)
            axs[0,column].plot(x,np.degrees(measured[mask,j]),label='Hardware')
            axs[0,column].plot(x,np.degrees(simulated[mask,j]),label='Calibrated model')
            axs[0,column].set(title=phase['mode'],ylabel='Joint angle (deg)');axs[0,column].legend()
            axs[1,column].plot(x,np.degrees(simulated[mask,j]-measured[mask,j]))
            axs[1,column].axhline(0,color='black',lw=.7);axs[1,column].set(xlabel='Time in phase (s)',ylabel='Model − hardware (deg)')
        for ax in axs.ravel():ax.grid(alpha=.2)
        fig.suptitle(joint+' — '+('empty-hand replay' if experiment['object_condition']=='empty_hand' else 'loaded hardware / hand-only model'))
        fig.savefig(directory/f'{joint}.png',dpi=180);fig.savefig(directory/f'{joint}.pdf');plt.close(fig)


def write_index(output,reports,limit_sensitivity=None):
    lines=['<!doctype html><meta charset="utf-8"><title>Calibrated hand replay validation</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}</style>',
        '<h1>Calibrated hand: recorded-target replay</h1><p>Offline CPU MuJoCo replay. Actual bridge-written targets and times are used. One measured initial state per recording, then continuous prediction. No fitted offset, time shift, gain or parameter update.</p>',
        '<p>Empty-hand is the direct validation. Loaded recordings are hand-only diagnostic comparisons: no initial cube poses or contact states were logged, so those cannot support a matched loaded validation.</p>']
    if limit_sensitivity is not None:
        lines+=['<h2>Joint-limit discrepancy</h2>',
            '<p>The measured mf_mcp angle is approximately −2.1°, below the model’s 0° lower joint limit. Disabling only this modeled limit in an isolated replay reduced its mean prediction error from '+
            f'{limit_sensitivity["baseline_mf_mcp_prediction_error_deg"]["mae"]:.3f}° to {limit_sensitivity["diagnostic_mf_mcp_prediction_error_deg"]["mae"]:.3f}°.'+
            ' This identifies a constraint-driven discrepancy at this pose; it does not justify removing physical limits or establish whether joint zero or range definitions are wrong.</p>',
            '<img src="mf_mcp_limit_sensitivity.png"><p><a href="mf_mcp_limit_sensitivity.json">Constraint diagnostic details</a></p>']
    for report in reports:
        name=report['run'];primary=report['alignments']['midpoint']
        lines+=['<h2>'+html.escape(name)+'</h2><p>'+html.escape(report['scope'])+'</p>',
            f'<p>All-joint prediction MAE {primary["all_joints_prediction_error_deg"]["mae"]:.3f}°, RMSE {primary["all_joints_prediction_error_deg"]["rmse"]:.3f}°. These compare simulation to hardware, not target to hardware.</p>',
            '<table><tr><th>Joint</th><th>Fast prediction MAE (deg)</th><th>Hardware lag (ms)</th><th>Model lag (ms)</th><th>Hardware amplitude ratio</th><th>Model amplitude ratio</th></tr>']
        for phase in primary['phases']:
            if phase['mode']!='fast':continue
            hw=phase['hardware_response'];sim=phase['model_response']
            lines.append(f'<tr><td>{phase["joint"]}</td><td>{phase["prediction_error_deg"]["mae"]:.3f}</td><td>{hw["lag_ms"]:.1f}</td><td>{sim["lag_ms"]:.1f}</td><td>{hw["amplitude_ratio"]:.3f}</td><td>{sim["amplitude_ratio"]:.3f}</td></tr>')
        lines+=['</table>',f'<p><a href="{name}/report.json">Full metrics and timing sensitivity</a> · <a href="{name}/joints.csv">All 16 joints</a> · <a href="{name}/trajectories.npz">Recorded and predicted trajectories</a></p>']
        steady=[p for p in primary['phases'] if p['mode']=='settled']
        lines+=['<table><tr><th>Exercised joint</th><th>Settled hardware–target MAE (deg)</th><th>Settled model–target MAE (deg)</th><th>Settled model–hardware MAE (deg)</th></tr>']
        for p in steady:
            lines.append(f'<tr><td>{p["joint"]}</td><td>{p["settled_hardware_minus_target_deg"]["mae"]:.3f}</td><td>{p["settled_model_minus_target_deg"]["mae"]:.3f}</td><td>{p["settled_model_minus_hardware_deg"]["mae"]:.3f}</td></tr>')
        lines+=['</table><p>Encoder acquisition time is uncertain within each read. The report includes start/midpoint/end sampling assumptions. Primary scores use midpoint; no best-fit timing shift is applied.</p>']
        for joint in ['if_pip','mf_pip','rf_pip','th_mcp']:lines.append(f'<h3>{joint}</h3><img src="{name}/{joint}.png">')
    (output/'index.html').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings',nargs='+',type=Path,default=DEFAULT_RUNS)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();reports=[]
    for directory in args.recordings:
        reports.append(validate_run(directory,args.output))
        (args.output/'progress.json').write_text(json.dumps(dict(completed_runs=len(reports),elapsed_s=time.monotonic()-start))+'\n')
    empty=next((directory for directory in args.recordings if json.loads((directory/'experiment.json').read_text())['object_condition']=='empty_hand'),None)
    sensitivity=diagnose_mcp_limit(empty,args.output) if empty is not None else None
    (args.output/'report.json').write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_wall_s=time.monotonic()-start,hardware_access=False,runs=reports,
        constraint_sensitivity=sensitivity),indent=2,allow_nan=False)+'\n')
    write_index(args.output,reports,sensitivity)
    print('Results:',args.output/'index.html',flush=True)


if __name__=='__main__':main()
