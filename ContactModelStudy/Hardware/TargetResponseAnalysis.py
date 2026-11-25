"""Offline analysis of the isolated LEAP target-response experiment."""
import csv
import html
import json
from pathlib import Path

import mujoco
import numpy as np


def stats(values):
    values=np.asarray(values,dtype=float)
    if not values.size:return None
    return dict(count=int(values.size),mean=float(values.mean()),
        p95=float(np.percentile(values,95)),max=float(values.max()))


def harmonic_lag(times,target,measured,update_period,samples_per_cycle=4):
    """Fundamental response to +a,0,-a,0; relative to actual held targets.

    Bias and gain are fitted separately. This descriptive periodic phase lag
    does not identify a unique motor model. Read acquisition timing is
    reported separately from the recorded read intervals.
    """
    times,target,measured=map(np.asarray,(times,target,measured))
    if len(times)<12:return None
    omega=2*np.pi/(samples_per_cycle*update_period)
    centered=times-times[0]
    design=np.column_stack([np.sin(omega*centered),np.cos(omega*centered),np.ones(len(times))])
    a=np.linalg.lstsq(design,target,rcond=None)[0]
    b=np.linalg.lstsq(design,measured,rcond=None)[0]
    target_amplitude=np.hypot(*a[:2]);response_amplitude=np.hypot(*b[:2])
    if target_amplitude<1e-5:return None
    delta=(np.arctan2(a[1],a[0])-np.arctan2(b[1],b[0])+np.pi)%(2*np.pi)-np.pi
    residual=measured-design@b
    return dict(fundamental_hz=float(omega/(2*np.pi)),lag_ms=float(delta/omega*1000),
        amplitude_ratio=float(response_amplitude/target_amplitude),
        response_fit_residual_rms_rad=float(np.sqrt(np.mean(residual**2))),
        measurement_timing='Read midpoint used; per-motor acquisition instant is unknown within each read.')


def analyze_response(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    experiment=json.loads((directory/'experiment.json').read_text())
    if not (directory/'bridge_telemetry.jsonl').exists():
        return None
    events=[json.loads(line) for line in (directory/'bridge_telemetry.jsonl').read_text().splitlines()]
    cycles=[r for r in events if r.get('event')=='cycle' and 'q' in r]
    model=mujoco.MjModel.from_xml_path(experiment['model']);data=mujoco.MjData(model)
    sites=[model.site(name).id for name in ('if_tip','mf_tip','rf_tip','th_tip')]
    def tips(q):
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        return data.site_xpos[sites].copy()
    for r in cycles:
        r['read_midpoint_us']=(r['read_started_us']+r['read_finished_us'])/2
    read_timing=stats([(r['read_finished_us']-r['read_started_us'])*.001 for r in cycles])
    feedback_timing=stats(np.diff([r['state_timestamp_us'] for r in cycles])*.001)
    timing_note=(f"Recorded mean read span {read_timing['mean']:.2f} ms; "
                 f"mean feedback interval {feedback_timing['mean']:.2f} ms."
                 if read_timing is not None and feedback_timing is not None else
                 'Insufficient feedback to summarize read timing.')
    phases=[];endpoints=[];plot_data={}
    for phase in experiment['phases']:
        if 'ended_us' not in phase:continue
        j=phase['joint_index'];finger=sites.index(model.site(phase['fingertip']).id)
        selected=[r for r in cycles if phase['started_us']<=r['read_started_us'] and r['read_finished_us']<=phase['ended_us']]
        if not selected:continue
        times=np.array([r['read_midpoint_us']*1e-6 for r in selected])
        q=np.array([r['q'][j] for r in selected]);u=np.array([r['held_target_before_read'][j] for r in selected])
        actual=np.array([tips(r['q'])[finger] for r in selected]);desired=np.array([tips(r['held_target_before_read'])[finger] for r in selected])
        tip_error=np.linalg.norm(actual-desired,axis=1)*1000
        commands=[c for c in experiment['commands'] if c.get('phase')==phase['index'] and c['kind']=='new_target']
        writes={r['written_action_timestamp_us']:r for r in cycles if r.get('write_succeeded') and r.get('new_action')}
        latencies=[(writes[c['timestamp_us']]['write_started_us']-c['timestamp_us'])*.001 for c in commands if c['timestamp_us'] in writes]
        entry=dict(**phase,frames=len(selected),joint_abs_error_deg=stats(np.degrees(abs(q-u))),
            fingertip_error_mm=stats(tip_error),publish_to_write_ms=stats(latencies),
            new_targets=len(commands),matched_writes=sum(c['timestamp_us'] in writes for c in commands))
        if phase['mode']=='fast':
            actual_period=float(np.mean(np.diff([c['timestamp_us'] for c in commands])))*1e-6 if len(commands)>1 else experiment['period_s']
            entry['periodic_response']=harmonic_lag(times,u,q,actual_period,phase.get('samples_per_cycle',4))
            entry['publication_interval_ms']=stats(np.diff([c['timestamp_us'] for c in commands])*.001)
        else:
            errors=[];tip_errors=[];settling_times=[];rises={k:[] for k in (.1,.5,.9)}
            for index,command in enumerate(commands):
                end=commands[index+1]['timestamp_us'] if index+1<len(commands) else phase['ended_us']
                tail=[r for r in selected if end-250000<=r['read_started_us'] and r['read_finished_us']<=end
                      and np.allclose(r['held_target_before_read'],command['target'],rtol=0,atol=2e-7)]
                if not tail:continue
                terminal_q=np.mean([r['q'] for r in tail],axis=0)
                terminal_tip=np.mean([tips(r['q'])[finger] for r in tail],axis=0)
                target_tip=tips(command['target'])[finger]
                gap=terminal_q[j]-command['target'][j]
                tip_gap=float(np.linalg.norm(terminal_tip-target_tip)*1000)
                errors.append(abs(gap));tip_errors.append(tip_gap)
                settling=None;crossings={}
                # The immediate identical heartbeat can supersede the initial
                # packet in the bridge's queue of length one. Match values and
                # time to find the first actual write of this step target.
                matching=[r for r in cycles if r.get('write_succeeded')
                    and command['timestamp_us']<=r['write_started_us']<end
                    and np.allclose(r['written_target'],command['target'],rtol=0,atol=2e-7)]
                if matching:
                    first=matching[0]
                    following=[r for r in selected if first['write_finished_us']<=r['read_started_us'] and r['read_finished_us']<=end]
                    initial=first['q'][j];movement=terminal_q[j]-initial
                    if abs(movement)>.01:
                        for fraction in rises:
                            crossing=next((r for r in following if (r['q'][j]-initial)/movement>=fraction),None)
                            if crossing is not None:
                                value=(crossing['read_midpoint_us']-first['write_finished_us'])*.001
                                rises[fraction].append(value);crossings[str(fraction)]=value
                    band=max(.005,.1*abs(terminal_q[j]-first['q'][j]))
                    for offset,r in enumerate(following):
                        if end-r['read_finished_us']>=200000 and all(abs(x['q'][j]-terminal_q[j])<=band for x in following[offset:]):
                            settling=(r['read_midpoint_us']-first['write_finished_us'])*.001;break
                if settling is not None:settling_times.append(settling)
                endpoints.append(dict(phase=phase['index'],joint=phase['joint'],target=command['target'],
                    measured_q=terminal_q.tolist(),target_tip_m=target_tip.tolist(),measured_tip_m=terminal_tip.tolist(),
                    joint_error_deg=float(np.degrees(gap)),fingertip_error_mm=tip_gap,settling_ms=settling,
                    observed_fractional_response_ms=crossings))
            entry['settled_joint_abs_error_deg']=stats(np.degrees(errors))
            entry['settled_fingertip_error_mm']=stats(tip_errors)
            entry['settling_to_measured_endpoint_ms']=stats(settling_times)
            entry['observed_fractional_response_ms']={str(k):stats(v) for k,v in rises.items()}
        phases.append(entry);plot_data[phase['joint'],phase['mode']]=(times-phase['started_us']*1e-6,q,u,tip_error,actual)
    # Compare fast motion directly to the measured fingertip endpoints from
    # the slow test, separating transient following error from static FK bias.
    endpoint_errors={}
    for phase in phases:
        if phase['mode']!='fast':continue
        joint=phase['joint'];j=phase['joint_index']
        previous=[e for e in endpoints if e['joint']==joint]
        if not previous:continue
        keys=sorted(set(round(e['target'][j],5) for e in previous))
        point_by_value=np.array([np.mean([e['measured_tip_m'] for e in previous
            if round(e['target'][j],5)==key],axis=0) for key in keys])
        joint_by_value=np.array([np.mean([e['measured_q'][j] for e in previous
            if round(e['target'][j],5)==key]) for key in keys])
        t,q,u,tip_error,actual=plot_data[joint,'fast']
        steady_tips=np.column_stack([np.interp(u,keys,point_by_value[:,axis]) for axis in range(3)])
        error=np.linalg.norm(actual-steady_tips,axis=1)*1000
        phase['distance_to_slow_measured_fingertip_endpoint_mm']=stats(error)
        phase['joint_gap_to_slow_measured_endpoint_deg']=stats(np.degrees(abs(q-np.interp(u,keys,joint_by_value))))
        endpoint_errors[joint]=error
    for joint in dict.fromkeys(p['joint'] for p in phases):
        fig,axs=plt.subplots(2,2,figsize=(12,6.5),constrained_layout=True)
        for column,mode in enumerate(('settled','fast')):
            if (joint,mode) not in plot_data:continue
            t,q,u,error,actual=plot_data[joint,mode]
            axs[0,column].plot(t,q,label='Measured joint')
            axs[0,column].step(t,u,where='mid',label='Previously written target',alpha=.8)
            axs[0,column].set(title=mode,ylabel='Joint position (rad)');axs[0,column].legend()
            axs[1,column].plot(t,error,label='Versus FK of target joints')
            if mode=='fast' and joint in endpoint_errors:
                axs[1,column].plot(t,endpoint_errors[joint],label='Versus measured slow endpoint')
            axs[1,column].set(xlabel='Time in phase (s)',ylabel='Fingertip target gap (mm)');axs[1,column].legend()
        for ax in axs.ravel():ax.grid(alpha=.2)
        title=f'{joint}: settled vs {experiment["rate_hz"]:.2f} Hz target updates'
        if experiment.get('waveform')=='sine':
            title+=f'; sine {experiment["rate_hz"]/experiment["sine_samples"]:.2f} Hz'
        fig.suptitle(title+'; encoder-based fingertip FK')
        fig.savefig(directory/f'{joint}_response.png',dpi=180);fig.savefig(directory/f'{joint}_response.pdf');plt.close(fig)
    observation_path=directory/'operator_observation.json'
    observation=json.loads(observation_path.read_text()) if observation_path.exists() else None
    report=dict(status=experiment['status'],rate_hz=experiment['rate_hz'],phases=phases,endpoints=endpoints,
        read_span_ms=read_timing,feedback_interval_ms=feedback_timing,
        object_condition=experiment.get('object_condition','empty_hand'),
        excitation=dict(amplitude_rad=experiment['amplitude_rad'],closing_only=experiment.get('closing_only',False),
                        cycles=experiment['cycles']),operator_observation=observation,
        bridge_errors=[r['error'] for r in cycles if r.get('error')],
        torque_off_verified=experiment.get('torque_off_verified'),
        caveats=['Fingertip positions use encoder feedback and calibrated FK, not optical measurements.',
            'The same absolute target sequences are used in both conditions; no measured-position rebasing or MPPI.',
            'Periodic phase lag is relative to held bridge targets; publish-to-write latency is reported separately.',
            timing_note+' Fitted lags are descriptive, not submillisecond measurements. Samples per input cycle are recorded per phase.',
            'Settling is relative to each measured terminal endpoint; residual error to the requested target is reported separately.',
            ('This test uses a cube grasp; comparison of load effects requires an empty-hand control with the same waveform, range and frequency.'
             if experiment.get('object_condition')=='cube' else
             'This empty-hand test characterizes unloaded following; object contact can change the response.')])
    (directory/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    comparison=[]
    def mean_or_none(value):
        return None if value is None else value['mean']
    for joint in dict.fromkeys(p['joint'] for p in phases):
        slow=next((p for p in phases if p['joint']==joint and p['mode']=='settled'),None)
        fast=next((p for p in phases if p['joint']==joint and p['mode']=='fast'),None)
        if slow is None or fast is None:continue
        comparison.append(dict(joint=joint,
            settled_mean_joint_error_deg=slow['settled_joint_abs_error_deg']['mean'],
            fast_mean_joint_error_deg=fast['joint_abs_error_deg']['mean'],
            fast_phase_lag_ms=(fast.get('periodic_response') or {}).get('lag_ms'),
            fast_amplitude_ratio=(fast.get('periodic_response') or {}).get('amplitude_ratio'),
            fast_mean_fingertip_gap_to_settled_endpoint_mm=fast['distance_to_slow_measured_fingertip_endpoint_mm']['mean'],
            write_to_10pct_response_ms=mean_or_none(slow['observed_fractional_response_ms']['0.1']),
            write_to_90pct_response_ms=mean_or_none(slow['observed_fractional_response_ms']['0.9']),
            fast_mean_publish_to_write_ms=fast['publish_to_write_ms']['mean'],
            achieved_fast_update_rate_hz=1000/fast['publication_interval_ms']['mean']))
    if comparison:
        with (directory/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    condition='Cube-grasp' if experiment.get('object_condition')=='cube' else 'Empty-hand'
    movement=f'closing offsets 0..{experiment["amplitude_rad"]:.2f}' if experiment.get('closing_only') else f'±{experiment["amplitude_rad"]:.2f}'
    body=['<!doctype html><meta charset="utf-8"><title>LEAP target response</title>',
        '<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}</style>',
        f'<h1>{condition} target-following experiment</h1>',
        f'<p>Status: {html.escape(report["status"])}. Torque-off verified: {report["torque_off_verified"]}.</p>',
        f'<p>One fingertip flexion joint at a time, {movement} rad. Target samples were held at least one second until quiet, then repeated at approximately {experiment["rate_hz"]:.2f} updates per second. Waveform: {experiment.get("waveform","steps")}; {experiment.get("sine_samples",16) if experiment.get("waveform")=="sine" else 4} updates per input cycle.</p>',
        '<table><tr><th>Joint</th><th>Settled error (deg)</th><th>Fast error (deg)</th><th>Periodic phase lag (ms)</th><th>Fast tip gap to measured slow endpoint (mm)</th></tr>']
    for row in comparison:
        body.append('<tr><td>'+row['joint']+'</td>'+''.join('<td>'+('n/a' if row[k] is None else f'{row[k]:.2f}')+'</td>' for k in [
            'settled_mean_joint_error_deg','fast_mean_joint_error_deg','fast_phase_lag_ms',
            'fast_mean_fingertip_gap_to_settled_endpoint_mm'])+'</tr>')
    body+=['</table><p>The fitted phase lag describes command-to-encoder behavior relative to targets already written by the bridge. Publish-to-write delay is separate. It cannot split physical motor response from internal encoder/reporting latency. '+timing_note+'</p>']
    if observation is not None:
        body.append('<p>Operator observation: '+html.escape(observation['user_text'])+'</p>')
    for row in comparison:body.append(f'<h2>{row["joint"]}</h2><img src="{row["joint"]}_response.png">')
    body+=['<h2>Scope and limitations</h2><ul>']+['<li>'+html.escape(c)+'</li>' for c in report['caveats']]
    body+=['</ul><p><a href="comparison.csv">Comparison CSV</a> · <a href="summary.json">Full analysis</a> · <a href="experiment.json">Commands and experiment metadata</a></p>']
    (directory/'index.html').write_text('\n'.join(body))
    print('Response analysis:',directory/'summary.json',flush=True)
    return report
